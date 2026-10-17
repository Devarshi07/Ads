"""Chaos scenarios against the docker-compose stack (design doc: Failure Scenarios A-J).

Starts the API (infra backend) and the outbox relay as subprocesses, then stops
and restarts real containers and checks the documented behaviour:

    redis down      → ads stop (fail closed), readiness 503; after restart, ads resume
    rabbitmq down   → serving and events unaffected (outbox); after restart the relay drains it
    postgres down   → no ad is served (503 or NO_AD); after restart, ads resume
    kill switch     → every process stops serving ads within the cache window

Every stopped container is started again in a `finally` block. Needs Docker, the
compose services running and migrated (`theraflo-ads migrate && theraflo-ads seed-demo`),
and THERAFLO_ADS_{TOKEN,USER_HASH,SESSION}_KEY in the environment.

    python scripts/chaos.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy import create_engine, text

from theraflo_ads.api.container import DEV_ADMIN_TOKENS
from theraflo_ads.experiments.assignment import Holdout, HoldoutConfig
from theraflo_ads.infra.settings import Settings
from theraflo_ads.security.identifiers import UserHasher
from theraflo_ads.security.sessions import SessionTokens

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("CHAOS_PORT", "8767"))
BASE = f"http://127.0.0.1:{PORT}"
ADMIN = {"Authorization": f"Bearer {next(iter(DEV_ADMIN_TOKENS))}"}


def compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], cwd=ROOT, check=True, capture_output=True)


class Users:
    """A fresh user per check (frequency caps), never one in a permanent holdout."""

    def __init__(self) -> None:
        self._sessions = SessionTokens(bytes.fromhex(os.environ["THERAFLO_ADS_SESSION_KEY"]))
        key = bytes.fromhex(os.environ["THERAFLO_ADS_USER_HASH_KEY"])
        self._hasher = UserHasher("ads", {"k1": key}, "k1")

    def headers(self) -> dict[str, str]:
        while True:
            user = f"chaos-{uuid.uuid4().hex[:12]}"
            if HoldoutConfig().assign(self._hasher.hash(user)) is Holdout.NONE:
                token = self._sessions.issue(user, datetime.now(UTC))
                return {"Authorization": f"Bearer {token}"}


def rank(client: httpx.Client, users: Users) -> httpx.Response:
    return client.post(
        "/v1/ads/rank",
        json={"session_id": "chaos", "placement": "home_feed"},
        headers=users.headers(),
    )


def serves_ad(client: httpx.Client, users: Users) -> bool:
    try:
        response = rank(client, users)
    except httpx.HTTPError:
        return False
    return response.status_code == 200 and bool(response.json()["ads"])


def wait_until(predicate, timeout: float = 90.0, interval: float = 0.5) -> float | None:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if predicate():
            return round(time.monotonic() - started, 1)
        time.sleep(interval)
    return None


def unpublished(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM outbox WHERE published_at IS NULL")).scalar()


@contextmanager
def process(*args: str):
    env = dict(os.environ, THERAFLO_ADS_BACKEND="infra", THERAFLO_ADS_RATE_LIMITS="off")
    proc = subprocess.Popen(
        [sys.executable, "-m", *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield proc
    finally:
        proc.terminate()
        proc.wait(timeout=20)


@contextmanager
def stopped(service: str):
    compose("stop", service)
    try:
        yield
    finally:
        compose("start", service)


def scenario(results: list, name: str, checks: dict, **extra) -> None:
    passed = all(checks.values())
    results.append({"scenario": name, "passed": passed, "checks": checks, **extra})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {json.dumps(checks)} {json.dumps(extra)}")


def main() -> int:
    users = Users()
    engine = create_engine(Settings().database_url, connect_args={"timeout": 5})
    results: list[dict] = []
    api = (
        "uvicorn",
        "--factory",
        "theraflo_ads.api.app:create_app",
        "--port",
        str(PORT),
        "--log-level",
        "warning",
    )
    with (
        process(*api),
        process("theraflo_ads.cli", "relay"),
        httpx.Client(base_url=BASE, timeout=5.0) as client,
    ):
        if wait_until(lambda: _ready(client), timeout=60) is None:
            print("API did not become ready")
            return 2
        scenario(results, "baseline", {"serves_ads": serves_ad(client, users)})

        with stopped("redis"):
            time.sleep(1)
            response = rank(client, users)
            checks = {
                "rank_ok": response.status_code == 200,
                "no_ads_fail_closed": response.status_code == 200 and not response.json()["ads"],
                "readiness_503": client.get("/readyz").status_code == 503,
            }
        recovery = wait_until(lambda: serves_ad(client, users))
        checks["recovered"] = recovery is not None
        scenario(results, "redis down", checks, recovery_s=recovery)

        with stopped("rabbitmq"):
            time.sleep(1)
            before = unpublished(engine)
            checks = {"serves_ads": serves_ad(client, users)}
            for _ in range(5):
                serves_ad(client, users)
            checks["outbox_buffers"] = unpublished(engine) > before
        drained = wait_until(lambda: unpublished(engine) == 0, timeout=120)
        checks["outbox_drained"] = drained is not None
        scenario(results, "rabbitmq down", checks, drain_s=drained)

        with stopped("postgres"):
            time.sleep(1)
            try:
                response = rank(client, users)
                no_ad = response.status_code == 503 or (
                    response.status_code == 200 and not response.json()["ads"]
                )
                status = response.status_code
            except httpx.HTTPError:
                no_ad, status = True, "timeout"
            checks = {"no_ad_served": no_ad}
        engine.dispose()
        recovery = wait_until(lambda: serves_ad(client, users), timeout=120)
        checks["recovered"] = recovery is not None
        scenario(results, "postgres down", checks, status=status, recovery_s=recovery)

        body = {"enabled": True, "reason": "chaos drill"}
        client.put("/v1/admin/kill-switch", json=body, headers=ADMIN)
        stop = wait_until(lambda: not serves_ad(client, users), timeout=10, interval=0.2)
        client.put(
            "/v1/admin/kill-switch", json={"enabled": False, "reason": "drill over"}, headers=ADMIN
        )
        resume = wait_until(lambda: serves_ad(client, users), timeout=10, interval=0.2)
        scenario(
            results,
            "kill switch",
            {"stops": stop is not None, "resumes": resume is not None},
            stop_s=stop,
            resume_s=resume,
        )

    out = ROOT / "reports" / "chaos.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(r["passed"] for r in results) else 1


def _ready(client: httpx.Client) -> bool:
    try:
        return client.get("/readyz").status_code == 200
    except httpx.HTTPError:
        return False


if __name__ == "__main__":
    sys.exit(main())
