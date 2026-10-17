"""Load test for the serving path (design doc: Latency Budget, Capacity Planning).

Virtual users (concurrent clients) loop: rank → (70%) viewable event → (2%) click,
each request as a user drawn from a population (so frequency caps behave like
production instead of capping a handful of users). Session tokens are minted with
THERAFLO_ADS_SESSION_KEY, which must match the server's.

    THERAFLO_ADS_SESSION_KEY=<hex> THERAFLO_ADS_RATE_LIMITS=off theraflo-ads serve
    THERAFLO_ADS_SESSION_KEY=<hex> python scripts/loadtest.py --users 50 --duration 30

Exits non-zero when the rank p99 or the error rate misses its SLO.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta

import httpx

from theraflo_ads.security.sessions import SessionTokens

PLACEMENTS = ("home_feed", "home_feed", "content_library")


class Stats:
    def __init__(self) -> None:
        self.latency: dict[str, list[float]] = defaultdict(list)
        self.status: Counter = Counter()
        self.served_by: Counter = Counter()
        self.errors: Counter = Counter()

    def record(self, endpoint: str, seconds: float, status: int) -> None:
        self.latency[endpoint].append(seconds * 1000)
        self.status[f"{endpoint}:{status}"] += 1


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[index]


async def virtual_user(
    client: httpx.AsyncClient, tokens: list[str], deadline: float, stats: Stats, rng: random.Random
) -> None:
    while time.perf_counter() < deadline:
        headers = {"Authorization": f"Bearer {rng.choice(tokens)}"}
        body = {"session_id": f"s{rng.randrange(10**6)}", "placement": rng.choice(PLACEMENTS)}
        started = time.perf_counter()
        try:
            response = await client.post("/v1/ads/rank", json=body, headers=headers)
        except httpx.HTTPError as exc:
            stats.errors[type(exc).__name__] += 1
            continue
        stats.record("rank", time.perf_counter() - started, response.status_code)
        if response.status_code != 200:
            await asyncio.sleep(0.05)
            continue
        ranked = response.json()
        stats.served_by[ranked["served_by"]] += 1
        for ad in ranked["ads"]:
            if rng.random() < 0.7:
                event = {
                    "event_id": f"evt_{uuid.uuid4().hex}",
                    "event_type": "AD_VIEWABLE",
                    "impression_id": ad["impression_id"],
                    "ad_id": ad["ad_id"],
                    "impression_token": ad["impression_token"],
                    "client_timestamp": datetime.now(UTC).isoformat(),
                }
                started = time.perf_counter()
                r = await client.post("/v1/events", json=event, headers=headers)
                stats.record("events", time.perf_counter() - started, r.status_code)
            if rng.random() < 0.02:
                started = time.perf_counter()
                r = await client.get(ad["click_url"], follow_redirects=False)
                stats.record("click", time.perf_counter() - started, r.status_code)
        await asyncio.sleep(rng.uniform(0.0, 0.02))


async def run(args: argparse.Namespace) -> dict:
    key = os.environ.get("THERAFLO_ADS_SESSION_KEY")
    if not key:
        raise SystemExit("set THERAFLO_ADS_SESSION_KEY to the server's session key (hex)")
    sessions = SessionTokens(bytes.fromhex(key), ttl=timedelta(hours=2))
    now = datetime.now(UTC)
    tokens = [sessions.issue(f"load-user-{i}", now) for i in range(args.population)]
    stats = Stats()
    limits = httpx.Limits(max_connections=args.users, max_keepalive_connections=args.users)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=5.0, limits=limits) as client:
        (await client.get("/readyz")).raise_for_status()
        started = time.perf_counter()
        deadline = started + args.duration
        await asyncio.gather(
            *(
                virtual_user(client, tokens, deadline, stats, random.Random(i))
                for i in range(args.users)
            )
        )
        elapsed = time.perf_counter() - started
    total = sum(len(v) for v in stats.latency.values())
    failures = sum(n for k, n in stats.status.items() if int(k.split(":")[1]) >= 500)
    failures += sum(stats.errors.values())
    report = {
        "concurrency": args.users,
        "population": args.population,
        "duration_s": round(elapsed, 1),
        "requests": total,
        "throughput_rps": round(total / elapsed, 1),
        "rank_rps": round(len(stats.latency["rank"]) / elapsed, 1),
        "latency_ms": {
            endpoint: {
                "p50": round(percentile(values, 50), 2),
                "p95": round(percentile(values, 95), 2),
                "p99": round(percentile(values, 99), 2),
                "mean": round(statistics.fmean(values), 2) if values else 0.0,
            }
            for endpoint, values in stats.latency.items()
        },
        "status": dict(stats.status),
        "client_errors": dict(stats.errors),
        "error_rate": round(failures / max(total, 1), 4),
        "served_by": dict(stats.served_by),
    }
    rank_p99 = report["latency_ms"].get("rank", {}).get("p99", float("inf"))
    report["slo"] = {
        "rank_p99_ms": args.slo_p99_ms,
        "max_error_rate": args.max_error_rate,
        "passed": rank_p99 <= args.slo_p99_ms and report["error_rate"] <= args.max_error_rate,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=50, help="concurrent virtual users")
    parser.add_argument("--population", type=int, default=5000, help="distinct app users")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--slo-p99-ms", type=float, default=100.0)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--out", default="reports/loadtest.json")
    args = parser.parse_args()
    report = asyncio.run(run(args))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["slo"]["passed"] else 1)


if __name__ == "__main__":
    main()
