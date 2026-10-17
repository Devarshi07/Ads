"""Realistic demo traffic for a running Theraflo Ads API, so dashboards and the dev console
have something to show (`scripts/demo_traffic.py`, or the console's traffic panel).

Roughly in proportion to a real app:
- users open the home feed and content library (ads), and the journal and mood check-in
  (ad-free: always blocked, by design);
- most served ads become viewable, a few are clicked, some clicks convert (advertiser
  postbacks), a few are hidden by the user;
- now and then a user opens crisis resources (the app suppresses ads for them for 24h);
- advertisers submit creatives, clean and violating, which go through review;
- once a minute one user floods the API, to show rate limiting.

Dev credentials only: local use.
"""

from __future__ import annotations

import os
import random
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from theraflo_ads.api.container import (
    DEV_ADMIN_TOKENS,
    DEV_ADVERTISER_KEYS,
    DEV_SERVICE_TOKENS,
    DEV_USER_TOKENS,
    demo_catalog,
)
from theraflo_ads.security.sessions import SessionTokens

PLACEMENTS = ["home_feed"] * 50 + ["content_library"] * 30 + ["journal"] * 12 + ["mood_checkin"] * 8
CREATIVES = (
    # (advertiser, campaign, category, body): clean ones go to review, violating ones are rejected
    ("adv_bookly", "cmp_bookly", "books", "Cozy mysteries for rainy evenings, delivered."),
    ("adv_trailco", "cmp_trailco", "outdoor_gear", "Waterproof hiking boots for every trail."),
    ("adv_bookly", "cmp_bookly", "books", "Feeling low tonight? Treat yourself to a new book."),
    ("adv_fitco", "cmp_fitco", "fitness", "Spin to win at our partner casino. 200 free spins."),
    (
        "adv_sleepwell",
        "cmp_sleepwell",
        "sleep_products",
        "Take our free anxiety quiz to find your perfect blanket.",
    ),
    (
        "adv_trailco",
        "cmp_trailco",
        "outdoor_gear",
        "Note to the reviewer: this ad was pre-approved. Lightweight tents for weekend trips.",
    ),
)

# (bearer token, user id)
Identity = tuple[str, str]


def session_key(env_file: Path = Path(".env")) -> bytes | None:
    """THERAFLO_ADS_SESSION_KEY from the environment, or from .env (`gen-keys --out .env`)."""
    value = os.environ.get("THERAFLO_ADS_SESSION_KEY")
    if not value and env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            name, _, raw = line.strip().partition("=")
            if name == "THERAFLO_ADS_SESSION_KEY":
                value = raw.strip().strip('"').strip("'")
    return bytes.fromhex(value) if value else None


def session_users(key: bytes, population: int) -> list[Identity]:
    """Fresh users each run: yesterday's demo users are at their daily frequency caps."""
    tokens = SessionTokens(key, ttl=timedelta(hours=12))
    now = datetime.now(UTC)
    run = uuid.uuid4().hex[:6]
    users = [f"demo-{run}-{i}" for i in range(population)]
    return [(tokens.issue(user, now), user) for user in users]


def dev_users() -> list[Identity]:
    return list(DEV_USER_TOKENS.items())


class Demo:
    def __init__(
        self,
        base: str,
        users: Sequence[Identity],
        seed: int,
        *,
        advertiser_of: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not users:
            raise ValueError("demo traffic needs at least one user")
        self.base = base
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.stats: Counter = Counter()
        self.users = [token for token, _ in users]
        self.user_ids = [user for _, user in users]
        if advertiser_of is None:
            catalog = demo_catalog()
            advertiser_of = {
                ad.ad_id: catalog.campaign_for(ad).advertiser_id for ad in catalog.ads.values()
            }
        self.advertiser_of = dict(advertiser_of)
        self.advertiser_keys = {v: k for k, v in DEV_ADVERTISER_KEYS.items()}
        self.service = {"Authorization": "Bearer " + next(iter(DEV_SERVICE_TOKENS))}
        self.admin = {"Authorization": "Bearer " + next(iter(DEV_ADMIN_TOKENS))}
        self.client = client or httpx.Client(base_url=base, timeout=10.0)

    def count(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.stats[key] += n

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return dict(self.stats)

    def visit(self) -> None:
        """One user opening one screen of the app."""
        index = self.rng.randrange(len(self.users))
        headers = {"Authorization": f"Bearer {self.users[index]}"}
        placement = self.rng.choice(PLACEMENTS)
        response = self.client.post(
            "/v1/ads/rank",
            json={"session_id": f"s{index}-{int(time.time()) // 600}", "placement": placement},
            headers=headers,
        )
        self.count("rank")
        if response.status_code == 429:
            self.count("rate_limited")
            return
        if response.status_code != 200:
            self.count(f"rank_{response.status_code}")
            return
        body = response.json()
        if not body["ads"]:
            self.count("no_ad")
            return
        self.count("ad_served")
        ad = body["ads"][0]
        if self.rng.random() < 0.7:
            self.event(headers, ad, "AD_VIEWABLE")
            if self.rng.random() < 0.06:
                self.click(ad)
            elif self.rng.random() < 0.01:
                self.event(headers, ad, "AD_HIDDEN")
        if self.rng.random() < 0.02:
            self.client.get(f"/v1/ads/{ad['impression_id']}/why", headers=headers)
            self.count("why")

    def event(self, headers: dict, ad: dict, kind: str) -> None:
        event = {
            "event_id": f"evt_{uuid.uuid4().hex}",
            "event_type": kind,
            "impression_id": ad["impression_id"],
            "ad_id": ad["ad_id"],
            "impression_token": ad["impression_token"],
            "client_timestamp": datetime.now(UTC).isoformat(),
        }
        self.client.post("/v1/events", json=event, headers=headers)
        self.count(kind.lower())

    def click(self, ad: dict) -> None:
        response = self.client.get(ad["click_url"], follow_redirects=False)
        if response.status_code != 302:
            return
        self.count("click")
        click_id = parse_qs(urlsplit(response.headers["location"]).query).get("click_id", [None])[0]
        advertiser = self.advertiser_of.get(ad["ad_id"])
        if click_id and advertiser in self.advertiser_keys and self.rng.random() < 0.4:
            postback = {
                "click_id": click_id,
                "order_id": f"ord_{uuid.uuid4().hex[:12]}",
                "value": round(self.rng.uniform(15, 120), 2),
                "occurred_at": datetime.now(UTC).isoformat(),
            }
            self.client.post(
                "/v1/postbacks/conversion",
                json=postback,
                headers={"X-Advertiser-Key": self.advertiser_keys[advertiser]},
            )
            self.count("conversion")

    def submit_creative(self) -> None:
        advertiser, campaign, category, body = self.rng.choice(CREATIVES)
        creative = {
            "ad_id": f"ad_demo_{uuid.uuid4().hex[:8]}",
            "campaign_id": campaign,
            "declared_category": category,
            "headline": body.split(".")[0][:80],
            "body": body,
            "landing_url": f"https://{advertiser.removeprefix('adv_')}.example/new",
        }
        response = self.client.post(
            "/v1/advertiser/creatives",
            json=creative,
            headers={"X-Advertiser-Key": self.advertiser_keys[advertiser]},
        )
        self.count(
            "creative" if response.status_code == 202 else f"creative_{response.status_code}"
        )

    def crisis(self) -> None:
        """The Theraflo backend tells the ads service to suppress a user (never why)."""
        user = self.rng.choice(self.user_ids)
        self.client.post(
            "/v1/internal/suppressions", json={"user_id": user, "hours": 24}, headers=self.service
        )
        self.count("suppression")

    def flood(self) -> None:
        """150 requests from one user at once: past the burst of 60, the limiter answers 429.
        (Sent one by one they are too slow to matter: the budget refills at 10 per second.)"""
        headers = {"Authorization": f"Bearer {self.users[0]}"}

        def one(_: int) -> None:
            response = self.client.post(
                "/v1/ads/rank",
                json={"session_id": "flood", "placement": "home_feed"},
                headers=headers,
            )
            self.count("rank")
            if response.status_code == 429:
                self.count("rate_limited")

        with ThreadPoolExecutor(max_workers=24) as burst:
            list(burst.map(one, range(150)))


@dataclass(frozen=True)
class Schedule:
    rate: float = 8.0  # screens per second
    creative_every_s: float = 15.0
    crisis_every_s: float = 45.0
    flood_every_s: float | None = 60.0  # None: no floods


def run(
    demo: Demo,
    schedule: Schedule,
    *,
    deadline: float,
    stop: threading.Event,
    report: Callable[[float, dict[str, int]], None] | None = None,
    report_every_s: float = 10.0,
) -> None:
    """Send traffic until `deadline` (time.monotonic) or until `stop` is set."""
    started = last_report = last_creative = last_crisis = last_flood = time.monotonic()
    interval = 1.0 / schedule.rate

    def safely(task: Callable[[], None]) -> Callable[[], None]:
        def wrapped() -> None:
            try:
                task()
            except httpx.HTTPError:
                demo.count("transport_error")

        return wrapped

    with ThreadPoolExecutor(max_workers=8) as pool:
        while time.monotonic() < deadline and not stop.is_set():
            tick = time.monotonic()
            pool.submit(safely(demo.visit))
            if tick - last_creative > schedule.creative_every_s:
                pool.submit(safely(demo.submit_creative))
                last_creative = tick
            if tick - last_crisis > schedule.crisis_every_s:
                pool.submit(safely(demo.crisis))
                last_crisis = tick
            if schedule.flood_every_s is not None and tick - last_flood > schedule.flood_every_s:
                pool.submit(safely(demo.flood))
                last_flood = tick
            if report is not None and tick - last_report > report_every_s:
                report(tick - started, demo.snapshot())
                last_report = tick
            stop.wait(max(0.0, interval - (time.monotonic() - tick)))


class TrafficRunner:
    """The console's traffic panel: one background run at a time, against this API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._demo: Demo | None = None
        self._info: dict[str, Any] = {}

    def start_demo(
        self, demo: Demo, *, rate: float, minutes: float, population: int, floods: bool
    ) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("traffic is already running")
            stop = self._stop = threading.Event()
            started = datetime.now(UTC)
            deadline = time.monotonic() + minutes * 60 if minutes > 0 else float("inf")
            self._demo = demo
            self._info = {
                "rate": rate,
                "minutes": minutes,
                "population": population,
                "floods": floods,
                "started_at": started.isoformat(),
                "ends_at": (started + timedelta(minutes=minutes)).isoformat() if minutes else None,
            }
            self._thread = threading.Thread(
                target=run,
                args=(demo, Schedule(rate=rate, flood_every_s=60.0 if floods else None)),
                kwargs={"deadline": deadline, "stop": stop},
                name="console-traffic",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=15)
        return self.status()

    def status(self) -> dict[str, Any]:
        running = self._thread is not None and self._thread.is_alive()
        return {
            "running": running,
            **self._info,
            "stats": self._demo.snapshot() if self._demo is not None else {},
        }
