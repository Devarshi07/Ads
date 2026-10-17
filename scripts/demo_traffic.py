"""Realistic demo traffic for a running Theraflo Ads API, so the dashboards have something to
show. Leave it running and watch Grafana (http://localhost:3000) or the dev console
(http://localhost:8000/ui), whose Overview page can also start and stop this traffic.

    .venv/Scripts/python.exe scripts/demo_traffic.py               # 10 minutes, ~8 req/s
    .venv/Scripts/python.exe scripts/demo_traffic.py --minutes 0   # until Ctrl+C

What it sends is described in `theraflo_ads.devtools.traffic`: feed and ad-free screens,
viewable events, clicks, conversions, hidden ads, crisis suppressions, clean and violating
creatives, and one burst a minute that the rate limiter cuts off.

Users get session tokens minted with THERAFLO_ADS_SESSION_KEY, read from the environment or
from .env (written by `theraflo-ads gen-keys --out .env`). Without one, it falls back to the
three dev users, who hit their frequency caps quickly. Dev credentials only: local use.
"""

from __future__ import annotations

import argparse
import threading
import time

import httpx

from theraflo_ads.devtools.traffic import Demo, Schedule, dev_users, run, session_key, session_users


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--minutes", type=float, default=10.0, help="0 = until Ctrl+C")
    parser.add_argument("--rate", type=float, default=8.0, help="screens per second")
    parser.add_argument("--population", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    key = session_key()
    users = session_users(key, args.population) if key is not None else dev_users()
    demo = Demo(args.base_url, users, args.seed)
    try:
        ready = demo.client.get("/readyz")
    except httpx.HTTPError as exc:
        raise SystemExit(f"cannot reach {args.base_url}: {exc}. Is the API running?") from None
    if ready.status_code != 200:
        raise SystemExit(f"{args.base_url}/readyz says {ready.text}")
    mode = (
        f"{len(users)} users (session tokens)"
        if key is not None
        else "3 dev users (no THERAFLO_ADS_SESSION_KEY found)"
    )
    print(f"sending traffic to {args.base_url} as {mode}; Ctrl+C to stop")

    def report(elapsed: float, s: dict[str, int]) -> None:
        print(
            f"[{elapsed:5.0f}s] screens {s.get('rank', 0)}  "
            f"ads {s.get('ad_served', 0)}  no-ad {s.get('no_ad', 0)}  "
            f"viewable {s.get('ad_viewable', 0)}  clicks {s.get('click', 0)}  "
            f"conversions {s.get('conversion', 0)}  creatives "
            f"{s.get('creative', 0)}  rate-limited {s.get('rate_limited', 0)}",
            flush=True,
        )

    deadline = time.monotonic() + args.minutes * 60 if args.minutes > 0 else float("inf")
    try:
        run(
            demo, Schedule(rate=args.rate), deadline=deadline, stop=threading.Event(), report=report
        )
    except KeyboardInterrupt:
        print("stopping")
    print({k: v for k, v in sorted(demo.snapshot().items())})


if __name__ == "__main__":
    main()
