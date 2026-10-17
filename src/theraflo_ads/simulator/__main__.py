"""Run the simulator: python -m theraflo_ads.simulator --users 3000 --days 14"""

from __future__ import annotations

import argparse

from theraflo_ads.simulator.report import format_report
from theraflo_ads.simulator.runner import run_simulation
from theraflo_ads.simulator.world import SimConfig


def main(argv: list[str] | None = None) -> None:
    defaults = SimConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=defaults.n_users)
    parser.add_argument("--days", type=int, default=defaults.n_days)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--epsilon", type=float, default=defaults.epsilon)
    parser.add_argument("--no-ads-holdout", type=float, default=defaults.no_ads_holdout_share)
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="disable fatigue, churn, caps, and budgets (the setting where OPE is unbiased)",
    )
    args = parser.parse_args(argv)
    config = SimConfig(
        n_users=args.users,
        n_days=args.days,
        seed=args.seed,
        epsilon=args.epsilon,
        no_ads_holdout_share=args.no_ads_holdout,
        stateless=args.stateless,
    )
    result = run_simulation(config)
    print(format_report(result))
    # Hard-constraint violations fail the run, so CI can gate on it.
    if any(result.violations.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
