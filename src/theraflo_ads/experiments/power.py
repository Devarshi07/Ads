"""Power, sample size, and MDE (design doc: Statistical Testing → Minimum Detectable Effect).

All proportion calculations are for a two-sided two-proportion z-test with
separate (unpooled) arm variances. The design doc's table uses the cruder
2·p(1−p) approximation, so these figures run slightly higher, increasingly so
at large lifts; both are lower bounds once user-level clustering is counted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class MDERow:
    relative_lift: float
    units_per_arm: int
    days: tuple[float, ...]


def _check_unit_interval(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value}")


def _z_sum_squared(alpha: float, power: float) -> float:
    _check_unit_interval("alpha", alpha)
    _check_unit_interval("power", power)
    return (norm.isf(alpha / 2) + norm.ppf(power)) ** 2


def _exact_units(p_baseline: float, relative_lift: float, z2: float, ratio: float) -> float:
    p1 = p_baseline
    p2 = p_baseline * (1.0 + relative_lift)
    variance = p1 * (1.0 - p1) + p2 * (1.0 - p2) / ratio
    return z2 * variance / (p2 - p1) ** 2


def sample_size_two_proportions(
    p_baseline: float,
    relative_lift: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
) -> int:
    """Control-arm units needed; the treatment arm needs ``ratio`` times as many."""
    _check_unit_interval("p_baseline", p_baseline)
    if relative_lift == 0.0:
        raise ValueError("relative_lift must be non-zero")
    p_treatment = p_baseline * (1.0 + relative_lift)
    if not 0.0 < p_treatment < 1.0:
        raise ValueError(f"treatment rate {p_treatment} is outside (0, 1)")
    if ratio <= 0.0:
        raise ValueError("ratio must be positive")
    z2 = _z_sum_squared(alpha, power)
    return math.ceil(_exact_units(p_baseline, relative_lift, z2, ratio))


def mde_two_proportions(
    p_baseline: float, n_per_arm: float, *, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest positive relative lift detectable with ``n_per_arm`` units in each arm."""
    _check_unit_interval("p_baseline", p_baseline)
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    z2 = _z_sum_squared(alpha, power)
    # Required units fall monotonically as the lift grows, so one root exists if any does.
    lowest = 1e-12
    highest = (1.0 - p_baseline) / p_baseline * (1.0 - 1e-12)
    if _exact_units(p_baseline, highest, z2, 1.0) > n_per_arm:
        raise ValueError(f"no lift is detectable with {n_per_arm} units per arm")
    return float(
        brentq(
            lambda lift: _exact_units(p_baseline, lift, z2, 1.0) - n_per_arm,
            lowest,
            highest,
            xtol=1e-14,
            rtol=1e-12,
        )
    )


def days_to_detect(
    p_baseline: float,
    relative_lift: float,
    daily_units_total: float,
    *,
    split: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Days until both arms reach the required size; ``split`` is the treatment share."""
    if daily_units_total <= 0:
        raise ValueError("daily_units_total must be positive")
    _check_unit_interval("split", split)
    control_share = 1.0 - split
    n_control = sample_size_two_proportions(
        p_baseline, relative_lift, alpha=alpha, power=power, ratio=split / control_share
    )
    return n_control / (daily_units_total * control_share)


def sample_size_mean(sd: float, mde_abs: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """Units per arm for a two-sided test on a difference in means with equal arm SDs."""
    if sd <= 0:
        raise ValueError("sd must be positive")
    if mde_abs == 0:
        raise ValueError("mde_abs must be non-zero")
    z2 = _z_sum_squared(alpha, power)
    return math.ceil(2.0 * z2 * sd**2 / mde_abs**2)


def mde_table(
    p_baseline: float,
    lifts: Sequence[float],
    daily_units_options: Sequence[float],
    *,
    split: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
) -> list[MDERow]:
    """One row per lift: control-arm units and days at each daily-traffic option."""
    _check_unit_interval("split", split)
    ratio = split / (1.0 - split)
    return [
        MDERow(
            relative_lift=lift,
            units_per_arm=sample_size_two_proportions(
                p_baseline, lift, alpha=alpha, power=power, ratio=ratio
            ),
            days=tuple(
                days_to_detect(p_baseline, lift, daily, split=split, alpha=alpha, power=power)
                for daily in daily_units_options
            ),
        )
        for lift in lifts
    ]
