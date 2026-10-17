"""Feature and prediction drift (design doc: Data Drift, Prediction Drift, Drift Pitfalls).

At this scale a drift alert needs three things at once: a window large enough
to trust, an effect size above a threshold bootstrapped from the reference,
and statistical significance. Any one alone pages people for noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.spatial import distance

_PSI_EPS = 1e-4


class SeasonalMode(StrEnum):
    SAME_WEEKDAY = "same_weekday"
    SAME_PERIOD_LAST_YEAR = "same_period_last_year"


@dataclass(frozen=True)
class DriftAlertPolicy:
    effect_threshold: float
    alpha: float = 0.01
    min_samples: int = 2000

    def __post_init__(self) -> None:
        if self.effect_threshold < 0:
            raise ValueError("effect_threshold must be non-negative")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.min_samples < 0:
            raise ValueError("min_samples must be non-negative")


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    p_value: float
    n_reference: int
    n_current: int
    threshold: float
    alert: bool
    reason: str


def psi(
    reference: ArrayLike, current: ArrayLike, *, bins: int = 10, eps: float = _PSI_EPS
) -> float:
    ref = _as_sample(reference, "reference")
    cur = _as_sample(current, "current")
    if np.unique(ref).size < bins:
        # Discrete feature (one-hot flag, small count): bucket by exact value over both
        # windows. Quantile edges would collapse, and a value never seen in the reference
        # (a new device, a placement that was always 0) would land in the reference's bin.
        values = np.union1d(ref, cur)
        ref_shares = np.array([(ref == v).mean() for v in values])
        cur_shares = np.array([(cur == v).mean() for v in values])
        return _psi_from_shares(ref_shares, cur_shares, eps)
    edges = _quantile_edges(ref, bins)
    return _psi_from_shares(_bin_shares(ref, edges), _bin_shares(cur, edges), eps)


def psi_categorical(
    reference: Mapping[str, int], current: Mapping[str, int], *, eps: float = _PSI_EPS
) -> float:
    ref, cur = _aligned_counts(reference, current)
    return _psi_from_shares(ref / ref.sum(), cur / cur.sum(), eps)


def ks_test(reference: ArrayLike, current: ArrayLike) -> tuple[float, float]:
    result = stats.ks_2samp(_as_sample(reference, "reference"), _as_sample(current, "current"))
    return float(result.statistic), float(result.pvalue)


def js_divergence(p: ArrayLike, q: ArrayLike) -> float:
    p_arr = _as_distribution(p, "p")
    q_arr = _as_distribution(q, "q")
    if p_arr.shape != q_arr.shape:
        raise ValueError("p and q must have the same length")
    # scipy returns the JS distance, which is the square root of the divergence.
    return float(distance.jensenshannon(p_arr, q_arr, base=2) ** 2)


def wasserstein(reference: ArrayLike, current: ArrayLike) -> float:
    return float(
        stats.wasserstein_distance(
            _as_sample(reference, "reference"), _as_sample(current, "current")
        )
    )


def chi_square_categorical(
    reference_counts: Mapping[str, int], current_counts: Mapping[str, int]
) -> tuple[float, float]:
    ref, cur = _aligned_counts(reference_counts, current_counts)
    observed = np.vstack([ref, cur])
    observed = observed[:, observed.sum(axis=0) > 0]
    # No Yates correction, so two-category features get the same plain Pearson statistic.
    result = stats.chi2_contingency(observed, correction=False)
    return float(result.statistic), float(result.pvalue)


def bootstrap_psi_threshold(
    reference: ArrayLike,
    *,
    sample_size: int,
    n_boot: int = 200,
    quantile: float = 0.99,
    bins: int = 10,
    seed: int = 0,
) -> float:
    """PSI that same-size draws from the reference itself exceed with probability 1 - quantile.

    Smaller monitoring windows are noisier, so they get a higher threshold.
    """
    if sample_size < 1 or n_boot < 1:
        raise ValueError("sample_size and n_boot must be positive")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    ref = _as_sample(reference, "reference")
    edges = _quantile_edges(ref, bins)
    ref_shares = _bin_shares(ref, edges)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_boot):
        resample = rng.choice(ref, size=sample_size)
        null.append(_psi_from_shares(ref_shares, _bin_shares(resample, edges), _PSI_EPS))
    return float(np.quantile(null, quantile))


def evaluate_feature_drift(
    feature: str, reference: ArrayLike, current: ArrayLike, policy: DriftAlertPolicy
) -> FeatureDrift:
    ref = _as_sample(reference, "reference")
    cur = _as_sample(current, "current", allow_empty=True)
    threshold = policy.effect_threshold
    if cur.size == 0:
        value = ks_statistic = p_value = float("nan")
    else:
        value = psi(ref, cur)
        ks_statistic, p_value = ks_test(ref, cur)
    alert = False
    if cur.size == 0 or cur.size < policy.min_samples:
        reason = f"segment too small ({cur.size} < {policy.min_samples})"
    elif value < threshold:
        reason = f"effect too small (PSI {value:.4f} < {threshold:.4f})"
    elif p_value >= policy.alpha:
        reason = f"not significant (KS p={p_value:.3g} >= {policy.alpha:g})"
    else:
        alert = True
        reason = (
            f"drift (PSI {value:.4f} >= {threshold:.4f}, KS p={p_value:.3g} < {policy.alpha:g})"
        )
    return FeatureDrift(
        feature=feature,
        psi=value,
        ks_statistic=ks_statistic,
        p_value=p_value,
        n_reference=ref.size,
        n_current=cur.size,
        threshold=threshold,
        alert=alert,
        reason=reason,
    )


def prediction_drift(
    reference_scores: ArrayLike, current_scores: ArrayLike, policy: DriftAlertPolicy
) -> FeatureDrift:
    return evaluate_feature_drift("prediction", reference_scores, current_scores, policy)


def seasonal_reference(
    history: Mapping[date, Sequence[float]],
    current_day: date,
    *,
    mode: str = SeasonalMode.SAME_WEEKDAY,
    weeks: int = 4,
) -> list[float]:
    """Pool a reference window from comparable past days so expected seasonality does not alert.

    Wellness usage spikes in January, around exams, and over holidays, so the
    previous week is often the wrong baseline.
    """
    if SeasonalMode(mode) is SeasonalMode.SAME_WEEKDAY:
        if weeks < 1:
            raise ValueError("weeks must be positive")
        days = [current_day - timedelta(weeks=w) for w in range(weeks, 0, -1)]
    else:
        anchor = _one_year_earlier(current_day)
        days = [anchor + timedelta(days=d) for d in range(-3, 4)]
    pooled = [float(v) for day in days for v in history.get(day, ())]
    if not pooled:
        raise ValueError(f"no {mode} reference history for {current_day.isoformat()}")
    return pooled


def exclude_untrusted(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Drop degraded and invalid traffic so outages and fraud waves do not read as drift."""
    return [r for r in records if not r.get("degraded") and not r.get("invalid")]


def _as_sample(values: ArrayLike, name: str, *, allow_empty: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0 and not allow_empty:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_distribution(values: ArrayLike, name: str) -> np.ndarray:
    arr = _as_sample(values, name)
    if (arr < 0).any() or arr.sum() <= 0:
        raise ValueError(f"{name} must be non-negative with a positive sum")
    return arr / arr.sum()


def _quantile_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    # Interior edges only: the outer bins are open-ended so current values outside the
    # reference range still land somewhere. Ties in the reference collapse duplicate edges.
    return np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))


def _bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / len(values)


def _psi_from_shares(expected: np.ndarray, actual: np.ndarray, eps: float) -> float:
    e = np.maximum(expected, eps)
    a = np.maximum(actual, eps)
    return float(np.sum((a - e) * np.log(a / e)))


def _aligned_counts(
    reference: Mapping[str, int], current: Mapping[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    categories = sorted(set(reference) | set(current))
    ref = np.array([reference.get(c, 0) for c in categories], dtype=float)
    cur = np.array([current.get(c, 0) for c in categories], dtype=float)
    if (ref < 0).any() or (cur < 0).any():
        raise ValueError("category counts must be non-negative")
    if ref.sum() == 0 or cur.sum() == 0:
        raise ValueError("both reference and current need at least one observation")
    return ref, cur


def _one_year_earlier(day: date) -> date:
    try:
        return day.replace(year=day.year - 1)
    except ValueError:
        return day.replace(year=day.year - 1, day=28)
