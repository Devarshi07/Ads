"""Offline model evaluation and gates (design doc: Offline Metrics, Evals → Slice Evals).

A candidate is compared with a baseline on the same label-mature test window,
globally and per slice. Promotion gates on the worst slice, not the average.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from theraflo_ads.monitoring.calibration import brier_score, expected_calibration_error

_EPS = 1e-6


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    both_classes = 0 < y.sum() < len(y)
    return {
        "n": float(len(y)),
        "positives": float(y.sum()),
        "base_rate": float(y.mean()) if len(y) else 0.0,
        "mean_prediction": float(p.mean()) if len(p) else 0.0,
        "auc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both_classes else float("nan"),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else 0.0,
        "brier": float(brier_score(p, y)) if len(y) else 0.0,
        "ece": float(expected_calibration_error(p, y, bins=10)) if len(y) else 0.0,
    }


def slice_metrics(
    frame: pd.DataFrame,
    label: str,
    prediction: str,
    slices: Sequence[str],
    *,
    min_rows: int = 500,
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for column in slices:
        out[column] = {}
        for value, group in frame.groupby(column):
            if len(group) >= min_rows:
                out[column][str(value)] = classification_metrics(
                    group[label].to_numpy(), group[prediction].to_numpy()
                )
    return out


@dataclass(frozen=True)
class GateConfig:
    max_log_loss_regression: float = 0.0  # candidate log loss must not exceed baseline + this
    max_ece: float = 0.02
    max_slice_log_loss_regression: float = 0.02  # relative, per slice
    min_auc: float = 0.5
    min_slice_positives: int = 30  # below this, a slice cannot be judged either way


@dataclass(frozen=True)
class SliceRegression:
    column: str
    value: str
    rows: int
    positives: int
    regression: float  # relative log-loss change, candidate vs baseline (positive = worse)
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def slice_regressions(
    frame: pd.DataFrame,
    label: str,
    slices: Sequence[str],
    *,
    candidate: str = "candidate",
    baseline: str = "baseline",
    min_positives: int = 30,
    n_bootstrap: int = 300,
    seed: int = 0,
) -> list[SliceRegression]:
    """Paired bootstrap of the relative log-loss change in every slice with enough positives.
    Small slices are skipped: their point estimates are mostly noise (design doc: Drift
    Pitfalls → Small segments)."""
    rng = np.random.default_rng(seed)
    out = []
    for column in slices:
        for value, group in frame.groupby(column):
            y = group[label].to_numpy(float)
            if y.sum() < min_positives:
                continue
            loss_c = _pointwise_log_loss(y, group[candidate].to_numpy(float))
            loss_b = _pointwise_log_loss(y, group[baseline].to_numpy(float))
            point = loss_c.mean() / loss_b.mean() - 1
            n = len(y)
            boot = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, n)
                boot.append(loss_c[idx].mean() / loss_b[idx].mean() - 1)
            low, high = np.quantile(boot, [0.025, 0.975])
            out.append(
                SliceRegression(
                    column, str(value), n, int(y.sum()), float(point), float(low), float(high)
                )
            )
    return out


def evaluate_gates(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    regressions: Sequence[SliceRegression] = (),
    config: GateConfig | None = None,
) -> GateResult:
    config = config or GateConfig()
    reasons = []
    if candidate["log_loss"] > baseline["log_loss"] + config.max_log_loss_regression:
        reasons.append(
            f"log loss {candidate['log_loss']:.5f} worse than baseline {baseline['log_loss']:.5f}"
        )
    if candidate["ece"] > config.max_ece:
        reasons.append(f"ECE {candidate['ece']:.4f} > {config.max_ece}")
    if not np.isnan(candidate["auc"]) and candidate["auc"] < config.min_auc:
        reasons.append(f"AUC {candidate['auc']:.3f} < {config.min_auc}")
    for r in regressions:
        # Fail only when the slice is confidently worse than the allowed regression.
        if r.ci_low > config.max_slice_log_loss_regression:
            reasons.append(
                f"slice {r.column}={r.value}: log loss {r.regression:+.1%} "
                f"(95% CI {r.ci_low:+.1%} to {r.ci_high:+.1%}, {r.positives} positives)"
            )
    return GateResult(not reasons, tuple(reasons))


def _pointwise_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
