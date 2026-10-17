"""Calibration and concept drift on mature labels (design doc: Concept Drift, Calibration Drift).

Conversions arrive up to 7 days late, so callers pass only labels whose
attribution window has closed; click calibration is the early proxy.
Recalibration is the first response to calibration drift and full retraining
the second: a model that still ranks well only needs its probabilities remapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score


class CalibrationAction(StrEnum):
    NONE = "none"
    RECALIBRATE = "recalibrate"
    RETRAIN = "retrain"


@dataclass(frozen=True)
class ReliabilityBin:
    low: float
    high: float
    count: int
    mean_predicted: float
    observed_rate: float


@dataclass(frozen=True)
class CalibrationDrift:
    ece_reference: float
    ece_current: float
    auc_reference: float
    auc_current: float
    alert: bool
    action: CalibrationAction
    reason: str


@dataclass(frozen=True)
class ConceptDrift:
    auc_reference: float
    auc_current: float
    alert: bool
    reason: str


def reliability_table(
    predictions: ArrayLike, outcomes: ArrayLike, *, bins: int = 10
) -> list[ReliabilityBin]:
    if bins < 1:
        raise ValueError("bins must be positive")
    p, y = _validated(predictions, outcomes, probabilities=True)
    # Equal-width bins on [0, 1]; a prediction of exactly 1.0 belongs to the last bin.
    index = np.minimum((p * bins).astype(int), bins - 1)
    table = []
    for i in np.unique(index):
        in_bin = index == i
        table.append(
            ReliabilityBin(
                low=float(i / bins),
                high=float((i + 1) / bins),
                count=int(in_bin.sum()),
                mean_predicted=float(p[in_bin].mean()),
                observed_rate=float(y[in_bin].mean()),
            )
        )
    return table


def expected_calibration_error(
    predictions: ArrayLike, outcomes: ArrayLike, *, bins: int = 10
) -> float:
    table = reliability_table(predictions, outcomes, bins=bins)
    total = sum(b.count for b in table)
    return sum(b.count * abs(b.observed_rate - b.mean_predicted) for b in table) / total


def brier_score(predictions: ArrayLike, outcomes: ArrayLike) -> float:
    p, y = _validated(predictions, outcomes, probabilities=True)
    return float(np.mean((p - y) ** 2))


def calibration_drift(
    ref_pred: ArrayLike,
    ref_y: ArrayLike,
    cur_pred: ArrayLike,
    cur_y: ArrayLike,
    *,
    ece_increase_threshold: float = 0.02,
    max_auc_drop: float = 0.02,
    min_samples: int = 2000,
) -> CalibrationDrift:
    ece_reference = expected_calibration_error(ref_pred, ref_y)
    ece_current = expected_calibration_error(cur_pred, cur_y)
    auc_reference = _auc(ref_pred, ref_y)
    auc_current = _auc(cur_pred, cur_y)
    n_current = np.asarray(cur_pred).size
    increase = ece_current - ece_reference
    drop = auc_reference - auc_current
    if n_current < min_samples:
        action = CalibrationAction.NONE
        reason = f"not enough mature labels ({n_current} < {min_samples})"
    elif increase <= ece_increase_threshold:
        action = CalibrationAction.NONE
        reason = f"ECE stable ({increase:+.4f} <= {ece_increase_threshold:g})"
    elif drop > max_auc_drop:
        action = CalibrationAction.RETRAIN
        reason = f"ECE rose {increase:+.4f} and AUC dropped {drop:.4f} > {max_auc_drop:g}"
    else:
        # An undefined AUC (single-class window) also lands here: recalibration is the
        # cheaper first response when ranking damage cannot be shown.
        action = CalibrationAction.RECALIBRATE
        reason = f"ECE rose {increase:+.4f}; AUC drop {drop:.4f} within {max_auc_drop:g}"
    return CalibrationDrift(
        ece_reference=ece_reference,
        ece_current=ece_current,
        auc_reference=auc_reference,
        auc_current=auc_current,
        alert=action is not CalibrationAction.NONE,
        action=action,
        reason=reason,
    )


def concept_drift(
    ref_scores: ArrayLike,
    ref_y: ArrayLike,
    cur_scores: ArrayLike,
    cur_y: ArrayLike,
    *,
    max_auc_drop: float = 0.02,
    min_positives: int = 50,
) -> ConceptDrift:
    ref_s, ref_labels = _validated(ref_scores, ref_y, probabilities=False)
    cur_s, cur_labels = _validated(cur_scores, cur_y, probabilities=False)
    auc_reference = _auc(ref_s, ref_labels)
    auc_current = _auc(cur_s, cur_labels)
    shortfall = _label_shortfall(cur_labels, min_positives, "current") or _label_shortfall(
        ref_labels, min_positives, "reference"
    )
    if shortfall is not None:
        return ConceptDrift(auc_reference, auc_current, False, shortfall)
    drop = auc_reference - auc_current
    if drop > max_auc_drop:
        return ConceptDrift(
            auc_reference, auc_current, True, f"AUC dropped {drop:.4f} > {max_auc_drop:g}"
        )
    return ConceptDrift(
        auc_reference, auc_current, False, f"AUC stable (drop {drop:.4f} <= {max_auc_drop:g})"
    )


def _validated(
    scores: ArrayLike, outcomes: ArrayLike, *, probabilities: bool
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(outcomes, dtype=float).ravel()
    if s.size == 0 or s.size != y.size:
        raise ValueError("scores and outcomes must be non-empty and the same length")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if probabilities and ((s < 0).any() or (s > 1).any()):
        raise ValueError("predictions must be probabilities in [0, 1]")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("outcomes must be 0 or 1")
    return s, y


def _auc(scores: ArrayLike, outcomes: ArrayLike) -> float:
    y = np.asarray(outcomes, dtype=float).ravel()
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float).ravel()))


def _label_shortfall(outcomes: np.ndarray, min_positives: int, window: str) -> str | None:
    positives = int(outcomes.sum())
    if positives < min_positives:
        return f"not enough mature labels ({window}: {positives} positives < {min_positives})"
    if positives == outcomes.size:
        return f"not enough mature labels ({window}: no negatives)"
    return None
