"""Calibration drift (recalibrate vs retrain) and concept drift on mature labels."""

from __future__ import annotations

import numpy as np
import pytest

from theraflo_ads.monitoring import (
    CalibrationAction,
    ReliabilityBin,
    brier_score,
    calibration_drift,
    concept_drift,
    expected_calibration_error,
    reliability_table,
)

# Hand-worked example, 10 bins:
#   [0.0, 0.1): 0.05, 0.05 -> 0, 0          gap |0.00 - 0.05| = 0.05, weight 2/8
#   [0.2, 0.3): 0.2 x 4    -> 1, 0, 0, 0    gap |0.25 - 0.20| = 0.05, weight 4/8
#   [0.9, 1.0]: 1.0, 0.9   -> 1, 0          gap |0.50 - 0.95| = 0.45, weight 2/8
# ECE = (2 * 0.05 + 4 * 0.05 + 2 * 0.45) / 8 = 0.15
HAND_PREDICTIONS = [0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 1.0, 0.9]
HAND_OUTCOMES = [0, 0, 1, 0, 0, 0, 1, 0]


def calibrated(n=20_000, seed=0):
    """Predictions equal to the true conversion probability, with sampled outcomes."""
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 5, n)
    return p, (rng.random(n) < p).astype(int)


def test_reliability_table_hand_example():
    table = reliability_table(HAND_PREDICTIONS, HAND_OUTCOMES)
    assert [(b.low, b.high, b.count) for b in table] == [
        (0.0, 0.1, 2),
        (0.2, 0.3, 4),
        (0.9, 1.0, 2),
    ]
    assert table[1] == ReliabilityBin(0.2, 0.3, 4, pytest.approx(0.2), pytest.approx(0.25))
    assert table[2].mean_predicted == pytest.approx(0.95)
    assert table[2].observed_rate == pytest.approx(0.5)


def test_expected_calibration_error_hand_example():
    assert expected_calibration_error(HAND_PREDICTIONS, HAND_OUTCOMES) == pytest.approx(0.15)


def test_well_calibrated_model_has_small_ece():
    p, y = calibrated()
    assert expected_calibration_error(p, y) < 0.01


def test_brier_score():
    assert brier_score([0.2, 0.8], [0, 1]) == pytest.approx(0.04)
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0


@pytest.mark.parametrize(
    "predictions,outcomes",
    [([0.1, 0.2], [0]), ([], []), ([1.2], [1]), ([0.5], [2]), ([float("nan")], [0])],
)
def test_invalid_inputs_are_rejected(predictions, outcomes):
    with pytest.raises(ValueError):
        expected_calibration_error(predictions, outcomes)


def test_calibration_drift_none_when_still_calibrated():
    ref_p, ref_y = calibrated(seed=0)
    cur_p, cur_y = calibrated(seed=1)
    result = calibration_drift(ref_p, ref_y, cur_p, cur_y)
    assert result.action == "none"
    assert not result.alert


def test_calibration_drift_recalibrates_when_ranking_is_intact():
    ref_p, ref_y = calibrated()
    # A monotone distortion: probabilities are off, but the ordering (and AUC) is unchanged.
    result = calibration_drift(ref_p, ref_y, np.sqrt(ref_p), ref_y)
    assert result.ece_current - result.ece_reference > 0.02
    assert result.auc_current == pytest.approx(result.auc_reference)
    assert result.action is CalibrationAction.RECALIBRATE
    assert result.alert


def test_calibration_drift_retrains_when_ranking_also_degrades():
    ref_p, ref_y = calibrated()
    scrambled = np.random.default_rng(2).permutation(ref_p)
    result = calibration_drift(ref_p, ref_y, np.sqrt(scrambled), ref_y)
    assert result.auc_reference - result.auc_current > 0.02
    assert result.action == "retrain"
    assert result.alert


def test_calibration_drift_waits_for_enough_mature_labels():
    ref_p, ref_y = calibrated()
    result = calibration_drift(ref_p, ref_y, np.sqrt(ref_p[:500]), ref_y[:500])
    assert result.action == "none"
    assert not result.alert
    assert result.reason == "not enough mature labels (500 < 2000)"


def ranking_data(n, *, signal, seed):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=n)
    outcomes = (rng.random(n) < 1 / (1 + np.exp(-(signal * scores - 3)))).astype(int)
    return scores, outcomes


def test_concept_drift_detects_a_weaker_predictor():
    ref_s, ref_y = ranking_data(20_000, signal=1.5, seed=0)
    cur_s, cur_y = ranking_data(20_000, signal=0.5, seed=1)
    result = concept_drift(ref_s, ref_y, cur_s, cur_y)
    assert result.alert
    assert result.auc_reference - result.auc_current > 0.02


def test_concept_drift_stable_relationship_does_not_alert():
    ref_s, ref_y = ranking_data(20_000, signal=1.5, seed=0)
    cur_s, cur_y = ranking_data(20_000, signal=1.5, seed=1)
    result = concept_drift(ref_s, ref_y, cur_s, cur_y)
    assert not result.alert
    assert result.reason.startswith("AUC stable")


def test_concept_drift_needs_mature_positives():
    ref_s, ref_y = ranking_data(20_000, signal=1.5, seed=0)
    cur_s, cur_y = ranking_data(300, signal=0.5, seed=1)
    assert cur_y.sum() < 50
    result = concept_drift(ref_s, ref_y, cur_s, cur_y)
    assert not result.alert
    assert result.reason.startswith("not enough mature labels")


def test_concept_drift_needs_negatives_too():
    ref_s, ref_y = ranking_data(20_000, signal=1.5, seed=0)
    result = concept_drift(ref_s, ref_y, np.arange(100.0), np.ones(100))
    assert not result.alert
    assert result.reason == "not enough mature labels (current: no negatives)"
