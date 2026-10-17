"""Feature and prediction drift: statistics, alert policy, seasonal references."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from theraflo_ads.monitoring import (
    DriftAlertPolicy,
    bootstrap_psi_threshold,
    chi_square_categorical,
    evaluate_feature_drift,
    exclude_untrusted,
    js_divergence,
    ks_test,
    prediction_drift,
    psi,
    psi_categorical,
    seasonal_reference,
    wasserstein,
)


def normal(n, *, mean=0.0, seed=0):
    return np.random.default_rng(seed).normal(mean, 1.0, n)


@pytest.fixture(scope="module")
def reference():
    return normal(50_000, seed=1)


@pytest.fixture(scope="module")
def policy(reference):
    return DriftAlertPolicy(effect_threshold=bootstrap_psi_threshold(reference, sample_size=10_000))


def test_psi_matches_hand_computation():
    # Median edge 2.5: reference shares (0.5, 0.5), current shares (0.75, 0.25).
    expected = (0.75 - 0.5) * math.log(0.75 / 0.5) + (0.25 - 0.5) * math.log(0.25 / 0.5)
    assert psi([1, 2, 3, 4], [1, 1, 1, 4], bins=2) == pytest.approx(expected)


def test_psi_is_zero_for_identical_samples(reference):
    assert psi(reference, reference) == pytest.approx(0.0, abs=1e-12)


def test_psi_grows_with_the_shift(reference):
    small = psi(reference, normal(10_000, mean=0.1, seed=2))
    large = psi(reference, normal(10_000, mean=0.5, seed=2))
    assert 0 < small < large


def test_psi_outer_bins_are_open_ended():
    reference = np.repeat(np.arange(10.0), 100)  # exactly 10% per decile bin
    eps = 1e-4
    # Everything lands in the top bin, even though it is far outside the reference range.
    expected = 9 * (eps - 0.1) * math.log(eps / 0.1) + (1.0 - 0.1) * math.log(1.0 / 0.1)
    assert psi(reference, np.full(500, 50.0), eps=eps) == pytest.approx(expected)


def test_psi_handles_heavily_tied_reference():
    reference = np.array([0.0] * 900 + [1.0] * 100)
    assert psi(reference, reference) == pytest.approx(0.0, abs=1e-12)
    assert psi(reference, np.array([0.0] * 500 + [1.0] * 500)) > 0.3


def test_psi_rejects_empty_and_non_finite_input():
    with pytest.raises(ValueError):
        psi([], [1.0])
    with pytest.raises(ValueError):
        psi([1.0, 2.0], [float("nan")])


def test_psi_categorical_uses_the_union_of_categories():
    reference = {"a": 50, "b": 50}
    current = {"a": 50, "b": 25, "c": 25}
    eps = 1e-4
    expected = (0.25 - 0.5) * math.log(0.25 / 0.5) + (0.25 - eps) * math.log(0.25 / eps)
    assert psi_categorical(reference, current, eps=eps) == pytest.approx(expected)
    assert psi_categorical(reference, {"a": 5, "b": 5}) == pytest.approx(0.0)


def test_ks_and_wasserstein():
    x = normal(5000, seed=3)
    assert ks_test(x, x) == (0.0, 1.0)
    statistic, p_value = ks_test(x, x + 0.5)
    assert statistic > 0.1
    assert p_value < 1e-10
    assert wasserstein(x, x + 1.0) == pytest.approx(1.0)


def test_js_divergence_is_bounded_and_normalizes_inputs():
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1.0)
    assert js_divergence([2, 2], [1, 1]) == pytest.approx(0.0, abs=1e-12)
    assert 0 < js_divergence([0.5, 0.5], [0.9, 0.1]) < 1
    with pytest.raises(ValueError):
        js_divergence([1, 0], [1, 0, 0])


def test_chi_square_categorical():
    statistic, p_value = chi_square_categorical(
        {"ios": 100, "android": 100}, {"ios": 50, "android": 50}
    )
    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)
    _, p_value = chi_square_categorical({"ios": 100, "android": 100}, {"ios": 150, "android": 50})
    assert p_value < 0.01
    _, p_value = chi_square_categorical({"ios": 100}, {"ios": 90, "web": 30})
    assert p_value < 0.01


def test_bootstrap_threshold_is_higher_for_smaller_windows(reference):
    small = bootstrap_psi_threshold(reference, sample_size=500)
    large = bootstrap_psi_threshold(reference, sample_size=20_000)
    assert small > large > 0
    assert bootstrap_psi_threshold(reference, sample_size=500) == small


def test_identical_distributions_do_not_alert(reference, policy):
    result = evaluate_feature_drift("session_length", reference, normal(10_000, seed=4), policy)
    assert not result.alert
    assert result.n_reference == 50_000
    assert result.n_current == 10_000


def test_mean_shift_alerts(reference, policy):
    result = evaluate_feature_drift(
        "session_length", reference, normal(10_000, mean=0.3, seed=4), policy
    )
    assert result.alert
    assert result.reason.startswith("drift")
    assert result.psi >= result.threshold == policy.effect_threshold
    assert result.p_value < policy.alpha


def test_small_segment_does_not_alert(reference, policy):
    result = evaluate_feature_drift("session_length", reference, normal(412, mean=1.0), policy)
    assert result.psi > policy.effect_threshold
    assert not result.alert
    assert result.reason == "segment too small (412 < 2000)"


def test_empty_segment_does_not_alert(reference, policy):
    result = evaluate_feature_drift("session_length", reference, [], policy)
    assert not result.alert
    assert result.reason == "segment too small (0 < 2000)"


def test_significant_but_tiny_shift_does_not_alert():
    # At large n, KS flags a 0.03-sigma shift; the effect-size gate keeps it quiet.
    result = evaluate_feature_drift(
        "click_frequency",
        normal(200_000, seed=5),
        normal(200_000, mean=0.03, seed=6),
        DriftAlertPolicy(effect_threshold=0.1),
    )
    assert result.p_value < 0.01
    assert not result.alert
    assert result.reason.startswith("effect too small")


def test_insignificant_shift_does_not_alert(reference):
    result = evaluate_feature_drift(
        "click_frequency", reference, normal(10_000, seed=7), DriftAlertPolicy(effect_threshold=0)
    )
    assert not result.alert
    assert result.reason.startswith("not significant")


def test_invalid_alert_policy():
    with pytest.raises(ValueError):
        DriftAlertPolicy(effect_threshold=0.1, alpha=1.5)
    with pytest.raises(ValueError):
        DriftAlertPolicy(effect_threshold=-0.1)


def test_prediction_drift_flags_the_pcvr_example():
    # Design doc example: baseline mean pCVR 0.032, current 0.058.
    rng = np.random.default_rng(8)
    baseline = rng.beta(2, 60, 20_000)
    current = rng.beta(2, 33, 20_000)
    result = prediction_drift(baseline, current, DriftAlertPolicy(effect_threshold=0.1))
    assert result.feature == "prediction"
    assert result.alert


def daily_history(start, end):
    """One value per day, equal to the day's ordinal, so pooled values identify their day."""
    days = (start + timedelta(days=i) for i in range((end - start).days + 1))
    return {day: [float(day.toordinal())] for day in days}


def ordinals(*days):
    return sorted(float(d.toordinal()) for d in days)


def test_seasonal_reference_same_weekday():
    history = daily_history(date(2026, 8, 1), date(2026, 10, 5))
    today = date(2026, 10, 6)
    pooled = seasonal_reference(history, today)
    assert sorted(pooled) == ordinals(
        date(2026, 9, 8), date(2026, 9, 15), date(2026, 9, 22), date(2026, 9, 29)
    )
    assert all(date.fromordinal(int(v)).weekday() == today.weekday() for v in pooled)
    assert len(seasonal_reference(history, today, weeks=2)) == 2


def test_seasonal_reference_same_period_last_year():
    history = daily_history(date(2024, 12, 1), date(2025, 2, 1))
    pooled = seasonal_reference(history, date(2026, 1, 5), mode="same_period_last_year")
    assert sorted(pooled) == ordinals(*(date(2025, 1, 2) + timedelta(days=i) for i in range(7)))


def test_seasonal_reference_leap_day_falls_back_to_february_28():
    history = daily_history(date(2027, 2, 1), date(2027, 3, 31))
    pooled = seasonal_reference(history, date(2028, 2, 29), mode="same_period_last_year")
    assert sorted(pooled) == ordinals(*(date(2027, 2, 25) + timedelta(days=i) for i in range(7)))


def test_seasonal_reference_pools_every_value_and_skips_missing_days():
    history = {date(2026, 9, 29): [1.0, 2.0], date(2026, 9, 15): [3.0]}
    assert sorted(seasonal_reference(history, date(2026, 10, 6))) == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("mode", ["same_weekday", "same_period_last_year"])
def test_seasonal_reference_without_history_raises(mode):
    with pytest.raises(ValueError):
        seasonal_reference({}, date(2026, 10, 6), mode=mode)
    # History exists, but not on any comparable day.
    with pytest.raises(ValueError):
        seasonal_reference({date(2026, 10, 5): [1.0]}, date(2026, 10, 6), mode=mode)


def test_seasonal_reference_rejects_unknown_mode():
    with pytest.raises(ValueError):
        seasonal_reference({date(2026, 9, 29): [1.0]}, date(2026, 10, 6), mode="last_week")


def test_exclude_untrusted_drops_degraded_and_invalid_traffic():
    records = [
        {"request_id": "ok", "degraded": False, "invalid": False},
        {"request_id": "outage", "degraded": True},
        {"request_id": "fraud", "invalid": True},
        {"request_id": "both", "degraded": True, "invalid": True},
        {"request_id": "unflagged"},
    ]
    assert [r["request_id"] for r in exclude_untrusted(records)] == ["ok", "unflagged"]
    assert exclude_untrusted(iter(records[:1])) == records[:1]
