"""Data validation, the drift job, and the retraining controller."""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.helpers import NOW
from theraflo_ads.attribution.service import ImpressionLabel
from theraflo_ads.domain import PricingModel
from theraflo_ads.ml.validation import validate
from theraflo_ads.monitoring.jobs import DriftJobConfig, run_drift_job
from theraflo_ads.monitoring.retraining import ModelState
from theraflo_ads.orchestration.controller import RetrainingController
from theraflo_ads.simulator import SimConfig, run_simulation


def record(
    i,
    *,
    day=0,
    degraded=False,
    probs=None,
    features=None,
    clicked_label=None,
    ad="ad_1",
    p_ctr=0.02,
    placement="home_feed",
    ts=None,
):
    timestamp = ts or (NOW + timedelta(days=day, minutes=i % 1000))
    return {
        "request_id": f"r{i}",
        "timestamp": timestamp.isoformat(),
        "degraded": degraded,
        "blocked_reason": None,
        "holdout": "none",
        "placement": placement,
        "consent_scope": "personalized",
        "action_probabilities": probs if probs is not None else {ad: 0.9, "NO_AD": 0.1},
        "chosen_actions": [{"ad_id": ad, "position": 1}],
        "ope_eligible": True,
        "feature_schema_version": "features_v2",
        "impression_ids": [f"imp{i}"],
        "features": features if features is not None else {"daypart_morning": 1.0},
        "predicted": {"p_ctr": p_ctr},
    }


def label(i, clicked=False, ad="ad_1", revenue=0.01):
    return ImpressionLabel(
        f"imp{i}",
        f"r{i}",
        ad,
        "c",
        "adv",
        PricingModel.CPM,
        10.0,
        "c",
        100.0,
        None,
        NOW,
        click_id="x" if clicked else None,
        platform_revenue=revenue,
    )


# ----- validation -----


def test_clean_data_passes():
    records = [record(i) for i in range(100)]
    labels = {f"imp{i}": label(i) for i in range(100)}
    assert validate(records, labels, now=NOW + timedelta(days=1)).passed


@pytest.mark.parametrize(
    "broken,reason",
    [
        (lambda rs: rs + [dict(rs[0])], "duplicate"),
        (lambda rs: [dict(r, degraded=True) for r in rs[:10]] + rs[10:], "degraded share"),
        (lambda rs: [dict(rs[0], action_probabilities={"ad_1": 0.5})] + rs[1:], "propensities"),
        (
            lambda rs: [dict(rs[0], timestamp=(NOW + timedelta(days=30)).isoformat())] + rs[1:],
            "future",
        ),
        (lambda rs: [dict(rs[0], feature_schema_version="mystery")] + rs[1:], "schema"),
        (
            lambda rs: [dict(r, features={"x": float("nan")}) for r in rs[:5]] + rs[5:],
            "null features",
        ),
    ],
)
def test_broken_data_blocks_training(broken, reason):
    records = broken([record(i) for i in range(100)])
    labels = {f"imp{i}": label(i) for i in range(100)}
    report = validate(records, labels, now=NOW + timedelta(days=1))
    assert not report.passed
    assert any(reason in f for f in report.failures)


def test_impossible_labels_block_training():
    labels = {"imp0": label(0, revenue=-5.0)}
    assert not validate([record(0)], labels, now=NOW + timedelta(days=1)).passed


# ----- drift job -----


def window(start_day, n, *, p_ctr=0.02, click_every=50, offset=0, device=None, ad="ad_1"):
    records, labels = [], {}
    for k in range(n):
        i = offset + k
        features = {"daypart_morning": 1.0, "device_mobile": 1.0 if device and k % 2 else 0.0}
        ts = NOW + timedelta(days=start_day) + timedelta(seconds=k * (7 * 86400 // n))
        records.append(record(i, features=features, p_ctr=p_ctr, ad=ad, ts=ts))
        labels[f"imp{i}"] = label(i, clicked=(k % click_every == 0), ad=ad)
    return records, labels


def test_stable_traffic_raises_no_alert():
    ref, ref_labels = window(0, 4000)
    cur, cur_labels = window(7, 4000, offset=10_000)
    report = run_drift_job(
        ref + cur,
        ref_labels | cur_labels,
        now=NOW + timedelta(days=15),
        maturity=timedelta(days=1),
        categories={"ad_1": "books"},
        config=DriftJobConfig(click_maturity=timedelta(days=1)),
    )
    assert not report.drift_detected, report.to_dict()


def test_context_shift_and_calibration_shift_are_detected():
    ref, ref_labels = window(0, 4000)
    cur, cur_labels = window(7, 4000, offset=10_000, device="mobile", click_every=12)
    report = run_drift_job(
        ref + cur,
        ref_labels | cur_labels,
        now=NOW + timedelta(days=14),
        maturity=timedelta(days=0),
        categories={"ad_1": "books"},
        config=DriftJobConfig(click_maturity=timedelta(0)),
    )
    assert "device_mobile" in report.drifted_features
    assert any(c.alert and c.category == "books" for c in report.category_calibration)
    assert report.drift_detected


def test_degraded_traffic_is_excluded():
    ref, ref_labels = window(0, 3000)
    outage = [
        dict(r, degraded=True, placement="content_library")
        for r in window(7, 3000, offset=10_000)[0]
    ]
    report = run_drift_job(
        ref + outage, ref_labels, now=NOW + timedelta(days=14), maturity=timedelta(0)
    )
    assert report.excluded_untrusted == 3000
    assert not report.categorical.get("placement", {}).get("alert", False)


def test_small_windows_never_alert():
    ref, ref_labels = window(0, 300)
    cur, cur_labels = window(7, 300, offset=10_000, device="mobile")
    report = run_drift_job(
        ref + cur, ref_labels | cur_labels, now=NOW + timedelta(days=14), maturity=timedelta(0)
    )
    assert report.drifted_features == []


# ----- controller -----


class FakeReport:
    def __init__(self, drift=True, degraded=True, significant=True, mature=True, calibration=False):
        self.drift_detected = drift
        self.business = {"degraded": degraded, "significant": significant}
        self.labels_mature = mature
        self.drifted_features = ["x"] if drift else []
        self.prediction = None
        self.calibration = None
        self.concept_alert = False
        self.categorical = {}
        self.category_calibration = (
            [type("C", (), {"alert": True, "category": "books"})()] if calibration else []
        )


class Quality:
    passed = True


def test_controller_full_path_to_shadow_then_production(tmp_path):
    trained = []
    controller = RetrainingController(
        tmp_path / "state.json", retrain=lambda: trained.append(1) or {"passed": True}
    )
    assert controller.step(FakeReport(), Quality()).state is ModelState.WARNING
    step = controller.step(FakeReport(), Quality())
    assert step.state is ModelState.SHADOW and trained == [1]
    assert step.promotion_frozen
    assert controller.advance(shadow_ok=True) is ModelState.CANARY
    assert controller.advance(ab_promote=True) is ModelState.HEALTHY
    transitions = [(h["from"], h["to"]) for h in controller.history()]
    assert ("canary", "production") in transitions


def test_false_alarm_clears(tmp_path):
    controller = RetrainingController(tmp_path / "s.json", retrain=lambda: {"passed": True})
    controller.step(FakeReport(), Quality())
    assert controller.step(FakeReport(drift=False), Quality()).state is ModelState.HEALTHY


def test_retraining_blocked_without_business_impact_but_recalibrates(tmp_path):
    recalibrated = []
    controller = RetrainingController(
        tmp_path / "s.json",
        retrain=lambda: pytest.fail("must not retrain"),
        recalibrate=lambda: recalibrated.append(1) or {"passed": True},
    )
    controller.step(FakeReport(degraded=False, calibration=True), Quality())
    step = controller.step(FakeReport(degraded=False, calibration=True), Quality())
    assert recalibrated == [1]
    assert step.state is ModelState.SHADOW


def test_failed_candidate_returns_to_healthy(tmp_path):
    controller = RetrainingController(
        tmp_path / "s.json", retrain=lambda: {"passed": False, "reasons": ["ECE too high"]}
    )
    controller.step(FakeReport(), Quality())
    step = controller.step(FakeReport(), Quality())
    assert step.state is ModelState.HEALTHY
    assert "ECE too high" in step.actions[-1]


def test_bad_data_blocks_retraining(tmp_path):
    controller = RetrainingController(tmp_path / "s.json", retrain=lambda: pytest.fail("no"))
    bad = type("Q", (), {"passed": False})()
    controller.step(FakeReport(), bad)
    step = controller.step(FakeReport(), bad)
    assert step.state is ModelState.DRIFT_CONFIRMED
    assert step.reason == "data quality gate failed"


def test_drift_job_catches_a_simulated_ctr_shift():
    config = SimConfig(
        n_users=2500,
        n_days=28,
        seed=7,
        drift_day=21,
        drift_ctr=(("streaming", 0.25), ("budgeting_apps", 3.0)),
    )
    result = run_simulation(config)
    records = [d.to_log() for d in result.decisions]
    labels = {lbl.impression_id: lbl for lbl in result.platform.store.attribution.labels()}
    cats = {a.ad_id: a.category for a in result.world.catalog.ads.values()}
    report = run_drift_job(
        records,
        labels,
        now=config.start + timedelta(days=28),
        maturity=result.platform.attribution_policy.maturity_delay,
        categories=cats,
        baseline=config.start + timedelta(days=14),
    )
    shifted = {c.category for c in report.category_calibration if c.alert}
    assert shifted & {"streaming", "budgeting_apps"}
