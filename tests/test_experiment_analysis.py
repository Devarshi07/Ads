"""Experiment analysis: aggregation, SRM, CUPED, guardrails, promotion, A/A in the simulator."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from tests.helpers import NOW, make_request, make_stack, make_user
from theraflo_ads.attribution.service import ImpressionLabel
from theraflo_ads.domain import PricingModel
from theraflo_ads.experiments.analysis import analyze_experiment, pre_period_revenue
from theraflo_ads.experiments.assignment import Experiment
from theraflo_ads.policy import PopularityPolicy
from theraflo_ads.simulator.ab import ABConfig, heuristic_arm, preregister_mde, run_ab_test
from theraflo_ads.simulator.world import SimConfig


def synthetic(n_control=2000, n_treatment=2000, lift=0.0, seed=0, ctr_t=0.02):
    """One record and label per impression; revenue ~ exponential, treatment scaled by lift."""
    rng = np.random.default_rng(seed)
    records, labels, pre = [], {}, {}
    for arm, n, scale, ctr in (
        ("control", n_control, 1.0, 0.02),
        ("treatment", n_treatment, 1 + lift, ctr_t),
    ):
        for u in range(n):
            user = f"{arm}-{u}"
            baseline = rng.exponential(1.0)
            pre[user] = baseline + rng.normal(0, 0.1)
            for k in range(5):
                impression = f"{user}-{k}"
                records.append(
                    {
                        "user_hash": user,
                        "experiment_cohort": f"exp:{arm}",
                        "holdout": "none",
                        "blocked_reason": None,
                        "impression_ids": [impression],
                    }
                )
                labels[impression] = ImpressionLabel(
                    impression,
                    impression,
                    "ad",
                    "c",
                    "adv",
                    PricingModel.CPM,
                    1.0,
                    "c",
                    1.0,
                    None,
                    NOW,
                    click_id="x" if rng.random() < ctr else None,
                    platform_revenue=baseline * scale / 5,
                )
    return records, labels, pre


def test_real_effect_is_promoted_and_cuped_tightens_the_interval():
    records, labels, pre = synthetic(lift=0.10)
    report = analyze_experiment(
        records,
        labels,
        experiment_id="exp",
        control="control",
        treatment="treatment",
        preregistered_mde=0.05,
        pre_period=pre,
    )
    plain = analyze_experiment(
        records,
        labels,
        experiment_id="exp",
        control="control",
        treatment="treatment",
        preregistered_mde=0.05,
    )
    assert report.srm.passed
    assert report.decision.promote, report.decision.reasons
    width = report.primary.ci_high - report.primary.ci_low
    assert width < 0.5 * (plain.primary.ci_high - plain.primary.ci_low)


def test_no_effect_is_not_promoted():
    records, labels, pre = synthetic(lift=0.0)
    report = analyze_experiment(
        records,
        labels,
        experiment_id="exp",
        control="control",
        treatment="treatment",
        preregistered_mde=0.05,
        pre_period=pre,
    )
    assert not report.decision.promote


def test_sample_ratio_mismatch_blocks_promotion():
    records, labels, pre = synthetic(n_control=2000, n_treatment=1700, lift=0.2)
    report = analyze_experiment(
        records,
        labels,
        experiment_id="exp",
        control="control",
        treatment="treatment",
        preregistered_mde=0.05,
        pre_period=pre,
    )
    assert not report.srm.passed
    assert any(r.startswith("srm") for r in report.decision.reasons)


def test_guardrail_and_safety_counters_block_promotion():
    records, labels, pre = synthetic(lift=0.10, ctr_t=0.005)
    report = analyze_experiment(
        records,
        labels,
        experiment_id="exp",
        control="control",
        treatment="treatment",
        preregistered_mde=0.05,
        pre_period=pre,
        safety_counters={"restricted_category": 1},
    )
    reasons = " | ".join(report.decision.reasons)
    assert "guardrail ctr" in reasons
    assert "safety" in reasons


def test_pre_period_excludes_cohorts_and_holdouts():
    records = [
        {
            "user_hash": "a",
            "experiment_cohort": None,
            "holdout": "none",
            "impression_ids": ["i1"],
            "blocked_reason": None,
        },
        {
            "user_hash": "b",
            "experiment_cohort": None,
            "holdout": "no_ads",
            "impression_ids": [],
            "blocked_reason": None,
        },
        {
            "user_hash": "a",
            "experiment_cohort": "exp:control",
            "holdout": "none",
            "impression_ids": ["i2"],
            "blocked_reason": None,
        },
    ]
    labels = {
        i: ImpressionLabel(
            i, i, "ad", "c", "adv", PricingModel.CPM, 1.0, "c", 1.0, None, NOW, platform_revenue=v
        )
        for i, v in (("i1", 2.0), ("i2", 5.0))
    }
    assert pre_period_revenue(records, labels) == {"a": 2.0}


def test_experiment_only_assigns_inside_its_window():
    experiment = Experiment("exp", (("a", 1.0), ("b", 1.0)), starts_at=NOW + timedelta(days=1))
    stack = make_stack(
        experiment=experiment,
        arm_chains={"a": [PopularityPolicy(["ad_cpm"])], "b": [PopularityPolicy(["ad_cpm"])]},
    )
    before = stack.service.decide(make_request(user=make_user("k1:x"))).decision
    after = stack.service.decide(
        make_request(user=make_user("k1:x"), now=NOW + timedelta(days=2), request_id="r2")
    ).decision
    assert before.experiment_cohort is None
    assert after.experiment_cohort in {"exp:a", "exp:b"}


def test_preregistered_mde_shrinks_with_more_users():
    values = np.random.default_rng(0).exponential(1.0, 5000)
    assert preregister_mde(values, 4000, alpha=0.05, power=0.8) < preregister_mde(
        values, 400, alpha=0.05, power=0.8
    )


def test_simulated_a_a_test_never_promotes():
    config = ABConfig(SimConfig(n_users=800, n_days=10, seed=3, run_seed=4), pre_days=3)
    report, context = run_ab_test(config, heuristic_arm)
    assert report.srm.passed
    assert not report.decision.promote
    assert context["safety_violations"] == {}
    with pytest.raises(ValueError):
        analyze_experiment(
            [], {}, experiment_id="none", control="a", treatment="b", preregistered_mde=0.1
        )
