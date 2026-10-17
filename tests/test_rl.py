"""Offline RL dataset, fitted-Q iteration, PPO updates, and the safety evals."""

from __future__ import annotations

import random
from datetime import timedelta

import numpy as np
import pytest

from tests.helpers import NOW, PRIORS, make_catalog, make_request, make_stack, make_user
from theraflo_ads.attribution.service import ImpressionLabel
from theraflo_ads.domain import NO_AD, PricingModel
from theraflo_ads.experiments import HoldoutConfig
from theraflo_ads.ml.features import FeatureBuilder
from theraflo_ads.policy import PopularityPolicy
from theraflo_ads.policy.decision import ServedBy
from theraflo_ads.policy.policies import validate_distribution
from theraflo_ads.rl.dataset import build_transitions
from theraflo_ads.rl.fqi import ActionFeaturizer, FittedQ, FQIConfig, QPolicy, request_from
from theraflo_ads.rl.ppo import LinearSoftmaxPolicy, PPOConfig, _Step, ppo_update
from theraflo_ads.rl.safety import evaluate_policy_safety
from theraflo_ads.simulator.runner import CohortStats

CATALOG = make_catalog()
SAFE = [CATALOG.ads[a] for a in ("ad_cpm", "ad_cpc", "ad_cpa")]


def featurizer():
    return ActionFeaturizer(CATALOG, FeatureBuilder(CATALOG, PRIORS))


def logged(n_users=40, visits=6, reward_ad="ad_cpc"):
    """A uniform logging policy over the safe ads; `reward_ad` pays, others do not."""
    stack = make_stack(
        chain=lambda c: [PopularityPolicy(["ad_cpm"])],
        features=FeatureBuilder(CATALOG, PRIORS),
        holdouts=HoldoutConfig(uniform_exploration_share=0.999, no_ads_share=0.0),
        seed=3,
    )
    for u in range(n_users):
        for k in range(visits):
            stack.service.decide(
                make_request(
                    user=make_user(f"k1:u{u}"),
                    now=NOW + timedelta(minutes=11 * k + 200 * u),
                    request_id=f"r{u}-{k}",
                )
            )
    records = [d.to_log() for d in stack.decisions]
    labels = {}
    for d in stack.decisions:
        if d.impression_id:
            labels[d.impression_id] = ImpressionLabel(
                d.impression_id,
                d.request_id,
                d.chosen_action,
                "c",
                "a",
                PricingModel.CPC,
                1.0,
                "c",
                100.0,
                None,
                d.timestamp,
                platform_revenue=1.0 if d.chosen_action == reward_ad else 0.0,
            )
    return stack, records, labels


def test_transitions_follow_users_and_end_episodes():
    _, records, labels = logged(n_users=3, visits=4)
    transitions = build_transitions(records, labels, ad_cost=0.1)
    assert len(transitions) == 12
    ends = [t for t in transitions if t.done]
    assert len(ends) == 3
    t = transitions[0]
    assert t.next_index == 1 and transitions[1].decision.user_hash == t.decision.user_hash
    expected = (1.0 if t.action == "ad_cpc" else 0.0) - (0.1 if t.decision.impression_id else 0)
    assert t.reward == pytest.approx(expected)


def test_blocked_and_degraded_decisions_are_not_transitions():
    _, records, labels = logged(n_users=1, visits=2)
    records.append(dict(records[0], request_id="blocked", blocked_reason="placement_not_allowed"))
    records.append(dict(records[0], request_id="degraded", degraded=True))
    assert len(build_transitions(records, labels)) == 2


def test_no_ad_row_has_no_ad_features():
    f = featurizer()
    matrix = f.matrix(make_request(), ["ad_cpm", NO_AD])
    names = f.names
    assert matrix[1, names.index("is_no_ad")] == 1.0
    assert matrix[1, names.index("log_price")] == 0.0
    assert matrix[0, names.index("log_price")] > 0


def test_fqi_prefers_the_rewarding_action():
    _, records, labels = logged()
    transitions = build_transitions(records, labels)
    q = FittedQ(featurizer(), FQIConfig(iterations=3, algorithm="ridge")).fit(
        transitions, range(len(transitions))
    )
    request = request_from(transitions[0].decision)
    values = dict(
        zip(
            ["ad_cpm", "ad_cpc", "ad_cpa", NO_AD],
            q.q_values(request, ["ad_cpm", "ad_cpc", "ad_cpa", NO_AD]),
            strict=True,
        )
    )
    assert max(values, key=values.get) == "ad_cpc"
    policy = QPolicy(q, epsilon=0.1)
    dist = policy.distribution(request, SAFE)
    validate_distribution(dist, {a.ad_id for a in SAFE})
    assert policy.served_by is ServedBy.RL
    assert dist.probabilities["ad_cpc"] > 0.9


def test_ppo_update_moves_probability_toward_the_advantaged_action():
    f = featurizer()
    policy = LinearSoftmaxPolicy(f, rng=random.Random(0))
    phi = f.matrix(make_request(), ["ad_cpm", "ad_cpc", NO_AD])
    before = policy.probabilities(phi)
    chosen, rewards = {}, {}
    for i in range(200):
        action = ["ad_cpm", "ad_cpc", NO_AD][i % 3]
        policy.buffer[f"r{i}"] = _Step(
            f"r{i}", f"u{i}", float(i), phi, ["ad_cpm", "ad_cpc", NO_AD], before
        )
        chosen[f"r{i}"] = action
        rewards[f"r{i}"] = 1.0 if action == "ad_cpc" else 0.0
    for _ in range(5):
        for i in range(200):
            policy.buffer[f"r{i}"] = _Step(
                f"r{i}",
                f"u{i}",
                float(i),
                phi,
                ["ad_cpm", "ad_cpc", NO_AD],
                policy.probabilities(phi),
            )
        ppo_update(policy, chosen, rewards, PPOConfig(learning_rate=0.5))
    after = policy.probabilities(phi)
    assert after[1] > before[1]
    assert np.all(after >= 0.05 / 3 - 1e-12)  # the exploration floor survives training


def test_safety_eval_rejects_targeting_the_vulnerable_cohort():
    def result(vulnerable, other, violations=None):
        return type(
            "R",
            (),
            {"cohorts": {"vulnerable": vulnerable, "other": other}, "violations": violations or {}},
        )()

    fair = result(CohortStats(1000, 950, 200), CohortStats(10000, 9500, 2000))
    assert evaluate_policy_safety(fair).passed
    targeted = result(CohortStats(1000, 990, 400), CohortStats(10000, 9300, 1900))
    report = evaluate_policy_safety(targeted)
    assert not report.passed
    assert any("impulse" in r for r in report.reasons)
    assert any("ad load" in r for r in report.reasons)
    assert not evaluate_policy_safety(
        result(CohortStats(1, 1, 0), CohortStats(1, 1, 0), {"restricted_category": 1})
    ).passed
