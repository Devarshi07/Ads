"""OPE estimators on a synthetic bandit with a known answer."""

import math
import random

import numpy as np
import pytest

from tests.helpers import make_request, make_stack
from theraflo_ads.ope import (
    LoggedSample,
    doubly_robust,
    effective_sample_size,
    ips,
    samples_from_decisions,
    snips,
)

ACTIONS = ("a", "b", "c")
LOGGING = {"a": 0.5, "b": 0.3, "c": 0.2}
TRUE_MEAN = {"a": 0.10, "b": 0.30, "c": 0.20}


def logged(n=20_000, seed=0):
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        action = rng.choices(ACTIONS, weights=[LOGGING[a] for a in ACTIONS])[0]
        reward = float(rng.random() < TRUE_MEAN[action])
        samples.append(LoggedSample(ACTIONS, action, LOGGING[action], reward))
    return samples


def always(action):
    return lambda sample: {a: float(a == action) for a in ACTIONS}


def mixed(sample):
    return {"a": 0.0, "b": 0.6, "c": 0.4}


def reward_model(sample, action):
    return {"a": 0.12, "b": 0.25, "c": 0.22}[action]  # deliberately biased


def within_three_se(estimate, truth):
    se = (estimate.ci_high - estimate.ci_low) / (2 * 1.96)
    return abs(estimate.value - truth) <= 3 * se


@pytest.mark.parametrize("target,truth", [(always("b"), 0.30), (mixed, 0.6 * 0.3 + 0.4 * 0.2)])
def test_estimators_recover_the_true_value(target, truth):
    samples = logged()
    for estimate in (
        ips(samples, target),
        snips(samples, target),
        doubly_robust(samples, target, reward_model),
    ):
        assert estimate.conclusive, estimate
        assert estimate.ci_low <= truth <= estimate.ci_high or within_three_se(estimate, truth)


def test_dr_has_lower_variance_than_ips_with_a_decent_model():
    samples = logged()
    plain = ips(samples, always("c"))
    dr = doubly_robust(samples, always("c"), reward_model)
    assert (dr.ci_high - dr.ci_low) < (plain.ci_high - plain.ci_low)


def test_effective_sample_size():
    assert effective_sample_size(np.ones(100)) == pytest.approx(100)
    assert effective_sample_size(np.array([100.0] + [0.0] * 99)) == pytest.approx(1)


def test_too_little_effective_data_is_inconclusive():
    estimate = ips(logged(n=300), always("c"), min_ess=200)
    assert not estimate.conclusive
    assert "effective sample size" in estimate.reason


def test_wide_interval_is_inconclusive():
    estimate = ips(logged(n=2_000), always("c"), min_ess=10, max_ci_width=0.001)
    assert not estimate.conclusive
    assert "confidence interval" in estimate.reason


def test_wrong_logged_propensities_are_inconclusive():
    """Logged propensities that understate how often an action was taken (as on the Open
    Bandit Dataset, where Monte Carlo floors logged 1e-6 for items that were shown) make
    the importance weights average more than 1."""
    samples = [
        LoggedSample(s.eligible_actions, s.action, s.propensity / 2, s.reward)
        if s.action == "c"
        else s
        for s in logged()
    ]
    estimate = snips(samples, mixed)
    assert estimate.weight_mean > 1.05 and not estimate.conclusive
    assert "importance weights average" in estimate.reason
    honest = snips(logged(), mixed)
    assert abs(honest.weight_mean - 1) < 0.05 and honest.conclusive


def test_small_real_deviations_in_the_weights_do_not_block_an_estimate():
    """A logger that is 2% off its nominal propensities moves the estimate by about 2%:
    reported, but not a reason to call the estimate inconclusive."""
    samples = [
        LoggedSample(s.eligible_actions, s.action, s.propensity * 1.02, s.reward)
        for s in logged(n=50_000)
    ]
    estimate = ips(samples, mixed)
    assert estimate.weight_mean == pytest.approx(1 / 1.02, abs=0.01) and estimate.conclusive


def test_cluster_bootstrap_widens_intervals_for_correlated_rows():
    """Samples from one user (or hour) share effects; resampling rows treats them as
    independent and understates the uncertainty."""
    rng = random.Random(3)
    samples, clusters = [], []
    for user in range(300):
        boost = rng.choice([0.0, 0.4])  # a strong per-user effect
        for _ in range(40):
            action = rng.choices(ACTIONS, weights=[LOGGING[a] for a in ACTIONS])[0]
            reward = float(rng.random() < TRUE_MEAN[action] + boost)
            samples.append(LoggedSample(ACTIONS, action, LOGGING[action], reward))
            clusters.append(f"user{user}")
    rows = ips(samples, mixed)
    users = ips(samples, mixed, clusters=clusters)
    assert users.value == rows.value
    assert (users.ci_high - users.ci_low) > 1.5 * (rows.ci_high - rows.ci_low)


def test_cluster_bootstrap_needs_enough_clusters():
    samples = logged(n=1_000)
    with pytest.raises(ValueError, match="at least 20 clusters"):
        ips(samples, mixed, clusters=[i % 5 for i in range(1_000)])
    with pytest.raises(ValueError, match="one id per sample"):
        ips(samples, mixed, clusters=[0, 1])


def test_target_mass_outside_the_logged_eligible_set_is_an_error():
    samples = logged(n=10)
    with pytest.raises(ValueError, match="outside the logged eligible set"):
        ips(samples, lambda s: {"d": 1.0})


def test_zero_propensity_is_an_error():
    with pytest.raises(ValueError, match="positive"):
        ips([LoggedSample(ACTIONS, "a", 0.0, 1.0)], always("a"))


def test_no_data():
    estimate = snips([], always("a"))
    assert not estimate.conclusive
    assert math.isnan(estimate.value)


def test_only_ope_eligible_decisions_become_samples():
    stack = make_stack(seed=1)
    stack.service.decide(make_request(request_id="served"))
    stack.service.decide(make_request("journal", request_id="blocked"))
    samples = samples_from_decisions(stack.decisions, {"served": 0.5})
    assert len(samples) == 1
    assert samples[0].reward == 0.5
    assert samples[0].propensity == stack.decisions[0].propensity
