"""Simulator-backed evals: safety, determinism, OPE validity, cohort parity."""

from datetime import timedelta

import pytest

from theraflo_ads.ope import ips, samples_from_decisions, snips
from theraflo_ads.ope.targets import target_from_policy
from theraflo_ads.policy import EpsilonGreedyPolicy, HeuristicScorer, UniformPolicy
from theraflo_ads.simulator import SimConfig, run_simulation


@pytest.fixture(scope="module")
def stateful():
    return run_simulation(SimConfig(n_users=1500, n_days=7, seed=11))


def test_no_safety_violations_and_nothing_billed_above_budget(stateful):
    assert stateful.violations == {}
    served = sum(1 for d in stateful.decisions if d.impression_id)
    assert served > 1000
    config = stateful.world.config
    for campaign in stateful.world.catalog.campaigns.values():
        for offset in range(config.n_days + 40):
            day = (config.start + timedelta(days=offset)).date()
            billed = stateful.budget.billed(campaign.campaign_id, day)
            assert billed <= campaign.daily_budget + 1e-9


def test_every_decision_is_logged_and_blocked_requests_never_serve(stateful):
    for d in stateful.decisions:
        if d.blocked_reason is not None:
            assert d.impression_id is None


def test_simulation_is_deterministic():
    config = SimConfig(n_users=300, n_days=3, seed=5)
    a, b = run_simulation(config), run_simulation(config)
    assert [d.chosen_action for d in a.decisions] == [d.chosen_action for d in b.decisions]
    assert a.platform_revenue == pytest.approx(b.platform_revenue)


def test_non_personalized_policy_keeps_vulnerable_cohort_parity(stateful):
    vulnerable, other = stateful.cohorts["vulnerable"], stateful.cohorts["other"]
    assert abs(vulnerable.ad_load - other.ad_load) < 0.03
    assert abs(vulnerable.impulse_share - other.impulse_share) < 0.08


def test_no_ads_holdout_measures_retention_cost():
    result = run_simulation(SimConfig(n_users=3000, n_days=14, seed=3, no_ads_holdout_share=0.2))
    with_ads = result.retention["none"]
    without = result.retention["no_ads"]
    assert without[0] / without[1] > with_ads[0] / with_ads[1]


def test_ope_recovers_on_policy_value_in_the_stateless_setting():
    """Logs from the epsilon-greedy policy must predict what the greedy policy does when
    actually run. Clicks are the reward to keep the check tight."""
    logging = run_simulation(
        SimConfig(
            n_users=2500, n_days=10, seed=21, stateless=True, epsilon=0.3, no_ads_holdout_share=0.0
        )
    )
    world = logging.world
    greedy = EpsilonGreedyPolicy(HeuristicScorer(world.catalog, world.priors), epsilon=0.0)

    # Same world (seed), independent randomness (run_seed), greedy policy for everyone.
    truth_run = run_simulation(
        SimConfig(
            n_users=2500,
            n_days=20,
            seed=21,
            run_seed=99,
            stateless=True,
            uniform_holdout_share=0.0,
            no_ads_holdout_share=0.0,
        ),
        policies=lambda w: ((greedy,), UniformPolicy()),
    )
    ran = [d for d in truth_run.decisions if d.blocked_reason is None]
    truth = sum(truth_run.clicks.get(d.request_id, 0) for d in ran) / len(ran)
    truth_se = (truth * (1 - truth) / len(ran)) ** 0.5

    samples = samples_from_decisions(logging.decisions, logging.clicks)
    target = target_from_policy(greedy, world.catalog)
    for estimate in (ips(samples, target), snips(samples, target)):
        estimate_se = (estimate.ci_high - estimate.ci_low) / (2 * 1.96)
        assert estimate.conclusive
        assert abs(estimate.value - truth) <= 3 * (estimate_se**2 + truth_se**2) ** 0.5
