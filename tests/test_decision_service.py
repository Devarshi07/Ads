import json
from datetime import timedelta

import pytest

from tests.helpers import NOW, heuristic_chain, make_request, make_stack, make_user
from theraflo_ads.domain import NO_AD
from theraflo_ads.experiments import Experiment, HoldoutConfig
from theraflo_ads.policy import ActionDistribution, PopularityPolicy, ServedBy


class Exploding:
    served_by = ServedBy.BANDIT
    version = "exploding_v1"

    def distribution(self, request, eligible):
        raise TimeoutError("model server timed out")


class OutOfBounds:
    """Puts mass on an ad that is not eligible: must be treated as a failure."""

    served_by = ServedBy.RANKER
    version = "oob_v1"

    def distribution(self, request, eligible):
        return ActionDistribution({"ad_gamble": 1.0}, stochastic=False)


def popularity(catalog):
    return PopularityPolicy(["ad_cpa", "ad_cpm"])


def test_served_decision_logs_full_propensities_and_a_verifiable_token():
    stack = make_stack(seed=3)
    outcome = stack.service.decide(make_request())
    decision, served = outcome.decision, outcome.served
    assert served is not None
    assert decision is stack.decisions[0]
    assert decision.served_by is ServedBy.HEURISTIC
    assert set(decision.eligible_actions) == {
        "ad_cpm",
        "ad_cpc",
        "ad_cpa",
        "ad_dating",
        "ad_feed",
        NO_AD,
    }
    assert sum(decision.action_probabilities.values()) == pytest.approx(1.0)
    assert decision.ope_eligible
    assert decision.chosen_action == served.ad.ad_id == decision.policy_action
    claims = stack.signer.verify(served.token, NOW)
    assert (claims.impression_id, claims.ad_id) == (served.impression_id, served.ad.ad_id)
    assert stack.budget.reserved(served.campaign.campaign_id, NOW.date()) > 0
    assert stack.log.for_user("k1:user", NOW - timedelta(hours=1))[0].ad_id == served.ad.ad_id
    json.dumps(decision.to_log())  # the decision log is plain JSON


def test_blocked_request_is_logged_as_no_ad_without_degradation():
    stack = make_stack()
    decision = stack.service.decide(make_request("journal")).decision
    assert decision.chosen_action == NO_AD
    assert decision.blocked_reason == "placement_not_allowed"
    assert not decision.degraded
    assert not decision.ope_eligible
    assert decision.action_probabilities == {NO_AD: 1.0}


def test_counter_failure_blocks_and_marks_degraded():
    stack = make_stack()
    stack.frequency.available = False
    decision = stack.service.decide(make_request()).decision
    assert decision.served_by is ServedBy.NO_AD
    assert decision.degraded_reasons == ("blocked:frequency_unavailable",)


def test_failing_level_falls_back_and_is_excluded_from_ope():
    stack = make_stack(chain=lambda catalog: [Exploding(), popularity(catalog)])
    outcome = stack.service.decide(make_request())
    decision = outcome.decision
    assert outcome.served.ad.ad_id == "ad_cpa"
    assert decision.served_by is ServedBy.POPULARITY
    assert decision.degraded_reasons == ("bandit_failed:TimeoutError",)
    assert decision.action_probabilities["ad_cpa"] == 1.0
    assert not decision.ope_eligible


def test_invalid_distribution_is_treated_as_a_failure():
    stack = make_stack(chain=lambda catalog: [OutOfBounds(), popularity(catalog)])
    outcome = stack.service.decide(make_request())
    assert outcome.served.ad.ad_id == "ad_cpa"
    assert outcome.decision.degraded_reasons == ("ranker_failed:InvalidDistribution",)


def test_all_levels_failing_serves_no_ad():
    stack = make_stack(chain=lambda catalog: [Exploding()])
    decision = stack.service.decide(make_request()).decision
    assert decision.chosen_action == NO_AD
    assert decision.degraded_reasons[-1] == "all_levels_failed"


def test_budget_race_after_sampling_serves_no_ad(monkeypatch):
    stack = make_stack(chain=lambda catalog: [popularity(catalog)])
    monkeypatch.setattr(stack.budget, "reserve", lambda *args, **kwargs: None)
    outcome = stack.service.decide(make_request())
    assert outcome.served is None
    assert outcome.decision.policy_action == "ad_cpa"
    assert outcome.decision.chosen_action == NO_AD
    assert outcome.decision.degraded_reasons == ("budget_race",)


def test_frequency_write_failure_releases_the_reservation():
    stack = make_stack(chain=lambda catalog: [popularity(catalog)])
    stack.service.decide(make_request(request_id="warm"))  # load the user's counters
    stack.log.available = False
    # Past the 10-minute min gap, before the first reservation's 15-minute TTL.
    outcome = stack.service.decide(make_request(now=NOW + timedelta(minutes=11)))
    assert outcome.served is None
    assert outcome.decision.degraded_reasons == ("frequency_unavailable",)
    # Only the first (served) request still holds a reservation.
    assert stack.budget.reserved("c_cpa", NOW.date()) == pytest.approx(20 * 0.02 * 0.05)


def test_decision_log_failure_serves_nothing_and_releases_budget():
    stack = make_stack()
    stack.store.available = False  # system of record down: the decision cannot be recorded
    outcome = stack.service.decide(make_request())
    assert outcome.served is None
    assert "decision_log_unavailable" in outcome.decision.degraded_reasons
    assert stack.service.sink_failures == 1
    assert all(stack.budget.reserved(c, NOW.date()) == 0 for c in stack.catalog.campaigns)
    assert stack.decisions == []


def test_served_decision_records_label_and_decision_message_atomically():
    stack = make_stack(chain=lambda catalog: [popularity(catalog)])
    outcome = stack.service.decide(make_request())
    label = stack.labels.label(outcome.served.impression_id)
    assert (label.ad_id, label.budget_key, label.reservation_id) == (
        "ad_cpa",
        "c_cpa",
        outcome.served.reservation_id,
    )
    assert [m.type.value for m in stack.bus.published] == ["decision"]


def test_budget_split_experiment_gives_each_arm_its_own_slice():
    experiment = Experiment("exp_b", (("a", 1.0), ("b", 3.0)), budget_split=True)
    stack = make_stack(
        experiment=experiment,
        arm_chains={"a": [popularity(None)], "b": [popularity(None)]},
    )
    keys = set()
    for i in range(20):
        user = make_user(f"k1:user{i}")
        served = stack.service.decide(make_request(user=user, request_id=f"r{i}")).served
        label = stack.labels.label(served.impression_id)
        keys.add((label.budget_key, label.budget_limit))
    assert keys == {("c_cpa@exp_b:a", 200.0 * 0.25), ("c_cpa@exp_b:b", 200.0 * 0.75)}


def test_no_ads_holdout():
    stack = make_stack(holdouts=HoldoutConfig(uniform_exploration_share=0.0, no_ads_share=0.99))
    decision = stack.service.decide(make_request()).decision
    assert decision.served_by is ServedBy.NO_ADS_HOLDOUT
    assert not decision.ope_eligible


def test_uniform_holdout_is_stochastic_and_ope_eligible():
    stack = make_stack(holdouts=HoldoutConfig(uniform_exploration_share=0.99, no_ads_share=0.0))
    decision = stack.service.decide(make_request()).decision
    assert decision.served_by is ServedBy.UNIFORM_HOLDOUT
    assert len(set(decision.action_probabilities.values())) == 1
    assert decision.ope_eligible


def test_experiment_arms_choose_chains_and_holdout_users_have_no_cohort():
    experiment = Experiment("exp_1", (("control", 1.0), ("popularity", 1.0)))
    stack = make_stack(
        experiment=experiment,
        arm_chains={
            "control": heuristic_chain(make_stack().catalog),
            "popularity": [PopularityPolicy(["ad_cpa"])],
        },
    )
    seen = {}
    for i in range(40):
        user = make_user(f"k1:user{i}")
        decision = stack.service.decide(make_request(user=user, request_id=f"r{i}")).decision
        seen[decision.experiment_cohort] = decision.served_by
    assert seen == {"exp_1:control": ServedBy.HEURISTIC, "exp_1:popularity": ServedBy.POPULARITY}

    held_out = make_stack(
        experiment=experiment,
        arm_chains={"control": [PopularityPolicy([])], "popularity": [PopularityPolicy([])]},
        holdouts=HoldoutConfig(uniform_exploration_share=0.99, no_ads_share=0.0),
    )
    assert held_out.service.decide(make_request()).decision.experiment_cohort is None


def test_experiment_without_chain_for_every_arm_is_rejected():
    with pytest.raises(ValueError, match="no policy chain"):
        make_stack(experiment=Experiment("exp", (("a", 1.0),)), arm_chains={})
