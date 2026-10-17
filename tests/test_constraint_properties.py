"""Constraint evals (design doc: Evals → Constraint Evals).

Random request sequences, random user state, random infrastructure failures,
and every fallback level: no served ad may ever break a hard constraint.
"""

from collections import defaultdict
from datetime import timedelta

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from tests.helpers import (
    NO_HOLDOUTS,
    NOW,
    PRIORS,
    heuristic_chain,
    make_request,
    make_stack,
    make_user,
)
from tests.test_decision_service import Exploding
from theraflo_ads.domain import AgeBand, ReviewStatus
from theraflo_ads.experiments import HoldoutConfig
from theraflo_ads.policy import PopularityPolicy
from theraflo_ads.policy.bandits import BetaThompsonPolicy, ClickCounts
from theraflo_ads.safety.engine import restricted_categories
from theraflo_ads.safety.policy import AD_FREE_PLACEMENTS

PLACEMENTS = ["home_feed", "content_library", *sorted(AD_FREE_PLACEMENTS), "unknown"]
AD_IDS = ["ad_cpm", "ad_cpc", "ad_cpa", "ad_gamble", "ad_dating", "ad_feed"]

CHAINS = {
    "heuristic": heuristic_chain,
    "bandit_down": lambda catalog: [Exploding(), *heuristic_chain(catalog)],
    "popularity_only": lambda catalog: [PopularityPolicy(sorted(catalog.ads))],
    "bandit": lambda catalog: [
        BetaThompsonPolicy(catalog, PRIORS, ClickCounts(), samples=50),
        *heuristic_chain(catalog),
    ],
}

# Each strategy leans toward the serving path (one_of picks branches evenly),
# so served ads are common enough for the invariants to be exercised.
users = st.builds(
    lambda i, age, suppressed_minutes, hidden: make_user(
        f"k1:u{i}",
        age_band=age,
        suppressed_until=(
            None if suppressed_minutes is None else NOW + timedelta(minutes=suppressed_minutes)
        ),
        hidden_ads=frozenset(hidden),
    ),
    st.integers(0, 3),
    st.just(AgeBand.ADULT) | st.sampled_from(list(AgeBand)),
    st.none() | st.none() | st.integers(-600, 600),
    st.sets(st.sampled_from(AD_IDS), max_size=2),
)
placements = st.sampled_from(["home_feed", "content_library"]) | st.sampled_from(PLACEMENTS)
faults = st.just("none") | st.sampled_from(
    ["kill_switch", "flush_cache", "counters_down", "budget_down"]
)

steps = st.lists(
    st.tuples(users, placements, st.integers(0, 15), faults),  # minutes since previous request
    min_size=1,
    max_size=40,
)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    steps=steps,
    chain=st.sampled_from(sorted(CHAINS)),
    uniform_holdout=st.booleans(),
    seed=st.integers(0, 10_000),
)
def test_no_served_ad_ever_breaks_a_hard_constraint(steps, chain, uniform_holdout, seed):
    holdouts = (
        HoldoutConfig(uniform_exploration_share=0.5, no_ads_share=0.2)
        if uniform_holdout
        else NO_HOLDOUTS
    )
    stack = make_stack(chain=CHAINS[chain], holdouts=holdouts, seed=seed)
    caps = stack.frequency.caps
    policy = stack.safety.current()
    served_history = defaultdict(list)
    now = NOW

    for i, (user, placement, gap, fault) in enumerate(steps):
        now += timedelta(minutes=gap)
        stack.safety.set_kill_switch(fault == "kill_switch")
        stack.frequency.available = fault != "counters_down"
        stack.budget.available = fault != "budget_down"
        if fault == "flush_cache":
            stack.frequency.flush()

        outcome = stack.service.decide(make_request(placement, user, now, f"r{i}"))
        served = outcome.served
        if served is None:
            continue
        event(f"served by {outcome.decision.served_by.value}")

        ad, campaign = served.ad, served.campaign
        assert fault not in {"kill_switch", "counters_down", "budget_down"}
        assert placement in policy.ad_placements
        assert user.age_band is not AgeBand.UNDER_13
        assert user.suppressed_until is None or now >= user.suppressed_until
        assert ad.category not in restricted_categories(policy, user.age_band)
        assert ad.review_status is ReviewStatus.APPROVED
        assert campaign.active
        assert campaign.placements is None or placement in campaign.placements
        assert ad.ad_id not in user.hidden_ads

        window = [h for h in served_history[user.user_hash] if h[0] > now - caps.window]
        assert sum(h[1] == ad.ad_id for h in window) < caps.per_ad_24h
        assert sum(h[2] == campaign.advertiser_id for h in window) < caps.per_advertiser_24h
        assert len(window) < caps.per_user_24h
        assert all(now - h[0] >= caps.min_gap_same_ad for h in window if h[1] == ad.ad_id)
        served_history[user.user_hash].append((now, ad.ad_id, campaign.advertiser_id))

        stack.budget.available = True
        for c in stack.catalog.campaigns.values():
            spent = stack.budget.billed(c.campaign_id, now.date())
            assert spent + stack.budget.reserved(c.campaign_id, now.date()) <= c.daily_budget

    # Every decision, served or not, made it to the decision log.
    assert len(stack.decisions) == len(steps)
