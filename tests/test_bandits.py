"""Thompson-sampling bandits, their updater, and the Redis feature store."""

from __future__ import annotations

import random
from datetime import UTC, datetime

import fakeredis
import numpy as np
import pytest

from tests.helpers import NOW, PRIORS, make_catalog, make_request, make_stack, make_user
from theraflo_ads.domain import NO_AD, ConsentScope
from theraflo_ads.events.bus import Message, MessageType
from theraflo_ads.infra.redis_features import (
    FeatureAggregatorWorker,
    RedisClickCounts,
    RedisInteractionStore,
)
from theraflo_ads.ml.features import FeatureBuilder, InMemoryInteractionStore
from theraflo_ads.policy.bandits import (
    BanditUpdater,
    BetaThompsonPolicy,
    ClickCounts,
    LinearBanditState,
    LinearThompsonPolicy,
)
from theraflo_ads.policy.decision import ServedBy
from theraflo_ads.policy.policies import validate_distribution
from theraflo_ads.simulator import PolicySetup, SimConfig, run_simulation
from theraflo_ads.simulator.ab import heuristic_arm

CATALOG = make_catalog()
SAFE = [CATALOG.ads[a] for a in ("ad_cpm", "ad_cpc", "ad_cpa")]


def beta(counts=None, **options):
    return BetaThompsonPolicy(
        CATALOG, PRIORS, counts or ClickCounts(), rng=random.Random(1), **options
    )


def test_beta_distribution_is_valid_with_floor_and_no_ad():
    dist = beta(floor=0.04).distribution(make_request(), SAFE)
    validate_distribution(dist, {a.ad_id for a in SAFE})
    assert set(dist.probabilities) == {"ad_cpm", "ad_cpc", "ad_cpa", NO_AD}
    assert min(dist.probabilities.values()) >= 0.04 / 4 - 1e-12
    assert dist.stochastic


def test_beta_posterior_moves_mass_to_the_better_ad():
    counts = ClickCounts()
    for _ in range(5000):
        counts.add("ad_cpa", impressions=1, clicks=0)
    for _ in range(5000):
        counts.add("ad_cpc", impressions=1, clicks=1 if random.random() < 0.06 else 0)
    dist = beta(counts).distribution(make_request(), SAFE)
    assert dist.probabilities["ad_cpc"] > 0.9


def test_linear_warm_start_predicts_the_prior():
    features = FeatureBuilder(CATALOG, PRIORS)
    state = LinearBanditState(features.names)
    mean, _ = state.posterior()
    x = np.array([features.build(make_request(), SAFE)[0][n] for n in features.names])
    assert float(x @ mean) == pytest.approx(PRIORS.p_ctr("fitness"))


def test_linear_ts_learns_from_clicks():
    features = FeatureBuilder(CATALOG, PRIORS)
    state = LinearBanditState(features.names)
    policy = LinearThompsonPolicy(CATALOG, PRIORS, features, state, rng=random.Random(3))
    rows = {
        r_ad.ad_id: r for r_ad, r in zip(SAFE, features.build(make_request(), SAFE), strict=True)
    }
    for _ in range(3000):
        state.update(np.array([rows["ad_cpa"][n] for n in features.names]), 1.0)
        state.update(np.array([rows["ad_cpc"][n] for n in features.names]), 0.0)
    dist = policy.distribution(make_request(), SAFE)
    validate_distribution(dist, {a.ad_id for a in SAFE})
    assert dist.probabilities["ad_cpa"] > 0.9


def test_updater_learns_from_served_impressions_and_clicks():
    counts = ClickCounts()
    policy = beta(counts)
    stack = make_stack(chain=lambda c: [policy])
    BanditUpdater(beta=policy).subscribe(stack.bus)
    served = stack.service.decide(make_request()).served
    stack.events.click(served.impression_id, NOW)
    impressions, clicks = counts.snapshot()
    assert impressions[served.ad.ad_id] == 1
    assert clicks[served.ad.ad_id] == 1
    assert stack.decisions[0].served_by is ServedBy.BANDIT
    assert stack.decisions[0].ope_eligible


def test_redis_counts_and_interactions():
    client = fakeredis.FakeRedis()
    counts = RedisClickCounts(client)
    counts.add("ad_1", impressions=3, clicks=1)
    assert counts.snapshot() == ({"ad_1": 3.0}, {"ad_1": 1.0})
    store = RedisInteractionStore(client)
    store.record_impression("u1", "books")
    store.record_click("u1", "books")
    assert store.get("u1").clicks == {"books": 1.0}
    assert client.ttl("feat:u1") > 0  # retention window
    store.forget("u1")
    assert store.get("u1").clicks == {}


def test_feature_aggregator_worker_is_idempotent_and_consent_scoped():
    client = fakeredis.FakeRedis()
    interactions, counts = RedisInteractionStore(client), RedisClickCounts(client)
    worker = FeatureAggregatorWorker(client, CATALOG, interactions, counts)
    now = datetime(2026, 3, 2, tzinfo=UTC)

    def decision(request_id, user, consent):
        return Message(
            f"decision:{request_id}",
            MessageType.DECISION,
            {
                "impression_ids": [f"imp-{request_id}"],
                "degraded": False,
                "chosen_actions": [{"ad_id": "ad_cpc", "position": 1}],
                "consent_scope": consent,
                "user_hash": user,
            },
            now,
        )

    consented = decision("a", "u1", ConsentScope.PERSONALIZED.value)
    worker(consented)
    worker(consented)  # redelivered
    worker(decision("b", "u2", ConsentScope.CONTEXTUAL_ONLY.value))
    worker(Message("click:imp-a", MessageType.AD_CLICKED, {"impression_id": "imp-a"}, now))
    assert counts.snapshot() == ({"ad_cpc": 2.0}, {"ad_cpc": 1.0})
    assert interactions.get("u1").clicks == {"books": 1.0}
    assert interactions.get("u2") == type(interactions.get("u2"))()  # no data without consent


def test_bandit_beats_the_heuristic_in_the_simulator():
    def setup(world):
        policy = BetaThompsonPolicy(
            world.catalog, world.priors, ClickCounts(), rng=random.Random(5)
        )
        return PolicySetup(
            (policy, *heuristic_arm(world)), subscribers=[BanditUpdater(beta=policy).subscribe]
        )

    config = SimConfig(n_users=2500, n_days=21, seed=7, run_seed=301)
    heuristic = run_simulation(config)
    bandit = run_simulation(config, setup)
    assert bandit.violations == {} and heuristic.violations == {}
    assert bandit.platform_revenue > heuristic.platform_revenue
    vulnerable, other = bandit.cohorts["vulnerable"], bandit.cohorts["other"]
    assert abs(vulnerable.impulse_share - other.impulse_share) < 0.05


def test_contextual_users_cannot_be_targeted_by_linear_ts():
    store = InMemoryInteractionStore()
    for _ in range(50):
        store.record_click("k1:ctx", "sleep_products")
    features = FeatureBuilder(CATALOG, PRIORS, interactions=store)
    contextual = make_request(user=make_user("k1:ctx", consent=ConsentScope.CONTEXTUAL_ONLY))
    assert features.build(contextual, SAFE)[2]["user_total_clicks"] == 0
