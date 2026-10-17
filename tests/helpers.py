"""Builders for an in-memory platform used across tests."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from theraflo_ads.attribution.service import AttributionService
from theraflo_ads.counters.frequency import FrequencyCaps
from theraflo_ads.domain import (
    Ad,
    AdCatalog,
    AdRequest,
    AgeBand,
    Campaign,
    ConsentScope,
    PricingModel,
    ReviewStatus,
    UserContext,
)
from theraflo_ads.experiments.assignment import HoldoutConfig
from theraflo_ads.platform import Platform, build_memory_platform
from theraflo_ads.policy.decision import Decision
from theraflo_ads.policy.policies import EpsilonGreedyPolicy, Policy, PopularityPolicy
from theraflo_ads.policy.scoring import HeuristicScorer
from theraflo_ads.pricing import CategoryPriors, PriorCostEstimator
from theraflo_ads.security.identifiers import UserHasher
from theraflo_ads.security.tokens import ImpressionTokenSigner

NOW = datetime(2026, 3, 2, 12, 0, tzinfo=UTC)
HASH_KEY = b"h" * 32
TOKEN_KEY = b"t" * 32
NO_HOLDOUTS = HoldoutConfig(uniform_exploration_share=0.0, no_ads_share=0.0)


def make_catalog() -> AdCatalog:
    campaigns = [
        Campaign("c_cpm", "adv_a", PricingModel.CPM, 10.0, 100.0, priority=5),
        Campaign("c_cpc", "adv_b", PricingModel.CPC, 1.0, 50.0, priority=3),
        Campaign("c_cpa", "adv_c", PricingModel.CPA, 20.0, 200.0, priority=1),
        Campaign("c_gamble", "adv_d", PricingModel.CPM, 50.0, 100.0, priority=9),
        Campaign("c_dating", "adv_e", PricingModel.CPM, 9.0, 100.0, priority=2),
        Campaign(
            "c_feed", "adv_f", PricingModel.CPM, 8.0, 100.0, placements=frozenset({"home_feed"})
        ),
        Campaign("c_inactive", "adv_g", PricingModel.CPM, 99.0, 100.0, active=False),
    ]
    ads = [
        Ad("ad_cpm", "c_cpm", "fitness", "https://partner.example/cpm"),
        Ad("ad_cpc", "c_cpc", "books", "https://partner.example/cpc"),
        Ad("ad_cpa", "c_cpa", "sleep_products", "https://partner.example/cpa"),
        Ad("ad_gamble", "c_gamble", "gambling", "https://partner.example/g"),
        Ad("ad_dating", "c_dating", "dating", "https://partner.example/d"),
        Ad("ad_feed", "c_feed", "travel", "https://partner.example/f"),
        Ad("ad_inactive", "c_inactive", "books", "https://partner.example/i"),
        Ad(
            "ad_pending",
            "c_cpm",
            "fitness",
            "https://partner.example/p",
            review_status=ReviewStatus.PENDING,
        ),
    ]
    return AdCatalog({c.campaign_id: c for c in campaigns}, {a.ad_id: a for a in ads})


PRIORS = CategoryPriors(
    ctr={"fitness": 0.02, "books": 0.03, "sleep_products": 0.02, "travel": 0.01, "dating": 0.02},
    cvr={"fitness": 0.05, "books": 0.04, "sleep_products": 0.05, "travel": 0.02, "dating": 0.03},
)


@dataclass
class Stack:
    """A Platform plus the short names tests use."""

    platform: Platform
    hasher: UserHasher

    def __getattr__(self, name):
        return getattr(self.platform, name)

    @property
    def service(self):
        return self.platform.decisions

    @property
    def log(self):
        return self.platform.impression_log

    @property
    def decisions(self) -> list[Decision]:
        return self.platform.decision_log()

    @property
    def labels(self) -> AttributionService:
        return self.platform.attribution()


def heuristic_chain(catalog: AdCatalog, epsilon: float = 0.1) -> list[Policy]:
    scorer = HeuristicScorer(catalog, PRIORS)
    ranked = sorted(catalog.ads, key=lambda a: -catalog.campaign_for(catalog.ads[a]).priority)
    return [EpsilonGreedyPolicy(scorer, epsilon=epsilon), PopularityPolicy(ranked)]


def make_stack(
    *,
    catalog: AdCatalog | None = None,
    chain: Callable[[AdCatalog], Sequence[Policy]] | None = None,
    holdouts: HoldoutConfig = NO_HOLDOUTS,
    caps: FrequencyCaps | None = None,
    seed: int = 0,
    **platform_options,
) -> Stack:
    catalog = catalog or make_catalog()
    platform = build_memory_platform(
        catalog=catalog,
        cost=PriorCostEstimator(PRIORS),
        chain=(chain or heuristic_chain)(catalog),
        signer=ImpressionTokenSigner(TOKEN_KEY),
        holdouts=holdouts,
        caps=caps,
        rng=random.Random(seed),
        **platform_options,
    )
    return Stack(platform, UserHasher("ads", {"k1": HASH_KEY}, "k1"))


def make_user(user_hash: str = "k1:user", **overrides) -> UserContext:
    options = {"consent": ConsentScope.PERSONALIZED, "age_band": AgeBand.ADULT} | overrides
    return UserContext(user_hash=user_hash, **options)


def make_request(
    placement: str = "home_feed",
    user: UserContext | None = None,
    now: datetime = NOW,
    request_id: str = "r1",
) -> AdRequest:
    return AdRequest(request_id, "s1", placement, user or make_user(), now)
