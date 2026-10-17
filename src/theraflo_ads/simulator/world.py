"""Synthetic world for the simulator (design doc: Data Sources & Simulation).

True response rates live here and are never shown to the policy. The
heuristic gets deliberately noisy priors, so the logging policy is not an
oracle. A synthetic vulnerable cohort over-responds to impulse categories:
the safety evals check that serving does not drift toward it.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from theraflo_ads.domain import (
    Ad,
    AdCatalog,
    AgeBand,
    Campaign,
    ConsentScope,
    PricingModel,
    ReviewStatus,
    UserContext,
)
from theraflo_ads.pricing import CategoryPriors
from theraflo_ads.security.identifiers import UserHasher

SAFE_CATEGORIES = (
    "fitness",
    "sleep_products",
    "books",
    "meal_kits",
    "outdoor_gear",
    "online_courses",
    "travel",
    "budgeting_apps",
    "streaming",
    "home_goods",
)
# Present in inventory so the evals can prove they are never served.
RESTRICTED_INVENTORY = ("gambling", "alcohol", "weight_loss", "health_supplements", "crypto")
MINOR_RESTRICTED_INVENTORY = ("dating",)
ALL_CATEGORIES = SAFE_CATEGORIES + RESTRICTED_INVENTORY + MINOR_RESTRICTED_INVENTORY

IMPULSE_CATEGORIES = frozenset({"streaming", "travel", "gambling", "crypto"})

PLACEMENT_MIX = (
    ("home_feed", 0.45),
    ("content_library", 0.25),
    ("journal", 0.15),
    ("mood_checkin", 0.10),
    ("therapy_chat", 0.05),
)

TRUE_CTR = {
    "fitness": 0.020,
    "sleep_products": 0.025,
    "books": 0.018,
    "meal_kits": 0.022,
    "outdoor_gear": 0.015,
    "online_courses": 0.017,
    "travel": 0.020,
    "budgeting_apps": 0.016,
    "streaming": 0.030,
    "home_goods": 0.014,
    "gambling": 0.040,
    "alcohol": 0.030,
    "weight_loss": 0.035,
    "health_supplements": 0.030,
    "crypto": 0.035,
    "dating": 0.030,
}
TRUE_CVR = {
    "fitness": 0.040,
    "sleep_products": 0.050,
    "books": 0.060,
    "meal_kits": 0.035,
    "outdoor_gear": 0.030,
    "online_courses": 0.025,
    "travel": 0.020,
    "budgeting_apps": 0.045,
    "streaming": 0.070,
    "home_goods": 0.040,
    "gambling": 0.080,
    "alcohol": 0.060,
    "weight_loss": 0.070,
    "health_supplements": 0.060,
    "crypto": 0.050,
    "dating": 0.050,
}

SIM_HASH_KEY = b"simulator-user-hash-key-not-secret!!"
SIM_TOKEN_KEY = b"simulator-token-key-not-secret-at-all"


@dataclass(frozen=True)
class SimConfig:
    n_users: int = 3000
    n_days: int = 14
    visit_prob: float = 0.3
    opportunities_per_visit: int = 3
    crisis_prob: float = 0.01
    vulnerable_share: float = 0.10
    personalized_consent_share: float = 0.4
    epsilon: float = 0.1
    # Disables fatigue, churn, caps, and budget limits so decisions are
    # independent of policy history: the setting in which OPE is unbiased.
    stateless: bool = False
    uniform_holdout_share: float = 0.02
    no_ads_holdout_share: float = 0.02
    budget_scale: float = 1.0
    viewable_rate: float = 0.92
    mean_conversion_delay_days: float = 1.5
    # `seed` builds the world (catalog, priors, users); `run_seed` drives visits,
    # responses, and policy sampling. Same seed + different run_seed = an
    # independent run in the same world.
    seed: int = 7
    run_seed: int | None = None
    # Impulse-prone categories get no behavioral personalization (the safety policy's
    # default). Turn off only to demonstrate what the safety evals catch.
    personalization_exclusions: bool = True
    # Concept drift scenario: from drift_day on, true CTRs of these categories are
    # multiplied (e.g. a competitor launch, a seasonal shift).
    drift_day: int | None = None
    drift_ctr: tuple[tuple[str, float], ...] = ()
    start: datetime = datetime(2026, 1, 5, tzinfo=UTC)
    click_window: timedelta = timedelta(days=7)


@dataclass
class SimUser:
    index: int
    user_id: str
    user_hash: str
    affinity: dict[str, float]
    vulnerable: bool
    age_band: AgeBand
    consent: ConsentScope
    suppressed_until: datetime | None = None
    churned: bool = False
    recent: deque[datetime] = field(default_factory=deque)

    def context(self) -> UserContext:
        return UserContext(
            user_hash=self.user_hash,
            consent=self.consent,
            age_band=self.age_band,
            suppressed_until=self.suppressed_until,
        )

    def exposures(self, now: datetime) -> int:
        while self.recent and self.recent[0] <= now - timedelta(hours=24):
            self.recent.popleft()
        return len(self.recent)


@dataclass
class World:
    config: SimConfig
    catalog: AdCatalog
    priors: CategoryPriors
    users: list[SimUser]
    hasher: UserHasher
    # Set by the runner: the FeatureBuilder serving and logging use in this run.
    features: Any = None

    def p_click(self, user: SimUser, ad: Ad, exposures: int, when: datetime | None = None) -> float:
        p = TRUE_CTR[ad.category] * user.affinity[ad.category]
        config = self.config
        if (
            config.drift_day is not None
            and when is not None
            and when >= config.start + timedelta(days=config.drift_day)
        ):
            p *= dict(config.drift_ctr).get(ad.category, 1.0)
        if user.vulnerable and ad.category in IMPULSE_CATEGORIES:
            p *= 2.0
        if not self.config.stateless:
            p /= 1.0 + 0.15 * exposures
        return min(p, 0.5)

    def p_convert(self, user: SimUser, ad: Ad) -> float:
        p = TRUE_CVR[ad.category]
        if user.vulnerable and ad.category in IMPULSE_CATEGORIES:
            p *= 1.5
        return min(p, 0.9)


def build_world(config: SimConfig) -> World:
    rng = random.Random(config.seed)
    budget_scale = config.budget_scale * (1e6 if config.stateless else 1.0)
    campaigns: dict[str, Campaign] = {}
    ads: dict[str, Ad] = {}
    n = 0
    for category in ALL_CATEGORIES:
        for _ in range(2 if category in SAFE_CATEGORIES else 1):
            n += 1
            model = rng.choices(
                [PricingModel.CPM, PricingModel.CPC, PricingModel.CPA], weights=[0.4, 0.4, 0.2]
            )[0]
            price = {
                PricingModel.CPM: rng.uniform(4, 12),
                PricingModel.CPC: rng.uniform(0.3, 1.5),
                PricingModel.CPA: rng.uniform(8, 30),
            }[model]
            # Budgets are sized relative to price and small enough that popular
            # campaigns exhaust them daily, so pacing is exercised.
            budget = {
                PricingModel.CPM: rng.uniform(1, 6),
                PricingModel.CPC: price * rng.uniform(5, 15),
                PricingModel.CPA: price * rng.uniform(3, 8),
            }[model]
            campaign = Campaign(
                campaign_id=f"cmp_{n:03d}",
                advertiser_id=f"adv_{n % 12:02d}",
                pricing_model=model,
                price=round(price, 2),
                daily_budget=round(budget * budget_scale, 2),
                priority=rng.randint(0, 10),
            )
            campaigns[campaign.campaign_id] = campaign
            ad = Ad(f"ad_{n:03d}", campaign.campaign_id, category, f"https://partner.example/{n}")
            ads[ad.ad_id] = ad

    # Inventory that must never serve: a creative awaiting review and an
    # inactive campaign.
    campaigns["cmp_inactive"] = Campaign(
        "cmp_inactive", "adv_99", PricingModel.CPM, 50.0, 1e9, active=False
    )
    ads["ad_inactive"] = Ad("ad_inactive", "cmp_inactive", "books", "https://partner.example/x")
    ads["ad_pending"] = Ad(
        "ad_pending",
        "cmp_001",
        "fitness",
        "https://partner.example/p",
        review_status=ReviewStatus.PENDING,
    )

    noise = {c: rng.lognormvariate(0.0, 0.35) for c in ALL_CATEGORIES}
    priors = CategoryPriors(
        ctr={c: TRUE_CTR[c] * noise[c] for c in ALL_CATEGORIES},
        cvr={c: TRUE_CVR[c] * rng.lognormvariate(0.0, 0.35) for c in ALL_CATEGORIES},
    )

    hasher = UserHasher("ads", {"sim": SIM_HASH_KEY}, "sim")
    users = []
    for i in range(config.n_users):
        weights = [rng.gammavariate(2.0, 1.0) for _ in ALL_CATEGORIES]
        total = sum(weights)
        affinity = {
            c: w / total * len(ALL_CATEGORIES) for c, w in zip(ALL_CATEGORIES, weights, strict=True)
        }
        age_band = rng.choices(
            [AgeBand.ADULT, AgeBand.TEEN, AgeBand.UNDER_13, AgeBand.UNKNOWN],
            weights=[0.85, 0.07, 0.03, 0.05],
        )[0]
        consent = (
            ConsentScope.PERSONALIZED
            if rng.random() < config.personalized_consent_share
            else ConsentScope.CONTEXTUAL_ONLY
        )
        user_id = f"user_{i:05d}"
        users.append(
            SimUser(
                index=i,
                user_id=user_id,
                user_hash=hasher.hash(user_id),
                affinity=affinity,
                vulnerable=rng.random() < config.vulnerable_share,
                age_band=age_band,
                consent=consent,
            )
        )
    return World(config, AdCatalog(campaigns, ads), priors, users, hasher)
