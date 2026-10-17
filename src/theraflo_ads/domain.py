"""Core domain types shared by every layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

NO_AD = "NO_AD"

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class PricingModel(StrEnum):
    CPM = "cpm"
    CPC = "cpc"
    CPA = "cpa"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ConsentScope(StrEnum):
    CONTEXTUAL_ONLY = "contextual_only"
    PERSONALIZED = "personalized"


class AgeBand(StrEnum):
    UNDER_13 = "under_13"
    TEEN = "teen"
    ADULT = "adult"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    advertiser_id: str
    pricing_model: PricingModel
    # CPM: price per 1,000 impressions. CPC: price per click. CPA: payout per conversion.
    price: float
    daily_budget: float
    active: bool = True
    # None means "any placement on the safety allowlist".
    placements: frozenset[str] | None = None
    priority: int = 0


@dataclass(frozen=True)
class Ad:
    ad_id: str
    campaign_id: str
    category: str
    landing_url: str
    review_status: ReviewStatus = ReviewStatus.APPROVED


@dataclass
class AdCatalog:
    campaigns: dict[str, Campaign]
    ads: dict[str, Ad]

    def __post_init__(self) -> None:
        for ad in self.ads.values():
            if ad.campaign_id not in self.campaigns:
                raise ValueError(f"ad {ad.ad_id} references unknown campaign {ad.campaign_id}")

    def campaign_for(self, ad: Ad) -> Campaign:
        return self.campaigns[ad.campaign_id]

    def candidates(self) -> list[Ad]:
        return list(self.ads.values())


@dataclass(frozen=True)
class UserContext:
    """Ads-safe view of a user. Never carries wellness data."""

    user_hash: str
    consent: ConsentScope = ConsentScope.CONTEXTUAL_ONLY
    age_band: AgeBand = AgeBand.UNKNOWN
    region: str | None = None
    device: str | None = None
    suppressed_until: datetime | None = None
    hidden_ads: frozenset[str] = frozenset()
    hidden_advertisers: frozenset[str] = frozenset()


@dataclass(frozen=True)
class AdRequest:
    request_id: str
    session_id: str
    placement_id: str
    user: UserContext
    now: datetime
    # (slice name, share of each campaign's budget) when an experiment splits budgets.
    budget_slice: tuple[str, float] | None = None
    # Set by the DecisionService after eligibility, for features (FrequencySnapshot).
    frequency: Any = None
    # Logged, reproducible feature inputs (exposures, consented interactions, device).
    # When set, features are rebuilt from it instead of live stores (offline replay, OPE).
    feature_context: Any = None
