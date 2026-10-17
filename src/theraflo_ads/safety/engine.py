"""Hard constraint engine (design doc: Constraint Engine and Slate Assembly).

Runs before the policy samples, so logged propensities are computed over the
set of ads that can actually be served. Any unreadable safety-critical state
blocks the whole request (fail closed).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from theraflo_ads.counters.budget import BudgetStore
from theraflo_ads.counters.frequency import CounterUnavailable, FrequencySnapshot, FrequencyStore
from theraflo_ads.domain import (
    Ad,
    AdCatalog,
    AdRequest,
    AgeBand,
    Campaign,
    ConsentScope,
    ReviewStatus,
)
from theraflo_ads.pricing import budget_scope, max_event_cost
from theraflo_ads.safety.policy import SafetyPolicy, SafetyPolicyStore


class BlockReason(StrEnum):
    KILL_SWITCH = "kill_switch"
    PLACEMENT_NOT_ALLOWED = "placement_not_allowed"
    UNDER_13 = "under_13"
    SUPPRESSION_WINDOW = "suppression_window"
    AD_LOAD_CAP = "ad_load_cap"
    FREQUENCY_UNAVAILABLE = "frequency_unavailable"
    BUDGET_UNAVAILABLE = "budget_unavailable"
    NO_ELIGIBLE_ADS = "no_eligible_ads"


# Blocks caused by an infrastructure failure rather than a policy decision.
FAILURE_BLOCKS = frozenset({BlockReason.FREQUENCY_UNAVAILABLE, BlockReason.BUDGET_UNAVAILABLE})


class AdRejection(StrEnum):
    NOT_APPROVED = "not_approved"
    RESTRICTED_CATEGORY = "restricted_category"
    CAMPAIGN_INACTIVE = "campaign_inactive"
    PLACEMENT_TARGETING = "placement_targeting"
    HIDDEN_BY_USER = "hidden_by_user"
    FREQUENCY_CAP = "frequency_cap"
    BUDGET = "budget"


@dataclass(frozen=True)
class Eligibility:
    eligible: tuple[Ad, ...]
    effective_consent: ConsentScope
    safety_policy_version: str
    blocked: BlockReason | None = None
    rejections: Mapping[str, AdRejection] = field(default_factory=dict)
    snapshot: FrequencySnapshot | None = None

    @property
    def is_blocked(self) -> bool:
        return self.blocked is not None


CostEstimator = Callable[[Ad, Campaign], float]


class ConstraintEngine:
    def __init__(
        self,
        *,
        catalog: AdCatalog,
        policies: SafetyPolicyStore,
        frequency: FrequencyStore,
        budget: BudgetStore,
        cost_estimator: CostEstimator,
    ) -> None:
        self._catalog = catalog
        self._policies = policies
        self._frequency = frequency
        self._budget = budget
        self._cost = cost_estimator

    def evaluate(self, request: AdRequest) -> Eligibility:
        policy = self._policies.current()
        user = request.user
        consent = effective_consent(user.consent, user.age_band)

        def blocked(reason: BlockReason, rejections: Mapping[str, AdRejection] | None = None):
            return Eligibility((), consent, policy.version, reason, rejections or {})

        if policy.kill_switch:
            return blocked(BlockReason.KILL_SWITCH)
        if request.placement_id not in policy.ad_placements:
            return blocked(BlockReason.PLACEMENT_NOT_ALLOWED)
        if user.age_band is AgeBand.UNDER_13:
            return blocked(BlockReason.UNDER_13)
        if user.suppressed_until is not None and request.now < user.suppressed_until:
            return blocked(BlockReason.SUPPRESSION_WINDOW)

        try:
            snapshot = self._frequency.snapshot(user.user_hash, request.now)
        except CounterUnavailable:
            return blocked(BlockReason.FREQUENCY_UNAVAILABLE)
        if snapshot.user_at_cap(self._frequency.caps):
            return blocked(BlockReason.AD_LOAD_CAP)

        restricted = restricted_categories(policy, user.age_band)
        eligible: list[Ad] = []
        rejections: dict[str, AdRejection] = {}
        for ad in self._catalog.candidates():
            campaign = self._catalog.campaign_for(ad)
            reason = self._static_rejection(ad, campaign, request, restricted)
            if reason is None and not snapshot.allows(
                ad.ad_id, campaign.advertiser_id, self._frequency.caps, request.now
            ):
                reason = AdRejection.FREQUENCY_CAP
            if reason is None:
                budget_key, limit = budget_scope(campaign, request.budget_slice)
                try:
                    affordable = self._budget.can_reserve(
                        budget_key,
                        limit,
                        self._cost(ad, campaign),
                        request.now,
                        min_headroom=max_event_cost(campaign),
                    )
                except CounterUnavailable:
                    return blocked(BlockReason.BUDGET_UNAVAILABLE)
                if not affordable:
                    reason = AdRejection.BUDGET
            if reason is None:
                eligible.append(ad)
            else:
                rejections[ad.ad_id] = reason

        if not eligible:
            return blocked(BlockReason.NO_ELIGIBLE_ADS, rejections)
        return Eligibility(tuple(eligible), consent, policy.version, None, rejections, snapshot)

    @staticmethod
    def _static_rejection(
        ad: Ad, campaign: Campaign, request: AdRequest, restricted: frozenset[str]
    ) -> AdRejection | None:
        user = request.user
        if ad.review_status is not ReviewStatus.APPROVED:
            return AdRejection.NOT_APPROVED
        if ad.category in restricted:
            return AdRejection.RESTRICTED_CATEGORY
        if not campaign.active:
            return AdRejection.CAMPAIGN_INACTIVE
        if campaign.placements is not None and request.placement_id not in campaign.placements:
            return AdRejection.PLACEMENT_TARGETING
        if ad.ad_id in user.hidden_ads or campaign.advertiser_id in user.hidden_advertisers:
            return AdRejection.HIDDEN_BY_USER
        return None


def effective_consent(consent: ConsentScope, age_band: AgeBand) -> ConsentScope:
    """Minors and users of unknown age are always contextual-only."""
    if age_band is AgeBand.ADULT:
        return consent
    return ConsentScope.CONTEXTUAL_ONLY


def restricted_categories(policy: SafetyPolicy, age_band: AgeBand) -> frozenset[str]:
    if age_band is AgeBand.ADULT:
        return policy.restricted_categories
    return policy.restricted_categories | policy.minor_restricted_categories
