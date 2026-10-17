"""Platform revenue definitions (design doc: Pricing, Billing & Revenue Definition)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from theraflo_ads.domain import Ad, Campaign, PricingModel


def expected_revenue_per_impression(campaign: Campaign, p_ctr: float, p_cvr: float) -> float:
    """What Theraflo expects to bill for one impression. Not the advertiser's order value."""
    match campaign.pricing_model:
        case PricingModel.CPM:
            return campaign.price / 1000.0
        case PricingModel.CPC:
            return campaign.price * p_ctr
        case PricingModel.CPA:
            return campaign.price * p_ctr * p_cvr


def ecpm(campaign: Campaign, p_ctr: float, p_cvr: float) -> float:
    return 1000.0 * expected_revenue_per_impression(campaign, p_ctr, p_cvr)


def budget_scope(campaign: Campaign, budget_slice: tuple[str, float] | None) -> tuple[str, float]:
    """(budget key, daily limit) for a request: the whole campaign budget, or one experiment
    slice of it (design doc: Online Experimentation → Marketplace Interference)."""
    if budget_slice is None:
        return campaign.campaign_id, campaign.daily_budget
    name, share = budget_slice
    return f"{campaign.campaign_id}@{name}", campaign.daily_budget * share


def max_event_cost(campaign: Campaign) -> float:
    """The largest single charge one impression can trigger.

    Reservations hold expected cost, but a realized click or conversion bills
    the full price, so a campaign stays eligible only while its remaining
    budget can absorb one such event.
    """
    if campaign.pricing_model is PricingModel.CPM:
        return campaign.price / 1000.0
    return campaign.price


@dataclass(frozen=True)
class CategoryPriors:
    """Per-category CTR / post-click CVR priors used by the heuristic scorer and for budgeting."""

    ctr: Mapping[str, float] = field(default_factory=dict)
    cvr: Mapping[str, float] = field(default_factory=dict)
    default_ctr: float = 0.01
    default_cvr: float = 0.03

    def p_ctr(self, category: str) -> float:
        return self.ctr.get(category, self.default_ctr)

    def p_cvr(self, category: str) -> float:
        return self.cvr.get(category, self.default_cvr)


@dataclass(frozen=True)
class PriorCostEstimator:
    """Expected cost of one impression, used for budget eligibility and reservations."""

    priors: CategoryPriors

    def __call__(self, ad: Ad, campaign: Campaign) -> float:
        return expected_revenue_per_impression(
            campaign, self.priors.p_ctr(ad.category), self.priors.p_cvr(ad.category)
        )
