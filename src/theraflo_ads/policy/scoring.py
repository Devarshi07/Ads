"""Scoring (design doc: Initial Ranking)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from theraflo_ads.domain import Ad, AdCatalog, AdRequest, Campaign
from theraflo_ads.pricing import CategoryPriors, expected_revenue_per_impression


@dataclass(frozen=True)
class ScoredAd:
    ad: Ad
    campaign: Campaign
    p_ctr: float
    p_cvr: float
    expected_revenue: float


class HeuristicScorer:
    """Phase 1 baseline: expected platform revenue from category priors.

    The quality floor drops ads whose predicted CVR is too low (clickbait
    guard). It runs before the policy samples, like every filter that removes
    an ad.
    """

    def __init__(
        self,
        catalog: AdCatalog,
        priors: CategoryPriors,
        *,
        min_p_cvr: float = 0.0,
        version: str = "heuristic_scorer_v1",
    ) -> None:
        self._catalog = catalog
        self._priors = priors
        self._min_p_cvr = min_p_cvr
        self.version = version

    def score(self, request: AdRequest, ads: Sequence[Ad]) -> list[ScoredAd]:
        scored = []
        for ad in ads:
            campaign = self._catalog.campaign_for(ad)
            p_ctr = self._priors.p_ctr(ad.category)
            p_cvr = self._priors.p_cvr(ad.category)
            if p_cvr < self._min_p_cvr:
                continue
            revenue = expected_revenue_per_impression(campaign, p_ctr, p_cvr)
            scored.append(ScoredAd(ad, campaign, p_ctr, p_cvr, revenue))
        return scored
