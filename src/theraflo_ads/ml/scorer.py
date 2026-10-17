"""Serving-time ML scorer (design doc: Prediction Models, Initial Ranking).

Calibrated pCTR and post-click pCVR for every eligible ad, from the same
FeatureBuilder that writes the decision log. Expected platform revenue follows
the pricing model. Ads whose predicted CVR is below the quality floor are
dropped before the policy samples.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from theraflo_ads.domain import Ad, AdCatalog, AdRequest
from theraflo_ads.ml.bundle import ModelBundle
from theraflo_ads.ml.features import FeatureBuilder
from theraflo_ads.policy.scoring import ScoredAd
from theraflo_ads.pricing import CategoryPriors, expected_revenue_per_impression


class MLScorer:
    def __init__(
        self,
        catalog: AdCatalog,
        features: FeatureBuilder,
        ctr: ModelBundle,
        cvr: ModelBundle | None,
        *,
        priors: CategoryPriors | None = None,
        min_p_cvr: float = 0.0,
    ) -> None:
        if cvr is None and priors is None:
            raise ValueError("without a CVR model, priors are required for pCVR")
        self._catalog = catalog
        self._features = features
        self._ctr = ctr
        self._cvr = cvr
        self._priors = priors
        self._min_p_cvr = min_p_cvr
        self.version = f"ml:{ctr.version}+{cvr.version if cvr else 'prior'}"

    def score(self, request: AdRequest, ads: Sequence[Ad]) -> list[ScoredAd]:
        if not ads:
            return []
        rows = self._features.build(request, ads)
        p_ctr = self._ctr.model.predict(_matrix(rows, self._ctr.model.feature_names))
        if self._cvr is not None:
            p_cvr = self._cvr.model.predict(_matrix(rows, self._cvr.model.feature_names))
        else:
            # Hybrid: conversions are too sparse at this scale, so pCVR stays on the priors.
            if self._priors is None:
                raise ValueError("hybrid scoring needs category priors for pCVR")
            p_cvr = np.array([self._priors.p_cvr(ad.category) for ad in ads])
        scored = []
        for ad, ctr, cvr in zip(ads, p_ctr, p_cvr, strict=True):
            if cvr < self._min_p_cvr:
                continue
            campaign = self._catalog.campaign_for(ad)
            revenue = expected_revenue_per_impression(campaign, float(ctr), float(cvr))
            scored.append(ScoredAd(ad, campaign, float(ctr), float(cvr), revenue))
        return scored


def _matrix(rows: Sequence[dict[str, float]], names: Sequence[str]) -> np.ndarray:
    # Features the model was not trained on are ignored; missing ones default to 0.
    return np.array([[row.get(name, 0.0) for name in names] for row in rows], dtype=float)
