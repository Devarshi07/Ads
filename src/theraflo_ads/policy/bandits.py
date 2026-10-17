"""Thompson-sampling bandits (design doc: Contextual Bandits, RL Safety, Post-Training →
Bandit Warm Start).

Thompson-sampling action probabilities have no closed form, so each policy
estimates them by Monte Carlo, mixes in an exploration floor, and returns that
distribution; the DecisionService samples from exactly what it logs, so the
logged propensities are exact. NO_AD is always an action.

Rewards are clicks. An ad's value is its fixed revenue (CPM) plus
P(click) × value per click (CPC price, or CPA payout × prior pCVR).

Warm start: priors come from category CTR priors with a capped pseudo-count, so
the posterior moves quickly once real data arrives and exploration continues.
"""

from __future__ import annotations

import random
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from theraflo_ads.domain import NO_AD, Ad, AdCatalog, AdRequest, Campaign, PricingModel
from theraflo_ads.events.bus import Bus, Message, MessageType
from theraflo_ads.policy.decision import ServedBy
from theraflo_ads.policy.policies import ActionDistribution
from theraflo_ads.pricing import CategoryPriors


def fixed_revenue(campaign: Campaign) -> float:
    return campaign.price / 1000.0 if campaign.pricing_model is PricingModel.CPM else 0.0


def click_value(campaign: Campaign, category: str, priors: CategoryPriors) -> float:
    match campaign.pricing_model:
        case PricingModel.CPC:
            return campaign.price
        case PricingModel.CPA:
            return campaign.price * priors.p_cvr(category)
        case PricingModel.CPM:
            return 0.0


def _with_floor(
    counts: Mapping[str, int], actions: Sequence[str], floor: float
) -> dict[str, float]:
    total = sum(counts.values())
    k = len(actions)
    return {a: (1 - floor) * counts.get(a, 0) / total + floor / k for a in actions}


# ----- Beta-Bernoulli (non-contextual) -----


class CountStore(Protocol):
    """Per-ad impression and click counts (in memory, or a Redis hash in production)."""

    def snapshot(self) -> tuple[Mapping[str, float], Mapping[str, float]]: ...

    def add(self, ad_id: str, *, impressions: float = 0.0, clicks: float = 0.0) -> None: ...


@dataclass
class ClickCounts:
    impressions: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    clicks: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def snapshot(self) -> tuple[Mapping[str, float], Mapping[str, float]]:
        return self.impressions, self.clicks

    def add(self, ad_id: str, *, impressions: float = 0.0, clicks: float = 0.0) -> None:
        self.impressions[ad_id] += impressions
        self.clicks[ad_id] += clicks


class BetaThompsonPolicy:
    served_by = ServedBy.BANDIT

    def __init__(
        self,
        catalog: AdCatalog,
        priors: CategoryPriors,
        counts: CountStore,
        *,
        prior_strength: float = 30.0,
        samples: int = 200,
        floor: float = 0.02,
        rng: random.Random | None = None,
        version: str = "beta_ts_v1",
    ) -> None:
        self._catalog = catalog
        self._priors = priors
        self.counts = counts
        self._strength = prior_strength
        self._samples = samples
        self._floor = floor
        self._rng = np.random.default_rng((rng or random.Random()).getrandbits(64))
        self.version = version

    def distribution(self, request: AdRequest, eligible: Sequence[Ad]) -> ActionDistribution:
        actions = [ad.ad_id for ad in eligible] + [NO_AD]
        if not eligible:
            return ActionDistribution({NO_AD: 1.0}, stochastic=False)
        impressions, clicks = self.counts.snapshot()  # one read per request
        alpha, beta, fixed, value = [], [], [], []
        for ad in eligible:
            campaign = self._catalog.campaign_for(ad)
            prior = self._priors.p_ctr(ad.category)
            n, c = impressions.get(ad.ad_id, 0.0), clicks.get(ad.ad_id, 0.0)
            alpha.append(self._strength * prior + c)
            beta.append(self._strength * (1 - prior) + max(n - c, 0.0))
            fixed.append(fixed_revenue(campaign))
            value.append(click_value(campaign, ad.category, self._priors))
        draws = self._rng.beta(alpha, beta, size=(self._samples, len(eligible)))
        revenue = np.asarray(fixed) + draws * np.asarray(value)
        winners = np.argmax(revenue, axis=1)
        tally: dict[str, int] = defaultdict(int)
        for i in winners:
            tally[eligible[i].ad_id] += 1
        posterior_mean = np.asarray(fixed) + np.asarray(alpha) / (np.add(alpha, beta)) * value
        predictions = {
            ad.ad_id: {"expected_revenue": float(posterior_mean[i])}
            for i, ad in enumerate(eligible)
        }
        return ActionDistribution(_with_floor(tally, actions, self._floor), True, predictions)

    def update(self, ad_id: str, clicked: bool) -> None:
        self.counts.add(ad_id, impressions=1.0, clicks=1.0 if clicked else 0.0)


# ----- linear Thompson sampling (contextual) -----


class LinearBanditState:
    """Bayesian ridge regression on the ads-safe feature vector: A = λI + Σxxᵀ, b = Σrx.
    Warm-started so that P(click) starts at the category prior feature."""

    def __init__(
        self, names: Sequence[str], *, ridge: float = 1.0, warm_start_strength: float = 50.0
    ):
        self.names = tuple(names)
        d = len(self.names)
        self.a = np.eye(d) * ridge
        self.b = np.zeros(d)
        if "prior_ctr" in self.names:
            idx = self.names.index("prior_ctr")
            theta0 = np.zeros(d)
            theta0[idx] = 1.0
            # Capped pseudo-observations: the posterior mean starts exactly at θ0
            # ("predict the category prior") and real data overrides it quickly.
            self.a[idx, idx] += warm_start_strength
            self.b = self.a @ theta0
        self._lock = threading.Lock()
        self.updates = 0

    def update(self, x: np.ndarray, reward: float) -> None:
        with self._lock:
            self.a += np.outer(x, x)
            self.b += reward * x
            self.updates += 1

    def add_click(self, x: np.ndarray) -> None:
        """A click after the impression was already counted with reward 0."""
        with self._lock:
            self.b += x

    def posterior(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            cov = np.linalg.inv(self.a)
            return cov @ self.b, cov


class LinearThompsonPolicy:
    served_by = ServedBy.BANDIT

    def __init__(
        self,
        catalog: AdCatalog,
        priors: CategoryPriors,
        features,
        state: LinearBanditState,
        *,
        noise: float = 0.15,
        samples: int = 100,
        floor: float = 0.02,
        rng: random.Random | None = None,
        version: str = "lin_ts_v1",
    ) -> None:
        self._catalog = catalog
        self._priors = priors
        self._features = features
        self.state = state
        self._noise = noise
        self._samples = samples
        self._floor = floor
        self._rng = np.random.default_rng((rng or random.Random()).getrandbits(64))
        self.version = version

    def distribution(self, request: AdRequest, eligible: Sequence[Ad]) -> ActionDistribution:
        actions = [ad.ad_id for ad in eligible] + [NO_AD]
        if not eligible:
            return ActionDistribution({NO_AD: 1.0}, stochastic=False)
        rows = self._features.build(request, eligible)
        x = np.array([[row.get(n, 0.0) for n in self.state.names] for row in rows])
        mean, cov = self.state.posterior()
        thetas = self._rng.multivariate_normal(
            mean, (self._noise**2) * cov, size=self._samples, method="cholesky"
        )
        p_click = np.clip(thetas @ x.T, 0.0, 1.0)  # samples × ads
        fixed = np.array([fixed_revenue(self._catalog.campaign_for(ad)) for ad in eligible])
        value = np.array(
            [
                click_value(self._catalog.campaign_for(ad), ad.category, self._priors)
                for ad in eligible
            ]
        )
        winners = np.argmax(fixed + p_click * value, axis=1)
        tally: dict[str, int] = defaultdict(int)
        for i in winners:
            tally[eligible[i].ad_id] += 1
        mean_ctr = np.clip(x @ mean, 0.0, 1.0)
        predictions = {
            ad.ad_id: {
                "p_ctr": float(mean_ctr[i]),
                "expected_revenue": float(fixed[i] + mean_ctr[i] * value[i]),
            }
            for i, ad in enumerate(eligible)
        }
        return ActionDistribution(_with_floor(tally, actions, self._floor), True, predictions)


class BanditUpdater:
    """Learns from every served impression (any policy) and its click, via the bus.

    Impressions update with reward 0 at decision time; a click later adds reward 1, which
    is exactly ridge regression on {0, 1} rewards. Features come from the decision log.
    """

    def __init__(
        self, *, linear: LinearBanditState | None = None, beta: BetaThompsonPolicy | None = None
    ) -> None:
        self._linear = linear
        self._beta = beta
        self._pending: dict[str, tuple[str, np.ndarray | None]] = {}

    def subscribe(self, bus: Bus) -> None:
        bus.subscribe(MessageType.DECISION, self.on_decision)
        bus.subscribe(MessageType.AD_CLICKED, self.on_click)

    def on_decision(self, message: Message) -> None:
        payload = message.payload
        if not payload.get("impression_ids") or payload.get("degraded"):
            return
        ad_id = payload["chosen_actions"][0]["ad_id"]
        x = None
        if self._linear is not None and payload.get("features"):
            features = payload["features"]
            x = np.array([features.get(n, 0.0) for n in self._linear.names])
            self._linear.update(x, 0.0)
        if self._beta is not None:
            self._beta.update(ad_id, clicked=False)
        self._pending[payload["impression_ids"][0]] = (ad_id, x)

    def on_click(self, message: Message) -> None:
        known = self._pending.pop(message.payload["impression_id"], None)
        if known is None:
            return
        ad_id, x = known
        if self._linear is not None and x is not None:
            self._linear.add_click(x)
        if self._beta is not None:
            self._beta.counts.add(ad_id, clicks=1.0)
