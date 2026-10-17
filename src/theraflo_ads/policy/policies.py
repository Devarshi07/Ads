"""Serving policies (design doc: Contextual Bandits, Online Fallback Chain, RL Safety).

Every policy returns a full probability distribution over the eligible
actions, NO_AD included, so propensities can be logged for every candidate.
"""

from __future__ import annotations

import math
import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from theraflo_ads.domain import NO_AD, Ad, AdRequest
from theraflo_ads.policy.decision import ServedBy
from theraflo_ads.policy.scoring import HeuristicScorer, ScoredAd


class InvalidDistribution(ValueError):
    pass


@dataclass(frozen=True)
class ActionDistribution:
    probabilities: Mapping[str, float]
    stochastic: bool
    predictions: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


class Policy(Protocol):
    served_by: ServedBy
    version: str

    def distribution(self, request: AdRequest, eligible: Sequence[Ad]) -> ActionDistribution: ...


class EpsilonGreedyPolicy:
    """Greedy on expected revenue, with a uniform exploration floor of epsilon / k."""

    def __init__(
        self,
        scorer: HeuristicScorer,
        *,
        epsilon: float,
        no_ad_floor: float = 0.0,
        served_by: ServedBy = ServedBy.HEURISTIC,
        version: str = "heuristic_eps_v1",
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self._scorer = scorer
        self._epsilon = epsilon
        self._no_ad_floor = no_ad_floor
        self.served_by = served_by
        self.version = version

    def distribution(self, request: AdRequest, eligible: Sequence[Ad]) -> ActionDistribution:
        scored = self._scorer.score(request, eligible)
        actions = [s.ad.ad_id for s in scored] + [NO_AD]
        k = len(actions)
        probabilities = {a: self._epsilon / k for a in actions}
        probabilities[greedy_action(scored, self._no_ad_floor)] += 1.0 - self._epsilon
        return ActionDistribution(
            probabilities,
            stochastic=self._epsilon > 0 and k > 1,
            predictions=_predictions(scored),
        )


class UniformPolicy:
    """Uniform over the eligible set, NO_AD included. Serves the exploration holdout."""

    served_by = ServedBy.UNIFORM_HOLDOUT
    version = "uniform_v1"

    def distribution(self, request: AdRequest, eligible: Sequence[Ad]) -> ActionDistribution:
        actions = [ad.ad_id for ad in eligible] + [NO_AD]
        p = 1.0 / len(actions)
        return ActionDistribution({a: p for a in actions}, stochastic=len(actions) > 1)


class PopularityPolicy:
    """Deterministic fallback: the first eligible ad in a precomputed popularity list."""

    served_by = ServedBy.POPULARITY

    def __init__(self, ranked_ad_ids: Sequence[str], version: str = "popularity_v1") -> None:
        self._ranked = tuple(ranked_ad_ids)
        self.version = version

    def distribution(self, request: AdRequest, eligible: Sequence[Ad]) -> ActionDistribution:
        eligible_ids = {ad.ad_id for ad in eligible}
        choice = next((a for a in self._ranked if a in eligible_ids), NO_AD)
        probabilities = {ad.ad_id: 0.0 for ad in eligible} | {NO_AD: 0.0}
        probabilities[choice] = 1.0
        return ActionDistribution(probabilities, stochastic=False)


def greedy_action(scored: Sequence[ScoredAd], no_ad_floor: float) -> str:
    if not scored:
        return NO_AD
    best = max(scored, key=lambda s: (s.expected_revenue, s.ad.ad_id))
    return best.ad.ad_id if best.expected_revenue > no_ad_floor else NO_AD


def validate_distribution(dist: ActionDistribution, eligible_ids: Collection[str]) -> None:
    allowed = set(eligible_ids) | {NO_AD}
    unknown = set(dist.probabilities) - allowed
    if unknown:
        raise InvalidDistribution(f"actions outside the eligible set: {sorted(unknown)}")
    if any(p < 0 or math.isnan(p) for p in dist.probabilities.values()):
        raise InvalidDistribution("probabilities must be non-negative")
    if not math.isclose(sum(dist.probabilities.values()), 1.0, abs_tol=1e-9):
        raise InvalidDistribution("probabilities must sum to 1")


def sample(dist: ActionDistribution, rng: random.Random) -> str:
    point = rng.random()
    cumulative = 0.0
    last = NO_AD
    for action, p in dist.probabilities.items():
        if p <= 0:
            continue
        cumulative += p
        last = action
        if point < cumulative:
            return action
    return last  # guards against floating-point shortfall in the cumulative sum


def _predictions(scored: Sequence[ScoredAd]) -> dict[str, dict[str, float]]:
    return {
        s.ad.ad_id: {
            "p_ctr": s.p_ctr,
            "p_cvr": s.p_cvr,
            "expected_revenue": s.expected_revenue,
        }
        for s in scored
    }
