"""Offline fitted-Q iteration (design doc: Reinforcement Learning → offline RL, evaluated with
OPE before it receives any traffic).

Q(s, a) is regressed on φ(s, a): the shared ads-safe feature vector of the ad
(NO_AD gets the context features and an is_no_ad flag). Each iteration fits
    y = r + γ · max_{a' ∈ supported(s')} Q_k(s', a')
Support constraint (a conservative-Q stand-in): the max runs only over actions
the logging policy chose with probability >= `min_support`. Without it, the
max picks actions the data barely covers, where Q is pure extrapolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from theraflo_ads.domain import NO_AD, Ad, AdCatalog, AdRequest, UserContext
from theraflo_ads.ml.features import FeatureBuilder
from theraflo_ads.policy.decision import Decision, ServedBy
from theraflo_ads.policy.policies import ActionDistribution
from theraflo_ads.rl.dataset import Transition

AD_SPECIFIC_PREFIXES = ("cat_", "pricing_")
AD_SPECIFIC = (
    "log_price",
    "prior_ctr",
    "prior_cvr",
    "ad_exposures_24h",
    "user_cat_ctr",
    "user_cat_impressions",
)


def request_from(decision: Decision) -> AdRequest:
    return AdRequest(
        decision.request_id,
        decision.session_id,
        decision.placement,
        UserContext(decision.user_hash, decision.consent_scope),
        decision.timestamp,
        feature_context=dict(decision.feature_context) or None,
    )


class ActionFeaturizer:
    def __init__(self, catalog: AdCatalog, features: FeatureBuilder) -> None:
        self._catalog = catalog
        self._features = features
        self.names = (*features.names, "is_no_ad")
        self._ad_specific = [
            i
            for i, n in enumerate(features.names)
            if n.startswith(AD_SPECIFIC_PREFIXES) or n in AD_SPECIFIC
        ]

    def matrix(self, request: AdRequest, actions: Sequence[str]) -> np.ndarray:
        ads = [self._catalog.ads[a] for a in actions if a != NO_AD and a in self._catalog.ads]
        rows = self._features.build(request, ads or [next(iter(self._catalog.ads.values()))])
        by_ad = {ad.ad_id: row for ad, row in zip(ads, rows, strict=False)}
        template = rows[0]
        out = np.zeros((len(actions), len(self.names)))
        for i, action in enumerate(actions):
            if action == NO_AD:
                vector = np.array([template[n] for n in self._features.names], dtype=float)
                vector[self._ad_specific] = 0.0
                out[i, :-1] = vector
                out[i, -1] = 1.0
            else:
                out[i, :-1] = [by_ad[action][n] for n in self._features.names]
        return out


@dataclass(frozen=True)
class FQIConfig:
    gamma: float = 0.9
    iterations: int = 6
    min_support: float = 0.0
    ridge: float = 1.0
    algorithm: str = "lightgbm"  # lightgbm | ridge


class FittedQ:
    def __init__(self, featurizer: ActionFeaturizer, config: FQIConfig | None = None) -> None:
        self.featurizer = featurizer
        self.config = config or FQIConfig()
        self._model = None
        self.history: list[dict[str, float]] = []

    def fit(self, transitions: Sequence[Transition], indices: Sequence[int]) -> FittedQ:
        cfg = self.config
        chosen = np.vstack(
            [
                self.featurizer.matrix(
                    request_from(transitions[i].decision), [transitions[i].action]
                )
                for i in indices
            ]
        )
        rewards = np.array([transitions[i].reward for i in indices])
        # Candidate matrices for each next state are built once and reused every iteration.
        nexts: list[np.ndarray | None] = []
        for i in indices:
            t = transitions[i]
            if t.done or t.next_index is None:
                nexts.append(None)
                continue
            nd = transitions[t.next_index].decision
            supported = [a for a, p in nd.action_probabilities.items() if p >= cfg.min_support]
            nexts.append(self.featurizer.matrix(request_from(nd), supported or [NO_AD]))
        targets = rewards.copy()
        for iteration in range(cfg.iterations):
            self._model = self._fit(chosen, targets)
            future = np.array([0.0 if m is None else float(self._predict(m).max()) for m in nexts])
            new_targets = rewards + cfg.gamma * future
            self.history.append(
                {
                    "iteration": iteration,
                    "mean_target": float(new_targets.mean()),
                    "change": float(np.abs(new_targets - targets).mean()),
                }
            )
            targets = new_targets
        self._model = self._fit(chosen, targets)
        return self

    def q_values(self, request: AdRequest, actions: Sequence[str]) -> np.ndarray:
        return self._predict(self.featurizer.matrix(request, actions))

    def _fit(self, x: np.ndarray, y: np.ndarray):
        if self.config.algorithm == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=self.config.ridge).fit(x, y)
        import lightgbm as lgb

        params = {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 100,
            "verbose": -1,
            "seed": 7,
            "deterministic": True,
            "num_threads": 1,
        }
        return lgb.train(params, lgb.Dataset(x, y), 150)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(x), dtype=float)


class QPolicy:
    """ε-greedy over Q, with NO_AD always an action (served_by RL)."""

    served_by = ServedBy.RL

    def __init__(self, q: FittedQ, *, epsilon: float = 0.1, version: str = "fqi_v1") -> None:
        self._q = q
        self._epsilon = epsilon
        self.version = version

    def distribution(self, request: AdRequest, eligible: Sequence[Ad]) -> ActionDistribution:
        actions = [ad.ad_id for ad in eligible] + [NO_AD]
        values = self._q.q_values(request, actions)
        k = len(actions)
        probabilities = {a: self._epsilon / k for a in actions}
        probabilities[actions[int(np.argmax(values))]] += 1 - self._epsilon
        predictions = {a: {"q": float(v)} for a, v in zip(actions, values, strict=True)}
        return ActionDistribution(probabilities, self._epsilon > 0 and k > 1, predictions)
