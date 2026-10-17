"""Serving chains from the model registry (design doc: Online Fallback Chain, Shadow
Deployment, Model-Service Reliability).

    guarded ML ranker (ε-greedy, circuit breaker, latency budget)
      → heuristic (ε-greedy)  → popularity  → NO_AD

A production model that fails verification (checksum, signature, feature schema)
is never served: the chain simply starts at the heuristic. Candidates in the
shadow stage score every request and are logged, never served.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from theraflo_ads.domain import AdCatalog
from theraflo_ads.ml.bundle import BundleError, ModelBundle, load_bundle
from theraflo_ads.ml.features import FeatureBuilder
from theraflo_ads.ml.registry import ModelRegistry, Stage
from theraflo_ads.ml.scorer import MLScorer
from theraflo_ads.policy.decision import ServedBy
from theraflo_ads.policy.guard import CircuitBreaker, GuardedPolicy
from theraflo_ads.policy.policies import EpsilonGreedyPolicy, Policy, PopularityPolicy
from theraflo_ads.policy.scoring import HeuristicScorer
from theraflo_ads.pricing import CategoryPriors

log = logging.getLogger(__name__)


@dataclass
class ServingChain:
    chain: list[Policy]
    shadow: list[Policy] = field(default_factory=list)
    model_versions: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def build_serving_chain(
    catalog: AdCatalog,
    priors: CategoryPriors,
    features: FeatureBuilder,
    registry_root: Path | None,
    *,
    epsilon: float = 0.05,
    signing_key: bytes | None = None,
    require_signature: bool = False,
    latency_budget_ms: float = 40.0,
) -> ServingChain:
    heuristic = EpsilonGreedyPolicy(HeuristicScorer(catalog, priors), epsilon=epsilon)
    ranked = sorted(catalog.ads, key=lambda a: -catalog.campaign_for(catalog.ads[a]).priority)
    fallback: list[Policy] = [heuristic, PopularityPolicy(ranked)]
    result = ServingChain(chain=list(fallback))
    if registry_root is None or not (registry_root / "registry.json").exists():
        return result

    registry = ModelRegistry(registry_root)

    def load(name: str, stage: Stage) -> ModelBundle | None:
        info = (
            registry.production(name)
            if stage is Stage.PRODUCTION
            else _in_stage(registry, name, stage)
        )
        if info is None:
            return None
        try:
            return load_bundle(
                info.path,
                signing_key=signing_key,
                require_signature=require_signature,
                feature_schema_version=features.version,
            )
        except BundleError as exc:
            result.problems.append(f"{name}/{info.version}: {exc}")
            log.error("not serving %s/%s: %s", name, info.version, exc)
            return None

    ctr, cvr = load("ctr", Stage.PRODUCTION), load("cvr", Stage.PRODUCTION)
    if ctr is not None:
        ranker = EpsilonGreedyPolicy(
            MLScorer(catalog, features, ctr, cvr, priors=priors),
            epsilon=epsilon,
            served_by=ServedBy.RANKER,
            version=f"ranker:{ctr.version}+{cvr.version if cvr else 'prior'}",
        )
        guarded = GuardedPolicy(
            ranker, breaker=CircuitBreaker(), latency_budget_ms=latency_budget_ms
        )
        result.chain = [guarded, *fallback]
        result.model_versions = {"ctr": ctr.version, "cvr": cvr.version if cvr else "prior"}

    shadow_ctr = load("ctr", Stage.SHADOW)
    if shadow_ctr is not None:
        shadow_cvr = load("cvr", Stage.SHADOW) or cvr
        result.shadow.append(
            EpsilonGreedyPolicy(
                MLScorer(catalog, features, shadow_ctr, shadow_cvr, priors=priors),
                epsilon=epsilon,
                served_by=ServedBy.RANKER,
                version=f"shadow:{shadow_ctr.version}",
            )
        )
    return result


def _in_stage(registry: ModelRegistry, name: str, stage: Stage):
    matches: Sequence[str] = [v for v, s in registry.versions(name).items() if s is stage]
    return registry.info(name, matches[-1]) if matches else None
