"""Ranker lab for the dev console (design doc: Initial Ranking, Contextual Bandits, Online
Fallback Chain, Model Registry).

Every candidate policy scores the same eligible set for one request, and nothing is served:

    hard constraints  →  scores (category priors, ML ranker)  →  each policy's distribution

The constraint engine only reads counters, so a preview never spends budget or frequency.
The lab can also switch the serving policy at runtime (`DecisionService.set_chain`) and walk
a registry model into production, the same steps as `theraflo-ads ml promote`.

The API mounts the lab only with dev credentials, never in production. RL rerankers (FQI,
PPO) stay offline by design: they are trained and evaluated as jobs, and serve nothing until
their OPE and safety gates pass.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from theraflo_ads.domain import NO_AD, Ad, AdRequest
from theraflo_ads.ml.bundle import BundleError, ModelBundle, load_bundle
from theraflo_ads.ml.features import FeatureBuilder, daypart
from theraflo_ads.ml.registry import ModelRegistry, RegistryError, Stage
from theraflo_ads.ml.scorer import MLScorer
from theraflo_ads.platform import Platform
from theraflo_ads.policy.bandits import (
    BetaThompsonPolicy,
    CountStore,
    LinearBanditState,
    LinearThompsonPolicy,
)
from theraflo_ads.policy.decision import ServedBy
from theraflo_ads.policy.guard import GuardedPolicy
from theraflo_ads.policy.policies import (
    EpsilonGreedyPolicy,
    Policy,
    PopularityPolicy,
    UniformPolicy,
    validate_distribution,
)
from theraflo_ads.policy.scoring import HeuristicScorer, ScoredAd
from theraflo_ads.pricing import CategoryPriors, budget_scope
from theraflo_ads.safety.engine import Eligibility

log = logging.getLogger(__name__)

POLICIES: Mapping[str, str] = {
    "heuristic": "ε-greedy on expected revenue from category priors (the Phase 1 baseline)",
    "ranker": "ε-greedy on calibrated LightGBM pCTR × pCVR (the supervised ranker)",
    "beta_ts": "Beta-Bernoulli Thompson sampling on per-ad click counts",
    "lin_ts": "Linear Thompson sampling on the ads-safe feature vector",
    "popularity": "deterministic: the highest-priority eligible ad (a fallback level)",
    "uniform": "uniform over eligible ads and NO_AD (the exploration holdout)",
}
# The uniform policy serves the exploration holdout only: serving it to everyone would log
# every decision as a holdout decision.
SERVABLE = ("heuristic", "ranker", "beta_ts", "lin_ts", "popularity")
BETA_PRIOR_STRENGTH = 30.0
EPSILON = 0.05


class LabError(Exception):
    """A policy cannot be built or served (no model, no production model, unknown name)."""


@dataclass
class LoadedModels:
    ctr: ModelBundle
    cvr: ModelBundle | None
    source: str
    production: bool
    problems: list[str] = field(default_factory=list)


class PolicyLab:
    def __init__(
        self,
        *,
        platform: Platform,
        features: FeatureBuilder,
        counts: CountStore,
        linear: LinearBanditState,
        model_dir: Path,
        serving: str = "heuristic",
        linear_learns: bool = True,
    ) -> None:
        self._platform = platform
        self._features = features
        self._counts = counts
        self.linear = linear
        # False on the infra backend: workers update the Beta counts in Redis, but the
        # linear state lives in this process and only has its warm start.
        self.linear_learns = linear_learns
        self._model_dir = model_dir
        self.serving = serving
        self._lock = threading.Lock()
        self._models: LoadedModels | None = None
        self._models_key: tuple | None = None
        self._model_problems: list[str] = []
        self.beta = BetaThompsonPolicy(
            platform.catalog, self.priors(), counts, prior_strength=BETA_PRIOR_STRENGTH
        )
        self.lin = LinearThompsonPolicy(platform.catalog, self.priors(), features, linear)

    def priors(self) -> CategoryPriors:
        return getattr(self._platform.cost, "priors", None) or CategoryPriors()

    # ----- models -----

    def models(self) -> LoadedModels | None:
        """The ranker's models: registry production, else the newest registered version,
        else models/candidates. Reloaded when the registry or the candidates change."""
        key = self._model_key()
        with self._lock:
            if key != self._models_key:
                self._model_problems = []
                self._models = self._load_models()
                self._models_key = key
            return self._models

    def _model_key(self) -> tuple:
        paths = (
            self._model_dir / "registry" / "registry.json",
            self._model_dir / "candidates" / "ctr" / "manifest.json",
            self._model_dir / "candidates" / "cvr" / "manifest.json",
        )
        return tuple(p.stat().st_mtime_ns if p.exists() else None for p in paths)

    def _bundle(self, path: Path) -> ModelBundle | None:
        try:
            return load_bundle(path, feature_schema_version=self._features.version)
        except (BundleError, OSError, KeyError, ValueError) as exc:
            self._model_problems.append(f"{path.name}: {exc}")
            return None

    def _load_models(self) -> LoadedModels | None:
        registry_root = self._model_dir / "registry"
        if (registry_root / "registry.json").exists():
            registry = ModelRegistry(registry_root)
            production = registry.production("ctr")
            if production is not None:
                ctr = self._bundle(production.path)
                if ctr is not None:
                    cvr_info = registry.production("cvr")
                    cvr = self._bundle(cvr_info.path) if cvr_info else None
                    return LoadedModels(
                        ctr, cvr, f"registry ctr/{production.version} (production)", True
                    )
            live = [v for v, s in registry.versions("ctr").items() if s is not Stage.RETIRED]
            if live:
                info = registry.info("ctr", live[-1])
                ctr = self._bundle(info.path)
                if ctr is not None:
                    return LoadedModels(
                        ctr, None, f"registry ctr/{info.version} ({info.stage})", False
                    )
        candidates = self._model_dir / "candidates"
        if (candidates / "ctr" / "manifest.json").exists():
            ctr = self._bundle(candidates / "ctr")
            if ctr is not None:
                cvr = (
                    self._bundle(candidates / "cvr")
                    if (candidates / "cvr" / "manifest.json").exists()
                    else None
                )
                return LoadedModels(ctr, cvr, "models/candidates (not registered)", False)
        return None

    def model_report(self) -> dict[str, Any]:
        models = self.models()
        registry_root = self._model_dir / "registry"
        registry: dict[str, Any] = {}
        if (registry_root / "registry.json").exists():
            reg = ModelRegistry(registry_root)
            for name in ("ctr", "cvr", "creative_reviewer"):
                versions = reg.versions(name)
                if not versions:
                    continue
                production = reg.production(name)
                registry[name] = {
                    "production": production.version if production else None,
                    "versions": [
                        {
                            "version": version,
                            "stage": stage.value,
                            "metrics": reg.info(name, version).metrics,
                            "history": reg.history(name, version),
                        }
                        for version, stage in versions.items()
                    ],
                }
        loaded = None
        if models is not None:
            serving_names = set(self._features.names)
            model_names = models.ctr.model.feature_names
            booster = getattr(models.ctr.model.model, "booster", None)
            importance: list[dict[str, Any]] = []
            if booster is not None:
                gains = booster.feature_importance(importance_type="gain")
                ranked = sorted(zip(model_names, gains, strict=True), key=lambda x: -x[1])
                total = float(sum(gains)) or 1.0
                importance = [
                    {"feature": n, "share": float(g) / total} for n, g in ranked[:12] if g > 0
                ]
            loaded = {
                "source": models.source,
                "production": models.production,
                "ctr_version": models.ctr.version,
                "cvr_version": models.cvr.version if models.cvr else None,
                "algorithm": models.ctr.manifest.get("algorithm"),
                "feature_schema_version": models.ctr.manifest.get("feature_schema_version"),
                "metrics": models.ctr.manifest.get("metrics", {}),
                "trained_on": models.ctr.manifest.get("lineage", {}),
                # Model inputs the serving vector does not have (they score as 0): the
                # training catalog's categories differ from the live catalog's.
                "missing_features": [n for n in model_names if n not in serving_names],
                "model_features": len(model_names),
                "importance": importance,
            }
        return {
            "model_dir": str(self._model_dir),
            "loaded": loaded,
            "problems": list(self._model_problems),
            "registry": registry,
        }

    def promote(self, name: str, version: str) -> dict[str, Any]:
        registry = self._registry()
        try:
            registry.promote_to_production(name, version, "promoted from the dev console")
        except (RegistryError, KeyError) as exc:
            raise LabError(f"cannot promote {name}/{version}: {exc}") from None
        production = registry.production(name)
        if production is None or production.version != version:
            stage = registry.info(name, version).stage.value
            raise LabError(f"{name}/{version} is {stage} and cannot reach production")
        return self._after_registry_change()

    def rollback(self, name: str) -> dict[str, Any]:
        registry = self._registry()
        try:
            registry.rollback(name, "rolled back from the dev console")
        except (RegistryError, KeyError) as exc:
            raise LabError(str(exc)) from None
        return self._after_registry_change()

    def _registry(self) -> ModelRegistry:
        root = self._model_dir / "registry"
        if not (root / "registry.json").exists():
            raise LabError(f"no model registry at {root}")
        return ModelRegistry(root)

    def _after_registry_change(self) -> dict[str, Any]:
        if self.serving == "ranker":
            models = self.models()
            if models is None or not models.production:
                # Production was retired: serving falls back to the heuristic.
                self.serve("heuristic")
            else:
                self.serve("ranker")  # pick up the new production version
        return self.model_report()

    # ----- policies -----

    def policy(self, name: str) -> Policy:
        catalog, priors = self._platform.catalog, self.priors()
        match name:
            case "heuristic":
                return EpsilonGreedyPolicy(HeuristicScorer(catalog, priors), epsilon=EPSILON)
            case "ranker":
                models = self.models()
                if models is None:
                    raise LabError(
                        "no CTR model: train one with `theraflo-ads ml all` "
                        f"(looked in {self._model_dir})"
                    )
                return EpsilonGreedyPolicy(
                    MLScorer(catalog, self._features, models.ctr, models.cvr, priors=priors),
                    epsilon=EPSILON,
                    served_by=ServedBy.RANKER,
                    version=f"ranker:{models.ctr.version}+"
                    f"{models.cvr.version if models.cvr else 'prior'}",
                )
            case "beta_ts":
                return self.beta
            case "lin_ts":
                return self.lin
            case "popularity":
                return PopularityPolicy(self._popularity_order())
            case "uniform":
                return UniformPolicy()
        raise LabError(f"unknown policy {name!r}")

    def _popularity_order(self) -> list[str]:
        catalog = self._platform.catalog
        return sorted(catalog.ads, key=lambda a: -catalog.campaign_for(catalog.ads[a]).priority)

    def chain_for(self, name: str) -> list[Policy]:
        fallback: list[Policy] = [self.policy("heuristic"), self.policy("popularity")]
        if name == "heuristic":
            return fallback
        if name == "popularity":
            return [self.policy("popularity")]
        # The same guard the infra chains use: a failing or slow level opens its circuit.
        return [GuardedPolicy(self.policy(name)), *fallback]

    def serve(self, name: str) -> dict[str, Any]:
        if name not in SERVABLE:
            raise LabError(f"{name!r} cannot serve; choose one of {', '.join(SERVABLE)}")
        if name == "ranker":
            models = self.models()
            if models is None or not models.production:
                raise LabError(
                    "only a production model serves: promote a CTR model to production in the "
                    "registry first (Models panel, or `theraflo-ads ml promote ctr v1`)"
                )
        self._platform.decisions.set_chain(self.chain_for(name))
        self.serving = name
        log.warning("serving policy switched to %s from the dev console", name)
        return self.serving_state()

    def serving_state(self) -> dict[str, Any]:
        chain = self._platform.decisions.chain
        return {
            "policy": self.serving,
            "chain": [
                {"served_by": level.served_by.value, "version": level.version} for level in chain
            ],
            "servable": list(SERVABLE),
        }

    # ----- bandits -----

    def bandit_state(self) -> dict[str, Any]:
        from scipy.stats import beta as beta_distribution  # slow import; only this panel

        catalog, priors = self._platform.catalog, self.priors()
        impressions, clicks = self._counts.snapshot()
        arms = []
        for ad in catalog.ads.values():
            prior = priors.p_ctr(ad.category)
            n, c = float(impressions.get(ad.ad_id, 0.0)), float(clicks.get(ad.ad_id, 0.0))
            a = BETA_PRIOR_STRENGTH * prior + c
            b = BETA_PRIOR_STRENGTH * (1 - prior) + max(n - c, 0.0)
            low, high = beta_distribution.ppf([0.025, 0.975], a, b)
            arms.append(
                {
                    "ad_id": ad.ad_id,
                    "category": ad.category,
                    "review_status": ad.review_status.value,
                    "impressions": n,
                    "clicks": c,
                    "prior_ctr": prior,
                    "posterior_mean": a / (a + b),
                    "ci95": [float(low), float(high)],
                }
            )
        mean, cov = self.linear.posterior()
        order = np.argsort(-np.abs(mean))
        return {
            "beta_ts": {"prior_strength": BETA_PRIOR_STRENGTH, "arms": arms},
            "lin_ts": {
                "updates": self.linear.updates,
                "learns_online": self.linear_learns,
                "dimensions": len(self.linear.names),
                "weights": [
                    {
                        "feature": self.linear.names[i],
                        "mean": float(mean[i]),
                        "sd": float(np.sqrt(max(cov[i, i], 0.0))),
                    }
                    for i in order[:12]
                ],
            },
        }

    # ----- previews -----

    def preview(self, request: AdRequest, policies: Sequence[str] = tuple(POLICIES)) -> dict:
        """One request through every stage, served by nothing."""
        unknown = [p for p in policies if p not in POLICIES]
        if unknown:
            raise LabError(f"unknown policies: {', '.join(unknown)}")
        eligibility = self._platform.engine.evaluate(request)
        request = replace(request, frequency=eligibility.snapshot)
        if eligibility.blocked is None:
            request = replace(request, feature_context=self._features.context(request))
        result: dict[str, Any] = {
            "blocked": eligibility.blocked.value if eligibility.blocked else None,
            "effective_consent": eligibility.effective_consent.value,
            "safety_policy_version": eligibility.safety_policy_version,
            "daypart": daypart(request.now),
            "candidates": self._candidates(request, eligibility),
            "scores": [],
            "policies": [],
            "features": {},
            "reference": None,
            "serving": self.serving_state(),
        }
        if eligibility.blocked is not None:
            return result
        eligible = list(eligibility.eligible)
        ids = {ad.ad_id for ad in eligible}
        rows = self._features.build(request, eligible)
        result["features"] = {
            ad.ad_id: {k: v for k, v in row.items() if v}
            for ad, row in zip(eligible, rows, strict=True)
        }
        prior_scores = {
            s.ad.ad_id: s
            for s in HeuristicScorer(self._platform.catalog, self.priors()).score(request, eligible)
        }
        ml_scores, ml_error = self._ml_scores(request, eligible)
        result["scores"] = [
            {
                "ad_id": ad.ad_id,
                "prior": _score(prior_scores.get(ad.ad_id)),
                "ml": _score(ml_scores.get(ad.ad_id)) if ml_scores else None,
            }
            for ad in eligible
        ]
        reference = ml_scores or prior_scores
        result["reference"] = {
            "source": "ML ranker estimates" if ml_scores else "category priors",
            "ml_error": ml_error,
        }
        revenue = {ad_id: s.expected_revenue for ad_id, s in reference.items()}
        for name in policies:
            result["policies"].append(self._run_policy(name, request, eligible, ids, revenue))
        serving = next(
            (p for p in result["policies"] if p["name"] == self.serving and p["available"]), None
        )
        for entry in result["policies"]:
            if serving is not None and entry["available"]:
                entry["tv_from_serving"] = _tv(entry["probabilities"], serving["probabilities"])
        return result

    def _run_policy(
        self,
        name: str,
        request: AdRequest,
        eligible: Sequence[Ad],
        ids: set[str],
        revenue: Mapping[str, float],
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "name": name,
            "description": POLICIES[name],
            "servable": name in SERVABLE,
            "serving": name == self.serving,
        }
        try:
            policy = self.policy(name)
            started = time.perf_counter()
            dist = policy.distribution(request, eligible)
            elapsed_ms = (time.perf_counter() - started) * 1000
            validate_distribution(dist, ids)
        except LabError as exc:
            return {**entry, "available": False, "error": str(exc)}
        except Exception as exc:  # a broken policy is a finding, not a console crash
            return {**entry, "available": False, "error": f"{type(exc).__name__}: {exc}"}
        probabilities = {a: float(p) for a, p in dist.probabilities.items()}
        return {
            **entry,
            "available": True,
            "version": policy.version,
            "served_by": policy.served_by.value,
            "stochastic": dist.stochastic,
            "latency_ms": round(elapsed_ms, 3),
            "probabilities": probabilities,
            "predictions": {a: dict(p) for a, p in dist.predictions.items()},
            "top_action": max(probabilities, key=lambda a: (probabilities[a], a)),
            "p_no_ad": probabilities.get(NO_AD, 0.0),
            # Under one yardstick for every policy, so the cards compare.
            "expected_revenue": sum(p * revenue.get(a, 0.0) for a, p in probabilities.items()),
        }

    def _ml_scores(
        self, request: AdRequest, eligible: Sequence[Ad]
    ) -> tuple[dict[str, ScoredAd], str | None]:
        models = self.models()
        if models is None:
            return {}, "no CTR model found"
        try:
            scorer = MLScorer(
                self._platform.catalog,
                self._features,
                models.ctr,
                models.cvr,
                priors=self.priors(),
            )
            return {s.ad.ad_id: s for s in scorer.score(request, eligible)}, None
        except Exception as exc:
            return {}, f"{type(exc).__name__}: {exc}"

    def _candidates(self, request: AdRequest, eligibility: Eligibility) -> list[dict[str, Any]]:
        catalog, priors = self._platform.catalog, self.priors()
        eligible_ids = {ad.ad_id for ad in eligibility.eligible}
        snapshot = eligibility.snapshot
        rows = []
        for ad in catalog.candidates():
            campaign = catalog.campaign_for(ad)
            budget_key, limit = budget_scope(campaign, request.budget_slice)
            try:
                remaining = self._platform.budget.remaining(budget_key, limit, request.now)
            except Exception:
                remaining = None
            rejection = eligibility.rejections.get(ad.ad_id)
            rows.append(
                {
                    "ad_id": ad.ad_id,
                    "campaign_id": campaign.campaign_id,
                    "advertiser_id": campaign.advertiser_id,
                    "category": ad.category,
                    "review_status": ad.review_status.value,
                    "pricing_model": campaign.pricing_model.value,
                    "price": campaign.price,
                    "daily_budget": limit,
                    "budget_remaining": remaining,
                    "priority": campaign.priority,
                    "eligible": ad.ad_id in eligible_ids,
                    "rejection": rejection.value if rejection else None,
                    "prior_ctr": priors.p_ctr(ad.category),
                    "prior_cvr": priors.p_cvr(ad.category),
                    "exposures_24h": snapshot.ad_counts.get(ad.ad_id, 0) if snapshot else None,
                }
            )
        order = {True: 0, False: 1}
        return sorted(rows, key=lambda r: (order[r["eligible"]], r["ad_id"]))


def _score(scored: ScoredAd | None) -> dict[str, float] | None:
    if scored is None:
        return None  # dropped by the scorer's quality floor
    return {
        "p_ctr": scored.p_ctr,
        "p_cvr": scored.p_cvr,
        "expected_revenue": scored.expected_revenue,
        "ecpm": 1000.0 * scored.expected_revenue,
    }


def _tv(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Total-variation distance: the share of traffic two policies would serve differently."""
    return 0.5 * sum(abs(p.get(a, 0.0) - q.get(a, 0.0)) for a in set(p) | set(q))
