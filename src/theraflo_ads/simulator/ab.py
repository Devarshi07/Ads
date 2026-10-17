"""Simulated A/B test (design doc: Online Experimentation → canary / A/B stage).

The MDE is pre-registered from the pre-period alone, before any treatment data
exists, and the analysis then applies the promotion rule. Simulator results
exercise the machinery; they are never promotion evidence for production.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from theraflo_ads.experiments.analysis import (
    ExperimentReport,
    analyze_experiment,
    pre_period_revenue,
)
from theraflo_ads.experiments.assignment import Experiment
from theraflo_ads.ml.bundle import load_bundle
from theraflo_ads.ml.scorer import MLScorer
from theraflo_ads.policy.decision import ServedBy
from theraflo_ads.policy.policies import EpsilonGreedyPolicy, Policy
from theraflo_ads.policy.scoring import HeuristicScorer
from theraflo_ads.simulator.runner import PolicySetup, popularity_fallback, run_simulation
from theraflo_ads.simulator.world import SimConfig, World

TreatmentFactory = Callable[[World], Sequence[Policy]]


@dataclass(frozen=True)
class ABConfig:
    sim: SimConfig
    pre_days: int = 7
    experiment_id: str = "ranker_ab"
    alpha: float = 0.05
    power: float = 0.8


def heuristic_arm(world: World) -> list[Policy]:
    scorer = HeuristicScorer(world.catalog, world.priors)
    return [EpsilonGreedyPolicy(scorer, epsilon=world.config.epsilon), popularity_fallback(world)]


def ml_arm(ctr_bundle: Path, cvr_bundle: Path | None = None) -> TreatmentFactory:
    def build(world: World) -> list[Policy]:
        schema = world.features.version
        ctr = load_bundle(ctr_bundle, feature_schema_version=schema)
        cvr = load_bundle(cvr_bundle, feature_schema_version=schema) if cvr_bundle else None
        scorer = MLScorer(world.catalog, world.features, ctr, cvr, priors=world.priors)
        ranker = EpsilonGreedyPolicy(
            scorer, epsilon=world.config.epsilon, served_by=ServedBy.RANKER, version=scorer.version
        )
        return [ranker, *heuristic_arm(world)]

    return build


def preregister_mde(
    pre_values: np.ndarray, users_per_arm: int, *, alpha: float, power: float
) -> float:
    """Relative MDE on revenue per user from pre-period variance only."""
    mean = float(np.mean(pre_values)) if len(pre_values) else 0.0
    if mean <= 0 or users_per_arm <= 1:
        return float("inf")
    sd = float(np.std(pre_values, ddof=1))
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(z * sd * np.sqrt(2.0 / users_per_arm) / mean)


def run_ab_test(config: ABConfig, treatment: TreatmentFactory) -> tuple[ExperimentReport, dict]:
    sim = config.sim
    starts_at = sim.start + timedelta(days=config.pre_days)
    experiment = Experiment(
        config.experiment_id,
        (("control", 1.0), ("treatment", 1.0)),
        budget_split=True,
        starts_at=starts_at,
    )

    def setup(world: World) -> PolicySetup:
        control = heuristic_arm(world)
        return PolicySetup(
            chain=control,
            experiment=experiment,
            arm_chains={"control": control, "treatment": list(treatment(world))},
        )

    result = run_simulation(sim, setup)
    records = [d.to_log() for d in result.decisions]
    labels = {lbl.impression_id: lbl for lbl in result.platform.store.attribution.labels()}
    pre = [r for r in records if datetime.fromisoformat(r["timestamp"]) < starts_at]
    pre_revenue = pre_period_revenue(pre, labels)
    users = len({r["user_hash"] for r in records if r.get("experiment_cohort")})
    mde = preregister_mde(
        np.array(list(pre_revenue.values())),
        max(users // 2, 2),
        alpha=config.alpha,
        power=config.power,
    )
    retained = {u.user_hash: not u.churned for u in result.world.users}
    report = analyze_experiment(
        records,
        labels,
        experiment_id=config.experiment_id,
        control="control",
        treatment="treatment",
        preregistered_mde=mde,
        pre_period=pre_revenue,
        retained=retained,
        safety_counters=dict(result.violations),
        alpha=config.alpha,
    )
    context: dict[str, Any] = {
        "users_in_experiment": users,
        "pre_days": config.pre_days,
        "experiment_days": sim.n_days - config.pre_days,
        "served_by": dict(result.served_by),
        "safety_violations": dict(result.violations),
        "vulnerable_cohort": {
            k: {"ad_load": v.ad_load, "impulse_share": v.impulse_share}
            for k, v in result.cohorts.items()
        },
    }
    return report, context
