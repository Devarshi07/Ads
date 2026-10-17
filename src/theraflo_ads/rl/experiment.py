"""RL experiments (design doc: Phase 6 — RL).

    fqi   train offline from the logged decisions (data/raw), gate it with OPE on a later
          window, then run it frozen in the simulator with the safety evals
    ppo   train on-policy in the simulator, round by round, then evaluate the same way

Simulator numbers are for algorithm comparison and safety probing only.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from theraflo_ads.ml import pipeline
from theraflo_ads.ml.features import FeatureBuilder
from theraflo_ads.ope.estimators import (
    doubly_robust,
    samples_from_decisions,
    snips,
    user_clusters,
)
from theraflo_ads.ope.targets import category_reward_model, target_from_policy
from theraflo_ads.policy.decision import ServedBy, decision_from_record
from theraflo_ads.policy.policies import EpsilonGreedyPolicy, Policy
from theraflo_ads.policy.scoring import HeuristicScorer
from theraflo_ads.rl.dataset import build_transitions, window
from theraflo_ads.rl.fqi import ActionFeaturizer, FittedQ, FQIConfig, QPolicy
from theraflo_ads.rl.ppo import LinearSoftmaxPolicy, PPOConfig, ppo_update
from theraflo_ads.rl.safety import evaluate_policy_safety
from theraflo_ads.safety.policy import default_safety_policy
from theraflo_ads.simulator.ab import heuristic_arm
from theraflo_ads.simulator.runner import PolicySetup, run_simulation
from theraflo_ads.simulator.world import SimConfig, World

AD_COST = 0.002  # stands in for the UX cost of an impression


def _features(catalog, priors, interactions=None) -> FeatureBuilder:
    excluded = default_safety_policy().personalization_excluded_categories
    return FeatureBuilder(catalog, priors, interactions=interactions, excluded_categories=excluded)


def train_fqi(raw: Path, *, test_days: int = 6, config: FQIConfig | None = None) -> dict[str, Any]:
    catalog = pipeline.load_catalog_json(raw / "catalog.json")
    priors = pipeline.load_priors_json(raw / "priors.json")
    records = pipeline.load_records(raw)
    labels = pipeline.load_labels(raw)
    transitions = build_transitions(records, labels, ad_cost=AD_COST)
    end = max(t.decision.timestamp for t in transitions)
    split = end - timedelta(days=test_days)
    start = min(t.decision.timestamp for t in transitions)
    train_idx = window(transitions, start, split)
    featurizer = ActionFeaturizer(catalog, _features(catalog, priors))
    q = FittedQ(featurizer, config).fit(transitions, train_idx)
    test_decisions = [
        decision_from_record(r) for r in records if datetime.fromisoformat(r["timestamp"]) >= split
    ]
    rewards = {lbl.request_id: lbl.platform_revenue - AD_COST for lbl in labels.values()}
    samples = samples_from_decisions(test_decisions, rewards)
    users = user_clusters(samples)
    reward_model = category_reward_model(samples, catalog)
    candidates = {
        "fqi": QPolicy(q, epsilon=0.1),
        "heuristic": EpsilonGreedyPolicy(HeuristicScorer(catalog, priors), epsilon=0.1),
    }
    ope = {}
    for name, policy in candidates.items():
        target = target_from_policy(policy, catalog)
        ope[name] = {
            e.estimator: asdict(e)
            for e in (
                snips(samples, target, clusters=users),
                doubly_robust(samples, target, reward_model, clusters=users),
            )
        }
    return {
        "q": q,
        "ope": ope,
        "history": q.history,
        "train_transitions": len(train_idx),
        "test_samples": len(samples),
    }


def evaluate_in_simulator(
    make_policy: Callable[[World], Policy] | None,
    config: SimConfig,
    subscribers: Callable[[World], list] | None = None,
) -> dict[str, Any]:
    def setup(world: World) -> PolicySetup:
        chain = heuristic_arm(world)
        if make_policy is not None:
            chain = [make_policy(world), *chain]
        return PolicySetup(chain, subscribers=subscribers(world) if subscribers else ())

    result = run_simulation(config, setup)
    safety = evaluate_policy_safety(result)
    retained, total = result.retention["none"]
    return {
        "value_per_decision": result.policy_value(),
        "revenue": result.platform_revenue,
        "retention": retained / total if total else 0.0,
        "served_by": dict(result.served_by),
        "safety": asdict(safety),
    }


def train_ppo(
    base: SimConfig,
    *,
    rounds: int = 8,
    round_days: int = 7,
    users: int = 2000,
    config: PPOConfig | None = None,
) -> tuple[LinearSoftmaxPolicy, list[dict[str, float]]]:
    config = config or PPOConfig()
    holder: dict[str, LinearSoftmaxPolicy] = {}

    def make(world: World) -> LinearSoftmaxPolicy:
        if "policy" not in holder:
            featurizer = ActionFeaturizer(world.catalog, world.features)
            policy = LinearSoftmaxPolicy(featurizer, rng=random.Random(base.seed))
            _standardize(policy, world)
            holder["policy"] = policy
        policy = holder["policy"]
        policy.featurizer = ActionFeaturizer(world.catalog, world.features)
        return policy

    history = []
    for round_index in range(rounds):
        sim = SimConfig(
            n_users=users,
            n_days=round_days,
            seed=base.seed,
            run_seed=base.seed + 1000 + round_index,
            epsilon=base.epsilon,
        )

        def setup(world: World) -> PolicySetup:
            return PolicySetup([make(world), *heuristic_arm(world)])

        result = run_simulation(sim, setup)
        chosen = {
            d.request_id: d.chosen_action
            for d in result.decisions
            if d.served_by is ServedBy.RL and not d.degraded
        }
        rewards = {
            d.request_id: result.rewards.get(d.request_id, 0.0)
            - (AD_COST if d.impression_id else 0.0)
            for d in result.decisions
        }
        summary = ppo_update(holder["policy"], chosen, rewards, config)
        summary.update({"round": float(round_index), "value": result.policy_value()})
        history.append(summary)
    return holder["policy"], history


def _standardize(policy: LinearSoftmaxPolicy, world: World) -> None:
    """Feature scales from the catalog under an empty context, so no single feature dominates."""
    from theraflo_ads.domain import NO_AD, AdRequest, UserContext

    request = AdRequest("std", "std", "home_feed", UserContext("std"), world.config.start)
    actions = [*world.catalog.ads, NO_AD]
    phi = policy.featurizer.matrix(
        request, [a for a in actions if a in world.catalog.ads or a == NO_AD]
    )
    policy.standardize(phi.mean(axis=0), phi.std(axis=0))


def simulation_config(days: int = 21, users: int = 3000, run_seed: int = 501) -> SimConfig:
    return SimConfig(n_users=users, n_days=days, seed=7, run_seed=run_seed)


def summarize(results: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            k: (round(v, 5) if isinstance(v, float) else v)
            for k, v in r.items()
            if k in ("value_per_decision", "revenue", "retention")
        }
        | {"safety_passed": r["safety"]["passed"], "safety_reasons": r["safety"]["reasons"]}
        for name, r in results.items()
    }


def np_seed(seed: int) -> None:
    np.random.seed(seed)  # noqa: NPY002 - only for LightGBM-free code paths in tests
