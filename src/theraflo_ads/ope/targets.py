"""Adapters from serving policies to OPE target policies, plus a simple reward model."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from theraflo_ads.domain import NO_AD, AdCatalog, AdRequest, UserContext
from theraflo_ads.ope.estimators import LoggedSample, RewardModel, TargetPolicy
from theraflo_ads.policy.decision import Decision
from theraflo_ads.policy.policies import Policy


def target_from_policy(policy: Policy, catalog: AdCatalog) -> TargetPolicy:
    """Evaluate `policy` on the logged eligible set of each decision."""

    def target(sample: LoggedSample):
        decision = sample.context
        if not isinstance(decision, Decision):
            raise TypeError("target_from_policy needs samples built from decisions")
        eligible = [catalog.ads[a] for a in sample.eligible_actions if a != NO_AD]
        request = AdRequest(
            decision.request_id,
            decision.session_id,
            decision.placement,
            UserContext(decision.user_hash, decision.consent_scope),
            decision.timestamp,
            # Feature-based policies rebuild every candidate's features from the log.
            feature_context=dict(decision.feature_context) or None,
        )
        return policy.distribution(request, eligible).probabilities

    return target


def category_reward_model(samples: Sequence[LoggedSample], catalog: AdCatalog) -> RewardModel:
    """Mean logged reward per ad category; NO_AD is worth zero. Used by the DR estimator."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for s in samples:
        if s.action == NO_AD:
            continue
        category = catalog.ads[s.action].category
        sums[category] += s.reward
        counts[category] += 1
    total = sum(counts.values())
    overall = sum(sums.values()) / total if total else 0.0
    means = {c: sums[c] / counts[c] for c in counts}

    def model(sample: LoggedSample, action: str) -> float:
        if action == NO_AD:
            return 0.0
        return means.get(catalog.ads[action].category, overall)

    return model
