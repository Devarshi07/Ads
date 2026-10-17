"""Decision records (design doc: Counterfactual Logging)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from theraflo_ads.domain import NO_AD, Ad, Campaign, ConsentScope
from theraflo_ads.experiments.assignment import Holdout


class ServedBy(StrEnum):
    RL = "rl"
    BANDIT = "bandit"
    RANKER = "ranker"
    HEURISTIC = "heuristic"
    POPULARITY = "popularity"
    HOUSE_AD = "house_ad"
    NO_AD = "no_ad"
    UNIFORM_HOLDOUT = "uniform_holdout"
    NO_ADS_HOLDOUT = "no_ads_holdout"


@dataclass(frozen=True)
class Decision:
    request_id: str
    session_id: str
    user_hash: str
    placement: str
    consent_scope: ConsentScope
    holdout: Holdout
    experiment_cohort: str | None
    eligible_actions: tuple[str, ...]
    action_probabilities: Mapping[str, float]
    # What the policy sampled. Differs from chosen_action when a post-policy
    # step (budget race, counter failure) had to drop the ad.
    policy_action: str | None
    chosen_action: str
    position: int | None
    served_by: ServedBy
    stochastic: bool
    degraded_reasons: tuple[str, ...]
    blocked_reason: str | None
    policy_version: str | None
    safety_policy_version: str
    feature_schema_version: str
    impression_id: str | None
    timestamp: datetime
    predicted: Mapping[str, float] = field(default_factory=dict)
    # Feature vector of the chosen ad at decision time (point-in-time correct training data).
    features: Mapping[str, float] = field(default_factory=dict)
    # Inputs that rebuild the features of any eligible ad (for OPE of feature-based policies).
    feature_context: Mapping[str, Any] = field(default_factory=dict)
    # Candidate policies scored in shadow: version -> action probabilities. Never served.
    shadow: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return bool(self.degraded_reasons)

    @property
    def ope_eligible(self) -> bool:
        """Only stochastic, non-degraded decisions that served what the policy chose."""
        return (
            self.stochastic
            and not self.degraded
            and self.blocked_reason is None
            and self.policy_action == self.chosen_action
        )

    @property
    def propensity(self) -> float:
        return self.action_probabilities[self.chosen_action]

    def to_log(self) -> dict[str, Any]:
        served = self.impression_id is not None
        return {
            "request_id": self.request_id,
            "impression_ids": [self.impression_id] if served else [],
            "user_hash": self.user_hash,
            "session_id": self.session_id,
            "placement": self.placement,
            "consent_scope": self.consent_scope.value,
            "holdout": self.holdout.value,
            "experiment_cohort": self.experiment_cohort,
            "eligible_actions": list(self.eligible_actions),
            "action_probabilities": dict(self.action_probabilities),
            "policy_action": self.policy_action,
            "chosen_actions": (
                [{"ad_id": self.chosen_action, "position": self.position}] if served else []
            ),
            "served_by": self.served_by.value,
            "stochastic": self.stochastic,
            "degraded": self.degraded,
            "degraded_reasons": list(self.degraded_reasons),
            "blocked_reason": self.blocked_reason,
            "ope_eligible": self.ope_eligible,
            "policy_version": self.policy_version,
            "safety_policy_version": self.safety_policy_version,
            "feature_schema_version": self.feature_schema_version,
            "predicted": dict(self.predicted),
            "features": dict(self.features),
            "feature_context": dict(self.feature_context),
            "shadow": {k: dict(v) for k, v in self.shadow.items()},
            "timestamp": self.timestamp.isoformat(),
        }


def decision_from_record(record: Mapping[str, Any]) -> Decision:
    """Rebuild a Decision from its decision-log record (the lake / training-data format)."""
    chosen = record["chosen_actions"]
    return Decision(
        request_id=record["request_id"],
        session_id=record["session_id"],
        user_hash=record["user_hash"],
        placement=record["placement"],
        consent_scope=ConsentScope(record["consent_scope"]),
        holdout=Holdout(record["holdout"]),
        experiment_cohort=record["experiment_cohort"],
        eligible_actions=tuple(record["eligible_actions"]),
        action_probabilities=dict(record["action_probabilities"]),
        policy_action=record["policy_action"],
        chosen_action=chosen[0]["ad_id"] if chosen else NO_AD,
        position=chosen[0]["position"] if chosen else None,
        served_by=ServedBy(record["served_by"]),
        stochastic=record["stochastic"],
        degraded_reasons=tuple(record["degraded_reasons"]),
        blocked_reason=record["blocked_reason"],
        policy_version=record["policy_version"],
        safety_policy_version=record["safety_policy_version"],
        feature_schema_version=record["feature_schema_version"],
        impression_id=record["impression_ids"][0] if record["impression_ids"] else None,
        timestamp=datetime.fromisoformat(record["timestamp"]),
        predicted=dict(record.get("predicted", {})),
        features=dict(record.get("features", {})),
        feature_context=dict(record.get("feature_context", {})),
        shadow={k: dict(v) for k, v in record.get("shadow", {}).items()},
    )


@dataclass(frozen=True)
class ServedAd:
    impression_id: str
    ad: Ad
    campaign: Campaign
    position: int
    token: str
    expires_at: datetime
    reservation_id: str
    budget_key: str
    budget_limit: float


@dataclass(frozen=True)
class DecisionOutcome:
    decision: Decision
    served: ServedAd | None
