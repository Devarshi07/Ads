"""Retraining lifecycle (design doc: Retraining Strategy, Drift + Retraining State Machine).

Drift alone never retrains: business impact, mature labels, significance and a
passing data-quality gate must all hold. Once drift is confirmed, automatic
promotion stays frozen until the candidate has cleared canary (Production
Failure Scenarios → Scenario E).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelState(StrEnum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DRIFT_CONFIRMED = "drift_confirmed"
    RETRAINING = "retraining"
    CANDIDATE_EVALUATION = "candidate_evaluation"
    SHADOW = "shadow"
    CANARY = "canary"
    PRODUCTION = "production"


class InvalidTransition(ValueError):
    pass


_ALLOWED: dict[ModelState, frozenset[ModelState]] = {
    ModelState.HEALTHY: frozenset({ModelState.WARNING}),
    # WARNING -> HEALTHY is a false alarm.
    ModelState.WARNING: frozenset({ModelState.DRIFT_CONFIRMED, ModelState.HEALTHY}),
    ModelState.DRIFT_CONFIRMED: frozenset({ModelState.RETRAINING}),
    ModelState.RETRAINING: frozenset({ModelState.CANDIDATE_EVALUATION}),
    ModelState.CANDIDATE_EVALUATION: frozenset({ModelState.SHADOW}),
    ModelState.SHADOW: frozenset({ModelState.CANARY}),
    ModelState.CANARY: frozenset({ModelState.PRODUCTION}),
    # The promoted model becomes the new healthy baseline.
    ModelState.PRODUCTION: frozenset({ModelState.HEALTHY}),
}

_PROMOTION_FROZEN = frozenset(
    {
        ModelState.DRIFT_CONFIRMED,
        ModelState.RETRAINING,
        ModelState.CANDIDATE_EVALUATION,
        ModelState.SHADOW,
        ModelState.CANARY,
    }
)

Transition = tuple[ModelState, ModelState, str]


class RetrainingStateMachine:
    def __init__(self, state: ModelState = ModelState.HEALTHY) -> None:
        self._state = ModelState(state)
        self._history: list[Transition] = []

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def history(self) -> list[Transition]:
        return list(self._history)

    @property
    def promotion_frozen(self) -> bool:
        return self._state in _PROMOTION_FROZEN

    def transition(self, to: ModelState, reason: str) -> None:
        target = ModelState(to)
        if target not in _ALLOWED[self._state]:
            raise InvalidTransition(f"{self._state} -> {target} is not allowed")
        self._move(target, reason)

    def fail_quality_gate(self, reason: str) -> None:
        """Abandon the candidate from any state; the current production model keeps serving."""
        self._move(ModelState.HEALTHY, f"quality gate failed: {reason}")

    def _move(self, to: ModelState, reason: str) -> None:
        self._history.append((self._state, to, reason))
        self._state = to


@dataclass(frozen=True)
class RetrainingSignals:
    drift_detected: bool
    business_metric_degraded: bool
    labels_mature: bool
    degradation_significant: bool
    data_quality_passed: bool


def should_retrain(signals: RetrainingSignals) -> tuple[bool, str]:
    """Walk the decision chain in order and return the first blocking reason."""
    chain = (
        (signals.drift_detected, "no drift detected"),
        (signals.business_metric_degraded, "business metrics not affected"),
        (signals.labels_mature, "labels not mature (attribution window still open)"),
        (signals.degradation_significant, "degradation not statistically meaningful"),
        (signals.data_quality_passed, "data quality gate failed"),
    )
    for holds, blocking_reason in chain:
        if not holds:
            return False, blocking_reason
    return True, "retrain candidate"
