"""Retraining controller (design doc: Retraining Strategy, Drift + Retraining State Machine,
Scenario E: drift spike → freeze automatic promotion).

Each scheduled check feeds one drift report and one data-validation report into
the state machine:

    HEALTHY ──drift alert──▶ WARNING ──persists / business hit──▶ DRIFT_CONFIRMED
       ▲                        │ false alarm                         │ should_retrain()
       └────────────────────────┘                                     ▼
    PRODUCTION ◀── A/B passes ── CANARY ◀── SHADOW ◀── gates pass ── RETRAINING

Any failed gate returns to HEALTHY with the current production model untouched.
Promotion is frozen from DRIFT_CONFIRMED through CANARY.

"Recalibration is the first response to calibration drift; full retraining is the
second." When the drift is calibration drift and the only blocker for a full
retrain is that business metrics have not (yet) moved, the controller runs the
cheaper recalibration candidate through the same gates.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from theraflo_ads.ml.validation import ValidationReport
from theraflo_ads.monitoring.jobs import DriftReport
from theraflo_ads.monitoring.retraining import (
    ModelState,
    RetrainingSignals,
    RetrainingStateMachine,
    should_retrain,
)

# Returns {"passed": bool, "reasons": [...], ...} for the retrained candidate.
Retrain = Callable[[], Mapping[str, Any]]


@dataclass
class StepResult:
    state: ModelState
    actions: list[str] = field(default_factory=list)
    reason: str = ""
    promotion_frozen: bool = False
    signals: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "actions": self.actions,
            "reason": self.reason,
            "promotion_frozen": self.promotion_frozen,
            "signals": self.signals,
        }


class RetrainingController:
    def __init__(
        self,
        state_path: Path,
        *,
        retrain: Retrain,
        recalibrate: Retrain | None = None,
        confirm_after: int = 2,
    ) -> None:
        self._path = state_path
        self._retrain = retrain
        self._recalibrate = recalibrate
        self._confirm_after = confirm_after

    def step(self, drift: DriftReport, quality: ValidationReport) -> StepResult:
        saved = self._load()
        machine = RetrainingStateMachine(ModelState(saved["state"]))
        consecutive = saved["consecutive_drift"] + 1 if drift.drift_detected else 0
        signals = RetrainingSignals(
            drift_detected=drift.drift_detected,
            business_metric_degraded=bool(drift.business.get("degraded")),
            labels_mature=drift.labels_mature,
            degradation_significant=bool(drift.business.get("significant")),
            data_quality_passed=quality.passed,
        )
        result = StepResult(machine.state, signals=signals.__dict__.copy())

        if machine.state is ModelState.HEALTHY and drift.drift_detected:
            machine.transition(ModelState.WARNING, _drift_summary(drift))
            result.actions.append("alert: drift warning")
        elif machine.state is ModelState.WARNING:
            if not drift.drift_detected:
                machine.transition(ModelState.HEALTHY, "false alarm: drift did not persist")
                result.actions.append("cleared warning")
            elif consecutive >= self._confirm_after or signals.business_metric_degraded:
                machine.transition(ModelState.DRIFT_CONFIRMED, _drift_summary(drift))
                result.actions.append("drift confirmed: automatic promotion frozen")

        if machine.state is ModelState.DRIFT_CONFIRMED:
            ok, reason = should_retrain(signals)
            result.reason = reason
            business_only = reason in BUSINESS_BLOCKERS
            if ok:
                self._candidate(machine, result, self._retrain, "retraining", reason)
            elif business_only and self._recalibrate is not None and _calibration_drift(drift):
                self._candidate(
                    machine,
                    result,
                    self._recalibrate,
                    "recalibrating",
                    f"calibration drift; full retrain blocked ({reason})",
                )
            else:
                result.actions.append(f"retraining blocked: {reason}")

        self._save(machine, saved, consecutive)
        result.state = machine.state
        result.promotion_frozen = machine.promotion_frozen
        return result

    @staticmethod
    def _candidate(machine, result: StepResult, train: Retrain, action: str, reason: str) -> None:
        machine.transition(ModelState.RETRAINING, reason)
        result.actions.append(action)
        candidate = dict(train())
        machine.transition(ModelState.CANDIDATE_EVALUATION, "candidate trained")
        if candidate.get("passed"):
            machine.transition(ModelState.SHADOW, "offline gates passed")
            result.actions.append("candidate in shadow (awaiting canary + A/B)")
        else:
            reasons = "; ".join(candidate.get("reasons", [])) or "offline gates failed"
            machine.fail_quality_gate(reasons)
            result.actions.append(f"candidate rejected: {reasons}")

    def advance(
        self, *, shadow_ok: bool | None = None, ab_promote: bool | None = None, reason: str = ""
    ) -> ModelState:
        """Move a shadow candidate on from online evidence (shadow comparison, then A/B)."""
        saved = self._load()
        machine = RetrainingStateMachine(ModelState(saved["state"]))
        if machine.state is ModelState.SHADOW and shadow_ok is not None:
            if shadow_ok:
                machine.transition(ModelState.CANARY, reason or "shadow comparison passed")
            else:
                machine.fail_quality_gate(reason or "shadow comparison failed")
        elif machine.state is ModelState.CANARY and ab_promote is not None:
            if ab_promote:
                machine.transition(ModelState.PRODUCTION, reason or "A/B test passed")
                machine.transition(ModelState.HEALTHY, "new production baseline")
            else:
                machine.fail_quality_gate(reason or "A/B test did not pass")
        self._save(machine, saved, saved["consecutive_drift"])
        return machine.state

    def state(self) -> ModelState:
        return ModelState(self._load()["state"])

    def history(self) -> list[dict[str, Any]]:
        return self._load()["history"]

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"state": ModelState.HEALTHY.value, "history": [], "consecutive_drift": 0}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self, machine: RetrainingStateMachine, saved: dict, consecutive: int) -> None:
        now = datetime.now(UTC).isoformat()
        history = saved["history"] + [
            {"from": a.value, "to": b.value, "reason": why, "at": now}
            for a, b, why in machine.history
        ]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "state": machine.state.value,
                    "history": history,
                    "consecutive_drift": consecutive,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(self._path)


BUSINESS_BLOCKERS = frozenset(
    {"business metrics not affected", "degradation not statistically meaningful"}
)


def _calibration_drift(drift: DriftReport) -> bool:
    return bool(
        any(c.alert for c in drift.category_calibration)
        or (drift.calibration is not None and drift.calibration.alert)
    )


def _drift_summary(drift: DriftReport) -> str:
    parts = []
    if drift.drifted_features:
        parts.append(f"features {', '.join(drift.drifted_features[:5])}")
    if drift.prediction is not None and drift.prediction.alert:
        parts.append("prediction")
    if drift.calibration is not None and drift.calibration.alert:
        parts.append(f"calibration ({drift.calibration.action})")
    if drift.concept_alert:
        parts.append("concept")
    shifted = [c.category for c in drift.category_calibration if c.alert]
    if shifted:
        parts.append(f"calibration of {', '.join(shifted)}")
    parts += [f"{k} mix" for k, v in drift.categorical.items() if v["alert"]]
    return "drift: " + ", ".join(parts) if parts else "drift"
