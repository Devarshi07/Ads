"""Drift + retraining state machine and the retrain decision chain."""

from __future__ import annotations

import itertools

import pytest

from theraflo_ads.monitoring import (
    InvalidTransition,
    ModelState,
    RetrainingSignals,
    RetrainingStateMachine,
    should_retrain,
)

S = ModelState
HAPPY_PATH = [
    S.HEALTHY,
    S.WARNING,
    S.DRIFT_CONFIRMED,
    S.RETRAINING,
    S.CANDIDATE_EVALUATION,
    S.SHADOW,
    S.CANARY,
    S.PRODUCTION,
]
ALLOWED = {
    *itertools.pairwise(HAPPY_PATH),
    (S.WARNING, S.HEALTHY),
    (S.PRODUCTION, S.HEALTHY),
}
FROZEN = {S.DRIFT_CONFIRMED, S.RETRAINING, S.CANDIDATE_EVALUATION, S.SHADOW, S.CANARY}


def test_full_lifecycle_records_history():
    machine = RetrainingStateMachine()
    for target in [*HAPPY_PATH[1:], S.HEALTHY]:
        machine.transition(target, f"to {target}")
    assert machine.state is S.HEALTHY
    assert machine.history[0] == (S.HEALTHY, S.WARNING, "to warning")
    assert machine.history[-1] == (S.PRODUCTION, S.HEALTHY, "to healthy")
    assert len(machine.history) == len(HAPPY_PATH)


def test_false_alarm_returns_to_healthy():
    machine = RetrainingStateMachine()
    machine.transition(S.WARNING, "PSI above threshold on device mix")
    machine.transition(S.HEALTHY, "seasonal: same weekday last month looks the same")
    assert machine.state is S.HEALTHY
    assert not machine.promotion_frozen


@pytest.mark.parametrize("source,target", sorted(ALLOWED))
def test_allowed_transitions(source, target):
    machine = RetrainingStateMachine(source)
    machine.transition(target, "ok")
    assert machine.state is target
    assert machine.history == [(source, target, "ok")]


@pytest.mark.parametrize(
    "source,target",
    sorted(set(itertools.product(ModelState, repeat=2)) - ALLOWED),
)
def test_every_other_transition_is_rejected(source, target):
    machine = RetrainingStateMachine(source)
    with pytest.raises(InvalidTransition):
        machine.transition(target, "not allowed")
    assert machine.state is source
    assert machine.history == []


@pytest.mark.parametrize(
    "source,target",
    [
        (S.HEALTHY, S.DRIFT_CONFIRMED),  # skips the warning step
        (S.SHADOW, S.PRODUCTION),  # skips canary
        (S.CANARY, S.SHADOW),  # backwards
        (S.RETRAINING, S.HEALTHY),  # abandoning a candidate goes through the quality gate
    ],
)
def test_notable_disallowed_transitions(source, target):
    with pytest.raises(InvalidTransition):
        RetrainingStateMachine(source).transition(target, "shortcut")


def test_transition_accepts_state_values_and_rejects_unknown_ones():
    machine = RetrainingStateMachine()
    machine.transition("warning", "string value")
    assert machine.state is S.WARNING
    with pytest.raises(ValueError):
        machine.transition("rollback", "unknown")


def test_history_cannot_be_mutated_from_outside():
    machine = RetrainingStateMachine()
    machine.history.append((S.HEALTHY, S.PRODUCTION, "forged"))
    assert machine.history == []


@pytest.mark.parametrize("source", list(ModelState))
def test_failed_quality_gate_falls_back_to_healthy_from_any_state(source):
    machine = RetrainingStateMachine(source)
    machine.fail_quality_gate("null spike in session_length")
    assert machine.state is S.HEALTHY
    assert machine.history == [
        (source, S.HEALTHY, "quality gate failed: null spike in session_length")
    ]
    assert not machine.promotion_frozen


@pytest.mark.parametrize("state", list(ModelState))
def test_promotion_frozen_from_drift_confirmed_through_canary(state):
    assert RetrainingStateMachine(state).promotion_frozen is (state in FROZEN)


def test_promotion_freezes_on_drift_and_thaws_after_promotion():
    machine = RetrainingStateMachine()
    frozen = []
    for target in [*HAPPY_PATH[1:], S.HEALTHY]:
        machine.transition(target, "step")
        frozen.append(machine.promotion_frozen)
    assert frozen == [False, True, True, True, True, True, False, False]


ALL_CLEAR = dict(
    drift_detected=True,
    business_metric_degraded=True,
    labels_mature=True,
    degradation_significant=True,
    data_quality_passed=True,
)


def test_should_retrain_when_every_check_passes():
    assert should_retrain(RetrainingSignals(**ALL_CLEAR)) == (True, "retrain candidate")


@pytest.mark.parametrize(
    "failed,reason",
    [
        ("drift_detected", "no drift detected"),
        ("business_metric_degraded", "business metrics not affected"),
        ("labels_mature", "labels not mature (attribution window still open)"),
        ("degradation_significant", "degradation not statistically meaningful"),
        ("data_quality_passed", "data quality gate failed"),
    ],
)
def test_should_retrain_reports_each_blocking_reason(failed, reason):
    assert should_retrain(RetrainingSignals(**{**ALL_CLEAR, failed: False})) == (False, reason)


def test_should_retrain_reports_the_first_blocker_in_chain_order():
    signals = RetrainingSignals(
        drift_detected=True,
        business_metric_degraded=True,
        labels_mature=False,
        degradation_significant=False,
        data_quality_passed=False,
    )
    assert should_retrain(signals) == (False, "labels not mature (attribution window still open)")
    assert should_retrain(RetrainingSignals(*[False] * 5)) == (False, "no drift detected")
