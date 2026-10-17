import pytest

from theraflo_ads.experiments.promotion import (
    GuardrailCheck,
    PromotionDecision,
    PromotionKind,
    evaluate_promotion,
)
from theraflo_ads.experiments.stats import TestResult, srm_check


def _result(
    estimate=0.12,
    *,
    relative_lift=0.12,
    ci_low=0.05,
    ci_high=0.19,
    relative_ci_low=0.04,
    relative_ci_high=0.20,
):
    return TestResult(
        estimate=estimate,
        relative_lift=relative_lift,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=0.001,
        statistic=3.3,
        method="fixed",
        relative_ci_low=relative_ci_low,
        relative_ci_high=relative_ci_high,
    )


def _neutral(name, *, higher_is_better=True, max_degradation=0.01):
    flat = _result(
        0.0,
        relative_lift=0.0,
        ci_low=-0.001,
        ci_high=0.001,
        relative_ci_low=-0.005,
        relative_ci_high=0.005,
    )
    return GuardrailCheck(name, flat, max_degradation, higher_is_better)


CLEAN = {
    "primary": _result(),
    "kind": PromotionKind.SUPERIORITY,
    "preregistered_mde": 0.10,
    "guardrails": (_neutral("retention"), _neutral("abandonment", higher_is_better=False)),
    "safety_counters": {"crisis_session_ads": 0, "minor_ads": 0},
    "p95_latency_ms": 38.0,
    "latency_slo_ms": 50.0,
    "srm": srm_check(50_000, 50_050),
}


def _evaluate(**overrides):
    return evaluate_promotion(**{**CLEAN, **overrides})


def _single_reason(decision, prefix):
    assert not decision.promote
    assert len(decision.reasons) == 1, decision.reasons
    assert decision.reasons[0].startswith(prefix)
    return decision.reasons[0]


def test_clean_superiority_promotes():
    decision = _evaluate()
    assert decision == PromotionDecision(promote=True, reasons=())


def test_minimal_call_promotes():
    decision = evaluate_promotion(
        primary=_result(), kind=PromotionKind.SUPERIORITY, preregistered_mde=0.10
    )
    assert decision.promote


def test_lift_below_mde():
    reason = _single_reason(_evaluate(primary=_result(relative_lift=0.08)), "primary:")
    assert "MDE" in reason


def test_lift_exactly_at_mde_promotes():
    assert _evaluate(primary=_result(relative_lift=0.10)).promote


def test_superiority_needs_ci_above_zero():
    reason = _single_reason(_evaluate(primary=_result(ci_low=-0.01)), "primary:")
    assert "superiority" in reason
    _single_reason(_evaluate(primary=_result(ci_low=0.0)), "primary:")


def test_superiority_with_undefined_lift():
    reason = _single_reason(_evaluate(primary=_result(relative_lift=None)), "primary:")
    assert "undefined" in reason


def test_non_inferiority_passes_within_margin():
    retrain = _result(-0.001, relative_lift=-0.01, ci_low=-0.004, ci_high=0.002)
    decision = _evaluate(primary=retrain, kind=PromotionKind.NON_INFERIORITY, margin=0.005)
    assert decision.promote


def test_non_inferiority_fails_beyond_margin():
    retrain = _result(-0.003, relative_lift=-0.03, ci_low=-0.006, ci_high=0.0)
    decision = _evaluate(primary=retrain, kind=PromotionKind.NON_INFERIORITY, margin=0.005)
    assert "non-inferiority" in _single_reason(decision, "primary:")


def test_guardrail_degradation_blocks():
    hurt = GuardrailCheck("retention", _result(relative_ci_low=-0.03, relative_ci_high=0.01), 0.01)
    reason = _single_reason(_evaluate(guardrails=(hurt,)), "guardrail retention:")
    assert "lower bound" in reason


def test_guardrail_at_threshold_passes():
    edge = GuardrailCheck("retention", _result(relative_ci_low=-0.01), 0.01)
    assert _evaluate(guardrails=(edge,)).promote


def test_higher_is_worse_guardrail_checks_upper_bound():
    more_abandonment = GuardrailCheck(
        "abandonment",
        _result(relative_ci_low=0.01, relative_ci_high=0.04),
        0.02,
        higher_is_better=False,
    )
    reason = _single_reason(_evaluate(guardrails=(more_abandonment,)), "guardrail abandonment:")
    assert "upper bound" in reason


def test_guardrail_without_relative_interval():
    missing = GuardrailCheck(
        "ad_hides", _result(relative_lift=None, relative_ci_low=None, relative_ci_high=None), 0.01
    )
    _single_reason(_evaluate(guardrails=(missing,)), "guardrail ad_hides:")


def test_nonzero_safety_counter_blocks():
    counters = {"crisis_session_ads": 0, "minor_ads": 2}
    reason = _single_reason(_evaluate(safety_counters=counters), "safety:")
    assert "minor_ads" in reason


def test_latency_over_slo_blocks():
    _single_reason(_evaluate(p95_latency_ms=55.0), "latency:")
    assert _evaluate(p95_latency_ms=50.0).promote


def test_srm_failure_blocks():
    reason = _single_reason(_evaluate(srm=srm_check(50_000, 48_500)), "srm:")
    assert "mismatch" in reason


def test_every_failure_is_reported():
    decision = _evaluate(
        primary=_result(relative_lift=0.02, ci_low=-0.01),
        guardrails=(GuardrailCheck("retention", _result(relative_ci_low=-0.05), 0.01),),
        safety_counters={"crisis_session_ads": 1},
        p95_latency_ms=80.0,
        srm=srm_check(50_000, 48_500),
    )
    assert not decision.promote
    prefixes = [reason.split(":")[0] for reason in decision.reasons]
    assert prefixes == ["srm", "primary", "primary", "guardrail retention", "safety", "latency"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"preregistered_mde": -0.01},
        {"margin": -0.01},
        {"kind": PromotionKind.NON_INFERIORITY, "margin": 0.0},
        {"p95_latency_ms": 40.0, "latency_slo_ms": None},
        {"p95_latency_ms": None, "latency_slo_ms": 50.0},
    ],
)
def test_validation(overrides):
    with pytest.raises(ValueError):
        _evaluate(**overrides)


def test_guardrail_validation():
    with pytest.raises(ValueError):
        GuardrailCheck("retention", _result(), -0.01)


def test_kind_values():
    assert PromotionKind("non_inferiority") is PromotionKind.NON_INFERIORITY
