"""Promotion rule (design doc: Statistical Testing → Example experiment rule).

A change is promoted only when every condition holds. All conditions are
evaluated, not just the first failing one, so a rejection lists everything
that has to change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from theraflo_ads.experiments.stats import SRMResult, TestResult


class GuardrailMode(StrEnum):
    """NON_INFERIORITY proves the metric did not degrade beyond the limit (the CI bound must
    clear it): strict, and it needs a lot of traffic. NO_SIGNIFICANT_HARM fails only on a
    statistically significant degradation or a point estimate beyond the limit: what is
    feasible at ~100K MAU, where long-term harm is watched by the no-ads holdout instead."""

    NON_INFERIORITY = "non_inferiority"
    NO_SIGNIFICANT_HARM = "no_significant_harm"


class PromotionKind(StrEnum):
    SUPERIORITY = "superiority"
    NON_INFERIORITY = "non_inferiority"


@dataclass(frozen=True)
class GuardrailCheck:
    """``max_degradation`` is relative: 0.01 means the metric may not get more than 1% worse.

    For metrics where higher is worse (abandonment, ad hides), set
    ``higher_is_better=False`` and the CI upper bound is checked instead.
    """

    name: str
    result: TestResult
    max_degradation: float
    higher_is_better: bool = True
    mode: GuardrailMode = GuardrailMode.NON_INFERIORITY

    def __post_init__(self) -> None:
        if self.max_degradation < 0:
            raise ValueError("max_degradation must be non-negative")


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reasons: tuple[str, ...]


def _primary_reasons(
    primary: TestResult, kind: PromotionKind, preregistered_mde: float, margin: float
) -> list[str]:
    reasons = []
    if kind is PromotionKind.SUPERIORITY:
        if primary.relative_lift is None:
            reasons.append("primary: relative lift is undefined because the control mean is 0")
        elif primary.relative_lift < preregistered_mde:
            reasons.append(
                f"primary: relative lift {primary.relative_lift:+.2%} is below the "
                f"pre-registered MDE of {preregistered_mde:.2%}"
            )
        if primary.ci_low <= 0:
            reasons.append(
                f"primary: CI lower bound {primary.ci_low:.6g} is not above 0 (superiority)"
            )
    elif primary.ci_low <= -margin:
        reasons.append(
            f"primary: CI lower bound {primary.ci_low:.6g} is not above -{margin:.6g} "
            "(non-inferiority)"
        )
    return reasons


def _guardrail_reasons(guardrail: GuardrailCheck) -> list[str]:
    if guardrail.mode is GuardrailMode.NO_SIGNIFICANT_HARM:
        return _harm_reasons(guardrail)
    result = guardrail.result
    bound = result.relative_ci_low if guardrail.higher_is_better else result.relative_ci_high
    if bound is None:
        return [f"guardrail {guardrail.name}: no relative CI (control mean is 0)"]
    worst_change = bound if guardrail.higher_is_better else -bound
    if worst_change < -guardrail.max_degradation:
        side = "lower" if guardrail.higher_is_better else "upper"
        return [
            f"guardrail {guardrail.name}: CI {side} bound on relative change {bound:+.2%} "
            f"exceeds the allowed degradation of {guardrail.max_degradation:.2%}"
        ]
    return []


def _harm_reasons(guardrail: GuardrailCheck) -> list[str]:
    result = guardrail.result
    if result.relative_lift is None or result.relative_ci_low is None:
        return [f"guardrail {guardrail.name}: no relative estimate (control mean is 0)"]
    sign = 1 if guardrail.higher_is_better else -1
    change = sign * result.relative_lift  # positive = better
    reasons = []
    significant = (
        result.relative_ci_high < 0 if guardrail.higher_is_better else result.relative_ci_low > 0
    )
    if significant:
        reasons.append(
            f"guardrail {guardrail.name}: significant degradation "
            f"({result.relative_lift:+.2%}, CI {result.relative_ci_low:+.2%} to "
            f"{result.relative_ci_high:+.2%})"
        )
    elif change < -guardrail.max_degradation:
        reasons.append(
            f"guardrail {guardrail.name}: point estimate {result.relative_lift:+.2%} exceeds the "
            f"allowed degradation of {guardrail.max_degradation:.2%}"
        )
    return reasons


def evaluate_promotion(
    *,
    primary: TestResult,
    kind: PromotionKind,
    preregistered_mde: float,
    margin: float = 0.0,
    guardrails: Sequence[GuardrailCheck] = (),
    safety_counters: Mapping[str, int] | None = None,
    p95_latency_ms: float | None = None,
    latency_slo_ms: float | None = None,
    srm: SRMResult | None = None,
) -> PromotionDecision:
    """Apply the promotion rule; ``margin`` is absolute, on the primary estimate's scale."""
    if preregistered_mde < 0:
        raise ValueError("preregistered_mde must be non-negative")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if kind is PromotionKind.NON_INFERIORITY and margin == 0:
        raise ValueError("a non-inferiority test needs a positive margin")
    if (p95_latency_ms is None) != (latency_slo_ms is None):
        raise ValueError("p95_latency_ms and latency_slo_ms must be given together")

    reasons: list[str] = []
    # A sample-ratio mismatch means assignment is broken, so it is reported first.
    if srm is not None and not srm.passed:
        reasons.append(
            f"srm: sample-ratio mismatch (p={srm.p_value:.2g}, observed ratio "
            f"{srm.observed_ratio:.4f} vs expected {srm.expected_ratio:.4f}); results are invalid"
        )
    reasons.extend(_primary_reasons(primary, kind, preregistered_mde, margin))
    for guardrail in guardrails:
        reasons.extend(_guardrail_reasons(guardrail))
    for name, count in (safety_counters or {}).items():
        if count != 0:
            reasons.append(f"safety: counter {name} is {count}, must be 0")
    if (
        p95_latency_ms is not None
        and latency_slo_ms is not None
        and p95_latency_ms > latency_slo_ms
    ):
        reasons.append(
            f"latency: p95 {p95_latency_ms:.1f} ms exceeds the SLO of {latency_slo_ms:.1f} ms"
        )
    return PromotionDecision(promote=not reasons, reasons=tuple(reasons))
