"""Experiment analysis (design doc: Online Experimentation, Statistical Testing, Experiment
Hygiene, Permanent Holdouts).

Randomization is by user, so every metric is aggregated per user first:
- sample-ratio mismatch check on users per arm (assignment / logging health);
- primary metric: revenue per user, CUPED-adjusted with a pre-period covariate,
  winsorized against heavy tails;
- guardrails: CTR (delta-method ratio metric), ad load per decision (must not rise),
  retention (two-proportion test);
- safety counters must be zero;
- the promotion rule from evaluate_promotion, against a pre-registered MDE.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from theraflo_ads.attribution.service import ImpressionLabel
from theraflo_ads.experiments.promotion import (
    GuardrailCheck,
    GuardrailMode,
    PromotionDecision,
    PromotionKind,
    evaluate_promotion,
)
from theraflo_ads.experiments.stats import (
    SRMResult,
    TestResult,
    cuped_ttest,
    ratio_metric_delta,
    srm_check,
    two_proportion_ztest,
    welch_ttest,
)


@dataclass
class _User:
    decisions: int = 0
    impressions: int = 0
    clicks: int = 0
    revenue: float = 0.0


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    users: int
    decisions: int
    impressions: int
    clicks: int
    revenue: float
    revenue_per_user: float
    ctr: float
    ad_load: float
    retention: float | None


@dataclass(frozen=True)
class GuardrailLimits:
    ctr: float = 0.05  # CTR may not drop more than 5%
    ad_load: float = 0.05  # ads per decision may not rise more than 5%
    retention: float = 0.02  # retention may not drop more than 2%
    mode: GuardrailMode = GuardrailMode.NO_SIGNIFICANT_HARM


@dataclass(frozen=True)
class ExperimentReport:
    experiment_id: str
    control: str
    treatment: str
    arms: dict[str, ArmSummary]
    srm: SRMResult
    primary: TestResult
    primary_unadjusted: TestResult
    guardrails: list[GuardrailCheck]
    safety_counters: dict[str, int]
    preregistered_mde: float
    decision: PromotionDecision
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "control": self.control,
            "treatment": self.treatment,
            "arms": {k: asdict(v) for k, v in self.arms.items()},
            "srm": asdict(self.srm),
            "primary": asdict(self.primary),
            "primary_unadjusted": asdict(self.primary_unadjusted),
            "guardrails": [
                {
                    "name": g.name,
                    "max_degradation": g.max_degradation,
                    "higher_is_better": g.higher_is_better,
                    "result": asdict(g.result),
                }
                for g in self.guardrails
            ],
            "safety_counters": self.safety_counters,
            "preregistered_mde": self.preregistered_mde,
            "decision": {"promote": self.decision.promote, "reasons": list(self.decision.reasons)},
            "notes": self.notes,
        }


def per_user(
    records: Iterable[Mapping[str, Any]],
    labels: Mapping[str, ImpressionLabel],
    *,
    experiment_id: str | None,
) -> dict[str, dict[str, _User]]:
    """arm -> user_hash -> totals. experiment_id=None aggregates the pre-period (no cohort)."""
    arms: dict[str, dict[str, _User]] = defaultdict(lambda: defaultdict(_User))
    prefix = f"{experiment_id}:" if experiment_id else None
    for record in records:
        cohort = record.get("experiment_cohort")
        if prefix is None:
            if cohort is not None or record.get("holdout") != "none":
                continue
            arm = "pre"
        else:
            if not cohort or not cohort.startswith(prefix):
                continue
            arm = cohort[len(prefix) :]
        user = arms[arm][record["user_hash"]]
        if record.get("blocked_reason") is None:
            user.decisions += 1
        for impression_id in record.get("impression_ids") or ():
            user.impressions += 1
            label = labels.get(impression_id)
            if label is not None:
                user.clicks += int(label.click_id is not None)
                user.revenue += label.platform_revenue
    return arms


def analyze_experiment(
    records: Iterable[Mapping[str, Any]],
    labels: Mapping[str, ImpressionLabel],
    *,
    experiment_id: str,
    control: str,
    treatment: str,
    expected_ratio: float = 1.0,
    preregistered_mde: float,
    kind: PromotionKind = PromotionKind.SUPERIORITY,
    margin: float = 0.0,
    pre_period: Mapping[str, float] | None = None,
    retained: Mapping[str, bool] | None = None,
    safety_counters: Mapping[str, int] | None = None,
    limits: GuardrailLimits | None = None,
    winsorize_quantile: float = 0.995,
    alpha: float = 0.05,
) -> ExperimentReport:
    records = list(records)
    limits = limits or GuardrailLimits()
    arms = per_user(records, labels, experiment_id=experiment_id)
    c, t = arms.get(control, {}), arms.get(treatment, {})
    if not c or not t:
        raise ValueError(f"no users in arm {control if not c else treatment}")
    notes = []

    srm = srm_check(len(c), len(t), expected_ratio=expected_ratio)

    def values(arm: Mapping[str, _User], attr: str) -> np.ndarray:
        return np.array([getattr(u, attr) for u in arm.values()], dtype=float)

    # Winsorize at a quantile of the pooled data so one large order cannot decide the test.
    pooled = np.concatenate([values(c, "revenue"), values(t, "revenue")])
    cap = float(np.quantile(pooled, winsorize_quantile)) if len(pooled) else 0.0
    rev_c, rev_t = np.minimum(values(c, "revenue"), cap), np.minimum(values(t, "revenue"), cap)
    primary_unadjusted = welch_ttest(rev_c, rev_t, alpha=alpha)
    if pre_period:
        pre_c = np.array([pre_period.get(u, 0.0) for u in c], dtype=float)
        pre_t = np.array([pre_period.get(u, 0.0) for u in t], dtype=float)
        primary = cuped_ttest(
            rev_c, np.minimum(pre_c, cap), rev_t, np.minimum(pre_t, cap), alpha=alpha
        )
        notes.append("primary metric is CUPED-adjusted with pre-period revenue per user")
    else:
        primary = primary_unadjusted

    guardrails = [
        GuardrailCheck(
            "ctr",
            ratio_metric_delta(
                values(c, "clicks"),
                values(c, "impressions"),
                values(t, "clicks"),
                values(t, "impressions"),
                alpha=alpha,
            ),
            limits.ctr,
            mode=limits.mode,
        ),
        GuardrailCheck(
            "ad_load",
            ratio_metric_delta(
                values(c, "impressions"),
                values(c, "decisions"),
                values(t, "impressions"),
                values(t, "decisions"),
                alpha=alpha,
            ),
            limits.ad_load,
            higher_is_better=False,
            mode=limits.mode,
        ),
    ]
    retention_rates: dict[str, float | None] = {control: None, treatment: None}
    if retained is not None:
        kept_c = sum(bool(retained.get(u, True)) for u in c)
        kept_t = sum(bool(retained.get(u, True)) for u in t)
        retention_rates = {control: kept_c / len(c), treatment: kept_t / len(t)}
        guardrails.append(
            GuardrailCheck(
                "retention",
                two_proportion_ztest(kept_c, len(c), kept_t, len(t), alpha=alpha),
                limits.retention,
                mode=limits.mode,
            )
        )
    counters = {k: int(v) for k, v in (safety_counters or {}).items()}
    decision = evaluate_promotion(
        primary=primary,
        kind=kind,
        preregistered_mde=preregistered_mde,
        margin=margin,
        guardrails=guardrails,
        safety_counters=counters,
        srm=srm,
    )
    summaries = {
        name: _summary(name, arm, retention_rates.get(name))
        for name, arm in ((control, c), (treatment, t))
    }
    return ExperimentReport(
        experiment_id,
        control,
        treatment,
        summaries,
        srm,
        primary,
        primary_unadjusted,
        guardrails,
        counters,
        preregistered_mde,
        decision,
        notes,
    )


def pre_period_revenue(
    records: Iterable[Mapping[str, Any]], labels: Mapping[str, ImpressionLabel]
) -> dict[str, float]:
    """Revenue per user before the experiment started (the CUPED covariate)."""
    return {u: v.revenue for u, v in per_user(records, labels, experiment_id=None)["pre"].items()}


def _summary(name: str, users: Mapping[str, _User], retention: float | None) -> ArmSummary:
    decisions = sum(u.decisions for u in users.values())
    impressions = sum(u.impressions for u in users.values())
    clicks = sum(u.clicks for u in users.values())
    revenue = sum(u.revenue for u in users.values())
    return ArmSummary(
        arm=name,
        users=len(users),
        decisions=decisions,
        impressions=impressions,
        clicks=clicks,
        revenue=revenue,
        revenue_per_user=revenue / len(users) if users else 0.0,
        ctr=clicks / impressions if impressions else 0.0,
        ad_load=impressions / decisions if decisions else 0.0,
        retention=retention,
    )
