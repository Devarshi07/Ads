"""The drift job (design doc: Drift Monitoring Architecture, Drift Pitfalls at This Scale).

Three window families: features at `now`, clicks at `now - click_maturity`
(clicks arrive within the late-click grace, so they are the fast proxy), and
revenue at `now - maturity` (conversion revenue is only visible once
attribution closes). Each is compared with a reference: by default the window
just before it, or, with `baseline`, a fixed window ending at `baseline` (the
reference dataset from when the serving model was deployed). A step change is
visible to rolling windows in only one check; a fixed baseline keeps it visible
until it is dealt with.

What alerts, and why each check is shaped the way it is:
- Context-feature drift (daypart, device, placement, consent, exposure and
  interaction counts) on all trusted traffic. Features of the *chosen* ad move
  whenever the policy or budget exhaustion changes the served mix: that is
  policy-induced drift, so ad-level features are measured only on the
  uniform-exploration holdout (usually too small to alert at this scale).
- Placement / consent mix (PSI + chi-square).
- Per-category calibration: observed / predicted clicks, reference vs current,
  with a Poisson rate test. This catches "streaming CTR fell 70%" directly.
- Global calibration and concept drift; concept drift alerts only when a
  bootstrap shows the AUC drop is real, not noise on a few thousand clicks.
- Prediction drift is reported but informational (it is policy-dependent).
- The business signal: revenue per served impression on mature windows.

Degraded and invalid traffic is excluded first, so an outage or a fraud wave
never looks like drift.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from theraflo_ads.attribution.service import ImpressionLabel
from theraflo_ads.experiments.stats import welch_ttest
from theraflo_ads.monitoring.calibration import (
    CalibrationDrift,
    ConceptDrift,
    calibration_drift,
    concept_drift,
)
from theraflo_ads.monitoring.drift import (
    DriftAlertPolicy,
    FeatureDrift,
    bootstrap_psi_threshold,
    chi_square_categorical,
    evaluate_feature_drift,
    exclude_untrusted,
    psi_categorical,
)

CONTEXT_PREFIXES = ("daypart_", "device_", "placement_")
CONTEXT_FEATURES = ("personalized", "user_exposures_24h", "user_total_clicks")


@dataclass(frozen=True)
class DriftJobConfig:
    reference_days: int = 7
    current_days: int = 7
    alpha: float = 0.01
    min_samples: int = 2000
    min_effect: float = 0.05  # floor under the bootstrapped PSI threshold
    business_degradation: float = 0.05  # relative drop in revenue per impression
    click_maturity: timedelta = timedelta(days=1)
    calibration_shift: float = 0.30  # per-category observed/predicted ratio change
    min_category_clicks: int = 30
    concept_bootstrap: int = 200


@dataclass(frozen=True)
class CategoryCalibration:
    category: str
    ratio_reference: float  # observed clicks / predicted clicks
    ratio_current: float
    relative_change: float
    p_value: float
    clicks_reference: int
    clicks_current: int
    alert: bool


@dataclass
class DriftReport:
    generated_at: datetime
    windows: dict[str, tuple[str, str]]
    excluded_untrusted: int
    features: list[FeatureDrift] = field(default_factory=list)
    holdout_features: list[FeatureDrift] = field(default_factory=list)
    categorical: dict[str, dict[str, float | bool]] = field(default_factory=dict)
    category_calibration: list[CategoryCalibration] = field(default_factory=list)
    prediction: FeatureDrift | None = None
    calibration: CalibrationDrift | None = None
    concept: ConceptDrift | None = None
    concept_drop_ci: tuple[float, float] | None = None
    business: dict[str, Any] = field(default_factory=dict)
    labels_mature: bool = False

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature for f in (*self.features, *self.holdout_features) if f.alert]

    @property
    def concept_alert(self) -> bool:
        ci = self.concept_drop_ci
        return bool(self.concept is not None and self.concept.alert and ci and ci[0] > 0)

    @property
    def drift_detected(self) -> bool:
        return bool(
            self.drifted_features
            or any(c["alert"] for c in self.categorical.values())
            or any(c.alert for c in self.category_calibration)
            or (self.calibration is not None and self.calibration.alert)
            or self.concept_alert
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "windows": self.windows,
            "excluded_untrusted": self.excluded_untrusted,
            "drift_detected": self.drift_detected,
            "drifted_features": self.drifted_features,
            "features": [asdict(f) for f in self.features],
            "holdout_features": [asdict(f) for f in self.holdout_features],
            "categorical": self.categorical,
            "category_calibration": [asdict(c) for c in self.category_calibration],
            "prediction": asdict(self.prediction) if self.prediction else None,
            "calibration": asdict(self.calibration) if self.calibration else None,
            "concept": asdict(self.concept) if self.concept else None,
            "concept_alert": self.concept_alert,
            "concept_drop_ci": self.concept_drop_ci,
            "business": self.business,
            "labels_mature": self.labels_mature,
        }


def run_drift_job(
    records: Iterable[Mapping[str, Any]],
    labels: Mapping[str, ImpressionLabel],
    *,
    now: datetime,
    maturity: timedelta,
    config: DriftJobConfig | None = None,
    categories: Mapping[str, str] | None = None,
    baseline: datetime | None = None,
) -> DriftReport:
    """`categories` maps ad_id to category for per-category calibration; `baseline` fixes the
    reference windows to the period ending then."""
    config = config or DriftJobConfig()
    all_records = list(records)
    trusted = [r for r in exclude_untrusted(all_records) if r.get("blocked_reason") is None]
    span_cur, span_ref = timedelta(days=config.current_days), timedelta(days=config.reference_days)

    def pair(end: datetime) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
        current = (end - span_cur, end)
        if baseline is not None and baseline <= current[0]:
            return (baseline - span_ref, baseline), current
        return (current[0] - span_ref, current[0]), current

    feature_ref, feature_cur = pair(now)
    click_ref, click_cur = pair(now - config.click_maturity)
    revenue_ref, revenue_cur = pair(now - maturity)

    def within(span: tuple[datetime, datetime]) -> list[Mapping[str, Any]]:
        start, end = span[0].isoformat(), span[1].isoformat()
        return [r for r in trusted if start <= r["timestamp"] < end]

    report = DriftReport(
        generated_at=now,
        windows={
            name: (span[0].isoformat(), span[1].isoformat())
            for name, span in (
                ("feature_reference", feature_ref),
                ("feature_current", feature_cur),
                ("click_reference", click_ref),
                ("click_current", click_cur),
                ("revenue_reference", revenue_ref),
                ("revenue_current", revenue_cur),
            )
        },
        excluded_untrusted=len(all_records) - len(exclude_untrusted(all_records)),
    )
    floor = DriftAlertPolicy(config.min_effect, config.alpha, config.min_samples)
    ref, cur = within(feature_ref), within(feature_cur)
    served_ref = [r for r in ref if r.get("features")]
    served_cur = [r for r in cur if r.get("features")]

    names = _feature_names(served_ref)
    context = [n for n in names if n.startswith(CONTEXT_PREFIXES) or n in CONTEXT_FEATURES]
    report.features = _feature_drift(context, served_ref, served_cur, floor)
    holdout = "uniform_exploration"
    report.holdout_features = _feature_drift(
        [n for n in names if n not in context],
        [r for r in served_ref if r.get("holdout") == holdout],
        [r for r in served_cur if r.get("holdout") == holdout],
        floor,
    )

    for column in ("placement", "consent_scope"):
        counts_ref, counts_cur = Counter(r[column] for r in ref), Counter(r[column] for r in cur)
        if not counts_ref or not counts_cur:
            continue
        value = psi_categorical(counts_ref, counts_cur)
        _, p_value = chi_square_categorical(counts_ref, counts_cur)
        enough = sum(counts_cur.values()) >= config.min_samples
        report.categorical[column] = {
            "psi": value,
            "p_value": p_value,
            "alert": bool(enough and value >= 0.1 and p_value < config.alpha),
        }

    pred_ref, pred_cur = _predictions(served_ref), _predictions(served_cur)
    if pred_ref.size:
        report.prediction = evaluate_feature_drift(
            "predicted_p_ctr", pred_ref, pred_cur, _policy(pred_ref, pred_cur, floor)
        )

    clicks_ref = _labelled(within(click_ref), labels)
    clicks_cur = _labelled(within(click_cur), labels)
    if clicks_ref and clicks_cur:
        p_ref, y_ref = (np.array([v[i] for v in clicks_ref]) for i in (0, 1))
        p_cur, y_cur = (np.array([v[i] for v in clicks_cur]) for i in (0, 1))
        report.calibration = calibration_drift(
            p_ref, y_ref, p_cur, y_cur, min_samples=config.min_samples
        )
        report.concept = concept_drift(p_ref, y_ref, p_cur, y_cur)
        report.concept_drop_ci = _auc_drop_ci(p_ref, y_ref, p_cur, y_cur, config.concept_bootstrap)
        if categories:
            report.category_calibration = _category_calibration(
                clicks_ref, clicks_cur, categories, config
            )

    revenue_ref_rows = _labelled(within(revenue_ref), labels)
    revenue_cur_rows = _labelled(within(revenue_cur), labels)
    report.labels_mature = bool(revenue_cur_rows)
    if revenue_ref_rows and revenue_cur_rows:
        test = welch_ttest(
            np.array([v[2] for v in revenue_ref_rows]), np.array([v[2] for v in revenue_cur_rows])
        )
        change = test.relative_lift if test.relative_lift is not None else 0.0
        report.business = {
            "revenue_per_impression_reference": float(np.mean([v[2] for v in revenue_ref_rows])),
            "revenue_per_impression_current": float(np.mean([v[2] for v in revenue_cur_rows])),
            "relative_change": change,
            "ci": [test.relative_ci_low, test.relative_ci_high],
            "degraded": change < -config.business_degradation,
            "significant": test.relative_ci_high is not None and test.relative_ci_high < 0,
        }
    return report


def _feature_drift(names, ref_rows, cur_rows, floor) -> list[FeatureDrift]:
    out = []
    for name in names:
        reference = np.array([r["features"].get(name, 0.0) for r in ref_rows], dtype=float)
        current = np.array([r["features"].get(name, 0.0) for r in cur_rows], dtype=float)
        if reference.size == 0:
            continue
        out.append(
            evaluate_feature_drift(name, reference, current, _policy(reference, current, floor))
        )
    return out


def _policy(
    reference: np.ndarray, current: np.ndarray, floor: DriftAlertPolicy
) -> DriftAlertPolicy:
    """Threshold bootstrapped from the reference at the current window's size (small windows get
    higher thresholds), never below the configured floor."""
    if current.size < 2 or np.unique(reference).size < 2:
        return floor
    threshold = bootstrap_psi_threshold(reference, sample_size=current.size, n_boot=100)
    return DriftAlertPolicy(max(threshold, floor.effect_threshold), floor.alpha, floor.min_samples)


def _feature_names(records: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(records[0]["features"]) if records else []


def _predictions(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.array(
        [r["predicted"]["p_ctr"] for r in records if "p_ctr" in (r.get("predicted") or {})],
        dtype=float,
    )


def _labelled(
    records: Sequence[Mapping[str, Any]], labels: Mapping[str, ImpressionLabel]
) -> list[tuple[float, float, float, str]]:
    """(predicted pCTR, clicked, revenue, ad_id) for served impressions with a label."""
    out = []
    for r in records:
        impression_ids = r.get("impression_ids") or []
        predicted = (r.get("predicted") or {}).get("p_ctr")
        if not impression_ids or predicted is None:
            continue
        label = labels.get(impression_ids[0])
        if label is not None:
            out.append(
                (
                    float(predicted),
                    float(label.click_id is not None),
                    label.platform_revenue,
                    label.ad_id,
                )
            )
    return out


def _auc_drop_ci(
    p_ref, y_ref, p_cur, y_cur, n_boot: int, seed: int = 0
) -> tuple[float, float] | None:
    if len(np.unique(y_ref)) < 2 or len(np.unique(y_cur)) < 2:
        return None
    rng = np.random.default_rng(seed)
    drops = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_ref), len(y_ref))
        j = rng.integers(0, len(y_cur), len(y_cur))
        if len(np.unique(y_ref[i])) < 2 or len(np.unique(y_cur[j])) < 2:
            continue
        drops.append(roc_auc_score(y_ref[i], p_ref[i]) - roc_auc_score(y_cur[j], p_cur[j]))
    if not drops:
        return None
    low, high = np.quantile(drops, [0.025, 0.975])
    return float(low), float(high)


def _category_calibration(clicks_ref, clicks_cur, categories, config) -> list[CategoryCalibration]:
    def totals(rows):
        agg: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for predicted, clicked, _, ad_id in rows:
            entry = agg[categories.get(ad_id, "unknown")]
            entry[0] += clicked
            entry[1] += predicted
        return agg

    ref, cur = totals(clicks_ref), totals(clicks_cur)
    out = []
    for category in sorted(set(ref) & set(cur)):
        (c_ref, e_ref), (c_cur, e_cur) = ref[category], cur[category]
        if e_ref <= 0 or e_cur <= 0:
            continue
        r_ref, r_cur = c_ref / e_ref, c_cur / e_cur
        # Clicks ~ Poisson(ratio × expected): a two-sample test on the two rates.
        se = np.sqrt(max(r_ref, 1e-9) / e_ref + max(r_cur, 1e-9) / e_cur)
        p_value = float(2 * norm.sf(abs(r_cur - r_ref) / se))
        change = r_cur / r_ref - 1 if r_ref > 0 else float("inf")
        enough = min(c_ref, c_cur) >= config.min_category_clicks or (
            c_ref >= config.min_category_clicks and e_cur * r_ref >= config.min_category_clicks
        )
        alert = bool(enough and abs(change) >= config.calibration_shift and p_value < config.alpha)
        out.append(
            CategoryCalibration(
                category, r_ref, r_cur, change, p_value, int(c_ref), int(c_cur), alert
            )
        )
    return out
