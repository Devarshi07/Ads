"""Post-training, Track A (design doc: Post-Training → Calibration, Compression and Export).

Calibration is fitted on a separate label-mature window. Before registration, a
model must survive a round trip through its export format with parity gates:
scores within ε on the golden set, ranking agreement (Kendall τ), calibration
no worse, and p95 latency within the stage SLO.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from theraflo_ads.ml.bundle import load_bundle, save_bundle
from theraflo_ads.ml.models import CalibratedModel, IsotonicCalibrator, PlattCalibrator
from theraflo_ads.monitoring.calibration import expected_calibration_error


def fit_calibrator(
    model: CalibratedModel, x: np.ndarray, y: np.ndarray, *, kind: str = "isotonic"
) -> CalibratedModel:
    raw = model.model.predict_raw(x)
    if kind == "isotonic" and y.sum() >= 20:
        return CalibratedModel(model.model, IsotonicCalibrator.fit(raw, y))
    # Too few positives for a stable isotonic fit: a two-parameter Platt map.
    return CalibratedModel(model.model, PlattCalibrator.fit(raw, y))


@dataclass(frozen=True)
class ParityReport:
    passed: bool
    max_abs_diff: float
    kendall_tau: float
    ece_original: float
    ece_exported: float
    p95_latency_ms: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


def export_parity(
    model: CalibratedModel,
    golden_x: np.ndarray,
    recent_x: np.ndarray,
    recent_y: np.ndarray,
    *,
    feature_schema_version: str,
    epsilon: float = 1e-6,
    min_tau: float = 0.99,
    latency_budget_ms: float = 40.0,
    batch_size: int = 20,
) -> ParityReport:
    with tempfile.TemporaryDirectory() as tmp:
        save_bundle(
            Path(tmp),
            name="parity",
            version="check",
            model=model,
            feature_schema_version=feature_schema_version,
        )
        exported = load_bundle(Path(tmp), feature_schema_version=feature_schema_version).model
    original_golden = model.predict(golden_x)
    exported_golden = exported.predict(golden_x)
    max_diff = float(np.max(np.abs(original_golden - exported_golden))) if len(golden_x) else 0.0
    original_recent = model.predict(recent_x)
    exported_recent = exported.predict(recent_x)
    tau = (
        float(kendalltau(original_recent, exported_recent).statistic) if len(recent_x) > 1 else 1.0
    )
    ece_original = expected_calibration_error(original_recent, recent_y)
    ece_exported = expected_calibration_error(exported_recent, recent_y)
    latencies = []
    for start in range(0, min(len(recent_x), 2000), batch_size):
        batch = recent_x[start : start + batch_size]
        began = time.perf_counter()
        exported.predict(batch)
        latencies.append((time.perf_counter() - began) * 1000)
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

    reasons = []
    if max_diff >= epsilon:
        reasons.append(f"golden-set score diff {max_diff:.2e} >= {epsilon:.0e}")
    if np.isnan(tau) or tau < min_tau:
        reasons.append(f"ranking agreement tau {tau:.4f} < {min_tau}")
    if ece_exported > ece_original + 1e-9:
        reasons.append("calibration worse after export")
    if p95 > latency_budget_ms:
        reasons.append(f"p95 latency {p95:.1f}ms > {latency_budget_ms}ms per batch of {batch_size}")
    return ParityReport(not reasons, max_diff, tau, ece_original, ece_exported, p95, tuple(reasons))
