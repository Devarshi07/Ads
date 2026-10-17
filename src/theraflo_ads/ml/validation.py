"""Data validation before training (design doc: Data Validation, Data Contracts).

A failed check blocks training: a model trained on an outage, a logging bug, or a
fraud wave is worse than the model already serving.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from theraflo_ads.attribution.service import ImpressionLabel

KNOWN_SCHEMA_PREFIXES = ("features_v",)


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    checked_records: int
    checked_labels: int
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    stats: Mapping[str, Any] = field(default_factory=dict)


def validate(
    records: Iterable[Mapping[str, Any]],
    labels: Mapping[str, ImpressionLabel],
    *,
    now: datetime,
    max_degraded_share: float = 0.05,
    max_null_feature_share: float = 0.01,
    clock_skew: timedelta = timedelta(minutes=5),
) -> ValidationReport:
    records = list(records)
    failures: list[str] = []
    warnings: list[str] = []
    ids = Counter(r.get("request_id") for r in records)
    duplicates = sum(c - 1 for c in ids.values() if c > 1)
    if duplicates:
        failures.append(f"{duplicates} duplicate request_ids (idempotency broken upstream)")

    degraded = sum(1 for r in records if r.get("degraded"))
    if records and degraded / len(records) > max_degraded_share:
        failures.append(
            f"degraded share {degraded / len(records):.1%} > {max_degraded_share:.0%} "
            "(an outage window; do not train on it)"
        )

    bad_probabilities = future = bad_schema = served = null_features = 0
    for r in records:
        probs = r.get("action_probabilities") or {}
        total = sum(probs.values())
        if probs and (not math.isclose(total, 1.0, abs_tol=1e-6) or min(probs.values()) < 0):
            bad_probabilities += 1
        chosen = r.get("chosen_actions") or []
        if chosen and r.get("ope_eligible") and probs.get(chosen[0]["ad_id"], 0) <= 0:
            bad_probabilities += 1
        if datetime.fromisoformat(r["timestamp"]) > now + clock_skew:
            future += 1
        version = r.get("feature_schema_version", "")
        if not version.startswith(KNOWN_SCHEMA_PREFIXES):
            bad_schema += 1
        if r.get("impression_ids"):
            served += 1
            features = r.get("features") or {}
            if any(
                v is None or (isinstance(v, float) and not math.isfinite(v))
                for v in features.values()
            ):
                null_features += 1
    if bad_probabilities:
        failures.append(f"{bad_probabilities} decisions with invalid propensities")
    if future:
        failures.append(f"{future} decisions timestamped in the future")
    if bad_schema:
        failures.append(f"{bad_schema} decisions with an unknown feature schema")
    if served and null_features / served > max_null_feature_share:
        failures.append(f"{null_features / served:.1%} of served decisions have null features")

    negative_revenue = sum(1 for label in labels.values() if label.platform_revenue < -1e-9)
    impossible = sum(1 for label in labels.values() if label.conversions < 0)
    if negative_revenue or impossible:
        failures.append(f"{negative_revenue + impossible} labels with impossible values")
    orphans = len({i for r in records for i in r.get("impression_ids") or ()} - set(labels))
    if orphans:
        warnings.append(f"{orphans} served impressions without a label (recorded elsewhere?)")

    return ValidationReport(
        passed=not failures,
        checked_records=len(records),
        checked_labels=len(labels),
        failures=tuple(failures),
        warnings=tuple(warnings),
        stats={"degraded": degraded, "duplicates": duplicates, "served": served},
    )
