"""Training examples from the decision log (design doc: Training Pipeline, Delayed Rewards,
Training / Validation Split).

Rules:
- One example per served impression, with the features logged at decision time.
- Degraded decisions are excluded (fallback traffic is not normal traffic).
- Only label-mature impressions: served_at + click window + grace <= cutoff.
  An impression whose window is still open is not a negative.
- Time-based splits (train < validation < test), never random.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta

import pandas as pd

from theraflo_ads.attribution.service import ImpressionLabel
from theraflo_ads.policy.decision import Decision

FEATURE_PREFIX = "f_"
META_COLUMNS = (
    "request_id",
    "impression_id",
    "timestamp",
    "day",
    "user_hash",
    "ad_id",
    "category",
    "pricing_model",
    "placement",
    "consent_scope",
    "served_by",
    "holdout",
    "propensity",
    "ope_eligible",
)
LABEL_COLUMNS = ("clicked", "converted", "revenue")


def decision_rows(decisions: Iterable[Decision]) -> list[dict]:
    """Flatten decision records (the decision log / lake format) for storage."""
    return [d.to_log() for d in decisions]


def examples(
    decision_records: Iterable[Mapping],
    labels: Mapping[str, ImpressionLabel],
    *,
    categories: Mapping[str, str],
    pricing: Mapping[str, str],
    cutoff: datetime,
    maturity: timedelta,
) -> pd.DataFrame:
    """decision_records are decision-log dicts (Decision.to_log()); labels are keyed by
    impression_id; categories and pricing map ad_id to category / pricing model."""
    rows = []
    for record in decision_records:
        impression_ids = record.get("impression_ids") or []
        if not impression_ids or record.get("degraded") or not record.get("features"):
            continue
        label = labels.get(impression_ids[0])
        timestamp = datetime.fromisoformat(record["timestamp"])
        if label is None or timestamp + maturity > cutoff:
            continue
        ad_id = record["chosen_actions"][0]["ad_id"]
        row = {
            "request_id": record["request_id"],
            "impression_id": impression_ids[0],
            "timestamp": timestamp,
            "day": timestamp.date().isoformat(),
            "user_hash": record["user_hash"],
            "ad_id": ad_id,
            "category": categories.get(ad_id, "unknown"),
            "pricing_model": pricing.get(ad_id, "unknown"),
            "placement": record["placement"],
            "consent_scope": record["consent_scope"],
            "served_by": record["served_by"],
            "holdout": record["holdout"],
            "propensity": record["action_probabilities"][ad_id],
            "ope_eligible": record["ope_eligible"],
            "clicked": int(label.click_id is not None),
            "converted": int(label.conversions > 0),
            "revenue": label.platform_revenue,
        }
        row.update({f"{FEATURE_PREFIX}{k}": v for k, v in record["features"].items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith(FEATURE_PREFIX)]


def feature_names(frame: pd.DataFrame) -> list[str]:
    return [c[len(FEATURE_PREFIX) :] for c in feature_columns(frame)]


def time_split(
    frame: pd.DataFrame, *, validation_days: int, test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    days = sorted(frame["day"].unique())
    if len(days) < validation_days + test_days + 1:
        raise ValueError(f"need more than {validation_days + test_days} days, have {len(days)}")
    test_start = days[-test_days]
    validation_start = days[-(test_days + validation_days)]
    train = frame[frame["day"] < validation_start]
    validation = frame[(frame["day"] >= validation_start) & (frame["day"] < test_start)]
    test = frame[frame["day"] >= test_start]
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
