"""Advertiser reports (design doc: Advertiser Data Leakage, Reporting).

Even "this person uses a mental-health app" is a health-data disclosure, so:
- segments are ads-safe context only (placement, daypart, device): never a user
  attribute, never anything derived from wellness data;
- a segment is shown only with at least `min_users` distinct users and
  `min_conversions` conversions (design doc: 50 users / 10 conversions);
- complementary suppression: if hiding one segment still lets it be recovered
  as total minus the others, the next-smallest segment is hidden too.
Campaign totals (impressions, clicks, conversions, spend) are the advertiser's
own billing data and are always shown.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from theraflo_ads.attribution.service import ImpressionLabel, LabelState
from theraflo_ads.ml.features import daypart
from theraflo_ads.policy.decision import Decision

DIMENSIONS = ("placement", "daypart", "device")


@dataclass(frozen=True)
class ReportThresholds:
    min_users: int = 50
    min_conversions: int = 10


@dataclass(frozen=True)
class SegmentRow:
    dimension: str
    value: str
    impressions: int
    users: int
    clicks: int
    conversions: int
    spend: float

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "value": self.value,
            "impressions": self.impressions,
            "users": self.users,
            "clicks": self.clicks,
            "conversions": self.conversions,
            "spend": round(self.spend, 2),
            "ctr": round(self.ctr, 4),
        }


@dataclass(frozen=True)
class AdvertiserReport:
    advertiser_id: str
    start: datetime
    end: datetime
    impressions: int
    clicks: int
    conversions: int
    spend: float
    segments: tuple[SegmentRow, ...]
    suppressed: Mapping[str, int] = field(default_factory=dict)  # dimension -> hidden segments

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0

    @property
    def cost_per_conversion(self) -> float | None:
        return self.spend / self.conversions if self.conversions else None

    def to_dict(self) -> dict[str, Any]:
        cpa = self.cost_per_conversion
        return {
            "advertiser_id": self.advertiser_id,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "totals": {
                "impressions": self.impressions,
                "clicks": self.clicks,
                "conversions": self.conversions,
                "spend": round(self.spend, 2),
                "ctr": round(self.ctr, 4),
                "cost_per_conversion": round(cpa, 2) if cpa is not None else None,
            },
            "segments": [s.to_dict() for s in self.segments],
            "suppressed_segments": dict(self.suppressed),
        }


def _segment_value(dimension: str, decision: Decision | None) -> str:
    if decision is None:
        return "unknown"
    if dimension == "placement":
        return decision.placement
    if dimension == "daypart":
        return daypart(decision.timestamp)
    device = decision.feature_context.get("device") if decision.feature_context else None
    return device or "unknown"


def build_advertiser_report(
    advertiser_id: str,
    labels: Iterable[ImpressionLabel],
    decisions: Mapping[str, Decision],
    *,
    start: datetime,
    end: datetime,
    thresholds: ReportThresholds | None = None,
) -> AdvertiserReport:
    thresholds = thresholds or ReportThresholds()
    own = [
        label
        for label in labels
        if label.advertiser_id == advertiser_id
        and start <= label.served_at < end
        and label.state is not LabelState.REVERSED
    ]
    stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"impressions": 0, "users": set(), "clicks": 0, "conversions": 0, "spend": 0.0}
    )
    for label in own:
        decision = decisions.get(label.request_id)
        for dimension in DIMENSIONS:
            row = stats[(dimension, _segment_value(dimension, decision))]
            row["impressions"] += 1
            if decision is not None:
                row["users"].add(decision.user_hash)
            row["clicks"] += 1 if label.click_id else 0
            row["conversions"] += label.conversions
            row["spend"] += label.platform_revenue

    segments: list[SegmentRow] = []
    suppressed: dict[str, int] = {}
    for dimension in DIMENSIONS:
        rows = [
            SegmentRow(
                dimension,
                value,
                s["impressions"],
                len(s["users"]),
                s["clicks"],
                s["conversions"],
                s["spend"],
            )
            for (dim, value), s in sorted(stats.items())
            if dim == dimension
        ]
        shown, hidden = _suppress(rows, thresholds)
        segments.extend(shown)
        if hidden:
            suppressed[dimension] = hidden
    return AdvertiserReport(
        advertiser_id=advertiser_id,
        start=start,
        end=end,
        impressions=len(own),
        clicks=sum(1 for label in own if label.click_id),
        conversions=sum(label.conversions for label in own),
        spend=sum(label.platform_revenue for label in own),
        segments=tuple(segments),
        suppressed=suppressed,
    )


def _suppress(rows: list[SegmentRow], t: ReportThresholds) -> tuple[list[SegmentRow], int]:
    small = {r.value for r in rows if r.users < t.min_users or r.conversions < t.min_conversions}
    if len(small) == 1 and len(rows) > 1:
        # Total minus the shown segments would reveal the hidden one: hide one more.
        remaining = sorted((r for r in rows if r.value not in small), key=lambda r: r.users)
        small.add(remaining[0].value)
    shown = [r for r in rows if r.value not in small]
    return shown, len(small)
