"""Prometheus metrics (design doc: Monitoring, Alerting, Incident Response).

Labels are bounded: routes are templates (never raw paths, which carry tokens),
placements outside the known set collapse to "other", and nothing is labeled by
user, ad, or advertiser.

The safety counters are invariants checked on every served ad, after the
decision service and before the response leaves the process. Their expected
value is zero; any increment pages someone, and the ad is withheld (the last
fail-closed gate).

Every counter that an alert or panel reads is created at 0 for each known label
value at startup. A labeled series otherwise appears only with its first
increment, Prometheus never sees the step from 0, and `increase()` reads 0: the
first safety violation would not page anyone. (Found on the running stack, where
82 rate-limited requests showed as a rate of 0.)
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from theraflo_ads.domain import AdCatalog, ReviewStatus, UserContext
from theraflo_ads.events.ingestion import IngestStatus
from theraflo_ads.observability.ratelimit import DEFAULT_LIMITS
from theraflo_ads.policy.decision import Decision
from theraflo_ads.review.pipeline import Outcome
from theraflo_ads.safety.engine import BlockReason, restricted_categories
from theraflo_ads.safety.policy import AD_FREE_PLACEMENTS, DEFAULT_AD_PLACEMENTS, SafetyPolicy

KNOWN_PLACEMENTS = DEFAULT_AD_PLACEMENTS | AD_FREE_PLACEMENTS
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5)
SAFETY_VIOLATION_KINDS = (
    "served_during_kill_switch",
    "ad_free_placement",
    "suppression_window",
    "unknown_ad",
    "unapproved_ad",
    "restricted_category",
)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        r = self.registry = registry or CollectorRegistry()
        self.decisions = Counter(
            "theraflo_ads_decisions_total",
            "Ad decisions by who served them",
            ["served_by", "placement"],
            registry=r,
        )
        self.blocked = Counter(
            "theraflo_ads_blocked_total",
            "Requests blocked before the policy ran",
            ["reason"],
            registry=r,
        )
        self.degraded = Counter(
            "theraflo_ads_degraded_total",
            "Decisions that fell back or failed closed",
            ["reason"],
            registry=r,
        )
        self.safety_violations = Counter(
            "theraflo_ads_safety_violations_total",
            "Served ads that broke a hard constraint (expected: 0; the ad is withheld)",
            ["kind"],
            registry=r,
        )
        self.http_requests = Counter(
            "theraflo_ads_http_requests_total",
            "HTTP requests",
            ["route", "method", "status"],
            registry=r,
        )
        self.http_latency = Histogram(
            "theraflo_ads_http_request_duration_seconds",
            "HTTP request latency",
            ["route"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.client_events = Counter(
            "theraflo_ads_client_events_total",
            "Client events by ingestion status",
            ["status"],
            registry=r,
        )
        self.rate_limited = Counter(
            "theraflo_ads_rate_limited_total",
            "Requests refused by the rate limiter",
            ["scope"],
            registry=r,
        )
        self.rate_limiter_errors = Counter(
            "theraflo_ads_rate_limiter_errors_total",
            "Rate limiter unavailable (failed open)",
            registry=r,
        )
        self.review_outcomes = Counter(
            "theraflo_ads_review_outcomes_total",
            "Creative review outcomes",
            ["outcome"],
            registry=r,
        )
        self.kill_switch = Gauge(
            "theraflo_ads_kill_switch", "1 while the global kill switch is on", registry=r
        )
        self.dependency_up = Gauge(
            "theraflo_ads_dependency_up", "Readiness of each dependency", ["name"], registry=r
        )
        self.review_queue = Gauge(
            "theraflo_ads_review_queue_depth", "Open human review items", ["kind"], registry=r
        )
        self._collectors: list[Callable[[], None]] = []
        for kind in SAFETY_VIOLATION_KINDS:
            self.safety_violations.labels(kind)
        for reason in BlockReason:
            self.blocked.labels(reason.value)
        for scope in DEFAULT_LIMITS:
            self.rate_limited.labels(scope)
        for status in IngestStatus:
            self.client_events.labels(status.value)
        for outcome in Outcome:
            self.review_outcomes.labels(outcome.value)

    def on_scrape(self, collector: Callable[[], None]) -> None:
        self._collectors.append(collector)

    def render(self) -> bytes:
        for collect in self._collectors:
            # A broken gauge (e.g. a dependency down) must not break the scrape.
            with contextlib.suppress(Exception):
                collect()
        return generate_latest(self.registry)

    def summary(self) -> dict[str, Any]:
        """Totals since this process started, for the dev console (Prometheus has the rates).
        Counters are summed over every label but the one named."""
        totals: dict[str, dict[str, float]] = {}
        latency: dict[str, list[tuple[float, float]]] = {}
        wanted = {
            "theraflo_ads_decisions": "served_by",
            "theraflo_ads_blocked": "reason",
            "theraflo_ads_degraded": "reason",
            "theraflo_ads_safety_violations": "kind",
            "theraflo_ads_client_events": "status",
            "theraflo_ads_rate_limited": "scope",
            "theraflo_ads_review_outcomes": "outcome",
            "theraflo_ads_http_requests": "status",
        }
        for family in self.registry.collect():
            label = wanted.get(family.name)
            for sample in family.samples:
                if label is not None and sample.name.endswith("_total"):
                    by = totals.setdefault(family.name.removeprefix("theraflo_ads_"), {})
                    key = sample.labels.get(label, "")
                    by[key] = by.get(key, 0.0) + sample.value
                elif sample.name == "theraflo_ads_http_request_duration_seconds_bucket":
                    route = sample.labels["route"]
                    latency.setdefault(route, []).append((float(sample.labels["le"]), sample.value))
        return {
            "totals": totals,
            "rank_latency_ms": _quantiles(latency.get("/v1/ads/rank", [])),
        }

    def observe_decision(self, decision: Decision) -> None:
        placement = decision.placement if decision.placement in KNOWN_PLACEMENTS else "other"
        self.decisions.labels(decision.served_by.value, placement).inc()
        if decision.blocked_reason:
            self.blocked.labels(decision.blocked_reason).inc()
        for reason in decision.degraded_reasons:
            self.degraded.labels(reason).inc()


def _quantiles(buckets: list[tuple[float, float]]) -> dict[str, float | None]:
    """p50 / p95 / p99 from cumulative histogram buckets (bucket upper bounds, like
    Prometheus' histogram_quantile, interpolated linearly inside the bucket)."""
    buckets = sorted(buckets)
    count = buckets[-1][1] if buckets else 0.0
    result: dict[str, float | None] = {"count": count}
    for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        if not count:
            result[name] = None
            continue
        rank = q * count
        lower_bound, lower_count = 0.0, 0.0
        value = None
        for bound, cumulative in buckets:
            if cumulative >= rank:
                if bound == float("inf"):
                    value = lower_bound
                else:
                    inside = cumulative - lower_count
                    share = (rank - lower_count) / inside if inside else 0.0
                    value = lower_bound + (bound - lower_bound) * share
                break
            lower_bound, lower_count = bound, cumulative
        result[name] = None if value is None else value * 1000
    return result


def served_ad_violations(
    decision: Decision,
    catalog: AdCatalog,
    policy: SafetyPolicy,
    user: UserContext,
) -> list[str]:
    """Hard constraints re-checked on the final answer, independent of the code that made it."""
    if decision.impression_id is None:
        return []
    kinds = []
    ad = catalog.ads.get(decision.chosen_action)
    if policy.kill_switch:
        kinds.append("served_during_kill_switch")
    if decision.placement in AD_FREE_PLACEMENTS or decision.placement not in policy.ad_placements:
        kinds.append("ad_free_placement")
    if user.suppressed_until is not None and decision.timestamp < user.suppressed_until:
        kinds.append("suppression_window")
    if ad is None:
        kinds.append("unknown_ad")
        return kinds
    if ad.review_status is not ReviewStatus.APPROVED:
        kinds.append("unapproved_ad")
    if ad.category in restricted_categories(policy, user.age_band):
        kinds.append("restricted_category")
    return kinds
