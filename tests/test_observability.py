"""Metrics, the final served-ad safety gate, rate limiting, and log redaction."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from pathlib import Path

import pytest

from tests.helpers import NOW, make_request, make_stack, make_user
from tests.test_api import ALICE, BOB, make_env, rank
from theraflo_ads.domain import AgeBand, ReviewStatus
from theraflo_ads.observability.logs import JsonFormatter, RedactingFilter, redact
from theraflo_ads.observability.metrics import SAFETY_VIOLATION_KINDS, Metrics, served_ad_violations
from theraflo_ads.observability.ratelimit import InProcessRateLimiter, RateLimit, limit_key
from theraflo_ads.safety.policy import default_safety_policy


def scrape(client) -> str:
    response = client.get("/metrics")
    assert response.status_code == 200
    return response.text


def test_metrics_count_decisions_and_requests_by_route_template():
    client, _, _ = make_env()
    ad = rank(client)["ads"][0]
    client.get(f"/v1/click/{ad['impression_token']}", follow_redirects=False)
    text = scrape(client)
    assert 'theraflo_ads_decisions_total{placement="home_feed",served_by="popularity"} 1.0' in text
    assert 'route="/v1/click/{token}"' in text
    assert ad["impression_token"] not in text  # raw paths never become labels
    assert "theraflo_ads_kill_switch 0.0" in text
    assert 'theraflo_ads_dependency_up{name="frequency_store"} 1.0' in text
    assert 'theraflo_ads_review_queue_depth{kind="review"} 0.0' in text


def test_blocked_requests_are_counted_by_reason():
    client, _, _ = make_env()
    rank(client, placement="journal")
    assert 'theraflo_ads_blocked_total{reason="placement_not_allowed"} 1.0' in scrape(client)


def test_metrics_token(monkeypatch):
    monkeypatch.setenv("THERAFLO_ADS_METRICS_TOKEN", "scrape-secret")
    client, _, _ = make_env()
    assert client.get("/metrics").status_code == 401
    ok = client.get("/metrics", headers={"Authorization": "Bearer scrape-secret"})
    assert ok.status_code == 200


def test_request_ids_are_echoed_only_when_well_formed():
    client, _, _ = make_env()
    good = client.get("/healthz", headers={"X-Request-ID": "req_12345678"})
    assert good.headers["X-Request-ID"] == "req_12345678"
    bad = client.get("/healthz", headers={"X-Request-ID": "<script>"})
    assert bad.headers["X-Request-ID"] != "<script>"


# ----- the final safety gate -----


def test_served_ad_violations_rechecks_every_hard_constraint():
    stack = make_stack()
    decision = stack.service.decide(make_request()).decision
    catalog, policy, user = stack.catalog, default_safety_policy(), make_user()
    assert served_ad_violations(decision, catalog, policy, user) == []
    gamble = replace(decision, chosen_action="ad_gamble")
    assert served_ad_violations(gamble, catalog, policy, user) == ["restricted_category"]
    dating = replace(decision, chosen_action="ad_dating")
    assert served_ad_violations(dating, catalog, policy, make_user()) == []
    teen = make_user(age_band=AgeBand.TEEN)
    assert served_ad_violations(dating, catalog, policy, teen) == ["restricted_category"]
    pending = replace(decision, chosen_action="ad_pending")
    assert served_ad_violations(pending, catalog, policy, user) == ["unapproved_ad"]
    journal = replace(decision, placement="journal")
    assert "ad_free_placement" in served_ad_violations(journal, catalog, policy, user)
    suppressed = make_user(suppressed_until=NOW.replace(year=2027))
    assert "suppression_window" in served_ad_violations(decision, catalog, policy, suppressed)
    killed = replace(policy, kill_switch=True)
    assert "served_during_kill_switch" in served_ad_violations(decision, catalog, killed, user)
    assert served_ad_violations(replace(decision, impression_id=None), catalog, killed, user) == []


def test_api_withholds_an_ad_that_breaks_a_hard_constraint():
    """Simulates a bug upstream: the decision service serves an unapproved ad."""
    client, container, _ = make_env()
    real = container.platform.decisions.decide

    def buggy(request):
        outcome = real(request)
        ad_id = outcome.decision.chosen_action
        container.catalog.ads[ad_id] = replace(
            container.catalog.ads[ad_id], review_status=ReviewStatus.REJECTED
        )
        return outcome

    container.platform.decisions.decide = buggy
    body = rank(client)
    assert body["ads"] == []
    assert 'theraflo_ads_safety_violations_total{kind="unapproved_ad"} 1.0' in scrape(client)


# ----- rate limiting -----


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_gcra_allows_a_burst_then_the_sustained_rate():
    clock = Clock()
    limiter = InProcessRateLimiter({"rank": RateLimit(rate=2.0, burst=3)}, clock=clock)
    assert [limiter.check("rank", "u1").allowed for _ in range(4)] == [True, True, True, False]
    assert limiter.check("rank", "u2").allowed  # budgets are per principal
    denied = limiter.check("rank", "u1")
    assert 0 < denied.retry_after_s <= 0.5
    clock.t += 0.5
    assert limiter.check("rank", "u1").allowed and not limiter.check("rank", "u1").allowed
    assert limiter.check("unknown_scope", "u1").allowed
    assert "u1" not in limit_key("rank", "u1")


def test_api_returns_429_with_retry_after():
    client, container, _ = make_env()
    container.limiter = InProcessRateLimiter({"rank": RateLimit(rate=0.01, burst=2)})
    assert rank(client)["request_id"] and rank(client)["request_id"]
    response = client.post(
        "/v1/ads/rank", json={"session_id": "s1", "placement": "home_feed"}, headers=ALICE
    )
    assert response.status_code == 429 and int(response.headers["Retry-After"]) >= 1
    assert rank(client, headers=BOB)  # another user's budget is untouched
    assert 'theraflo_ads_rate_limited_total{scope="rank"} 1.0' in scrape(client)


# ----- logs -----


@pytest.mark.parametrize(
    "secret",
    [
        "Bearer dev-user-alice",
        "x-advertiser-key: dev-adv-bookly",
        "a" * 64,
        "sk-ant-api03-abcdefghijklmnop",
    ],
)
def test_secrets_are_redacted_from_logs(secret):
    cleaned = redact(f"auth failed for {secret} at /v1/ads/rank")
    assert "[redacted]" in cleaned
    assert "alice" not in cleaned and "bookly" not in cleaned
    assert "a" * 64 not in cleaned and "abcdefghijklmnop" not in cleaned


def test_json_formatter_emits_one_object_with_extras():
    record = logging.LogRecord(
        "theraflo_ads.access", logging.INFO, __file__, 1, "request with Bearer tok123", None, None
    )
    record.route = "/v1/ads/rank"
    RedactingFilter().filter(record)
    data = json.loads(JsonFormatter().format(record))
    assert data["msg"] == "request with Bearer [redacted]"
    assert data["route"] == "/v1/ads/rank" and data["level"] == "info"


# ----- alerts and dashboard reference real metrics -----

MONITORING = Path(__file__).resolve().parents[1] / "monitoring"


def referenced_metrics() -> set[str]:
    sources = [
        (MONITORING / "prometheus" / "alerts.yml").read_text(encoding="utf-8"),
        (MONITORING / "grafana" / "dashboards" / "theraflo-ads.json").read_text(encoding="utf-8"),
    ]
    names = set()
    for source in sources:
        for name in re.findall(r"theraflo_ads_[a-z_]+", source):
            names.add(re.sub(r"_(bucket|sum|count)$", "", name))
    return names


def test_every_metric_in_alerts_and_dashboards_is_exported():
    """promtool proves the rules fire on synthetic series; this proves the series exist,
    so a renamed metric cannot silently disable an alert."""
    client, _, _ = make_env()
    exported = set(re.findall(r"^# TYPE (\S+)", scrape(client), flags=re.MULTILINE))
    exported |= {f"{name}_total" for name in exported}  # counters: family vs. sample name
    referenced = referenced_metrics()
    assert len(referenced) >= 10
    assert referenced <= exported, sorted(referenced - exported)


def test_alerted_counters_exist_at_zero_before_anything_happens():
    """Prometheus cannot see a step from 0 it never sampled: a series that first appears at
    1 has increase() 0, so the first safety violation would not page. Found on the running
    stack, where 82 rate-limited requests showed as a rate of 0."""
    text = Metrics().render().decode()
    for kind in SAFETY_VIOLATION_KINDS:
        assert f'theraflo_ads_safety_violations_total{{kind="{kind}"}} 0.0' in text
    for name, label, value in (
        ("theraflo_ads_rate_limited_total", "scope", "rank"),
        ("theraflo_ads_blocked_total", "reason", "frequency_unavailable"),
        ("theraflo_ads_blocked_total", "reason", "budget_unavailable"),
        ("theraflo_ads_client_events_total", "status", "accepted"),
        ("theraflo_ads_review_outcomes_total", "outcome", "human_review"),
    ):
        assert f'{name}{{{label}="{value}"}} 0.0' in text, (name, value)


def test_the_gate_only_reports_kinds_that_are_initialized():
    stack = make_stack()
    decision = stack.service.decide(make_request()).decision
    policy, user = default_safety_policy(), make_user()
    found = set()
    for case, case_policy, case_user in (
        (replace(decision, chosen_action="ad_gamble"), policy, user),
        (replace(decision, chosen_action="ad_pending"), policy, user),
        (replace(decision, chosen_action="ad_unknown"), policy, user),
        (replace(decision, placement="journal"), policy, user),
        (decision, replace(policy, kill_switch=True), user),
        (decision, policy, make_user(suppressed_until=NOW.replace(year=2027))),
    ):
        found |= set(served_ad_violations(case, stack.catalog, case_policy, case_user))
    assert found == set(SAFETY_VIOLATION_KINDS)
