"""Generates monitoring/grafana/dashboards/theraflo-ads.json (run from the project root).
Edit the panels here, not the JSON; Grafana reloads the file within seconds."""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
panels = []


def panel(
    title,
    exprs,
    x,
    y,
    w=12,
    h=8,
    kind="timeseries",
    unit=None,
    legend=None,
    thresholds=None,
    description="",
):
    targets = [
        {
            "datasource": DS,
            "expr": e,
            "legendFormat": (legend or [None] * len(exprs))[i] or "",
            "refId": chr(65 + i),
        }
        for i, e in enumerate(exprs)
    ]
    defaults = {"unit": unit} if unit else {}
    if thresholds:
        defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
        defaults["color"] = {"mode": "thresholds"}
    panels.append(
        {
            "id": len(panels) + 1,
            "type": kind,
            "title": title,
            "description": description,
            "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": targets,
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {},
        }
    )


RED_IF_ANY = [{"color": "green", "value": None}, {"color": "red", "value": 1}]

panel(
    "Safety gate: withheld ads (expected 0)",
    ["sum(increase(theraflo_ads_safety_violations_total[1h])) or vector(0)"],
    0,
    0,
    6,
    5,
    kind="stat",
    thresholds=RED_IF_ANY,
    description="Served ads that broke a hard constraint and were withheld at the API.",
)
panel(
    "Kill switch",
    ["max(theraflo_ads_kill_switch) or vector(0)"],
    6,
    0,
    6,
    5,
    kind="stat",
    thresholds=RED_IF_ANY,
)
panel(
    "Dependencies ready",
    ["min by (name) (theraflo_ads_dependency_up)"],
    12,
    0,
    12,
    5,
    kind="stat",
    legend=["{{name}}"],
    thresholds=[{"color": "red", "value": None}, {"color": "green", "value": 1}],
)

panel(
    "Decisions by policy",
    ["sum by (served_by) (rate(theraflo_ads_decisions_total[1m]))"],
    0,
    5,
    legend=["{{served_by}}"],
    unit="reqps",
)
panel(
    "Fail-closed and blocked requests",
    ["sum by (reason) (rate(theraflo_ads_blocked_total[1m]))"],
    12,
    5,
    legend=["{{reason}}"],
    unit="reqps",
)

RANK_BUCKETS = 'theraflo_ads_http_request_duration_seconds_bucket{route="/v1/ads/rank"}'
panel(
    "Rank latency",
    [f"histogram_quantile({q}, sum by (le) (rate({RANK_BUCKETS}[1m])))" for q in (0.5, 0.95, 0.99)],
    0,
    13,
    legend=["p50", "p95", "p99"],
    unit="s",
)
panel(
    "HTTP responses by status",
    ["sum by (status) (rate(theraflo_ads_http_requests_total[1m]))"],
    12,
    13,
    legend=["{{status}}"],
    unit="reqps",
)

panel(
    "Client events",
    ["sum by (status) (rate(theraflo_ads_client_events_total[1m]))"],
    0,
    21,
    legend=["{{status}}"],
    unit="reqps",
)
panel(
    "Degraded decisions",
    ["sum by (reason) (rate(theraflo_ads_degraded_total[1m]))"],
    12,
    21,
    legend=["{{reason}}"],
    unit="reqps",
)

panel(
    "Creative review outcomes",
    ["sum by (outcome) (increase(theraflo_ads_review_outcomes_total[1h]))"],
    0,
    29,
    legend=["{{outcome}}"],
)
panel(
    "Human review queue",
    ["max by (kind) (theraflo_ads_review_queue_depth)"],
    12,
    29,
    legend=["{{kind}}"],
)

panel(
    "Rate limited",
    ["sum by (scope) (rate(theraflo_ads_rate_limited_total[1m]))"],
    0,
    37,
    legend=["{{scope}}"],
    unit="reqps",
)
panel(
    "Dead-letter queues",
    ['max by (queue) (rabbitmq_queue_messages_ready{queue=~".*\\\\.dlq"})'],
    12,
    37,
    legend=["{{queue}}"],
)

dashboard = {
    "uid": "theraflo-ads",
    "title": "Theraflo Ads",
    "tags": ["theraflo", "ads"],
    "timezone": "utc",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "10s",
    "time": {"from": "now-15m", "to": "now"},
    "panels": panels,
}
out = Path("monitoring/grafana/dashboards/theraflo-ads.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes((json.dumps(dashboard, indent=2) + "\n").encode("utf-8"))
print(len(panels), "panels")
