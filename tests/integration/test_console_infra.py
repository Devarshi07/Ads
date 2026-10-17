"""The dev console on the infra backend: decisions, the audit log, creatives, test users, and
the ranker lab all read Postgres and Redis."""

from __future__ import annotations

import pytest

from tests.integration.test_infra import ALICE, engine, rank  # noqa: F401 - fixture
from tests.integration.test_review_store import BOOKLY, api  # noqa: F401 - fixture
from theraflo_ads.api.container import DEV_ADMIN_TOKENS

pytestmark = pytest.mark.integration

ADMIN = {"Authorization": f"Bearer {next(iter(DEV_ADMIN_TOKENS))}"}


def test_decisions_are_looked_up_by_request_and_listed_newest_first(api):  # noqa: F811
    client, container = api
    assert container.dev_mode and container.backend == "infra"
    first = rank(client)
    second = rank(client, placement="journal")
    detail = client.get(f"/v1/dev/decisions/{first['request_id']}", headers=ADMIN).json()
    assert detail["decision"]["request_id"] == first["request_id"]
    assert detail["label"]["ad_id"] == first["ads"][0]["ad_id"]
    recent = client.get("/v1/dev/decisions", params={"limit": 2}, headers=ADMIN).json()
    assert [d["request_id"] for d in recent["decisions"]] == [
        second["request_id"],
        first["request_id"],
    ]
    assert recent["decisions"][0]["blocked_reason"] == "placement_not_allowed"


def test_the_audit_log_and_creatives_come_from_postgres(api):  # noqa: F811
    client, _ = api
    body = {"enabled": False, "reason": "integration check"}
    assert client.put("/v1/admin/kill-switch", json=body, headers=ADMIN).status_code == 200
    entries = client.get("/v1/admin/audit", params={"limit": 3}, headers=ADMIN).json()["entries"]
    assert entries[0]["event"] == "kill_switch"
    assert entries[0]["reason"] == "integration check"
    creative = {
        "ad_id": "ad_bookly_console",
        "campaign_id": "cmp_bookly",
        "declared_category": "books",
        "headline": "Cozy mysteries",
        "body": "Cozy mysteries for rainy evenings, delivered.",
        "landing_url": "https://bookly.example/new",
    }
    assert client.post("/v1/advertiser/creatives", json=creative, headers=BOOKLY).status_code == 202
    listed = client.get("/v1/advertiser/creatives", headers=BOOKLY).json()["creatives"]
    assert [c["ad_id"] for c in listed] == ["ad_bookly_console"]
    assert listed[0]["review_status"] == "pending"


def test_test_users_and_the_ranker_lab_on_postgres_and_redis(api):  # noqa: F811
    client, _ = api
    made = client.post("/v1/dev/users", json={"consent": "personalized"}, headers=ADMIN)
    assert made.status_code == 201
    user = made.json()
    headers = {"Authorization": f"Bearer {user['token']}"}
    ranked = client.post(
        "/v1/ads/rank", json={"session_id": "s1", "placement": "home_feed"}, headers=headers
    )
    assert ranked.status_code == 200
    profile = client.get(f"/v1/dev/users/{user['user_id']}", headers=ADMIN).json()
    assert profile["consent"] == "personalized"
    preview = client.post(
        "/v1/dev/rank-preview", json={"user_id": user["user_id"]}, headers=ADMIN
    ).json()
    assert preview["blocked"] is None
    assert {p["name"] for p in preview["policies"] if p["available"]} >= {
        "heuristic",
        "beta_ts",
        "lin_ts",
    }
    bandit = client.get("/v1/dev/bandit", headers=ADMIN).json()
    # Beta counts live in Redis (the features worker writes them); the linear state does not.
    assert bandit["lin_ts"]["learns_online"] is False
    switched = client.put("/v1/dev/serving-policy", json={"policy": "beta_ts"}, headers=ADMIN)
    assert switched.status_code == 200
    assert rank(client)["served_by"] in {"bandit", "no_ad"}
