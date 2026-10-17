"""End-to-end API flow on the dev container with a controllable clock."""

from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from tests.helpers import NOW
from theraflo_ads.api import build_dev_container, create_app
from theraflo_ads.experiments import HoldoutConfig
from theraflo_ads.policy import PopularityPolicy

ALICE = {"Authorization": "Bearer dev-user-alice"}
BOB = {"Authorization": "Bearer dev-user-bob"}
ADMIN = {"Authorization": "Bearer dev-admin"}


class FakeClock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now


def make_env(serving_order=("ad_bookly", "ad_sleepwell")):
    """Dev container with deterministic serving: the first eligible ad in serving_order."""
    clock = FakeClock()
    container = build_dev_container(
        clock=clock,
        user_hash_key=b"u" * 32,
        token_key=b"t" * 32,
        holdouts=HoldoutConfig(uniform_exploration_share=0.0, no_ads_share=0.0),
        chain=lambda catalog: [PopularityPolicy(serving_order)],
    )
    return TestClient(create_app(container)), container, clock


@pytest.fixture
def env():
    return make_env()


def rank(client, headers=ALICE, placement="home_feed"):
    response = client.post(
        "/v1/ads/rank", json={"session_id": "s1", "placement": placement}, headers=headers
    )
    assert response.status_code == 200, response.text
    return response.json()


def viewable(ad, event_id="evt_00000001", at=NOW):
    return {
        "event_id": event_id,
        "event_type": "AD_VIEWABLE",
        "impression_id": ad["impression_id"],
        "ad_id": ad["ad_id"],
        "impression_token": ad["impression_token"],
        "client_timestamp": at.isoformat(),
    }


def test_health_and_readiness(env):
    client, container, _ = env
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
    container.platform.budget.available = False
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["budget_store"] is False


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}])
def test_rank_requires_auth(env, headers):
    client, _, _ = env
    body = {"session_id": "s1", "placement": "home_feed"}
    assert client.post("/v1/ads/rank", json=body, headers=headers).status_code == 401


def fallback(token):
    return {"X-Theraflo-Authorization": token}


def test_the_fallback_auth_header_is_off_by_default(env):
    client, _, _ = env
    path = "/v1/admin/kill-switch"
    assert client.get(path, headers=ADMIN).status_code == 200
    assert client.get(path, headers=fallback("Bearer dev-admin")).status_code == 401


def test_the_fallback_auth_header_behind_a_proxy_that_consumes_authorization(monkeypatch):
    """LocalStack's Container Apps ingress deletes Authorization. Opted in, the same token in
    X-Theraflo-Authorization authenticates instead, with exactly the same checks."""
    monkeypatch.setenv("THERAFLO_ADS_AUTH_FALLBACK_HEADER", "on")
    client, _, _ = make_env()
    path = "/v1/admin/kill-switch"
    assert client.get(path, headers=fallback("Bearer dev-admin")).status_code == 200
    assert client.get(path, headers=fallback("Bearer nope")).status_code == 401
    assert client.get(path, headers=fallback("dev-admin")).status_code == 401  # needs the scheme
    assert client.get(path, headers=fallback("Bearer dev-user-alice")).status_code == 401
    # Authorization, when present, is the only credential read.
    both = {"Authorization": "Bearer nope", **fallback("Bearer dev-admin")}
    assert client.get(path, headers=both).status_code == 401
    # User routes too: the App simulator's calls.
    body = {"session_id": "s1", "placement": "home_feed"}
    ok = client.post("/v1/ads/rank", json=body, headers=fallback("Bearer dev-user-alice"))
    assert ok.status_code == 200


def test_the_fallback_auth_header_is_refused_in_production(monkeypatch):
    _, container, _ = make_env()
    monkeypatch.setenv("THERAFLO_ADS_AUTH_FALLBACK_HEADER", "on")
    monkeypatch.setenv("THERAFLO_ADS_ENV", "production")
    with pytest.raises(RuntimeError, match="emulators"):
        create_app(container)


def test_full_flow_rank_view_click_convert_refund(env):
    client, container, clock = env
    body = rank(client)
    assert body["served_by"] == "popularity"
    (ad,) = body["ads"]
    assert ad["ad_id"] == "ad_bookly"

    assert client.post("/v1/events", json=viewable(ad), headers=ALICE).status_code == 202
    assert client.post("/v1/events", json=viewable(ad), headers=ALICE).status_code == 200

    clock.now += timedelta(seconds=30)
    response = client.get(ad["click_url"], follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["referrer-policy"] == "no-referrer"
    location = urlsplit(response.headers["location"])
    assert location.netloc == "bookly.example"
    query = parse_qs(location.query)
    assert query["ref"] == ["theraflo"]  # existing parameters are preserved
    (click_id,) = query["click_id"]
    assert set(query) == {"ref", "click_id"}  # no user identifiers leave Theraflo

    postback = {
        "click_id": click_id,
        "order_id": "order_1",
        "value": 42.0,
        "occurred_at": (NOW + timedelta(hours=2)).isoformat(),
    }
    bookly = {"X-Advertiser-Key": "dev-adv-bookly"}
    response = client.post("/v1/postbacks/conversion", json=postback, headers=bookly)
    assert (response.status_code, response.json()) == (202, {"status": "attributed"})
    duplicate = client.post("/v1/postbacks/conversion", json=postback, headers=bookly)
    assert duplicate.json() == {"status": "duplicate"}
    label = container.platform.attribution().label(ad["impression_id"])
    assert label.platform_revenue == pytest.approx(12.0)

    refund = client.post("/v1/postbacks/reversal", json={"order_id": "order_1"}, headers=bookly)
    assert refund.json() == {"status": "reversed"}
    assert label.platform_revenue == pytest.approx(0.0)


def test_another_users_token_is_rejected(env):
    client, container, _ = env
    (ad,) = rank(client)["ads"]
    response = client.post("/v1/events", json=viewable(ad), headers=BOB)
    assert (response.status_code, response.json()) == (400, {"status": "rejected"})
    assert container.platform.store.events.rejected[-1].reason == "user_mismatch"


def test_forged_click_is_rejected(env):
    client, _, _ = env
    response = client.get("/v1/click/forged.token", follow_redirects=False)
    assert response.status_code == 400


def test_cross_tenant_postback_looks_like_an_unknown_click(env):
    client, container, _ = env
    (ad,) = rank(client)["ads"]
    location = client.get(ad["click_url"], follow_redirects=False).headers["location"]
    click_id = parse_qs(urlsplit(location).query)["click_id"][0]
    postback = {
        "click_id": click_id,
        "order_id": "o1",
        "value": 1.0,
        "occurred_at": NOW.isoformat(),
    }
    response = client.post(
        "/v1/postbacks/conversion", json=postback, headers={"X-Advertiser-Key": "dev-adv-fitco"}
    )
    assert (response.status_code, response.json()) == (404, {"status": "unknown_click"})
    assert container.audit[-1]["event"] == "cross_tenant_postback"
    missing = client.post("/v1/postbacks/conversion", json=postback, headers={})
    assert missing.status_code == 401


def test_ad_free_placement_returns_no_ads(env):
    client, _, _ = env
    body = rank(client, placement="journal")
    assert body["ads"] == []
    assert body["served_by"] == "no_ad"


def test_teen_never_gets_restricted_categories():
    client, _, _ = make_env(serving_order=("ad_luckyspin", "ad_bookly"))
    body = rank(client, headers={"Authorization": "Bearer dev-user-teen"})
    assert [ad["ad_id"] for ad in body["ads"]] == ["ad_bookly"]


def test_kill_switch_requires_admin_and_stops_serving(env):
    client, container, _ = env
    payload = {"enabled": True, "reason": "incident drill"}
    assert client.put("/v1/admin/kill-switch", json=payload, headers=ALICE).status_code == 401
    response = client.put("/v1/admin/kill-switch", json=payload, headers=ADMIN)
    assert response.json()["kill_switch"] is True
    assert rank(client)["ads"] == []
    assert container.audit[-1]["reason"] == "incident drill"


def test_hidden_ad_is_not_served_again(env):
    client, _, _ = env
    (ad,) = rank(client)["ads"]
    hide = viewable(ad) | {"event_type": "AD_HIDDEN"}
    assert client.post("/v1/events", json=hide, headers=ALICE).status_code == 202
    (next_ad,) = rank(client)["ads"]
    assert next_ad["ad_id"] == "ad_sleepwell"


def test_billing_outage_does_not_fail_the_client():
    client, container, _ = make_env(serving_order=("ad_sleepwell",))  # CPM bills on view
    (ad,) = rank(client)["ads"]
    platform = container.platform
    platform.budget.available = False
    # Accepted and queued: billing runs in the worker, not on the request path.
    assert client.post("/v1/events", json=viewable(ad), headers=ALICE).status_code == 202
    assert platform.bus.counts()["pending"] == 1
    platform.budget.available = True
    platform.bus.retry_pending()
    assert platform.attribution().label(ad["impression_id"]).viewable


def test_storage_outage_returns_retryable_503():
    client, container, _ = make_env(serving_order=("ad_sleepwell",))
    (ad,) = rank(client)["ads"]
    container.platform.store.available = False
    response = client.post("/v1/events", json=viewable(ad), headers=ALICE)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    container.platform.store.available = True
    assert client.post("/v1/events", json=viewable(ad), headers=ALICE).status_code == 202


def test_schema_validation(env):
    client, _, _ = env
    bad = {"session_id": "s 1; drop", "placement": "home_feed"}
    assert client.post("/v1/ads/rank", json=bad, headers=ALICE).status_code == 422
    naive_time = {
        "event_id": "evt_00000009",
        "event_type": "AD_VIEWABLE",
        "impression_id": "imp",
        "ad_id": "ad",
        "impression_token": "x",
        "client_timestamp": "2026-03-02T12:00:00",
    }
    assert client.post("/v1/events", json=naive_time, headers=ALICE).status_code == 422


def test_dev_container_refuses_production(monkeypatch):
    monkeypatch.setenv("THERAFLO_ADS_ENV", "production")
    with pytest.raises(RuntimeError, match="production"):
        build_dev_container()
