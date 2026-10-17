"""User controls, internal suppression, and session-token authentication."""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from tests.helpers import NOW
from theraflo_ads.api import build_dev_container, create_app
from theraflo_ads.api.container import hashed_key_authenticator
from theraflo_ads.experiments import HoldoutConfig
from theraflo_ads.policy import PopularityPolicy
from theraflo_ads.security.sessions import SessionTokens

ALICE = {"Authorization": "Bearer dev-user-alice"}
BACKEND = {"Authorization": "Bearer dev-theraflo-backend"}


class FakeClock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now


@pytest.fixture
def env():
    clock = FakeClock()
    container = build_dev_container(
        clock=clock,
        user_hash_key=b"u" * 32,
        token_key=b"t" * 32,
        holdouts=HoldoutConfig(uniform_exploration_share=0.0, no_ads_share=0.0),
        # Deterministic serving: the default chain explores NO_AD now and then.
        chain=lambda catalog: [PopularityPolicy(["ad_sleepwell", "ad_bookly"])],
    )
    return TestClient(create_app(container)), container, clock


def rank(client, headers=ALICE):
    body = {"session_id": "s1", "placement": "home_feed"}
    return client.post("/v1/ads/rank", json=body, headers=headers).json()


def test_consent_toggle(env):
    client, container, _ = env
    response = client.put(
        "/v1/users/me/consent", json={"consent": "contextual_only"}, headers=ALICE
    )
    assert response.json() == {"consent": "contextual_only"}
    assert container.profiles.context("user-alice").consent.value == "contextual_only"
    assert container.audit[-1]["event"] == "consent_changed"
    bad = client.put("/v1/users/me/consent", json={"consent": "everything"}, headers=ALICE)
    assert bad.status_code == 422


def test_deletion_forgets_the_profile(env):
    client, container, _ = env
    container.profiles.hide_ad("user-alice", "ad_bookly")
    assert client.delete("/v1/users/me", headers=ALICE).status_code == 202
    assert container.profiles.context("user-alice").hidden_ads == frozenset()
    events = [e["event"] for e in container.audit]
    assert "user_deletion_requested" in events and "user_deletion_completed" in events
    user_hash = container.hasher.hash("user-alice")
    assert "user-alice" not in str(container.audit) and user_hash not in str(container.audit)


def test_backend_suppression_stops_ads(env):
    client, _, clock = env
    assert rank(client)["ads"]
    payload = {"user_id": "user-alice", "hours": 24}
    assert (
        client.post("/v1/internal/suppressions", json=payload, headers=BACKEND).status_code == 204
    )
    assert rank(client)["ads"] == []
    clock.now += timedelta(hours=25)
    assert rank(client)["ads"]


def test_suppression_requires_a_service_credential(env):
    client, _, _ = env
    payload = {"user_id": "user-alice", "hours": 24}
    assert client.post("/v1/internal/suppressions", json=payload, headers=ALICE).status_code == 401


def test_session_tokens():
    tokens = SessionTokens(b"s" * 32, ttl=timedelta(minutes=30))
    token = tokens.issue("user-42", NOW)
    assert tokens.verify(token, NOW + timedelta(minutes=29)) == "user-42"
    assert tokens.verify(token, NOW + timedelta(minutes=31)) is None
    assert tokens.verify(token[:-2] + "xx", NOW) is None
    assert tokens.verify("garbage", NOW) is None
    assert SessionTokens(b"x" * 32).verify(token, NOW) is None
    with pytest.raises(ValueError):
        tokens.issue("a|b", NOW)


def test_session_token_authenticates_the_api(monkeypatch):
    key = b"k" * 32
    monkeypatch.setenv("THERAFLO_ADS_SESSION_KEY", key.hex())
    clock = FakeClock()
    container = build_dev_container(clock=clock, user_hash_key=b"u" * 32, token_key=b"t" * 32)
    client = TestClient(create_app(container))
    token = SessionTokens(key).issue("user-alice", NOW)
    body = {"session_id": "s1", "placement": "home_feed"}
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post("/v1/ads/rank", json=body, headers=headers).status_code == 200


def test_hashed_api_keys():
    import hashlib

    authenticate = hashed_key_authenticator({hashlib.sha256(b"secret").hexdigest(): "adv_1"})
    assert authenticate("secret") == "adv_1"
    assert authenticate("wrong") is None
    assert authenticate(None) is None
