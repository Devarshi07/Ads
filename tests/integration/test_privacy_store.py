"""Crypto-shredding on Postgres and Redis: the deletion job end to end."""

from __future__ import annotations

import pytest
import redis
from sqlalchemy import select

from tests.integration.conftest import REDIS_URL
from tests.integration.test_infra import ALICE, app, engine, rank  # noqa: F401 - fixtures
from theraflo_ads.cli import run_deletions
from theraflo_ads.infra import schema
from theraflo_ads.infra.privacy_store import PgUserKeyStore
from theraflo_ads.privacy.shredding import SHREDDED, Pseudonymizer

pytestmark = pytest.mark.integration


def test_user_keys_are_stable_and_shreddable(engine):  # noqa: F811
    keys = PgUserKeyStore(engine, cache_seconds=0)
    first = Pseudonymizer(keys).pseudonym("k1:alice")
    assert Pseudonymizer(PgUserKeyStore(engine)).pseudonym("k1:alice") == first
    assert keys.shred("k1:alice") and not keys.shred("k1:alice")
    assert Pseudonymizer(keys).pseudonym("k1:alice") != first


def test_deletion_job_erases_postgres_and_redis_state(app, engine):  # noqa: F811
    client, container, _ = app
    for _ in range(2):
        rank(client)
    user_hash = container.hasher.hash("user-alice")
    PgUserKeyStore(engine).key_for(user_hash)
    r = redis.Redis.from_url(REDIS_URL)
    assert r.exists(f"freq:{user_hash}")

    assert client.delete("/v1/users/me", headers=ALICE).status_code == 202
    with engine.connect() as conn:
        pending = conn.execute(select(schema.deletion_requests)).mappings().one()
    assert pending["completed_at"] is None and pending["user_hash"] == user_hash

    run_deletions()

    with engine.connect() as conn:
        done = conn.execute(select(schema.deletion_requests)).mappings().one()
        decisions = conn.execute(select(schema.decisions)).mappings().all()
        impressions = conn.execute(
            select(schema.impressions).where(schema.impressions.c.user_hash == user_hash)
        ).all()
        keys = conn.execute(select(schema.user_keys)).all()
        profiles = conn.execute(select(schema.user_profiles)).all()
        outbox = conn.execute(select(schema.outbox.c.body)).scalars().all()
        audit = conn.execute(select(schema.audit_log.c.details)).scalars().all()
    assert done["completed_at"] is not None and done["user_hash"] == SHREDDED
    assert done["steps"]["user_key_shredded"] is True
    assert decisions and all(d["user_hash"] == SHREDDED for d in decisions)
    assert all(d["record"]["user_hash"] == SHREDDED for d in decisions)
    assert all(d["record"]["session_id"] == SHREDDED for d in decisions)
    assert impressions == [] and keys == [] and profiles == []
    assert not any(user_hash in str(body) for body in outbox)
    assert not any(user_hash in str(details) for details in audit)
    assert not r.exists(f"freq:{user_hash}") and not r.exists(f"freq:ready:{user_hash}")
