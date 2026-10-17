"""Postgres and Redis adapters for the deletion job (crypto-shredding)."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import redis
from sqlalchemy import Engine, delete, select, text, update
from sqlalchemy.dialects.postgresql import insert

from theraflo_ads.infra import schema
from theraflo_ads.infra.postgres import transaction
from theraflo_ads.privacy.shredding import SHREDDED, DeletionRequest, Eraser


class PgUserKeyStore:
    """Per-user keys in `user_keys`, the one table no dataset snapshot ever includes.

    Keys are cached briefly per process; a shredded key can therefore keep being used
    for at most `cache_seconds` by a process that already held it, never longer.
    """

    def __init__(self, engine: Engine, *, cache_seconds: float = 60.0) -> None:
        self._engine = engine
        self._cache_seconds = cache_seconds
        self._cache: dict[str, tuple[float, bytes]] = {}
        self._lock = threading.Lock()

    def key_for(self, user_hash: str) -> bytes:
        with self._lock:
            hit = self._cache.get(user_hash)
            if hit and time.monotonic() - hit[0] < self._cache_seconds:
                return hit[1]
        t = schema.user_keys
        with transaction(self._engine) as conn:
            conn.execute(
                insert(t)
                .values(user_hash=user_hash, key=secrets.token_hex(32))
                .on_conflict_do_nothing(index_elements=["user_hash"])
            )
            key = bytes.fromhex(
                conn.execute(select(t.c.key).where(t.c.user_hash == user_hash)).scalar_one()
            )
        with self._lock:
            self._cache[user_hash] = (time.monotonic(), key)
        return key

    def shred(self, user_hash: str) -> bool:
        with self._lock:
            self._cache.pop(user_hash, None)
        t = schema.user_keys
        with transaction(self._engine) as conn:
            return conn.execute(delete(t).where(t.c.user_hash == user_hash)).rowcount > 0


class PgDeletionRequests:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def add(self, request: DeletionRequest) -> None:
        with transaction(self._engine) as conn:
            conn.execute(
                insert(schema.deletion_requests).values(
                    request_id=request.request_id,
                    user_hash=request.user_hash,
                    requested_at=request.requested_at,
                )
            )

    def pending(self) -> list[DeletionRequest]:
        t = schema.deletion_requests
        query = select(t).where(t.c.completed_at.is_(None)).order_by(t.c.requested_at)
        with transaction(self._engine) as conn:
            return [
                DeletionRequest(r["request_id"], r["user_hash"], r["requested_at"])
                for r in conn.execute(query).mappings()
            ]

    def complete(self, request_id: str, at: datetime, steps: Mapping[str, Any]) -> None:
        t = schema.deletion_requests
        with transaction(self._engine) as conn:
            # The user hash itself is erased from the request once the work is done.
            conn.execute(
                update(t)
                .where(t.c.request_id == request_id)
                .values(completed_at=at, steps=dict(steps), user_hash=SHREDDED)
            )


_ANONYMIZE_DECISIONS = text(
    """
    UPDATE decisions
       SET user_hash = :shredded,
           record = jsonb_set(jsonb_set(record, '{user_hash}', to_jsonb(CAST(:shredded AS text))),
                              '{session_id}', to_jsonb(CAST(:shredded AS text)))
     WHERE user_hash = :user_hash
    """
)
_ANONYMIZE_OUTBOX = text(
    """
    UPDATE outbox
       SET body = jsonb_set(jsonb_set(body, '{payload,user_hash}',
                                      to_jsonb(CAST(:shredded AS text))),
                            '{payload,session_id}', to_jsonb(CAST(:shredded AS text)))
     WHERE body #>> '{payload,user_hash}' = :user_hash
    """
)


def pg_erasers(engine: Engine, redis_client: redis.Redis | None) -> dict[str, Eraser]:
    def decisions(user_hash: str) -> int:
        params = {"user_hash": user_hash, "shredded": SHREDDED}
        with transaction(engine) as conn:
            count = conn.execute(_ANONYMIZE_DECISIONS, params).rowcount
            conn.execute(_ANONYMIZE_OUTBOX, params)
        return count

    def impressions(user_hash: str) -> int:
        t = schema.impressions
        with transaction(engine) as conn:
            return conn.execute(delete(t).where(t.c.user_hash == user_hash)).rowcount

    def profile(user_hash: str) -> int:
        t = schema.user_profiles
        with transaction(engine) as conn:
            return conn.execute(delete(t).where(t.c.user_hash == user_hash)).rowcount

    erasers: dict[str, Eraser] = {
        "decisions_anonymized": decisions,
        "impressions_deleted": impressions,
        "profile_deleted": profile,
    }
    if redis_client is not None:
        erasers["redis_keys_deleted"] = lambda h: int(
            redis_client.delete(f"freq:{h}", f"freq:ready:{h}", f"feat:{h}")
        )
    return erasers
