"""Right to deletion with crypto-shredding (design doc: DVC Edge Cases, Security & Threat Model).

Dataset snapshots (the lake, DVC-versioned training data) are immutable, so a
user cannot be deleted from them row by row. Instead, snapshots never contain
the user hash: they contain a pseudonym HMAC(user_key, user_hash), and the
per-user key lives only in the `user_keys` table, outside every snapshot.
Deleting the key makes every snapshot row of that user permanently unlinkable,
in every past and future copy of the data.

The deletion job also erases what the operational stores hold:
- the user key (crypto-shredding of snapshots),
- the ads profile (consent, hidden ads, suppression windows),
- frequency and interaction-feature state,
- the user hash and session ID in decision records, which are kept, anonymized,
  because billing and attribution audits need them.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Protocol

from theraflo_ads.domain import Clock, utc_now
from theraflo_ads.events.bus import Message

SHREDDED = "shredded"


class UserKeyStore(Protocol):
    def key_for(self, user_hash: str) -> bytes:
        """The user's key, created on first use."""
        ...

    def shred(self, user_hash: str) -> bool: ...


class InMemoryUserKeyStore:
    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def key_for(self, user_hash: str) -> bytes:
        with self._lock:
            return self._keys.setdefault(user_hash, secrets.token_bytes(32))

    def shred(self, user_hash: str) -> bool:
        with self._lock:
            return self._keys.pop(user_hash, None) is not None

    def __contains__(self, user_hash: str) -> bool:
        return user_hash in self._keys


class Pseudonymizer:
    """user_hash → a snapshot pseudonym that only the (deletable) user key can produce."""

    def __init__(self, keys: UserKeyStore) -> None:
        self._keys = keys

    def pseudonym(self, user_hash: str) -> str:
        digest = hmac.new(self._keys.key_for(user_hash), user_hash.encode(), hashlib.sha256)
        return "p:" + digest.hexdigest()[:32]

    def message(self, message: Message) -> Message:
        """A copy safe to write to a snapshot: user hash replaced, session ID dropped."""
        payload = dict(message.payload)
        if "user_hash" in payload:
            payload["user_hash"] = self.pseudonym(payload["user_hash"])
        if "session_id" in payload:
            payload["session_id"] = None
        return replace(message, payload=payload)


@dataclass(frozen=True)
class DeletionRequest:
    request_id: str
    user_hash: str
    requested_at: datetime
    completed_at: datetime | None = None
    steps: Mapping[str, Any] = field(default_factory=dict)


class DeletionRequests(Protocol):
    def add(self, request: DeletionRequest) -> None: ...

    def pending(self) -> list[DeletionRequest]: ...

    def complete(self, request_id: str, at: datetime, steps: Mapping[str, Any]) -> None: ...


class InMemoryDeletionRequests:
    def __init__(self) -> None:
        self.requests: dict[str, DeletionRequest] = {}

    def add(self, request: DeletionRequest) -> None:
        self.requests[request.request_id] = request

    def pending(self) -> list[DeletionRequest]:
        return [r for r in self.requests.values() if r.completed_at is None]

    def complete(self, request_id: str, at: datetime, steps: Mapping[str, Any]) -> None:
        self.requests[request_id] = replace(
            self.requests[request_id], completed_at=at, steps=dict(steps)
        )


# Each eraser removes or anonymizes one store's data for a user hash and reports a count.
Eraser = Callable[[str], int]


class DeletionService:
    def __init__(
        self,
        *,
        requests: DeletionRequests,
        keys: UserKeyStore,
        erasers: Mapping[str, Eraser],
        clock: Clock = utc_now,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._requests = requests
        self._keys = keys
        self._erasers = dict(erasers)
        self._clock = clock
        self._on_event = on_event

    def request(self, user_hash: str) -> DeletionRequest:
        request = DeletionRequest(uuid.uuid4().hex, user_hash, self._clock())
        self._requests.add(request)
        return request

    def run_pending(self) -> list[str]:
        """Completes every pending request; a failing step leaves the request pending, so the
        next run retries it (every step is idempotent)."""
        done = []
        for request in self._requests.pending():
            steps: dict[str, Any] = {"user_key_shredded": self._keys.shred(request.user_hash)}
            for name, erase in self._erasers.items():
                steps[name] = erase(request.user_hash)
            self._requests.complete(request.request_id, self._clock(), steps)
            done.append(request.request_id)
            if self._on_event is not None:
                # The audit trail records that a deletion ran, not whose.
                self._on_event(
                    "user_deletion_completed", {"request_id": request.request_id, **steps}
                )
        return done


def memory_erasers(platform, interactions=None) -> dict[str, Eraser]:
    """Erasers for the in-memory platform (dev API, tests)."""

    def decisions(user_hash: str) -> int:
        records = platform.store.decisions.records
        count = 0
        for i, decision in enumerate(records):
            if decision.user_hash == user_hash:
                records[i] = replace(decision, user_hash=SHREDDED, session_id=SHREDDED)
                count += 1
        return count

    def impressions(user_hash: str) -> int:
        count = platform.impression_log.forget(user_hash)
        forget_counters = getattr(platform.frequency, "forget", None)
        if forget_counters is not None:
            forget_counters(user_hash)
        return count

    erasers: dict[str, Eraser] = {
        "decisions_anonymized": decisions,
        "impressions_deleted": impressions,
    }
    if interactions is not None:
        erasers["interaction_features"] = lambda h: interactions.forget(h)
    return erasers
