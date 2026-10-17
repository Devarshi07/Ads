"""Frequency caps (design doc: Frequency Capping → Failure Semantics).

Counters are incremented at serve time, the durable impressions log is the
source of truth, and a missing counter is never read as zero.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from theraflo_ads.errors import TransientError


class CounterUnavailable(TransientError):
    """Cap or budget state cannot be read or written. Callers must fail closed."""


@dataclass(frozen=True)
class FrequencyCaps:
    per_ad_24h: int = 3
    per_advertiser_24h: int = 8
    per_user_24h: int = 12
    min_gap_same_ad: timedelta = timedelta(minutes=10)
    window: timedelta = timedelta(hours=24)


@dataclass(frozen=True)
class ImpressionRecord:
    user_hash: str
    ad_id: str
    advertiser_id: str
    served_at: datetime
    # Unique per impression, so a rebuilt counter set and live writes never double count.
    impression_id: str = ""


@dataclass(frozen=True)
class FrequencySnapshot:
    ad_counts: Mapping[str, int]
    advertiser_counts: Mapping[str, int]
    user_count: int
    last_served: Mapping[str, datetime]

    def allows(self, ad_id: str, advertiser_id: str, caps: FrequencyCaps, now: datetime) -> bool:
        if self.ad_counts.get(ad_id, 0) >= caps.per_ad_24h:
            return False
        if self.advertiser_counts.get(advertiser_id, 0) >= caps.per_advertiser_24h:
            return False
        last = self.last_served.get(ad_id)
        return last is None or now - last >= caps.min_gap_same_ad

    def user_at_cap(self, caps: FrequencyCaps) -> bool:
        return self.user_count >= caps.per_user_24h


class FrequencyStore(Protocol):
    caps: FrequencyCaps

    def healthy(self) -> bool: ...

    def snapshot(self, user_hash: str, now: datetime) -> FrequencySnapshot: ...

    def record(self, record: ImpressionRecord) -> None: ...


class DurableImpressionLog(Protocol):
    """The impressions table of record. Frequency counters are rebuilt from it."""

    def healthy(self) -> bool: ...

    def append(self, record: ImpressionRecord) -> None: ...

    def for_user(self, user_hash: str, since: datetime) -> list[ImpressionRecord]: ...


class ImpressionLog:
    """In-memory DurableImpressionLog (a Postgres table in production)."""

    def __init__(self) -> None:
        self._by_user: dict[str, list[ImpressionRecord]] = defaultdict(list)
        self.available = True

    def healthy(self) -> bool:
        return self.available

    def append(self, record: ImpressionRecord) -> None:
        if not self.available:
            raise CounterUnavailable("impression log unavailable")
        self._by_user[record.user_hash].append(record)

    def for_user(self, user_hash: str, since: datetime) -> list[ImpressionRecord]:
        if not self.available:
            raise CounterUnavailable("impression log unavailable")
        return [r for r in self._by_user.get(user_hash, ()) if r.served_at >= since]

    def forget(self, user_hash: str) -> int:
        """Right to deletion: drop the user's impressions."""
        return len(self._by_user.pop(user_hash, ()))


class InMemoryFrequencyStore:
    """Stand-in for Redis: a derived, rebuildable copy of the impressions log.

    A user whose counters are missing (cold cache, restart) is rebuilt from the
    durable log. If the log is unreachable too, CounterUnavailable propagates
    and the caller serves NO_AD.
    """

    def __init__(self, log: DurableImpressionLog, caps: FrequencyCaps | None = None) -> None:
        self._log = log
        self.caps = caps or FrequencyCaps()
        self._cache: dict[str, list[ImpressionRecord]] = {}
        self.available = True

    def healthy(self) -> bool:
        return self.available

    def snapshot(self, user_hash: str, now: datetime) -> FrequencySnapshot:
        records = self._records(user_hash, now)
        last_served: dict[str, datetime] = {}
        for r in records:
            if r.ad_id not in last_served or r.served_at > last_served[r.ad_id]:
                last_served[r.ad_id] = r.served_at
        return FrequencySnapshot(
            ad_counts=Counter(r.ad_id for r in records),
            advertiser_counts=Counter(r.advertiser_id for r in records),
            user_count=len(records),
            last_served=last_served,
        )

    def record(self, record: ImpressionRecord) -> None:
        if not self.available:
            raise CounterUnavailable("frequency store unavailable")
        self._log.append(record)  # source of truth first
        cached = self._cache.get(record.user_hash)
        if cached is not None:
            cached.append(record)

    def flush(self) -> None:
        """Simulate a Redis restart: every cached counter is lost."""
        self._cache.clear()

    def forget(self, user_hash: str) -> int:
        return 1 if self._cache.pop(user_hash, None) is not None else 0

    def _records(self, user_hash: str, now: datetime) -> list[ImpressionRecord]:
        if not self.available:
            raise CounterUnavailable("frequency store unavailable")
        since = now - self.caps.window
        cached = self._cache.get(user_hash)
        if cached is None:
            cached = self._log.for_user(user_hash, since)
            self._cache[user_hash] = cached
        cached[:] = [r for r in cached if r.served_at >= since]
        return cached
