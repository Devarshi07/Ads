"""Redis adapters for hot serving state (design doc: Cache Reliability, Frequency Capping →
Failure Semantics, Spend Reservation).

Redis only ever holds derived state:
- frequency counters, rebuilt from the Postgres impressions table;
- open budget reservations (ephemeral by design) and a mirror of billed totals,
  rebuilt from the Postgres billing ledger.
A "ready" marker key says the derived copy is complete. Without it, the state is
rebuilt before use; it is never read as zero. Any Redis error fails closed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import redis

from theraflo_ads.counters.frequency import (
    CounterUnavailable,
    DurableImpressionLog,
    FrequencyCaps,
    FrequencySnapshot,
    ImpressionRecord,
)
from theraflo_ads.infra.postgres import PgBillingLedger

_SEP = "|"


def _ts(when: datetime) -> float:
    return when.timestamp()


class RedisFrequencyStore:
    def __init__(
        self,
        client: redis.Redis,
        log: DurableImpressionLog,
        caps: FrequencyCaps | None = None,
    ) -> None:
        self._redis = client
        self._log = log
        self.caps = caps or FrequencyCaps()
        self._ttl = int((self.caps.window + timedelta(hours=1)).total_seconds())
        # The ready marker is short-lived: if a live write was ever missed (a Redis blip
        # after the Postgres insert), the set is rebuilt from Postgres within minutes.
        self._ready_ttl = 600

    def healthy(self) -> bool:
        try:
            return bool(self._redis.ping())
        except redis.RedisError:
            return False

    def snapshot(self, user_hash: str, now: datetime) -> FrequencySnapshot:
        key, ready = f"freq:{user_hash}", f"freq:ready:{user_hash}"
        since = now - self.caps.window
        try:
            if not self._redis.exists(ready):
                self._rebuild(user_hash, since)
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, "-inf", f"({_ts(since)}")
            pipe.zrangebyscore(key, _ts(since), "+inf", withscores=True)
            _, members = pipe.execute()
        except redis.RedisError as exc:
            raise CounterUnavailable(f"redis: {exc}") from exc

        ad_counts: dict[str, int] = {}
        advertiser_counts: dict[str, int] = {}
        last_served: dict[str, datetime] = {}
        for member, score in members:
            _, ad_id, advertiser_id = member.decode().split(_SEP)
            ad_counts[ad_id] = ad_counts.get(ad_id, 0) + 1
            advertiser_counts[advertiser_id] = advertiser_counts.get(advertiser_id, 0) + 1
            served = datetime.fromtimestamp(score, UTC)
            if ad_id not in last_served or served > last_served[ad_id]:
                last_served[ad_id] = served
        return FrequencySnapshot(ad_counts, advertiser_counts, len(members), last_served)

    def record(self, record: ImpressionRecord) -> None:
        self._log.append(record)  # durable first; raises CounterUnavailable if Postgres is down
        try:
            pipe = self._redis.pipeline()
            pipe.zadd(f"freq:{record.user_hash}", {_member(record): _ts(record.served_at)})
            pipe.expire(f"freq:{record.user_hash}", self._ttl)
            pipe.execute()
        except redis.RedisError as exc:
            # The impression is already recorded durably, so caps stay correct after the
            # next rebuild; this request serves NO_AD.
            raise CounterUnavailable(f"redis: {exc}") from exc

    def _rebuild(self, user_hash: str, since: datetime) -> None:
        records = self._log.for_user(user_hash, since)
        key = f"freq:{user_hash}"
        pipe = self._redis.pipeline()
        # ZADD is idempotent per member, so live writes racing the rebuild are kept.
        if records:
            pipe.zadd(key, {_member(r): _ts(r.served_at) for r in records})
        pipe.expire(key, self._ttl)
        pipe.set(f"freq:ready:{user_hash}", 1, ex=self._ready_ttl)
        pipe.execute()


def _member(record: ImpressionRecord) -> str:
    impression_id = record.impression_id or f"{_ts(record.served_at)}-{record.ad_id}"
    return _SEP.join((impression_id, record.ad_id, record.advertiser_id))


# Atomically expire stale reservations, then reserve if the budget can hold `amount` and
# still absorb a `min_headroom` charge. Returns -1 if the billed mirror is not ready.
_RESERVE = """
local res, exp, billed_key, ready = KEYS[1], KEYS[2], KEYS[3], KEYS[4]
local now, limit, amount = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local min_headroom, margin = tonumber(ARGV[4]), tonumber(ARGV[5])
local rid, expires_at, dry_run, ttl = ARGV[6], ARGV[7], ARGV[8], tonumber(ARGV[9])
if redis.call('EXISTS', ready) == 0 then return -1 end
local stale = redis.call('ZRANGEBYSCORE', exp, '-inf', now)
for _, id in ipairs(stale) do redis.call('HDEL', res, id) end
redis.call('ZREMRANGEBYSCORE', exp, '-inf', now)
local reserved = 0
for _, v in ipairs(redis.call('HVALS', res)) do reserved = reserved + tonumber(v) end
local billed = tonumber(redis.call('GET', billed_key) or '0')
local headroom = limit - billed - reserved - margin * limit
local need = math.max(amount, min_headroom)
if headroom < need then return 0 end
if dry_run == '1' then return 1 end
redis.call('HSET', res, rid, amount)
redis.call('ZADD', exp, expires_at, rid)
redis.call('EXPIRE', res, ttl)
redis.call('EXPIRE', exp, ttl)
return 1
"""


class RedisBudgetStore:
    """BudgetStore over Redis (reservations, billed mirror) and Postgres (billing ledger)."""

    def __init__(
        self,
        client: redis.Redis,
        ledger: PgBillingLedger,
        *,
        safety_margin: float = 0.02,
        reservation_ttl: timedelta = timedelta(minutes=15),
        mirror_ttl: timedelta = timedelta(minutes=5),
    ) -> None:
        self._redis = client
        self._ledger = ledger
        self._margin = safety_margin
        self._reservation_ttl = reservation_ttl
        # A short-lived mirror bounds how stale it can get if a mirror update is ever missed.
        self._mirror_ttl = int(mirror_ttl.total_seconds())
        self._reserve_script = client.register_script(_RESERVE)

    def healthy(self) -> bool:
        try:
            return bool(self._redis.ping()) and self._ledger.healthy()
        except redis.RedisError:
            return False

    def remaining(self, budget_key: str, limit: float, now: datetime) -> float:
        day = now.date()
        billed = self._billed_mirror(budget_key, day)
        return limit - billed - self.reserved(budget_key, day)

    def can_reserve(
        self,
        budget_key: str,
        limit: float,
        amount: float,
        now: datetime,
        *,
        min_headroom: float = 0.0,
    ) -> bool:
        return self._run_reserve(budget_key, limit, amount, now, min_headroom, None) is not None

    def reserve(
        self,
        budget_key: str,
        limit: float,
        amount: float,
        now: datetime,
        *,
        min_headroom: float = 0.0,
    ) -> str | None:
        return self._run_reserve(budget_key, limit, amount, now, min_headroom, uuid4().hex)

    def settle(
        self,
        reservation_id: str | None,
        budget_key: str,
        amount: float,
        limit: float,
        now: datetime,
        *,
        idempotency_key: str,
    ) -> float:
        day = self._reservation_day(reservation_id) or now.date()
        billed, applied = self._ledger.bill(
            idempotency_key=idempotency_key,
            budget_key=budget_key,
            day=day,
            amount=amount,
            limit=limit,
        )
        self._after_billing(budget_key, day, billed if applied else 0.0, reservation_id)
        return billed

    def release(self, reservation_id: str | None) -> None:
        self._after_billing(None, None, 0.0, reservation_id)

    def charge(
        self, budget_key: str, amount: float, limit: float, now: datetime, *, idempotency_key: str
    ) -> float:
        day = now.date()
        billed, applied = self._ledger.bill(
            idempotency_key=idempotency_key,
            budget_key=budget_key,
            day=day,
            amount=amount,
            limit=limit,
        )
        self._after_billing(budget_key, day, billed if applied else 0.0, None)
        return billed

    def credit(
        self, budget_key: str, amount: float, now: datetime, *, idempotency_key: str
    ) -> None:
        self._ledger.credit(
            idempotency_key=idempotency_key, budget_key=budget_key, day=now.date(), amount=amount
        )

    def billed(self, budget_key: str, day: date) -> float:
        return self._ledger.totals(budget_key, day)["billed"]

    def overdelivery(self, budget_key: str, day: date) -> float:
        return self._ledger.totals(budget_key, day)["overdelivery"]

    def credited(self, budget_key: str, day: date) -> float:
        return self._ledger.totals(budget_key, day)["credited"]

    def reserved(self, budget_key: str, day: date) -> float:
        try:
            return sum(float(v) for v in self._redis.hvals(_keys(budget_key, day)[0]))
        except redis.RedisError as exc:
            raise CounterUnavailable(f"redis: {exc}") from exc

    # ----- internals -----

    def _run_reserve(
        self,
        budget_key: str,
        limit: float,
        amount: float,
        now: datetime,
        min_headroom: float,
        reservation_id: str | None,
    ) -> str | None:
        day = now.date()
        keys = _keys(budget_key, day)
        expires_at = now + self._reservation_ttl
        args = [
            _ts(now),
            limit,
            amount,
            min_headroom,
            self._margin,
            reservation_id or "",
            _ts(expires_at),
            "1" if reservation_id is None else "0",
            int(timedelta(days=2).total_seconds()),
        ]
        try:
            result = self._reserve_script(keys=keys, args=args)
            if result == -1:
                self._rebuild_mirror(budget_key, day)
                result = self._reserve_script(keys=keys, args=args)
            if result == 1 and reservation_id is not None:
                self._redis.set(
                    f"bud:rid:{reservation_id}",
                    _SEP.join((budget_key, day.isoformat())),
                    ex=int(self._reservation_ttl.total_seconds()) + 60,
                )
        except redis.RedisError as exc:
            raise CounterUnavailable(f"redis: {exc}") from exc
        if result != 1:
            return None
        return reservation_id or "dry-run"

    def _billed_mirror(self, budget_key: str, day: date) -> float:
        _, _, billed_key, ready = _keys(budget_key, day)
        try:
            if not self._redis.exists(ready):
                self._rebuild_mirror(budget_key, day)
            return float(self._redis.get(billed_key) or 0.0)
        except redis.RedisError as exc:
            raise CounterUnavailable(f"redis: {exc}") from exc

    def _rebuild_mirror(self, budget_key: str, day: date) -> None:
        billed = self._ledger.totals(budget_key, day)["billed"]
        _, _, billed_key, ready = _keys(budget_key, day)
        pipe = self._redis.pipeline()
        pipe.set(billed_key, billed, ex=self._mirror_ttl + 60)
        pipe.set(ready, 1, ex=self._mirror_ttl)
        pipe.execute()

    def _reservation_day(self, reservation_id: str | None) -> date | None:
        if not reservation_id:
            return None
        try:
            value = self._redis.get(f"bud:rid:{reservation_id}")
        except redis.RedisError:
            return None
        if value is None:
            return None
        return date.fromisoformat(value.decode().split(_SEP)[1])

    def _after_billing(
        self, budget_key: str | None, day: date | None, billed: float, reservation_id: str | None
    ) -> None:
        """Mirror a committed charge and drop the settled reservation. Best effort: the ledger
        already committed, and the short mirror TTL forces a rebuild from it anyway."""
        try:
            pipe = self._redis.pipeline()
            if reservation_id:
                value = self._redis.get(f"bud:rid:{reservation_id}")
                if value is not None:
                    key, day_text = value.decode().split(_SEP)
                    res, exp, _, _ = _keys(key, date.fromisoformat(day_text))
                    pipe.hdel(res, reservation_id)
                    pipe.zrem(exp, reservation_id)
                    pipe.delete(f"bud:rid:{reservation_id}")
            if budget_key is not None and day is not None and billed:
                _, _, billed_key, ready = _keys(budget_key, day)
                if self._redis.exists(ready):
                    pipe.incrbyfloat(billed_key, billed)
            pipe.execute()
        except redis.RedisError:
            pass


def _keys(budget_key: str, day: date) -> list[str]:
    suffix = f"{budget_key}:{day.isoformat()}"
    return [f"bud:res:{suffix}", f"bud:exp:{suffix}", f"bud:billed:{suffix}", f"bud:ready:{suffix}"]
