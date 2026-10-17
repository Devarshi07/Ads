"""Redis adapters against fakeredis (Lua scripts run for real via lupa)."""

from datetime import date, timedelta

import fakeredis
import pytest
import redis

from tests.helpers import NOW
from theraflo_ads.counters.frequency import CounterUnavailable, ImpressionLog, ImpressionRecord
from theraflo_ads.infra.redis_stores import RedisBudgetStore, RedisFrequencyStore


class FakeLedger:
    """In-memory stand-in for PgBillingLedger with the same idempotency and cap semantics."""

    def __init__(self):
        self.days: dict[tuple[str, date], dict[str, float]] = {}
        self.applied: dict[str, float] = {}
        self.available = True

    def healthy(self):
        return self.available

    def _day(self, key, day):
        return self.days.setdefault(
            (key, day), {"billed": 0.0, "overdelivery": 0.0, "credited": 0.0}
        )

    def bill(self, *, idempotency_key, budget_key, day, amount, limit):
        if not self.available:
            raise CounterUnavailable("ledger down")
        if idempotency_key in self.applied:
            return self.applied[idempotency_key], False
        row = self._day(budget_key, day)
        billable = max(0.0, min(amount, limit - row["billed"]))
        row["billed"] += billable
        row["overdelivery"] += amount - billable
        self.applied[idempotency_key] = billable
        return billable, True

    def credit(self, *, idempotency_key, budget_key, day, amount):
        if idempotency_key in self.applied:
            return False
        self.applied[idempotency_key] = -amount
        self._day(budget_key, day)["credited"] += amount
        return True

    def totals(self, budget_key, day):
        return dict(self._day(budget_key, day))


@pytest.fixture
def client():
    return fakeredis.FakeRedis()


def impression(i, ad_id="ad_1", minutes_ago=0, user="u1"):
    return ImpressionRecord(user, ad_id, "adv_1", NOW - timedelta(minutes=minutes_ago), f"imp{i}")


class TestRedisFrequencyStore:
    def test_counts_and_window(self, client):
        store = RedisFrequencyStore(client, ImpressionLog())
        store.record(impression(1, minutes_ago=60 * 25))
        store.record(impression(2, minutes_ago=30))
        store.record(impression(3, "ad_2", minutes_ago=20))
        snapshot = store.snapshot("u1", NOW)
        assert snapshot.ad_counts == {"ad_1": 1, "ad_2": 1}
        assert snapshot.advertiser_counts == {"adv_1": 2}
        assert snapshot.last_served["ad_2"] == NOW - timedelta(minutes=20)

    def test_flushall_rebuilds_from_the_durable_log(self, client):
        log = ImpressionLog()
        store = RedisFrequencyStore(client, log)
        for i in range(3):
            store.record(impression(i, minutes_ago=10 * (i + 1)))
        client.flushall()  # Redis restart without persistence
        assert store.snapshot("u1", NOW).ad_counts == {"ad_1": 3}

    def test_rebuild_and_live_writes_never_double_count(self, client):
        log = ImpressionLog()
        store = RedisFrequencyStore(client, log)
        store.record(impression(1))
        client.delete("freq:ready:u1")  # marker gone, set still present
        assert store.snapshot("u1", NOW).user_count == 1

    def test_missing_counters_with_postgres_down_fail_closed(self, client):
        log = ImpressionLog()
        store = RedisFrequencyStore(client, log)
        log.available = False
        with pytest.raises(CounterUnavailable):
            store.snapshot("u1", NOW)

    def test_redis_errors_fail_closed(self):
        broken = redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.05)
        store = RedisFrequencyStore(broken, ImpressionLog())
        assert not store.healthy()
        with pytest.raises(CounterUnavailable):
            store.snapshot("u1", NOW)


class TestRedisBudgetStore:
    def test_reserve_settle_release(self, client):
        ledger = FakeLedger()
        store = RedisBudgetStore(client, ledger, safety_margin=0.0)
        rid = store.reserve("c", 10.0, 1.0, NOW)
        assert rid is not None
        assert store.reserved("c", NOW.date()) == pytest.approx(1.0)
        assert store.settle(rid, "c", 0.8, 10.0, NOW, idempotency_key="k1") == pytest.approx(0.8)
        assert store.reserved("c", NOW.date()) == 0
        assert store.remaining("c", 10.0, NOW) == pytest.approx(9.2)
        other = store.reserve("c", 10.0, 2.0, NOW)
        store.release(other)
        assert store.reserved("c", NOW.date()) == 0

    def test_reservations_expire(self, client):
        store = RedisBudgetStore(client, FakeLedger(), safety_margin=0.0)
        store.reserve("c", 10.0, 9.0, NOW)
        assert not store.can_reserve("c", 10.0, 2.0, NOW)
        assert store.can_reserve("c", 10.0, 2.0, NOW + timedelta(minutes=16))

    def test_headroom_and_margin(self, client):
        store = RedisBudgetStore(client, FakeLedger(), safety_margin=0.1)
        assert store.can_reserve("c", 10.0, 9.0, NOW)
        assert not store.can_reserve("c", 10.0, 9.01, NOW)
        assert not store.can_reserve("c", 10.0, 0.01, NOW, min_headroom=9.5)

    def test_billing_is_idempotent_and_capped(self, client):
        ledger = FakeLedger()
        store = RedisBudgetStore(client, ledger, safety_margin=0.0)
        assert store.charge("c", 8.0, 10.0, NOW, idempotency_key="a") == 8.0
        assert store.charge("c", 8.0, 10.0, NOW, idempotency_key="a") == 8.0
        assert store.charge("c", 5.0, 10.0, NOW, idempotency_key="b") == 2.0
        assert store.billed("c", NOW.date()) == pytest.approx(10.0)
        assert store.overdelivery("c", NOW.date()) == pytest.approx(3.0)
        assert not store.can_reserve("c", 10.0, 0.01, NOW)

    def test_mirror_rebuilds_from_the_ledger_after_a_flush(self, client):
        ledger = FakeLedger()
        store = RedisBudgetStore(client, ledger, safety_margin=0.0)
        store.charge("c", 9.5, 10.0, NOW, idempotency_key="a")
        client.flushall()
        # Billed spend comes back from the ledger, never as zero.
        assert not store.can_reserve("c", 10.0, 1.0, NOW)

    def test_redis_down_fails_closed(self):
        broken = redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.05)
        store = RedisBudgetStore(broken, FakeLedger())
        with pytest.raises(CounterUnavailable):
            store.can_reserve("c", 10.0, 1.0, NOW)
