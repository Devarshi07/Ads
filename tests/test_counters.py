from datetime import timedelta

import pytest

from tests.helpers import NOW
from theraflo_ads.counters import (
    BudgetLedger,
    CounterUnavailable,
    FrequencyCaps,
    ImpressionLog,
    ImpressionRecord,
    InMemoryFrequencyStore,
)


def record(ad_id="ad_1", advertiser="adv_1", minutes_ago=0, user="u1"):
    return ImpressionRecord(user, ad_id, advertiser, NOW - timedelta(minutes=minutes_ago))


class TestFrequencyStore:
    def test_counts_within_window_only(self):
        store = InMemoryFrequencyStore(ImpressionLog(), FrequencyCaps())
        store.record(record(minutes_ago=60 * 25))  # outside the 24h window
        store.record(record(minutes_ago=30))
        store.record(record("ad_2", minutes_ago=20))
        snapshot = store.snapshot("u1", NOW)
        assert snapshot.ad_counts == {"ad_1": 1, "ad_2": 1}
        assert snapshot.advertiser_counts == {"adv_1": 2}
        assert snapshot.user_count == 2

    def test_restart_rebuilds_from_the_durable_log(self):
        log = ImpressionLog()
        store = InMemoryFrequencyStore(log)
        for minutes in (10, 20, 30):
            store.record(record(minutes_ago=minutes))
        store.snapshot("u1", NOW)
        store.flush()  # Redis restart
        assert store.snapshot("u1", NOW).ad_counts["ad_1"] == 3

    def test_missing_counter_is_never_read_as_zero(self):
        log = ImpressionLog()
        store = InMemoryFrequencyStore(log)
        store.record(record())
        store.flush()
        log.available = False
        with pytest.raises(CounterUnavailable):
            store.snapshot("u1", NOW)

    def test_unavailable_store_raises_on_read_and_write(self):
        store = InMemoryFrequencyStore(ImpressionLog())
        store.available = False
        with pytest.raises(CounterUnavailable):
            store.snapshot("u1", NOW)
        with pytest.raises(CounterUnavailable):
            store.record(record())

    def test_write_goes_to_the_log_first(self):
        log = ImpressionLog()
        store = InMemoryFrequencyStore(log)
        log.available = False
        with pytest.raises(CounterUnavailable):
            store.record(record())
        log.available = True
        assert store.snapshot("u1", NOW).user_count == 0


class TestBudgetLedger:
    def test_reserve_and_settle(self):
        ledger = BudgetLedger(safety_margin=0.0)
        rid = ledger.reserve("c", 10.0, 1.0, NOW)
        assert rid is not None
        assert ledger.remaining("c", 10.0, NOW) == pytest.approx(9.0)
        assert ledger.settle(rid, "c", 0.8, 10.0, NOW, idempotency_key="s1") == pytest.approx(0.8)
        assert ledger.remaining("c", 10.0, NOW) == pytest.approx(9.2)

    def test_release_returns_headroom(self):
        ledger = BudgetLedger(safety_margin=0.0)
        rid = ledger.reserve("c", 10.0, 4.0, NOW)
        ledger.release(rid)
        assert ledger.remaining("c", 10.0, NOW) == pytest.approx(10.0)

    def test_reservations_expire(self):
        ledger = BudgetLedger(safety_margin=0.0, reservation_ttl=timedelta(minutes=15))
        ledger.reserve("c", 10.0, 4.0, NOW)
        assert ledger.remaining("c", 10.0, NOW + timedelta(minutes=16)) == pytest.approx(10.0)

    def test_safety_margin_and_headroom(self):
        ledger = BudgetLedger(safety_margin=0.1)
        assert ledger.can_reserve("c", 10.0, 9.0, NOW)
        assert not ledger.can_reserve("c", 10.0, 9.01, NOW)
        assert not ledger.can_reserve("c", 10.0, 0.01, NOW, min_headroom=9.5)

    def test_never_bills_above_budget(self):
        ledger = BudgetLedger(safety_margin=0.0)
        assert ledger.charge("c", 8.0, 10.0, NOW, idempotency_key="c1") == pytest.approx(8.0)
        assert ledger.charge("c", 5.0, 10.0, NOW, idempotency_key="c2") == pytest.approx(2.0)
        assert ledger.billed("c", NOW.date()) == pytest.approx(10.0)
        assert ledger.overdelivery("c", NOW.date()) == pytest.approx(3.0)

    def test_settling_an_expired_reservation_bills_directly(self):
        ledger = BudgetLedger(safety_margin=0.0)
        later = NOW + timedelta(hours=1)
        assert ledger.settle(
            "missing", "c", 1.0, 10.0, later, idempotency_key="s1"
        ) == pytest.approx(1.0)
        assert ledger.billed("c", later.date()) == pytest.approx(1.0)

    def test_full_budget_refuses_reservations(self):
        ledger = BudgetLedger(safety_margin=0.0)
        ledger.charge("c", 10.0, 10.0, NOW, idempotency_key="c1")
        assert ledger.reserve("c", 10.0, 0.01, NOW) is None

    def test_credits_do_not_reopen_pacing(self):
        ledger = BudgetLedger(safety_margin=0.0)
        ledger.charge("c", 10.0, 10.0, NOW, idempotency_key="c1")
        ledger.credit("c", 5.0, NOW, idempotency_key="r1")
        assert ledger.remaining("c", 10.0, NOW) == pytest.approx(0.0)

    def test_unavailable_ledger_raises(self):
        ledger = BudgetLedger()
        ledger.available = False
        with pytest.raises(CounterUnavailable):
            ledger.can_reserve("c", 10.0, 1.0, NOW)
        with pytest.raises(CounterUnavailable):
            ledger.charge("c", 1.0, 10.0, NOW, idempotency_key="c1")

    def test_replayed_charges_bill_once(self):
        ledger = BudgetLedger(safety_margin=0.0)
        rid = ledger.reserve("c", 10.0, 1.0, NOW)
        assert ledger.settle(rid, "c", 1.0, 10.0, NOW, idempotency_key="view:i1") == 1.0
        assert ledger.settle(rid, "c", 1.0, 10.0, NOW, idempotency_key="view:i1") == 1.0
        assert ledger.charge("c", 2.0, 10.0, NOW, idempotency_key="conv:o1") == 2.0
        assert ledger.charge("c", 2.0, 10.0, NOW, idempotency_key="conv:o1") == 2.0
        ledger.credit("c", 2.0, NOW, idempotency_key="rev:o1")
        ledger.credit("c", 2.0, NOW, idempotency_key="rev:o1")
        assert ledger.billed("c", NOW.date()) == pytest.approx(3.0)
        assert ledger.credited("c", NOW.date()) == pytest.approx(2.0)

    def test_invalid_margin(self):
        with pytest.raises(ValueError):
            BudgetLedger(safety_margin=1.0)
