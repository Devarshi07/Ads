"""Budget reservations (design doc: Advertiser Budget and Pacing → Spend Reservation).

Spend computed only from asynchronous events lags reality, so expected cost is
reserved at serve time and settled or released later. Advertisers are never
billed above budget; anything beyond it is recorded as overdelivery.

A budget key is a campaign ID, or "campaign@slice" when an experiment splits
the budget between arms (design doc: Marketplace Interference). Every charge
carries an idempotency key, so a retried event never bills twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Protocol
from uuid import uuid4

from theraflo_ads.counters.frequency import CounterUnavailable


class BudgetStore(Protocol):
    def healthy(self) -> bool: ...

    def remaining(self, budget_key: str, limit: float, now: datetime) -> float: ...

    def can_reserve(
        self,
        budget_key: str,
        limit: float,
        amount: float,
        now: datetime,
        *,
        min_headroom: float = 0.0,
    ) -> bool: ...

    def reserve(
        self,
        budget_key: str,
        limit: float,
        amount: float,
        now: datetime,
        *,
        min_headroom: float = 0.0,
    ) -> str | None: ...

    def settle(
        self,
        reservation_id: str | None,
        budget_key: str,
        amount: float,
        limit: float,
        now: datetime,
        *,
        idempotency_key: str,
    ) -> float: ...

    def release(self, reservation_id: str | None) -> None: ...

    def charge(
        self, budget_key: str, amount: float, limit: float, now: datetime, *, idempotency_key: str
    ) -> float: ...

    def credit(
        self, budget_key: str, amount: float, now: datetime, *, idempotency_key: str
    ) -> None: ...

    def billed(self, budget_key: str, day: date) -> float: ...

    def overdelivery(self, budget_key: str, day: date) -> float: ...

    def reserved(self, budget_key: str, day: date) -> float: ...


@dataclass
class _Reservation:
    budget_key: str
    day: date
    amount: float
    expires_at: datetime


@dataclass
class _BudgetDay:
    billed: float = 0.0
    overdelivery: float = 0.0
    credited: float = 0.0
    open: dict[str, _Reservation] = field(default_factory=dict)

    @property
    def reserved(self) -> float:
        return sum(r.amount for r in self.open.values())


class BudgetLedger:
    """In-memory BudgetStore. Production splits it into Redis (reservations, a mirror of
    billed totals) and Postgres (the billing ledger of record)."""

    def __init__(
        self,
        *,
        safety_margin: float = 0.02,
        reservation_ttl: timedelta = timedelta(minutes=15),
    ) -> None:
        if not 0.0 <= safety_margin < 1.0:
            raise ValueError("safety_margin must be in [0, 1)")
        self._margin = safety_margin
        self._ttl = reservation_ttl
        self._days: dict[tuple[str, date], _BudgetDay] = {}
        self._reservations: dict[str, _Reservation] = {}
        self._applied: dict[str, float] = {}
        self.available = True

    def healthy(self) -> bool:
        return self.available

    def remaining(self, budget_key: str, limit: float, now: datetime) -> float:
        self._check()
        self._expire(budget_key, now)
        day = self._day(budget_key, now.date())
        return limit - day.billed - day.reserved

    def can_reserve(
        self,
        budget_key: str,
        limit: float,
        amount: float,
        now: datetime,
        *,
        min_headroom: float = 0.0,
    ) -> bool:
        """True if the budget can hold `amount` and still absorb a `min_headroom` charge."""
        headroom = self.remaining(budget_key, limit, now) - self._margin * limit
        return headroom >= max(amount, min_headroom)

    def reserve(
        self,
        budget_key: str,
        limit: float,
        amount: float,
        now: datetime,
        *,
        min_headroom: float = 0.0,
    ) -> str | None:
        """Reserve expected cost. Returns None when the budget cannot cover it."""
        if not self.can_reserve(budget_key, limit, amount, now, min_headroom=min_headroom):
            return None
        reservation_id = uuid4().hex
        reservation = _Reservation(budget_key, now.date(), amount, now + self._ttl)
        self._day(budget_key, reservation.day).open[reservation_id] = reservation
        self._reservations[reservation_id] = reservation
        return reservation_id

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
        """Turn a reservation into billed spend. Returns the billable amount.

        A reservation that already expired is billed as a direct charge.
        """
        self._check()
        reservation = self._pop(reservation_id)
        if idempotency_key in self._applied:
            return self._applied[idempotency_key]
        day = reservation.day if reservation else now.date()
        return self._bill(idempotency_key, self._day(budget_key, day), amount, limit)

    def release(self, reservation_id: str | None) -> None:
        self._check()
        self._pop(reservation_id)

    def charge(
        self, budget_key: str, amount: float, limit: float, now: datetime, *, idempotency_key: str
    ) -> float:
        """Bill without a reservation (e.g., CPA conversions that arrive days later)."""
        self._check()
        if idempotency_key in self._applied:
            return self._applied[idempotency_key]
        return self._bill(idempotency_key, self._day(budget_key, now.date()), amount, limit)

    def credit(
        self, budget_key: str, amount: float, now: datetime, *, idempotency_key: str
    ) -> None:
        """Record a refund / chargeback credit. Does not reopen pacing headroom."""
        self._check()
        if idempotency_key in self._applied:
            return
        self._applied[idempotency_key] = -amount
        self._day(budget_key, now.date()).credited += amount

    def billed(self, budget_key: str, day: date) -> float:
        return self._day(budget_key, day).billed

    def overdelivery(self, budget_key: str, day: date) -> float:
        return self._day(budget_key, day).overdelivery

    def reserved(self, budget_key: str, day: date) -> float:
        return self._day(budget_key, day).reserved

    def credited(self, budget_key: str, day: date) -> float:
        return self._day(budget_key, day).credited

    def _bill(self, idempotency_key: str, day: _BudgetDay, amount: float, limit: float) -> float:
        billable = max(0.0, min(amount, limit - day.billed))
        day.billed += billable
        day.overdelivery += amount - billable
        self._applied[idempotency_key] = billable
        return billable

    def _pop(self, reservation_id: str | None) -> _Reservation | None:
        if reservation_id is None:
            return None
        reservation = self._reservations.pop(reservation_id, None)
        if reservation is not None:
            self._day(reservation.budget_key, reservation.day).open.pop(reservation_id, None)
        return reservation

    def _expire(self, budget_key: str, now: datetime) -> None:
        day = self._day(budget_key, now.date())
        for reservation_id, reservation in list(day.open.items()):
            if reservation.expires_at <= now:
                self._pop(reservation_id)

    def _day(self, budget_key: str, day: date) -> _BudgetDay:
        return self._days.setdefault((budget_key, day), _BudgetDay())

    def _check(self) -> None:
        if not self.available:
            raise CounterUnavailable("budget ledger unavailable")
