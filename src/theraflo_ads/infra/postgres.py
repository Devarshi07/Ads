"""Postgres adapters: repositories, unit of work with transactional outbox, impression log,
and the billing ledger (design doc: Database Reliability, Event Queue → transactional outbox).

Rows read for update inside a unit of work are locked (SELECT ... FOR UPDATE),
so concurrent workers cannot lose updates to the same label or order.
Connection-level failures surface as StorageUnavailable (transient).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime
from types import TracebackType
from typing import Any

from sqlalchemy import Connection, Engine, create_engine, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

from theraflo_ads.attribution.service import (
    OPEN_STATES,
    ConversionOutcome,
    ConversionRecord,
    ImpressionLabel,
    LabelState,
)
from theraflo_ads.counters.frequency import CounterUnavailable, ImpressionRecord
from theraflo_ads.domain import PricingModel
from theraflo_ads.errors import StorageUnavailable
from theraflo_ads.events.bus import Message
from theraflo_ads.infra import schema
from theraflo_ads.infra.settings import Settings
from theraflo_ads.policy.decision import Decision
from theraflo_ads.storage.uow import RejectedEvent

CONNECTION_ERRORS = (OperationalError, InterfaceError)


def make_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_pool_size,
        pool_pre_ping=True,
        pool_timeout=2,
        connect_args={"timeout": 5},  # pg8000 socket timeout, seconds
    )


def ping(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except DBAPIError:
        return False


@contextmanager
def transaction(
    engine: Engine, error: type[Exception] = StorageUnavailable
) -> Iterator[Connection]:
    """A short transaction whose connection failures become `error` (a transient error)."""
    try:
        with engine.begin() as conn:
            yield conn
    except CONNECTION_ERRORS as exc:
        raise error(str(exc.orig) if hasattr(exc, "orig") else str(exc)) from exc


# ----- repositories (bound to one connection / transaction) -----


class PgAttributionRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def get_label(self, impression_id: str) -> ImpressionLabel | None:
        row = (
            self._conn.execute(
                select(schema.impression_labels)
                .where(schema.impression_labels.c.impression_id == impression_id)
                .with_for_update()
            )
            .mappings()
            .first()
        )
        return _label(row) if row else None

    def save_label(self, label: ImpressionLabel) -> None:
        values = _label_values(label)
        stmt = insert(schema.impression_labels).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["impression_id"],
            set_={k: stmt.excluded[k] for k in values if k != "impression_id"},
        )
        self._conn.execute(stmt)

    def impression_for_click(self, click_id: str) -> str | None:
        return self._conn.execute(
            select(schema.impression_labels.c.impression_id).where(
                schema.impression_labels.c.click_id == click_id
            )
        ).scalar_one_or_none()

    def save_click(self, click_id: str, impression_id: str) -> None:
        # The click_id lives on the label row (unique), written by save_label.
        return None

    def get_order(self, advertiser_id: str, order_id: str) -> ConversionRecord | None:
        row = (
            self._conn.execute(
                select(schema.conversions)
                .where(
                    schema.conversions.c.advertiser_id == advertiser_id,
                    schema.conversions.c.order_id == order_id,
                )
                .with_for_update()
            )
            .mappings()
            .first()
        )
        return _order(row) if row else None

    def insert_order(self, record: ConversionRecord) -> bool:
        result = self._conn.execute(
            insert(schema.conversions)
            .values(**_order_values(record))
            .on_conflict_do_nothing(index_elements=["advertiser_id", "order_id"])
        )
        return result.rowcount == 1

    def update_order(self, record: ConversionRecord) -> None:
        values = _order_values(record)
        self._conn.execute(
            update(schema.conversions)
            .where(
                schema.conversions.c.advertiser_id == record.advertiser_id,
                schema.conversions.c.order_id == record.order_id,
            )
            .values(**values)
        )

    def labels_served_before(
        self, cutoff: datetime, *, open_only: bool = False
    ) -> list[ImpressionLabel]:
        query = select(schema.impression_labels).where(
            schema.impression_labels.c.served_at <= cutoff
        )
        if open_only:
            query = query.where(
                schema.impression_labels.c.state.in_([s.value for s in OPEN_STATES])
            )
        return [_label(row) for row in self._conn.execute(query).mappings()]

    def orders(self) -> list[ConversionRecord]:
        return [_order(row) for row in self._conn.execute(select(schema.conversions)).mappings()]


class PgEventRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def accept(self, event_id: str, kind: str, payload: Mapping[str, Any]) -> bool:
        result = self._conn.execute(
            insert(schema.events)
            .values(event_id=event_id, kind=kind, payload=dict(payload))
            .on_conflict_do_nothing(index_elements=["event_id"])
        )
        return result.rowcount == 1

    def reject(self, rejected: RejectedEvent) -> None:
        self._conn.execute(
            insert(schema.rejected_events).values(
                event_id=rejected.event_id,
                reason=rejected.reason,
                received_at=rejected.received_at,
            )
        )


class PgDecisionRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def append(self, decision: Decision) -> None:
        self._conn.execute(
            insert(schema.decisions)
            .values(
                request_id=decision.request_id,
                user_hash=decision.user_hash,
                placement=decision.placement,
                served_by=decision.served_by.value,
                chosen_action=decision.chosen_action,
                impression_id=decision.impression_id,
                ope_eligible=decision.ope_eligible,
                degraded=decision.degraded,
                decided_at=decision.timestamp,
                record=decision.to_log(),
            )
            .on_conflict_do_nothing(index_elements=["request_id"])
        )


class PgUnitOfWork:
    """One Postgres transaction. Outbox rows are inserted in the same transaction as the
    data they describe, so a message exists if and only if its fact committed."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._messages: list[Message] = []

    def __enter__(self) -> PgUnitOfWork:
        try:
            self._conn = self._engine.connect()
            self._tx = self._conn.begin()
        except CONNECTION_ERRORS as exc:
            raise StorageUnavailable(str(exc)) from exc
        self._messages = []
        self.attribution = PgAttributionRepository(self._conn)
        self.events = PgEventRepository(self._conn)
        self.decisions = PgDecisionRepository(self._conn)
        return self

    def add_message(self, message: Message) -> None:
        self._messages.append(message)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if exc_type is None:
                if self._messages:
                    self._conn.execute(
                        insert(schema.outbox)
                        .values([_outbox_row(m) for m in self._messages])
                        .on_conflict_do_nothing(index_elements=["message_id"])
                    )
                self._tx.commit()
            else:
                self._tx.rollback()
        except CONNECTION_ERRORS as error:
            raise StorageUnavailable(str(error)) from error
        finally:
            self._conn.close()
        if exc_type is not None and issubclass(exc_type, CONNECTION_ERRORS):
            raise StorageUnavailable(str(exc)) from exc


def pg_uow_factory(engine: Engine):
    return lambda: PgUnitOfWork(engine)


# ----- durable impression log -----


class PgImpressionLog:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def healthy(self) -> bool:
        return ping(self._engine)

    def append(self, record: ImpressionRecord) -> None:
        with transaction(self._engine, CounterUnavailable) as conn:
            conn.execute(
                insert(schema.impressions)
                .values(
                    impression_id=record.impression_id,
                    user_hash=record.user_hash,
                    ad_id=record.ad_id,
                    advertiser_id=record.advertiser_id,
                    served_at=record.served_at,
                )
                .on_conflict_do_nothing(index_elements=["impression_id"])
            )

    def for_user(self, user_hash: str, since: datetime) -> list[ImpressionRecord]:
        with transaction(self._engine, CounterUnavailable) as conn:
            rows = conn.execute(
                select(schema.impressions)
                .where(
                    schema.impressions.c.user_hash == user_hash,
                    schema.impressions.c.served_at >= since,
                )
                .order_by(schema.impressions.c.served_at)
            ).mappings()
            return [
                ImpressionRecord(
                    r["user_hash"],
                    r["ad_id"],
                    r["advertiser_id"],
                    r["served_at"],
                    r["impression_id"],
                )
                for r in rows
            ]


# ----- billing ledger (the record of what advertisers are charged) -----


class PgBillingLedger:
    """Serializes billing per (budget_key, day) with a row lock, so concurrent workers can
    never bill past the limit. The idempotency key makes every charge exactly-once."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def healthy(self) -> bool:
        return ping(self._engine)

    def bill(
        self, *, idempotency_key: str, budget_key: str, day: date, amount: float, limit: float
    ) -> tuple[float, bool]:
        """Returns (billed amount, newly_applied)."""
        with transaction(self._engine, CounterUnavailable) as conn:
            existing = self._existing(conn, idempotency_key)
            if existing is not None:
                return existing, False
            row = self._lock_day(conn, budget_key, day)
            billable = max(0.0, min(amount, limit - row["billed"]))
            conn.execute(
                insert(schema.billing_ledger).values(
                    idempotency_key=idempotency_key,
                    budget_key=budget_key,
                    day=day,
                    kind="charge",
                    requested=amount,
                    billed=billable,
                )
            )
            conn.execute(
                update(schema.budget_days)
                .where(
                    schema.budget_days.c.budget_key == budget_key,
                    schema.budget_days.c.day == day,
                )
                .values(
                    billed=schema.budget_days.c.billed + billable,
                    overdelivery=schema.budget_days.c.overdelivery + (amount - billable),
                )
            )
            return billable, True

    def credit(self, *, idempotency_key: str, budget_key: str, day: date, amount: float) -> bool:
        with transaction(self._engine, CounterUnavailable) as conn:
            if self._existing(conn, idempotency_key) is not None:
                return False
            self._lock_day(conn, budget_key, day)
            conn.execute(
                insert(schema.billing_ledger).values(
                    idempotency_key=idempotency_key,
                    budget_key=budget_key,
                    day=day,
                    kind="credit",
                    requested=amount,
                    billed=-amount,
                )
            )
            conn.execute(
                update(schema.budget_days)
                .where(
                    schema.budget_days.c.budget_key == budget_key,
                    schema.budget_days.c.day == day,
                )
                .values(credited=schema.budget_days.c.credited + amount)
            )
            return True

    def totals(self, budget_key: str, day: date) -> Mapping[str, float]:
        with transaction(self._engine, CounterUnavailable) as conn:
            row = (
                conn.execute(
                    select(schema.budget_days).where(
                        schema.budget_days.c.budget_key == budget_key,
                        schema.budget_days.c.day == day,
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return {"billed": 0.0, "overdelivery": 0.0, "credited": 0.0}
        return {k: float(row[k]) for k in ("billed", "overdelivery", "credited")}

    def total_billed(self) -> float:
        with transaction(self._engine, CounterUnavailable) as conn:
            return float(
                conn.execute(
                    select(func.coalesce(func.sum(schema.budget_days.c.billed), 0))
                ).scalar_one()
            )

    @staticmethod
    def _existing(conn: Connection, idempotency_key: str) -> float | None:
        return conn.execute(
            select(schema.billing_ledger.c.billed).where(
                schema.billing_ledger.c.idempotency_key == idempotency_key
            )
        ).scalar_one_or_none()

    @staticmethod
    def _lock_day(conn: Connection, budget_key: str, day: date):
        conn.execute(
            insert(schema.budget_days)
            .values(budget_key=budget_key, day=day)
            .on_conflict_do_nothing(index_elements=["budget_key", "day"])
        )
        return (
            conn.execute(
                select(schema.budget_days)
                .where(
                    schema.budget_days.c.budget_key == budget_key,
                    schema.budget_days.c.day == day,
                )
                .with_for_update()
            )
            .mappings()
            .one()
        )


# ----- row mapping -----


def _label(row: Mapping[str, Any]) -> ImpressionLabel:
    return ImpressionLabel(
        impression_id=row["impression_id"],
        request_id=row["request_id"],
        ad_id=row["ad_id"],
        campaign_id=row["campaign_id"],
        advertiser_id=row["advertiser_id"],
        pricing_model=PricingModel(row["pricing_model"]),
        price=float(row["price"]),
        budget_key=row["budget_key"],
        budget_limit=float(row["budget_limit"]),
        reservation_id=row["reservation_id"],
        served_at=row["served_at"],
        viewable=row["viewable"],
        click_id=row["click_id"],
        clicked_at=row["clicked_at"],
        click_accounted=row["click_accounted"],
        conversions=row["conversions"],
        conversion_value=float(row["conversion_value"]),
        platform_revenue=float(row["platform_revenue"]),
        state=LabelState(row["state"]),
    )


def _label_values(label: ImpressionLabel) -> dict[str, Any]:
    return {
        "impression_id": label.impression_id,
        "request_id": label.request_id,
        "ad_id": label.ad_id,
        "campaign_id": label.campaign_id,
        "advertiser_id": label.advertiser_id,
        "pricing_model": label.pricing_model.value,
        "price": label.price,
        "budget_key": label.budget_key,
        "budget_limit": label.budget_limit,
        "reservation_id": label.reservation_id,
        "served_at": label.served_at,
        "viewable": label.viewable,
        "click_id": label.click_id,
        "clicked_at": label.clicked_at,
        "click_accounted": label.click_accounted,
        "conversions": label.conversions,
        "conversion_value": label.conversion_value,
        "platform_revenue": label.platform_revenue,
        "state": label.state.value,
    }


def _order(row: Mapping[str, Any]) -> ConversionRecord:
    return ConversionRecord(
        advertiser_id=row["advertiser_id"],
        order_id=row["order_id"],
        click_id=row["click_id"],
        impression_id=row["impression_id"],
        value=float(row["value"]),
        payout=float(row["payout"]),
        occurred_at=row["occurred_at"],
        received_at=row["received_at"],
        outcome=ConversionOutcome(row["outcome"]),
        accounted=row["accounted"],
        reversal_requested=row["reversal_requested"],
        reversed=row["reversed"],
    )


def _order_values(record: ConversionRecord) -> dict[str, Any]:
    return {
        "advertiser_id": record.advertiser_id,
        "order_id": record.order_id,
        "click_id": record.click_id,
        "impression_id": record.impression_id,
        "value": record.value,
        "payout": record.payout,
        "occurred_at": record.occurred_at,
        "received_at": record.received_at,
        "outcome": record.outcome.value,
        "accounted": record.accounted,
        "reversal_requested": record.reversal_requested,
        "reversed": record.reversed,
    }


def _outbox_row(message: Message) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "type": message.type.value,
        "body": json.loads(message.to_json()),
        "created_at": message.created_at,
    }
