"""In-memory repositories and unit of work (Postgres stand-ins for tests and the simulator).

Writes apply immediately (there is no rollback); outbox messages are only
published after the unit of work exits cleanly, matching commit semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from types import TracebackType
from typing import Any

from theraflo_ads.attribution.service import (
    OPEN_STATES,
    ConversionRecord,
    ImpressionLabel,
)
from theraflo_ads.errors import StorageUnavailable
from theraflo_ads.events.bus import Bus, Message
from theraflo_ads.policy.decision import Decision
from theraflo_ads.storage.uow import RejectedEvent, UnitOfWorkFactory


class InMemoryAttributionRepository:
    def __init__(self) -> None:
        self._labels: dict[str, ImpressionLabel] = {}
        self._clicks: dict[str, str] = {}
        self._orders: dict[tuple[str, str], ConversionRecord] = {}

    def get_label(self, impression_id: str) -> ImpressionLabel | None:
        return self._labels.get(impression_id)

    def save_label(self, label: ImpressionLabel) -> None:
        self._labels[label.impression_id] = label

    def impression_for_click(self, click_id: str) -> str | None:
        return self._clicks.get(click_id)

    def save_click(self, click_id: str, impression_id: str) -> None:
        self._clicks[click_id] = impression_id

    def get_order(self, advertiser_id: str, order_id: str) -> ConversionRecord | None:
        return self._orders.get((advertiser_id, order_id))

    def insert_order(self, record: ConversionRecord) -> bool:
        key = (record.advertiser_id, record.order_id)
        if key in self._orders:
            return False
        self._orders[key] = record
        return True

    def update_order(self, record: ConversionRecord) -> None:
        self._orders[(record.advertiser_id, record.order_id)] = record

    def labels_served_before(
        self, cutoff: datetime, *, open_only: bool = False
    ) -> list[ImpressionLabel]:
        return [
            label
            for label in self._labels.values()
            if label.served_at <= cutoff and (not open_only or label.state in OPEN_STATES)
        ]

    def orders(self) -> list[ConversionRecord]:
        return list(self._orders.values())

    def labels(self) -> list[ImpressionLabel]:
        return list(self._labels.values())


class InMemoryEventRepository:
    def __init__(self) -> None:
        self.accepted: dict[str, tuple[str, Mapping[str, Any]]] = {}
        self.rejected: list[RejectedEvent] = []

    def accept(self, event_id: str, kind: str, payload: Mapping[str, Any]) -> bool:
        if event_id in self.accepted:
            return False
        self.accepted[event_id] = (kind, dict(payload))
        return True

    def reject(self, rejected: RejectedEvent) -> None:
        self.rejected.append(rejected)


class InMemoryDecisionRepository:
    def __init__(self) -> None:
        self.records: list[Decision] = []
        self._by_impression: dict[str, Decision] = {}
        self._by_request: dict[str, Decision] = {}

    def append(self, decision: Decision) -> None:
        self.records.append(decision)
        self._by_request[decision.request_id] = decision
        if decision.impression_id is not None:
            self._by_impression[decision.impression_id] = decision

    def by_impression(self, impression_id: str) -> Decision | None:
        return self._by_impression.get(impression_id)

    def by_request(self, request_id: str) -> Decision | None:
        return self._by_request.get(request_id)

    def recent(self, limit: int) -> list[Decision]:
        """Newest first."""
        return self.records[-limit:][::-1]


class MemoryStore:
    """The shared in-memory 'database'. Set available=False to simulate an outage."""

    def __init__(self) -> None:
        self.attribution = InMemoryAttributionRepository()
        self.events = InMemoryEventRepository()
        self.decisions = InMemoryDecisionRepository()
        self.available = True

    def healthy(self) -> bool:
        return self.available


class InMemoryUnitOfWork:
    def __init__(self, store: MemoryStore, bus: Bus) -> None:
        self._store = store
        self._bus = bus
        self._messages: list[Message] = []
        self.attribution = store.attribution
        self.events = store.events
        self.decisions = store.decisions

    def add_message(self, message: Message) -> None:
        self._messages.append(message)

    def __enter__(self) -> InMemoryUnitOfWork:
        if not self._store.available:
            raise StorageUnavailable("store unavailable")
        self._messages = []
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        messages, self._messages = self._messages, []
        if exc_type is None:
            for message in messages:
                self._bus.publish(message)


def memory_uow_factory(store: MemoryStore, bus: Bus) -> UnitOfWorkFactory:
    return lambda: InMemoryUnitOfWork(store, bus)
