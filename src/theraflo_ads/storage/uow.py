"""Unit of work (design doc: Database Reliability, Event Queue → transactional outbox).

Every write path runs inside one unit of work: repository writes and outbox
messages commit together or not at all. Messages become visible to consumers
only after commit, so a queue outage never loses an accepted event.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Any, Protocol

from theraflo_ads.attribution.service import AttributionRepository
from theraflo_ads.events.bus import Message
from theraflo_ads.policy.decision import Decision


@dataclass(frozen=True)
class RejectedEvent:
    event_id: str | None
    reason: str
    received_at: datetime


class EventRepository(Protocol):
    def accept(self, event_id: str, kind: str, payload: Mapping[str, Any]) -> bool:
        """Record an accepted event. False if event_id was already accepted (idempotency)."""
        ...

    def reject(self, rejected: RejectedEvent) -> None: ...


class DecisionRepository(Protocol):
    def append(self, decision: Decision) -> None: ...


class UnitOfWork(Protocol):
    attribution: AttributionRepository
    events: EventRepository
    decisions: DecisionRepository

    def add_message(self, message: Message) -> None: ...

    def __enter__(self) -> UnitOfWork: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


UnitOfWorkFactory = Callable[[], UnitOfWork]
