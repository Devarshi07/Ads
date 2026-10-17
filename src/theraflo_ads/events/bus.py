"""Messages and the in-process bus (design doc: Event Infrastructure, Message Processing,
Delivery Semantics).

Production path: unit of work → outbox row (same transaction) → relay →
RabbitMQ → worker. The InlineBus is the in-process stand-in with the same
semantics: messages are published only after the unit of work commits,
delivery is at-least-once, a transient failure leaves the message pending for
retry, and a permanent failure (or too many attempts) dead-letters it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from theraflo_ads.errors import TRANSIENT_ERRORS


class MessageType(StrEnum):
    DECISION = "decision"
    AD_VIEWABLE = "ad_viewable"
    AD_CLICKED = "ad_clicked"
    AD_HIDDEN = "ad_hidden"
    AD_REPORTED = "ad_reported"
    CONVERSION = "conversion"
    CONVERSION_REVERSED = "conversion_reversed"


@dataclass(frozen=True)
class Message:
    """message_id doubles as the idempotency key for consumers."""

    message_id: str
    type: MessageType
    payload: Mapping[str, Any]
    created_at: datetime
    schema_version: str = "message.v1"

    def to_json(self) -> str:
        return json.dumps(
            {
                "message_id": self.message_id,
                "type": self.type.value,
                "payload": dict(self.payload),
                "created_at": self.created_at.isoformat(),
                "schema_version": self.schema_version,
            }
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> Message:
        data = json.loads(text)
        return cls(
            message_id=data["message_id"],
            type=MessageType(data["type"]),
            payload=data["payload"],
            created_at=datetime.fromisoformat(data["created_at"]),
            schema_version=data.get("schema_version", "message.v1"),
        )


Handler = Callable[[Message], None]


class Bus(Protocol):
    def publish(self, message: Message) -> None: ...

    def subscribe(self, message_type: MessageType, handler: Handler) -> None: ...


@dataclass
class Delivery:
    message: Message
    handler: Handler
    attempts: int = 0
    last_error: str | None = None


@dataclass
class InlineBus:
    max_attempts: int = 5
    keep_published: bool = True
    published: list[Message] = field(default_factory=list)
    pending: list[Delivery] = field(default_factory=list)
    dead_letters: list[Delivery] = field(default_factory=list)
    _handlers: dict[MessageType, list[Handler]] = field(default_factory=lambda: defaultdict(list))

    def subscribe(self, message_type: MessageType, handler: Handler) -> None:
        self._handlers[message_type].append(handler)

    def publish(self, message: Message) -> None:
        if self.keep_published:
            self.published.append(message)
        for handler in self._handlers.get(message.type, ()):
            self._deliver(Delivery(message, handler))

    def retry_pending(self) -> int:
        """Redeliver pending messages once. Returns how many succeeded."""
        pending, self.pending = self.pending, []
        return sum(self._deliver(delivery) for delivery in pending)

    def counts(self) -> Mapping[str, int]:
        return {"pending": len(self.pending), "dead_letters": len(self.dead_letters)}

    def _deliver(self, delivery: Delivery) -> bool:
        delivery.attempts += 1
        try:
            delivery.handler(delivery.message)
        except TRANSIENT_ERRORS as exc:
            delivery.last_error = f"{type(exc).__name__}: {exc}"
            if delivery.attempts >= self.max_attempts:
                self.dead_letters.append(delivery)
            else:
                self.pending.append(delivery)
            return False
        except Exception as exc:  # permanent: retrying cannot help
            delivery.last_error = f"{type(exc).__name__}: {exc}"
            self.dead_letters.append(delivery)
            return False
        return True
