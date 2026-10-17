"""Outbox relay (design doc: Failure Hierarchy → "Event queue unavailable").

Moves committed outbox rows to RabbitMQ. Rows are claimed with
FOR UPDATE SKIP LOCKED, so several relays can run side by side; a row is
marked published only after the broker confirms it. A crash between the two
republishes the row: delivery is at-least-once and consumers are idempotent.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from datetime import UTC, datetime

import pika
import pika.exceptions
from sqlalchemy import Engine, select, update
from sqlalchemy.exc import DBAPIError

from theraflo_ads.infra import schema
from theraflo_ads.infra.amqp import EXCHANGE, connect, declare_topology, persistent

log = logging.getLogger(__name__)


class OutboxRelay:
    def __init__(self, engine: Engine, amqp_url: str, *, batch_size: int = 200) -> None:
        self._engine = engine
        self._amqp_url = amqp_url
        self._batch = batch_size
        self._connection: pika.BlockingConnection | None = None
        self._channel = None
        self.published = 0

    def run_once(self) -> int:
        """Publish one batch. Returns how many rows were published."""
        with self._engine.begin() as conn:
            rows = conn.execute(
                select(
                    schema.outbox.c.id,
                    schema.outbox.c.type,
                    schema.outbox.c.body,
                    schema.outbox.c.message_id,
                )
                .where(schema.outbox.c.published_at.is_(None))
                .order_by(schema.outbox.c.id)
                .limit(self._batch)
                .with_for_update(skip_locked=True)
            ).all()
            if not rows:
                return 0
            channel = self._ensure_channel()
            for row in rows:
                # With confirms on, basic_publish blocks until the broker acks and raises
                # on a nack or an unroutable message, so a row is never marked lost.
                channel.basic_publish(
                    EXCHANGE,
                    routing_key=row.type,
                    body=json.dumps(row.body).encode(),
                    properties=persistent(row.message_id),
                    mandatory=True,
                )
            conn.execute(
                update(schema.outbox)
                .where(schema.outbox.c.id.in_([row.id for row in rows]))
                .values(published_at=datetime.now(UTC))
            )
        self.published += len(rows)
        return len(rows)

    def run_forever(self, stop: threading.Event | None = None, idle_sleep: float = 0.2) -> None:
        stop = stop or threading.Event()
        backoff = 0.5
        while not stop.is_set():
            try:
                if self.run_once() == 0:
                    stop.wait(idle_sleep)
                backoff = 0.5
            except (pika.exceptions.AMQPError, DBAPIError, OSError) as exc:
                log.warning("outbox relay: %s; retrying in %.1fs", exc, backoff)
                self.close()
                stop.wait(backoff + random.uniform(0, backoff / 2))
                backoff = min(backoff * 2, 30.0)

    def pending(self) -> int:
        with self._engine.connect() as conn:
            return len(
                conn.execute(
                    select(schema.outbox.c.id).where(schema.outbox.c.published_at.is_(None))
                ).all()
            )

    def close(self) -> None:
        try:
            if self._connection is not None and self._connection.is_open:
                self._connection.close()
        except pika.exceptions.AMQPError:
            pass
        self._connection = None
        self._channel = None

    def _ensure_channel(self):
        if self._channel is None or not self._channel.is_open:
            self._connection = connect(self._amqp_url)
            self._channel = self._connection.channel()
            declare_topology(self._channel)
            self._channel.confirm_delivery()
        return self._channel


def wait_for_drain(relay: OutboxRelay, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        relay.run_once()
        if relay.pending() == 0:
            return True
    return False
