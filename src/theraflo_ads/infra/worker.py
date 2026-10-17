"""RabbitMQ consumers (design doc: Message Processing, Backpressure, Bulkheads).

consume → handle → ack. A transient failure is republished to <queue>.retry with
exponential backoff and jitter; after max_attempts, or on a permanent failure,
the message goes to <queue>.dlq with the error attached. The original delivery
is acked only after its retry or dead-letter copy is published.
"""

from __future__ import annotations

import logging
import random
import threading
from collections.abc import Callable, Mapping

import pika
import pika.exceptions

from theraflo_ads.errors import TRANSIENT_ERRORS
from theraflo_ads.events.accounting import AccountingHandler
from theraflo_ads.events.bus import Message, MessageType
from theraflo_ads.infra.amqp import DLX, connect, declare_topology, persistent

log = logging.getLogger(__name__)

Handler = Callable[[Message], None]


def accounting_dispatcher(accounting: AccountingHandler) -> Handler:
    routes: Mapping[MessageType, Handler] = {
        MessageType.AD_VIEWABLE: accounting.on_viewable,
        MessageType.AD_CLICKED: accounting.on_click,
        MessageType.CONVERSION: accounting.on_conversion,
        MessageType.CONVERSION_REVERSED: accounting.on_reversal,
    }

    def handle(message: Message) -> None:
        handler = routes.get(message.type)
        if handler is not None:
            handler(message)

    return handle


class QueueWorker:
    def __init__(
        self,
        amqp_url: str,
        queue: str,
        handler: Handler,
        *,
        max_attempts: int = 6,
        base_delay_ms: int = 500,
        max_delay_ms: int = 60_000,
        prefetch: int = 32,
    ) -> None:
        self._amqp_url = amqp_url
        self.queue = queue
        self._handler = handler
        self._max_attempts = max_attempts
        self._base_delay = base_delay_ms
        self._max_delay = max_delay_ms
        self._prefetch = prefetch
        self.processed = 0
        self.retried = 0
        self.dead_lettered = 0

    def on_message(self, channel, method, properties, body: bytes) -> None:
        headers = dict(properties.headers or {})
        attempts = int(headers.get("x-attempts", 0)) + 1
        try:
            self._handler(Message.from_json(body))
            self.processed += 1
        except TRANSIENT_ERRORS as exc:
            if attempts >= self._max_attempts:
                self._dead_letter(channel, body, properties, headers, attempts, exc)
            else:
                self._retry(channel, body, properties, headers, attempts, exc)
        except Exception as exc:  # permanent: retrying cannot help
            self._dead_letter(channel, body, properties, headers, attempts, exc)
        channel.basic_ack(method.delivery_tag)

    def drain(self, timeout: float = 5.0) -> int:
        """Process whatever is queued, then return (for tests and batch runs)."""
        connection = connect(self._amqp_url)
        handled = 0
        try:
            channel = connection.channel()
            declare_topology(channel)
            for method, properties, body in channel.consume(self.queue, inactivity_timeout=timeout):
                if method is None:
                    break
                self.on_message(channel, method, properties, body)
                handled += 1
            channel.cancel()
        finally:
            connection.close()
        return handled

    def run_forever(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        backoff = 0.5
        while not stop.is_set():
            try:
                connection = connect(self._amqp_url)
                channel = connection.channel()
                declare_topology(channel)
                channel.basic_qos(prefetch_count=self._prefetch)
                for method, properties, body in channel.consume(self.queue, inactivity_timeout=1.0):
                    if stop.is_set():
                        break
                    if method is None:
                        continue
                    self.on_message(channel, method, properties, body)
                channel.cancel()
                connection.close()
                backoff = 0.5
            except (pika.exceptions.AMQPError, OSError) as exc:
                log.warning("worker %s: %s; reconnecting in %.1fs", self.queue, exc, backoff)
                stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)

    def _retry(self, channel, body, properties, headers, attempts, exc) -> None:
        delay = min(self._base_delay * 2 ** (attempts - 1), self._max_delay)
        delay = int(delay * random.uniform(0.8, 1.2))  # jitter avoids synchronized retries
        headers.update({"x-attempts": attempts, "x-last-error": _describe(exc)})
        channel.basic_publish(
            "",
            routing_key=f"{self.queue}.retry",
            body=body,
            properties=persistent(properties.message_id, headers, expiration=str(delay)),
        )
        self.retried += 1

    def _dead_letter(self, channel, body, properties, headers, attempts, exc) -> None:
        headers.update({"x-attempts": attempts, "x-error": _describe(exc)})
        channel.basic_publish(
            DLX,
            routing_key=self.queue,
            body=body,
            properties=persistent(properties.message_id, headers),
        )
        self.dead_lettered += 1
        log.error("worker %s: dead-lettered %s: %s", self.queue, properties.message_id, exc)


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]
