"""RabbitMQ topology (design doc: Event Infrastructure, Message Processing).

theraflo.events     topic exchange, routing key = message type
<queue>             consumer queue, bound to the event types it needs
<queue>.retry       holds a failed message for its backoff delay (per-message TTL),
                    then dead-letters it to theraflo.redeliver → back to <queue>
<queue>.dlq         permanent failures and exhausted retries (via theraflo.dlx)
"""

from __future__ import annotations

import pika
from pika.adapters.blocking_connection import BlockingChannel

from theraflo_ads.events.bus import MessageType

EXCHANGE = "theraflo.events"
REDELIVER = "theraflo.redeliver"
DLX = "theraflo.dlx"

QUEUES: dict[str, tuple[str, ...]] = {
    "accounting": (
        MessageType.AD_VIEWABLE.value,
        MessageType.AD_CLICKED.value,
        MessageType.CONVERSION.value,
        MessageType.CONVERSION_REVERSED.value,
    ),
    # Online features and bandit counts (the design doc's feature aggregator).
    "features": (MessageType.DECISION.value, MessageType.AD_CLICKED.value),
    # Every message, for the data lake (training data, drift monitoring, analytics).
    "analytics": ("#",),
}


def connect(url: str) -> pika.BlockingConnection:
    params = pika.URLParameters(url)
    params.heartbeat = 30
    params.blocked_connection_timeout = 30
    params.connection_attempts = 2
    params.retry_delay = 1
    return pika.BlockingConnection(params)


def declare_topology(channel: BlockingChannel) -> None:
    channel.exchange_declare(EXCHANGE, exchange_type="topic", durable=True)
    channel.exchange_declare(REDELIVER, exchange_type="direct", durable=True)
    channel.exchange_declare(DLX, exchange_type="direct", durable=True)
    for queue, routing_keys in QUEUES.items():
        channel.queue_declare(queue, durable=True)
        for key in routing_keys:
            channel.queue_bind(queue, EXCHANGE, routing_key=key)
        channel.queue_bind(queue, REDELIVER, routing_key=queue)
        channel.queue_declare(
            f"{queue}.retry",
            durable=True,
            arguments={"x-dead-letter-exchange": REDELIVER, "x-dead-letter-routing-key": queue},
        )
        channel.queue_declare(f"{queue}.dlq", durable=True)
        channel.queue_bind(f"{queue}.dlq", DLX, routing_key=queue)


def persistent(message_id: str | None = None, headers: dict | None = None, **extra):
    return pika.BasicProperties(
        delivery_mode=pika.DeliveryMode.Persistent,
        content_type="application/json",
        message_id=message_id,
        headers=headers or {},
        **extra,
    )
