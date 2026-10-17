"""Integration fixtures: a throwaway Postgres database, Redis db 15, and purged RabbitMQ
queues. Every test is skipped when `docker compose up -d postgres redis rabbitmq` is not
running."""

from __future__ import annotations

import os
import secrets

import pika
import pika.exceptions
import pytest
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from theraflo_ads.experiments.assignment import Holdout, HoldoutConfig
from theraflo_ads.infra.amqp import QUEUES, connect, declare_topology
from theraflo_ads.infra.settings import Settings
from theraflo_ads.security.identifiers import UserHasher

BASE_DB = os.environ.get(
    "THERAFLO_ADS_TEST_DATABASE_URL",
    "postgresql+pg8000://theraflo:theraflo@127.0.0.1:5432/theraflo_ads",
)
TEST_DB = "theraflo_ads_test"
REDIS_URL = os.environ.get("THERAFLO_ADS_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
AMQP_URL = os.environ.get(
    "THERAFLO_ADS_TEST_AMQP_URL", "amqp://theraflo:theraflo@127.0.0.1:5672/%2F"
)


def _services_up() -> bool:
    try:
        create_engine(BASE_DB, connect_args={"timeout": 2}).connect().close()
        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1).ping()
        connect(AMQP_URL).close()
        return True
    except Exception:
        return False


def _live_consumers() -> dict[str, int]:
    """Consumers on the shared queues before any test starts: the app stack's workers."""
    connection = connect(AMQP_URL)
    try:
        channel = connection.channel()
        counts = {}
        for queue in QUEUES:
            try:
                counts[queue] = channel.queue_declare(queue, passive=True).method.consumer_count
            except pika.exceptions.ChannelClosedByBroker:
                channel = connection.channel()  # queue not declared yet: no consumers
        return {q: n for q, n in counts.items() if n}
    finally:
        connection.close()


@pytest.fixture(scope="session")
def services():
    if not _services_up():
        pytest.skip("integration services are not running (docker compose up -d ...)")
    # The tests purge and consume the RabbitMQ queues; with the app stack's workers attached,
    # they would eat each other's messages. (Postgres and Redis get their own test db.)
    consumers = _live_consumers()
    if consumers:
        pytest.skip(
            f"the app stack's workers are consuming {sorted(consumers)}: stop them first "
            "(docker compose --profile app stop) and keep postgres, redis, rabbitmq running"
        )


@pytest.fixture(scope="session")
def database_url(services):
    admin = create_engine(make_url(BASE_DB).set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)"))
        conn.execute(text(f"CREATE DATABASE {TEST_DB}"))
    url = make_url(BASE_DB).set(database=TEST_DB).render_as_string(hide_password=False)
    previous = os.environ.get("THERAFLO_ADS_DATABASE_URL")
    os.environ["THERAFLO_ADS_DATABASE_URL"] = url
    from theraflo_ads.cli import migrate

    migrate()
    yield url
    if previous is None:
        os.environ.pop("THERAFLO_ADS_DATABASE_URL", None)
    else:
        os.environ["THERAFLO_ADS_DATABASE_URL"] = previous


@pytest.fixture
def settings(database_url, monkeypatch, tmp_path):
    monkeypatch.setenv("THERAFLO_ADS_DATABASE_URL", database_url)
    monkeypatch.setenv("THERAFLO_ADS_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("THERAFLO_ADS_AMQP_URL", AMQP_URL)
    monkeypatch.setenv("THERAFLO_ADS_LAKE_PATH", str(tmp_path / "lake"))
    monkeypatch.setenv("THERAFLO_ADS_USER_HASH_KEY", _hash_key_outside_holdouts("user-alice"))
    monkeypatch.setenv("THERAFLO_ADS_TOKEN_KEY", secrets.token_hex(32))
    monkeypatch.setenv("THERAFLO_ADS_BACKEND", "infra")
    _reset(database_url)
    return Settings()


def _hash_key_outside_holdouts(user_id: str) -> str:
    """A fresh key per test, chosen so the test user is not in a permanent holdout (4% of
    keys would put them in one and the test would see no ads)."""
    while True:
        key = secrets.token_bytes(32)
        user_hash = UserHasher("ads", {"k1": key}, "k1").hash(user_id)
        if HoldoutConfig().assign(user_hash) is Holdout.NONE:
            return key.hex()


def _reset(database_url: str) -> None:
    engine = create_engine(database_url)
    with engine.begin() as conn:
        tables = conn.execute(
            text(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname='public' AND tablename <> 'alembic_version'"
            )
        ).scalars()
        conn.execute(text(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE"))
    engine.dispose()
    redis.Redis.from_url(REDIS_URL).flushdb()
    connection = connect(AMQP_URL)
    try:
        channel = connection.channel()
        declare_topology(channel)
        for queue in QUEUES:
            for name in (queue, f"{queue}.retry", f"{queue}.dlq"):
                channel.queue_purge(name)
    except pika.exceptions.AMQPError:
        pass
    finally:
        connection.close()


def queue_depth(name: str) -> int:
    connection = connect(AMQP_URL)
    try:
        return connection.channel().queue_declare(name, passive=True).method.message_count
    finally:
        connection.close()
