"""End-to-end and failure-mode tests on real Postgres, Redis, and RabbitMQ."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import pika.exceptions
import pytest
import redis
from fastapi.testclient import TestClient
from sqlalchemy import select, text, update

from tests.integration.conftest import AMQP_URL, queue_depth
from theraflo_ads.api.app import create_app
from theraflo_ads.api.container import DEMO_PRIORS, build_infra_container, demo_catalog
from theraflo_ads.domain import AgeBand, ConsentScope
from theraflo_ads.errors import StorageUnavailable, TransientError
from theraflo_ads.events.bus import Message, MessageType
from theraflo_ads.infra import schema
from theraflo_ads.infra.amqp import connect
from theraflo_ads.infra.lake import LakeWriter, read_partitions
from theraflo_ads.infra.postgres import PgBillingLedger, PgUnitOfWork, make_engine
from theraflo_ads.infra.relay import OutboxRelay
from theraflo_ads.infra.stores import PgProfileStore, PgSafetyPolicyStore, save_catalog
from theraflo_ads.infra.worker import QueueWorker, accounting_dispatcher
from theraflo_ads.policy.policies import PopularityPolicy
from theraflo_ads.safety.policy import default_safety_policy

pytestmark = pytest.mark.integration

ALICE = {"Authorization": "Bearer dev-user-alice"}


class Clock:
    def __init__(self):
        self.now = datetime.now(UTC).replace(microsecond=0)

    def __call__(self):
        return self.now


@pytest.fixture
def engine(settings):
    engine = make_engine(settings)
    save_catalog(engine, demo_catalog(), DEMO_PRIORS)
    yield engine
    engine.dispose()


@pytest.fixture
def app(settings, engine):
    clock = Clock()
    # Deterministic serving: the CPM ad first (it bills on view), then the rest.
    order = ["ad_sleepwell", "ad_fitco", "ad_trailco", "ad_bookly"]
    container = build_infra_container(
        settings,
        clock=clock,
        chain_factory=lambda catalog, priors, ctx: [PopularityPolicy(order)],
        refresh_catalog=False,
    )
    PgProfileStore(engine, container.hasher).upsert(
        "user-alice", consent=ConsentScope.PERSONALIZED, age_band=AgeBand.ADULT
    )
    return TestClient(create_app(container)), container, clock


def rank(client, placement="home_feed"):
    response = client.post(
        "/v1/ads/rank", json={"session_id": "s1", "placement": placement}, headers=ALICE
    )
    assert response.status_code == 200, response.text
    return response.json()


def viewable(ad, event_id):
    return {
        "event_id": event_id,
        "event_type": "AD_VIEWABLE",
        "impression_id": ad["impression_id"],
        "ad_id": ad["ad_id"],
        "impression_token": ad["impression_token"],
        "client_timestamp": datetime.now(UTC).isoformat(),
    }


def accounting_worker(container, **options):
    return QueueWorker(
        AMQP_URL,
        "accounting",
        accounting_dispatcher(container.platform.accounting),
        base_delay_ms=10,
        **options,
    )


def label(engine, impression_id):
    with engine.connect() as conn:
        return (
            conn.execute(
                select(schema.impression_labels).where(
                    schema.impression_labels.c.impression_id == impression_id
                )
            )
            .mappings()
            .one()
        )


def test_readiness_covers_every_dependency(app):
    client, _, _ = app
    body = client.get("/readyz").json()
    assert body["ready"] is True
    assert set(body["checks"]) >= {"frequency_store", "impression_log", "budget_store", "postgres"}


def test_full_flow_through_outbox_rabbitmq_and_worker(app, engine, settings):
    client, container, clock = app
    ads_seen = []
    for i in range(3):
        body = rank(client)
        assert body["ads"], body
        ad = body["ads"][0]
        ads_seen.append(ad)
        assert (
            client.post("/v1/events", json=viewable(ad, f"evt_{i:08d}"), headers=ALICE).status_code
            == 202
        )
        clock.now += timedelta(minutes=11)
    location = client.get(ads_seen[0]["click_url"], follow_redirects=False).headers["location"]
    click_id = parse_qs(urlsplit(location).query)["click_id"][0]

    # Nothing is billed yet: billing happens in the worker, after the relay publishes.
    assert label(engine, ads_seen[0]["impression_id"])["viewable"] is False
    relay = OutboxRelay(engine, AMQP_URL)
    published = relay.run_once()
    assert published >= 3 + 3 + 1  # decisions, viewables, click
    worker = accounting_worker(container)
    worker.drain(timeout=2)
    for ad in ads_seen:
        assert label(engine, ad["impression_id"])["viewable"] is True
    with engine.connect() as conn:
        billed = conn.execute(select(schema.budget_days.c.billed)).scalars().all()
    assert sum(billed) == pytest.approx(3 * 8.0 / 1000)  # three CPM views at $8 CPM

    lake = LakeWriter(settings.lake_path)
    QueueWorker(AMQP_URL, "analytics", lake).drain(timeout=2)
    decisions = read_partitions(settings.lake_path, MessageType.DECISION.value)
    assert len(decisions) == 3
    assert queue_depth("accounting.dlq") == 0
    assert click_id


def test_redelivery_never_double_bills(app, engine):
    client, container, _ = app
    ad = rank(client)["ads"][0]
    client.post("/v1/events", json=viewable(ad, "evt_dup00001"), headers=ALICE)
    relay = OutboxRelay(engine, AMQP_URL)
    relay.run_once()
    with engine.begin() as conn:  # simulate a relay crash before marking rows published
        conn.execute(update(schema.outbox).values(published_at=None))
    relay.run_once()
    accounting_worker(container).drain(timeout=2)
    with engine.connect() as conn:
        rows = conn.execute(select(schema.billing_ledger)).mappings().all()
    viewable_charges = [r for r in rows if r["idempotency_key"].startswith("view:")]
    assert len(viewable_charges) <= 1  # CPM ads bill once; CPC/CPA ads bill nothing on view


def test_unit_of_work_rolls_back_facts_and_outbox_together(engine):
    message = Message("m1", MessageType.AD_HIDDEN, {"x": 1}, datetime.now(UTC))
    with pytest.raises(RuntimeError), PgUnitOfWork(engine) as uow:
        uow.events.accept("evt_rollback", "AD_HIDDEN", {})
        uow.add_message(message)
        raise RuntimeError("crash mid-request")
    with engine.connect() as conn:
        assert conn.execute(select(schema.events)).first() is None
        assert conn.execute(select(schema.outbox)).first() is None


def test_outbox_survives_a_broker_outage(app, engine):
    client, _, _ = app
    rank(client)
    broken = OutboxRelay(engine, "amqp://theraflo:theraflo@127.0.0.1:1/%2F")
    with pytest.raises((pika.exceptions.AMQPError, OSError)):
        broken.run_once()
    assert broken.pending() == 1  # still in the outbox, nothing lost
    assert OutboxRelay(engine, AMQP_URL).run_once() == 1


def test_redis_flush_rebuilds_caps_from_postgres(app, settings):
    client, container, clock = app
    served = []
    for _ in range(6):
        body = rank(client)
        served += [a["ad_id"] for a in body["ads"]]
        clock.now += timedelta(minutes=11)
    redis.Redis.from_url(settings.redis_url).flushdb()
    snapshot = container.platform.frequency.snapshot(container.hasher.hash("user-alice"), clock.now)
    assert snapshot.user_count == len(served)


def test_redis_outage_fails_closed(app):
    client, container, _ = app
    container.platform.frequency._redis = redis.Redis(port=1, socket_connect_timeout=0.05)
    body = rank(client)
    assert body["ads"] == []
    assert body["served_by"] == "no_ad"


def test_postgres_outage_returns_retryable_503(app):
    client, container, _ = app
    ad = rank(client)["ads"][0]
    real = container.platform.events._uow

    def down():
        raise StorageUnavailable("postgres down")

    container.platform.events._uow = down
    response = client.post("/v1/events", json=viewable(ad, "evt_pgdown01"), headers=ALICE)
    assert response.status_code == 503
    container.platform.events._uow = real
    assert (
        client.post("/v1/events", json=viewable(ad, "evt_pgdown01"), headers=ALICE).status_code
        == 202
    )


def test_unreachable_database_on_the_rank_path_is_a_retryable_503(app):
    """Found by scripts/chaos.py: a dead Postgres connection used to surface as a 500."""
    from sqlalchemy import create_engine

    client, container, _ = app
    container.profiles._engine = create_engine(
        "postgresql+pg8000://theraflo:theraflo@127.0.0.1:1/none", connect_args={"timeout": 1}
    )
    response = client.post(
        "/v1/ads/rank", json={"session_id": "s1", "placement": "home_feed"}, headers=ALICE
    )
    assert response.status_code == 503 and response.headers["Retry-After"]


def test_concurrent_charges_never_exceed_the_limit(engine):
    ledger = PgBillingLedger(engine)
    day = datetime.now(UTC).date()

    def charge(i):
        ledger.bill(idempotency_key=f"k{i}", budget_key="c", day=day, amount=1.0, limit=10.0)

    threads = [threading.Thread(target=charge, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    totals = ledger.totals("c", day)
    assert totals["billed"] == pytest.approx(10.0)
    assert totals["overdelivery"] == pytest.approx(20.0)


def test_concurrent_reservations_respect_the_budget(app):
    _, container, clock = app
    budget = container.platform.budget
    results = []

    def reserve():
        results.append(budget.reserve("tiny", 1.0, 0.1, clock.now))

    threads = [threading.Thread(target=reserve) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r is not None for r in results) == 9  # (1.00 − 2% margin) / 0.10
    assert budget.reserved("tiny", clock.now.date()) == pytest.approx(0.9)


def test_transient_failures_retry_then_dead_letter(settings):
    attempts = []

    def flaky(message):
        attempts.append(message.message_id)
        raise TransientError("budget store down")

    worker = QueueWorker(AMQP_URL, "accounting", flaky, max_attempts=2, base_delay_ms=50)
    connection = connect(AMQP_URL)
    channel = connection.channel()
    channel.basic_publish(
        "theraflo.events",
        routing_key="ad_viewable",
        body=Message("m-flaky", MessageType.AD_VIEWABLE, {}, datetime.now(UTC)).to_json(),
    )
    connection.close()
    worker.drain(timeout=0.5)  # first attempt → retry queue
    time.sleep(0.3)  # backoff TTL elapses, message dead-letters back to the queue
    worker.drain(timeout=1)  # second attempt → DLQ
    assert attempts == ["m-flaky", "m-flaky"]
    assert queue_depth("accounting.dlq") == 1


def test_permanent_failures_go_straight_to_the_dlq(settings):
    worker = QueueWorker(AMQP_URL, "accounting", lambda m: (_ for _ in ()).throw(KeyError("x")))
    connection = connect(AMQP_URL)
    connection.channel().basic_publish(
        "theraflo.events", routing_key="ad_clicked", body=b"not json"
    )
    connection.close()
    worker.drain(timeout=0.5)
    assert worker.dead_lettered == 1
    assert queue_depth("accounting.dlq") == 1


def test_kill_switch_reaches_every_process(engine, app):
    client, _, _ = app
    other_process = PgSafetyPolicyStore(engine, default_safety_policy(), cache_seconds=0)
    response = client.put(
        "/v1/admin/kill-switch",
        json={"enabled": True, "reason": "drill"},
        headers={"Authorization": "Bearer dev-admin"},
    )
    assert response.status_code == 200
    assert other_process.current().kill_switch is True
    with engine.connect() as conn:
        events = conn.execute(text("SELECT event FROM audit_log")).scalars().all()
    assert "kill_switch" in events
