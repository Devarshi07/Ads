"""Creative review on Postgres: the repository, the worker pass, catalog refresh across
processes, decision lookup for "why this ad", and advertiser reports."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from tests.integration.test_infra import ALICE, Clock, engine, rank  # noqa: F401 - fixture
from theraflo_ads.api.app import create_app
from theraflo_ads.api.container import build_infra_container
from theraflo_ads.domain import (
    Ad,
    AdCatalog,
    AgeBand,
    Campaign,
    ConsentScope,
    PricingModel,
    ReviewStatus,
)
from theraflo_ads.infra import schema
from theraflo_ads.infra.review_store import PgReviewRepository, ad_writer, refresh_catalog
from theraflo_ads.infra.stores import PgProfileStore, load_catalog, save_catalog
from theraflo_ads.policy.policies import PopularityPolicy
from theraflo_ads.pricing import CategoryPriors
from theraflo_ads.review.creative import Creative, SubstantiatedClaim, Verdict
from theraflo_ads.review.pipeline import Outcome, ReviewConfig
from theraflo_ads.review.simulated import SimulatedReviewModel
from theraflo_ads.review.store import HumanDecision, QueueItem, QueueKind
from theraflo_ads.review.wiring import build_review_service

pytestmark = pytest.mark.integration

BOOKLY = {"X-Advertiser-Key": "dev-adv-bookly"}


@pytest.fixture
def api(settings, engine):  # noqa: F811
    clock = Clock()
    order = ["ad_newco", "ad_bookly_new", "ad_bookly", "ad_sleepwell"]
    container = build_infra_container(
        settings,
        clock=clock,
        chain_factory=lambda catalog, priors, ctx: [PopularityPolicy(order)],
        refresh_catalog=False,
    )
    PgProfileStore(engine, container.hasher).upsert(
        "user-alice", consent=ConsentScope.PERSONALIZED, age_band=AgeBand.ADULT
    )
    return TestClient(create_app(container)), container


def worker_service(engine):  # noqa: F811
    """What `theraflo-ads review worker` runs: its own catalog, Postgres, a model reviewer."""
    return build_review_service(
        catalog=load_catalog(engine),
        repository=PgReviewRepository(engine),
        clock=lambda: datetime.now(UTC),
        lock=threading.Lock(),
        model=SimulatedReviewModel(),
        config=ReviewConfig(audit_rate=0.0),
        persist_ad=ad_writer(engine),
    )


def test_repository_round_trips(engine):  # noqa: F811
    repo = PgReviewRepository(engine)
    now = datetime.now(UTC).replace(microsecond=0)
    creative = Creative(
        "ad_x",
        "adv_bookly",
        "cmp_bookly",
        "books",
        "h",
        "b",
        "https://bookly.example/x",
        submitted_at=now,
    )
    repo.save_creative(creative)
    assert repo.creative("ad_x") == creative and repo.pending_review() == [creative]
    service = worker_service(engine)
    record = service.pipeline.review(creative)
    repo.save_review(record)
    assert repo.pending_review() == [] and repo.latest_review("ad_x") == record
    item = QueueItem("ad_x", "adv_bookly", QueueKind.AUDIT, record.review_id, ("audit",), now)
    repo.enqueue(item)
    repo.enqueue(item)  # idempotent
    assert repo.open_items(QueueKind.AUDIT) == [item] and repo.open_items(QueueKind.REVIEW) == []
    assert repo.close_item("ad_x", QueueKind.AUDIT) == item and repo.open_items() == []
    decision = HumanDecision(
        "ad_x",
        "adv_bookly",
        QueueKind.AUDIT,
        record.review_id,
        Verdict.APPROVE,
        "books",
        (),
        "h1",
        now,
    )
    repo.save_decision(decision)
    assert repo.decisions() == [decision]
    claim = SubstantiatedClaim("c1", "adv_bookly", "Rated 4.8 stars.", "doc.pdf")
    repo.add_claim(claim)
    repo.add_claim(claim)
    assert repo.claims_for("adv_bookly") == [claim] and repo.claims_for("adv_x") == []


def test_submission_review_and_refresh_across_processes(api, engine):  # noqa: F811
    client, container = api
    body = {
        "ad_id": "ad_bookly_new",
        "campaign_id": "cmp_bookly",
        "declared_category": "books",
        "headline": "Fall reads",
        "body": "Cozy mysteries for rainy evenings.",
        "landing_url": "https://bookly.example/fall",
    }
    assert client.post("/v1/advertiser/creatives", json=body, headers=BOOKLY).status_code == 202
    with engine.connect() as conn:
        status = conn.execute(
            select(schema.ads.c.review_status).where(schema.ads.c.ad_id == "ad_bookly_new")
        ).scalar_one()
    assert status == "pending"
    # Pending ads are never served, even first in the serving order.
    assert rank(client)["ads"][0]["ad_id"] != "ad_bookly_new"

    [record] = worker_service(engine).process_pending()
    assert record.outcome is Outcome.AUTO_APPROVED
    assert container.catalog.ads["ad_bookly_new"].review_status is ReviewStatus.PENDING
    assert refresh_catalog(engine, container.catalog, threading.Lock()) == 1
    assert container.catalog.ads["ad_bookly_new"].review_status is ReviewStatus.APPROVED
    assert rank(client)["ads"][0]["ad_id"] == "ad_bookly_new"
    status = client.get("/v1/advertiser/creatives/ad_bookly_new", headers=BOOKLY).json()
    assert status["review_status"] == "approved"


def test_why_this_ad_and_reports_read_postgres(api):
    client, _ = api
    ad = rank(client)["ads"][0]
    why = client.get(f"/v1/ads/{ad['impression_id']}/why", headers=ALICE)
    assert why.status_code == 200 and why.json()["factors"]
    now = datetime.now(UTC)
    params = {
        "start": (now - timedelta(hours=1)).isoformat(),
        "end": (now + timedelta(hours=1)).isoformat(),
    }
    report = client.get("/v1/advertiser/report", params=params, headers=BOOKLY).json()
    assert report["totals"]["impressions"] == 1 and report["segments"] == []


def test_refresh_picks_up_campaigns_created_after_startup(api, engine):  # noqa: F811
    """Found running the compose stack: the API loaded its catalog before `seed-demo` ran and
    never learned the campaigns, so it served nothing until restarted."""
    client, container = api
    newco = AdCatalog(
        {"cmp_newco": Campaign("cmp_newco", "adv_newco", PricingModel.CPM, 9.0, 80.0, priority=6)},
        {"ad_newco": Ad("ad_newco", "cmp_newco", "home_goods", "https://newco.example/")},
    )
    save_catalog(engine, newco, CategoryPriors(ctr={"home_goods": 0.031}, cvr={"home_goods": 0.04}))
    assert "cmp_newco" not in container.catalog.campaigns
    assert rank(client)["ads"][0]["ad_id"] != "ad_newco"
    priors = container.platform.cost.priors
    assert refresh_catalog(engine, container.catalog, threading.Lock(), priors) >= 3
    assert container.catalog.campaigns["cmp_newco"].daily_budget == 80.0
    assert priors.p_ctr("home_goods") == pytest.approx(0.031)
    assert rank(client)["ads"][0]["ad_id"] == "ad_newco"
