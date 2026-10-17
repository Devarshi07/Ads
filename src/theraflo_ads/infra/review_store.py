"""Postgres adapters for creative review, catalog refresh, and decision lookup.

The review worker (`theraflo-ads review worker`) and the API share these tables.
API replicas load the catalog at startup and refresh it every few seconds:
campaigns created or edited since (onboarding, seeding, budget changes), ads (new,
review status, category), and category priors. A decision made by a human or by
the worker, or a campaign created elsewhere, reaches every replica without a
restart. A refresh failure keeps the last known catalog: statuses only change by
review, and the kill switch covers emergencies.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, MutableMapping
from contextlib import AbstractContextManager
from datetime import UTC, datetime

from sqlalchemy import Engine, delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import DBAPIError

from theraflo_ads.domain import Ad, AdCatalog
from theraflo_ads.infra import schema
from theraflo_ads.infra.postgres import transaction
from theraflo_ads.infra.stores import load_catalog, load_priors
from theraflo_ads.policy.decision import Decision, decision_from_record
from theraflo_ads.pricing import CategoryPriors
from theraflo_ads.review.creative import Creative, SubstantiatedClaim
from theraflo_ads.review.pipeline import ReviewRecord
from theraflo_ads.review.store import HumanDecision, QueueItem, QueueKind

log = logging.getLogger(__name__)


class PgReviewRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_creative(self, creative: Creative) -> None:
        values = {
            "ad_id": creative.ad_id,
            "advertiser_id": creative.advertiser_id,
            "campaign_id": creative.campaign_id,
            "payload": creative.to_dict(),
            "submitted_at": creative.submitted_at or datetime.now(UTC),
            "pending": True,
        }
        stmt = insert(schema.creatives).values(**values)
        with transaction(self._engine) as conn:
            conn.execute(stmt.on_conflict_do_update(index_elements=["ad_id"], set_=values))

    def creative(self, ad_id: str) -> Creative | None:
        with transaction(self._engine) as conn:
            row = conn.execute(
                select(schema.creatives.c.payload).where(schema.creatives.c.ad_id == ad_id)
            ).first()
        return Creative.from_dict(row[0]) if row else None

    def pending_review(self) -> list[Creative]:
        query = (
            select(schema.creatives.c.payload)
            .where(schema.creatives.c.pending)
            .order_by(schema.creatives.c.submitted_at)
        )
        with transaction(self._engine) as conn:
            return [Creative.from_dict(r[0]) for r in conn.execute(query)]

    def save_review(self, record: ReviewRecord) -> None:
        with transaction(self._engine) as conn:
            conn.execute(
                insert(schema.creative_reviews).values(
                    review_id=record.review_id,
                    ad_id=record.ad_id,
                    outcome=record.outcome.value,
                    prompt_version=record.prompt_version,
                    model=record.model,
                    record=record.to_dict(),
                    reviewed_at=record.reviewed_at,
                )
            )
            conn.execute(
                update(schema.creatives)
                .where(schema.creatives.c.ad_id == record.ad_id)
                .values(pending=False)
            )

    def latest_review(self, ad_id: str) -> ReviewRecord | None:
        t = schema.creative_reviews
        query = select(t.c.record).where(t.c.ad_id == ad_id).order_by(t.c.reviewed_at.desc())
        with transaction(self._engine) as conn:
            row = conn.execute(query.limit(1)).first()
        return ReviewRecord.from_dict(row[0]) if row else None

    def reviews(self) -> list[ReviewRecord]:
        t = schema.creative_reviews
        with transaction(self._engine) as conn:
            rows = conn.execute(select(t.c.record).order_by(t.c.reviewed_at))
            return [ReviewRecord.from_dict(r[0]) for r in rows]

    def enqueue(self, item: QueueItem) -> None:
        stmt = insert(schema.review_queue).values(
            ad_id=item.ad_id,
            kind=item.kind.value,
            item=item.to_dict(),
            priority=item.priority,
            enqueued_at=item.enqueued_at,
        )
        with transaction(self._engine) as conn:
            conn.execute(stmt.on_conflict_do_nothing(index_elements=["ad_id", "kind"]))

    def open_items(self, kind: QueueKind | None = None) -> list[QueueItem]:
        t = schema.review_queue
        query = select(t.c.item).order_by(t.c.priority.desc(), t.c.enqueued_at)
        if kind is not None:
            query = query.where(t.c.kind == kind.value)
        with transaction(self._engine) as conn:
            return [_queue_item(r[0]) for r in conn.execute(query)]

    def close_item(self, ad_id: str, kind: QueueKind) -> QueueItem | None:
        t = schema.review_queue
        stmt = delete(t).where(t.c.ad_id == ad_id, t.c.kind == kind.value).returning(t.c.item)
        with transaction(self._engine) as conn:
            row = conn.execute(stmt).first()
        return _queue_item(row[0]) if row else None

    def save_decision(self, decision: HumanDecision) -> None:
        with transaction(self._engine) as conn:
            conn.execute(
                insert(schema.review_decisions).values(
                    ad_id=decision.ad_id,
                    kind=decision.kind.value,
                    decision=decision.to_dict(),
                    decided_at=decision.decided_at,
                )
            )

    def decisions(self) -> list[HumanDecision]:
        t = schema.review_decisions
        with transaction(self._engine) as conn:
            rows = conn.execute(select(t.c.decision).order_by(t.c.id))
            return [HumanDecision.from_dict(r[0]) for r in rows]

    def creatives_for(self, advertiser_id: str) -> list[Creative]:
        """Newest first."""
        t = schema.creatives
        query = (
            select(t.c.payload)
            .where(t.c.advertiser_id == advertiser_id)
            .order_by(t.c.submitted_at.desc())
        )
        with transaction(self._engine) as conn:
            return [Creative.from_dict(r[0]) for r in conn.execute(query)]

    def claims_for(self, advertiser_id: str) -> list[SubstantiatedClaim]:
        t = schema.substantiated_claims
        with transaction(self._engine) as conn:
            rows = conn.execute(select(t).where(t.c.advertiser_id == advertiser_id)).mappings()
            return [
                SubstantiatedClaim(r["claim_id"], r["advertiser_id"], r["text"], r["evidence"])
                for r in rows
            ]

    def add_claim(self, claim: SubstantiatedClaim) -> None:
        values = {
            "claim_id": claim.claim_id,
            "advertiser_id": claim.advertiser_id,
            "text": claim.text,
            "evidence": claim.evidence,
        }
        stmt = insert(schema.substantiated_claims).values(**values)
        with transaction(self._engine) as conn:
            conn.execute(stmt.on_conflict_do_update(index_elements=["claim_id"], set_=values))


def _queue_item(data: dict) -> QueueItem:
    return QueueItem(
        ad_id=data["ad_id"],
        advertiser_id=data["advertiser_id"],
        kind=QueueKind(data["kind"]),
        review_id=data["review_id"],
        reasons=tuple(data["reasons"]),
        enqueued_at=datetime.fromisoformat(data["enqueued_at"]),
        priority=data.get("priority", 0),
    )


def ad_writer(engine: Engine) -> Callable[[Ad], None]:
    """Persists review status and category to the ads table (the serving catalog)."""

    def persist(ad: Ad) -> None:
        values = {
            "ad_id": ad.ad_id,
            "campaign_id": ad.campaign_id,
            "category": ad.category,
            "landing_url": ad.landing_url,
            "review_status": ad.review_status.value,
        }
        stmt = insert(schema.ads).values(**values)
        with transaction(engine) as conn:
            conn.execute(stmt.on_conflict_do_update(index_elements=["ad_id"], set_=values))

    return persist


def refresh_catalog(
    engine: Engine,
    catalog: AdCatalog,
    lock: AbstractContextManager,
    priors: CategoryPriors | None = None,
) -> int:
    """Applies campaign, ad, and (when given) category-prior changes from Postgres to the
    in-memory catalog; returns how many entries changed. Rows deleted from the tables stay
    in memory: a campaign is retired by deactivating it, an ad by rejecting it."""
    fresh = load_catalog(engine)
    fresh_priors = load_priors(engine) if priors is not None else None
    changed = 0
    with lock:
        # Campaigns first, so a request never sees an ad whose campaign it cannot find.
        for campaign_id, campaign in fresh.campaigns.items():
            if catalog.campaigns.get(campaign_id) != campaign:
                catalog.campaigns[campaign_id] = campaign
                changed += 1
        for ad_id, ad in fresh.ads.items():
            if catalog.ads.get(ad_id) != ad:
                catalog.ads[ad_id] = ad
                changed += 1
        if priors is not None and fresh_priors is not None:
            for mine, theirs in ((priors.ctr, fresh_priors.ctr), (priors.cvr, fresh_priors.cvr)):
                if isinstance(mine, MutableMapping):
                    updates = {k: v for k, v in theirs.items() if mine.get(k) != v}
                    mine.update(updates)
                    changed += len(updates)
    return changed


class CatalogRefresher:
    def __init__(
        self,
        engine: Engine,
        catalog: AdCatalog,
        lock: AbstractContextManager,
        *,
        priors: CategoryPriors | None = None,
        interval_s: float = 5.0,
    ) -> None:
        self._engine = engine
        self._catalog = catalog
        self._lock = lock
        self._priors = priors
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="catalog-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                refresh_catalog(self._engine, self._catalog, self._lock, self._priors)
            except DBAPIError:
                log.warning("catalog refresh failed; keeping the last known catalog")


def decision_lookup(engine: Engine) -> Callable[[str], Decision | None]:
    def lookup(impression_id: str) -> Decision | None:
        t = schema.decisions
        with transaction(engine) as conn:
            row = conn.execute(
                select(t.c.record).where(t.c.impression_id == impression_id).limit(1)
            ).first()
        return decision_from_record(row[0]) if row else None

    return lookup


def request_lookup(engine: Engine) -> Callable[[str], Decision | None]:
    def lookup(request_id: str) -> Decision | None:
        t = schema.decisions
        with transaction(engine) as conn:
            row = conn.execute(select(t.c.record).where(t.c.request_id == request_id)).first()
        return decision_from_record(row[0]) if row else None

    return lookup


def recent_lookup(engine: Engine) -> Callable[[int], list[Decision]]:
    def recent(limit: int) -> list[Decision]:
        t = schema.decisions
        query = select(t.c.record).order_by(t.c.decided_at.desc()).limit(limit)
        with transaction(engine) as conn:
            return [decision_from_record(r[0]) for r in conn.execute(query)]

    return recent


def recent_audit(engine: Engine) -> Callable[[int], list[dict]]:
    def recent(limit: int) -> list[dict]:
        t = schema.audit_log
        query = select(t).order_by(t.c.id.desc()).limit(limit)
        with transaction(engine) as conn:
            return [
                {"event": r["event"], "actor": r["actor"], "at": r["at"], **r["details"]}
                for r in conn.execute(query).mappings()
            ]

    return recent


def pg_report_source(engine: Engine):
    """Labels and their decisions for one advertiser and period (advertiser reports)."""
    from theraflo_ads.infra.postgres import _label

    def source(advertiser_id: str, start: datetime, end: datetime):
        t = schema.impression_labels
        query = select(t).where(
            t.c.advertiser_id == advertiser_id, t.c.served_at >= start, t.c.served_at < end
        )
        decisions: dict[str, Decision] = {}
        with transaction(engine) as conn:
            labels = [_label(r) for r in conn.execute(query).mappings()]
            ids = [label.request_id for label in labels]
            for i in range(0, len(ids), 1000):
                rows = conn.execute(
                    select(schema.decisions.c.record).where(
                        schema.decisions.c.request_id.in_(ids[i : i + 1000])
                    )
                )
                for row in rows:
                    decision = decision_from_record(row[0])
                    decisions[decision.request_id] = decision
        return labels, decisions

    return source
