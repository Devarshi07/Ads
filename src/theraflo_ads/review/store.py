"""Creative submissions, review records, the human queue, and catalog updates
(design doc: Creative Review, Scenario I — LLM Reviewer Down).

`ReviewService` is the application layer: advertisers submit creatives (the ad
enters the catalog as PENDING and is never served), the review worker runs the
pipeline asynchronously, and humans resolve the queue. Live serving reads only
the catalog's review status, so a reviewer outage leaves serving untouched.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from theraflo_ads.domain import Ad, AdCatalog, Clock, ReviewStatus, utc_now
from theraflo_ads.review.creative import Creative, SubstantiatedClaim, Verdict, Violation
from theraflo_ads.review.pipeline import Outcome, ReviewPipeline, ReviewRecord


class QueueKind(StrEnum):
    REVIEW = "review"  # not live: a human decides
    AUDIT = "audit"  # already live (auto-approved): a human checks the machine's call


@dataclass(frozen=True)
class QueueItem:
    ad_id: str
    advertiser_id: str
    kind: QueueKind
    review_id: str
    reasons: tuple[str, ...]
    enqueued_at: datetime
    priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ad_id": self.ad_id,
            "advertiser_id": self.advertiser_id,
            "kind": self.kind.value,
            "review_id": self.review_id,
            "reasons": list(self.reasons),
            "enqueued_at": self.enqueued_at.isoformat(),
            "priority": self.priority,
        }


@dataclass(frozen=True)
class HumanDecision:
    ad_id: str
    advertiser_id: str
    kind: QueueKind
    review_id: str
    verdict: Verdict  # approve | reject
    category: str
    violations: tuple[Violation, ...]
    reviewer: str
    decided_at: datetime
    notes: str = ""
    machine_outcome: str | None = None

    @property
    def false_approval(self) -> bool:
        """A human overturned an auto-approval: the error the whole design is built around."""
        return (
            self.kind is QueueKind.AUDIT
            and self.machine_outcome == Outcome.AUTO_APPROVED.value
            and self.verdict is Verdict.REJECT
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            kind=self.kind.value,
            verdict=self.verdict.value,
            violations=[v.value for v in self.violations],
            decided_at=self.decided_at.isoformat(),
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanDecision:
        return cls(
            ad_id=data["ad_id"],
            advertiser_id=data["advertiser_id"],
            kind=QueueKind(data["kind"]),
            review_id=data["review_id"],
            verdict=Verdict(data["verdict"]),
            category=data["category"],
            violations=tuple(Violation(v) for v in data["violations"]),
            reviewer=data["reviewer"],
            decided_at=datetime.fromisoformat(data["decided_at"]),
            notes=data.get("notes", ""),
            machine_outcome=data.get("machine_outcome"),
        )


class ReviewRepository(Protocol):
    def save_creative(self, creative: Creative) -> None: ...

    def creative(self, ad_id: str) -> Creative | None: ...

    def pending_review(self) -> list[Creative]: ...

    def save_review(self, record: ReviewRecord) -> None: ...

    def latest_review(self, ad_id: str) -> ReviewRecord | None: ...

    def reviews(self) -> list[ReviewRecord]: ...

    def enqueue(self, item: QueueItem) -> None: ...

    def open_items(self, kind: QueueKind | None = None) -> list[QueueItem]: ...

    def close_item(self, ad_id: str, kind: QueueKind) -> QueueItem | None: ...

    def save_decision(self, decision: HumanDecision) -> None: ...

    def decisions(self) -> list[HumanDecision]: ...

    def claims_for(self, advertiser_id: str) -> list[SubstantiatedClaim]: ...

    def add_claim(self, claim: SubstantiatedClaim) -> None: ...

    def creatives_for(self, advertiser_id: str) -> list[Creative]: ...


class InMemoryReviewRepository:
    def __init__(self) -> None:
        self._creatives: dict[str, Creative] = {}
        self._reviews: dict[str, list[ReviewRecord]] = {}
        self._queue: dict[tuple[str, QueueKind], QueueItem] = {}
        self._decisions: list[HumanDecision] = []
        self._claims: dict[str, list[SubstantiatedClaim]] = {}
        self._pending: dict[str, None] = {}

    def save_creative(self, creative: Creative) -> None:
        self._creatives[creative.ad_id] = creative
        self._pending[creative.ad_id] = None

    def creative(self, ad_id: str) -> Creative | None:
        return self._creatives.get(ad_id)

    def creatives_for(self, advertiser_id: str) -> list[Creative]:
        """Newest first."""
        mine = [c for c in self._creatives.values() if c.advertiser_id == advertiser_id]
        return mine[::-1]

    def pending_review(self) -> list[Creative]:
        return [self._creatives[ad_id] for ad_id in self._pending]

    def save_review(self, record: ReviewRecord) -> None:
        self._reviews.setdefault(record.ad_id, []).append(record)
        self._pending.pop(record.ad_id, None)

    def latest_review(self, ad_id: str) -> ReviewRecord | None:
        history = self._reviews.get(ad_id)
        return history[-1] if history else None

    def reviews(self) -> list[ReviewRecord]:
        return [r for history in self._reviews.values() for r in history]

    def enqueue(self, item: QueueItem) -> None:
        self._queue.setdefault((item.ad_id, item.kind), item)

    def open_items(self, kind: QueueKind | None = None) -> list[QueueItem]:
        items = [i for (_, k), i in self._queue.items() if kind is None or k is kind]
        return sorted(items, key=lambda i: (-i.priority, i.enqueued_at))

    def close_item(self, ad_id: str, kind: QueueKind) -> QueueItem | None:
        return self._queue.pop((ad_id, kind), None)

    def save_decision(self, decision: HumanDecision) -> None:
        self._decisions.append(decision)

    def decisions(self) -> list[HumanDecision]:
        return list(self._decisions)

    def claims_for(self, advertiser_id: str) -> list[SubstantiatedClaim]:
        return list(self._claims.get(advertiser_id, []))

    def add_claim(self, claim: SubstantiatedClaim) -> None:
        self._claims.setdefault(claim.advertiser_id, []).append(claim)


class SubmissionError(ValueError):
    pass


class CatalogWriter(Protocol):
    """Persists an ad's review status and category (the ads table in production)."""

    def __call__(self, ad: Ad) -> None: ...


class ReviewService:
    def __init__(
        self,
        *,
        repository: ReviewRepository,
        pipeline: ReviewPipeline,
        catalog: AdCatalog,
        clock: Clock = utc_now,
        persist_ad: CatalogWriter | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        lock: AbstractContextManager | None = None,
    ) -> None:
        self.repository = repository
        self.pipeline = pipeline
        self._catalog = catalog
        self._clock = clock
        self._persist = persist_ad
        self._on_event = on_event
        # Guards catalog writes against the serving path (the API container's lock). Model
        # calls run outside it, so a slow reviewer never blocks serving.
        self._lock = lock if lock is not None else threading.Lock()

    # ----- advertisers -----

    def submit(self, creative: Creative) -> Ad:
        campaign = self._catalog.campaigns.get(creative.campaign_id)
        if campaign is None or campaign.advertiser_id != creative.advertiser_id:
            raise SubmissionError("unknown campaign")
        existing = self._catalog.ads.get(creative.ad_id)
        if existing is not None and existing.campaign_id != creative.campaign_id:
            raise SubmissionError("ad_id belongs to another campaign")
        creative = replace(creative, submitted_at=creative.submitted_at or self._clock())
        # Any edit to a live creative takes it out of rotation until it is reviewed again.
        ad = Ad(
            creative.ad_id,
            creative.campaign_id,
            creative.declared_category,
            creative.landing_url,
            ReviewStatus.PENDING,
        )
        with self._lock:
            self.repository.save_creative(creative)
            # A queued decision was about the previous version of the creative.
            self.repository.close_item(creative.ad_id, QueueKind.REVIEW)
            self._set_ad(ad)
        self._emit("creative_submitted", ad_id=ad.ad_id, advertiser_id=creative.advertiser_id)
        return ad

    def status(self, ad_id: str, advertiser_id: str) -> dict[str, Any] | None:
        creative = self.repository.creative(ad_id)
        if creative is None or creative.advertiser_id != advertiser_id:
            return None  # same answer as unknown: no probing of other advertisers' ads
        ad = self._catalog.ads.get(ad_id)
        record = self.repository.latest_review(ad_id)
        return {
            "ad_id": ad_id,
            "review_status": ad.review_status.value if ad else ReviewStatus.PENDING.value,
            # Policy labels only: rationale text and model details stay internal.
            "violations": [v.value for v in record.violations] if record else [],
            "category": ad.category if ad else creative.declared_category,
        }

    # ----- review worker -----

    def process_pending(self, limit: int | None = None) -> list[ReviewRecord]:
        records = []
        for creative in self.repository.pending_review()[:limit]:
            records.append(self.review(creative))
        return records

    def review(self, creative: Creative) -> ReviewRecord:
        record = self.pipeline.review(creative)
        with self._lock:
            self.repository.save_review(record)
            ad = self._catalog.ads.get(creative.ad_id)
            if ad is not None:
                self._set_ad(replace(ad, review_status=record.status, category=record.category))
            now = self._clock()
            if record.outcome is Outcome.HUMAN_REVIEW:
                self.repository.enqueue(
                    QueueItem(
                        creative.ad_id,
                        creative.advertiser_id,
                        QueueKind.REVIEW,
                        record.review_id,
                        record.reasons,
                        now,
                        priority=1 if record.error else 0,
                    )
                )
            if record.audited:
                self.repository.enqueue(
                    QueueItem(
                        creative.ad_id,
                        creative.advertiser_id,
                        QueueKind.AUDIT,
                        record.review_id,
                        ("audit sample",),
                        now,
                    )
                )
        self._emit(
            "creative_reviewed",
            ad_id=creative.ad_id,
            outcome=record.outcome.value,
            model=record.model,
            prompt_version=record.prompt_version,
        )
        return record

    # ----- humans -----

    def queue(self, kind: QueueKind | None = None) -> list[QueueItem]:
        return self.repository.open_items(kind)

    def decide(
        self,
        ad_id: str,
        *,
        kind: QueueKind,
        verdict: Verdict,
        reviewer: str,
        category: str | None = None,
        violations: Sequence[Violation] = (),
        notes: str = "",
    ) -> HumanDecision:
        if verdict is Verdict.ESCALATE:
            raise ValueError("a human decision is approve or reject")
        taxonomy = self.pipeline.taxonomy
        with self._lock:
            ad = self._catalog.ads.get(ad_id)
            if ad is None:
                raise KeyError(ad_id)
            category = category or ad.category
            if not taxonomy.is_known(category):
                raise ValueError(f"unknown category {category!r}")
            if verdict is Verdict.APPROVE and taxonomy.is_restricted(category):
                raise ValueError("a restricted category cannot be approved")
            item = self.repository.close_item(ad_id, kind)
            if item is None:
                raise KeyError(f"no open {kind.value} item for {ad_id}")
            record = self.repository.latest_review(ad_id)
            decision = HumanDecision(
                ad_id=ad_id,
                advertiser_id=item.advertiser_id,
                kind=kind,
                review_id=item.review_id,
                verdict=verdict,
                category=category,
                violations=tuple(violations),
                reviewer=reviewer,
                decided_at=self._clock(),
                notes=notes,
                machine_outcome=record.outcome.value if record else None,
            )
            self.repository.save_decision(decision)
            status = ReviewStatus.APPROVED if verdict is Verdict.APPROVE else ReviewStatus.REJECTED
            if kind is QueueKind.REVIEW or verdict is Verdict.REJECT:
                # An audit that rejects takes the live ad down immediately.
                self._set_ad(replace(ad, review_status=status, category=category))
        self._emit(
            "human_review_decision",
            ad_id=ad_id,
            kind=kind.value,
            verdict=verdict.value,
            false_approval=decision.false_approval,
        )
        return decision

    def audit_summary(self) -> dict[str, Any]:
        audits = [d for d in self.repository.decisions() if d.kind is QueueKind.AUDIT]
        from theraflo_ads.review.metrics import wilson_interval

        overturned = sum(d.false_approval for d in audits)
        low, high = wilson_interval(overturned, len(audits))
        return {
            "audited": len(audits),
            "overturned": overturned,
            "false_approval_rate": overturned / len(audits) if audits else None,
            "ci95": [low, high],
            "open_audits": len(self.repository.open_items(QueueKind.AUDIT)),
        }

    def _set_ad(self, ad: Ad) -> None:
        self._catalog.ads[ad.ad_id] = ad
        if self._persist is not None:
            self._persist(ad)

    def _emit(self, event: str, **details: Any) -> None:
        if self._on_event is not None:
            self._on_event(event, details)
