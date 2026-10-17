"""Selecting the reviewer and narrator, and the report data sources.

THERAFLO_ADS_REVIEWER   none (default) | simulated | anthropic
THERAFLO_ADS_NARRATOR   none (default: template) | anthropic

`none` means rules plus the human queue: nothing is auto-approved, which is the
pre-Phase-8 behaviour and the state the system falls back to when the model is
down. `anthropic` calls Claude (credentials from ANTHROPIC_API_KEY or an
`ant auth login` profile); every call is billed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any

from theraflo_ads.attribution.service import ImpressionLabel
from theraflo_ads.domain import AdCatalog, Clock
from theraflo_ads.platform import Platform
from theraflo_ads.policy.decision import Decision
from theraflo_ads.review.creative import SubstantiatedClaim
from theraflo_ads.review.llm import NarratorModel, ReviewModel
from theraflo_ads.review.pipeline import ReviewConfig, ReviewPipeline
from theraflo_ads.review.store import CatalogWriter, ReviewRepository, ReviewService
from theraflo_ads.review.taxonomy import Taxonomy, default_taxonomy

ReportSource = Callable[
    [str, datetime, datetime], tuple[list[ImpressionLabel], dict[str, Decision]]
]


def reviewer_from_env(taxonomy: Taxonomy | None = None) -> ReviewModel | None:
    name = os.environ.get("THERAFLO_ADS_REVIEWER", "none").strip().lower()
    taxonomy = taxonomy or default_taxonomy()
    if name in ("", "none", "rules"):
        return None
    if name == "simulated":
        from theraflo_ads.review.simulated import SimulatedReviewModel

        return SimulatedReviewModel(taxonomy=taxonomy)
    if name == "ollama":
        from theraflo_ads.review.local import OllamaReviewModel

        return OllamaReviewModel(taxonomy=taxonomy)
    if name == "anthropic":
        from theraflo_ads.review.llm import AnthropicReviewModel

        return AnthropicReviewModel(taxonomy=taxonomy)
    raise ValueError(
        f"THERAFLO_ADS_REVIEWER must be none, simulated, ollama, or anthropic, not {name!r}"
    )


def narrator_from_env() -> NarratorModel | None:
    name = os.environ.get("THERAFLO_ADS_NARRATOR", "none").strip().lower()
    if name in ("", "none", "template"):
        return None
    if name == "ollama":
        from theraflo_ads.review.local import OllamaNarratorModel

        return OllamaNarratorModel()
    if name == "anthropic":
        from theraflo_ads.review.llm import AnthropicNarratorModel

        return AnthropicNarratorModel()
    raise ValueError(f"THERAFLO_ADS_NARRATOR must be none, ollama, or anthropic, not {name!r}")


def build_review_service(
    *,
    catalog: AdCatalog,
    repository: ReviewRepository,
    clock: Clock,
    lock: AbstractContextManager,
    model: ReviewModel | None,
    taxonomy: Taxonomy | None = None,
    config: ReviewConfig | None = None,
    persist_ad: CatalogWriter | None = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> ReviewService:
    taxonomy = taxonomy or default_taxonomy()
    pipeline = ReviewPipeline(
        model=model, claims=repository.claims_for, taxonomy=taxonomy, config=config, clock=clock
    )
    return ReviewService(
        repository=repository,
        pipeline=pipeline,
        catalog=catalog,
        clock=clock,
        persist_ad=persist_ad,
        on_event=on_event,
        lock=lock,
    )


def memory_report_source(platform: Platform) -> ReportSource:
    def source(advertiser_id: str, start: datetime, end: datetime):
        labels = [
            label
            for label in platform.attribution_labels()
            if label.advertiser_id == advertiser_id and start <= label.served_at < end
        ]
        wanted = {label.request_id for label in labels}
        decisions = {d.request_id: d for d in platform.decision_log() if d.request_id in wanted}
        return labels, decisions

    return source


DEMO_CLAIMS = (
    SubstantiatedClaim(
        "clm_sleepwell_1",
        "adv_sleepwell",
        "In a 2024 study of 120 adults, 78% fell asleep faster with our weighted blanket.",
        "sleepwell-study-2024.pdf",
    ),
    SubstantiatedClaim(
        "clm_bookly_1", "adv_bookly", "Rated 4.8 stars by over 50,000 readers.", "app-store.pdf"
    ),
)
