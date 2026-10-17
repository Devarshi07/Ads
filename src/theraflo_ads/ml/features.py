"""Shared feature definitions (design doc: User and Ad Representations, Feature Store,
Training-Serving Skew).

One FeatureBuilder computes features online (scoring and the decision log) and
the decision log is the training data, so training and serving cannot diverge.

Allowed inputs are ads-safe only: the ad, the placement, coarse daypart, device,
ad exposure counts, consent scope, and ad-interaction history. Interaction
history is used only for users who opted into personalization, is never
recorded for anyone else, and is never used for personalization-excluded
categories.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from theraflo_ads.domain import Ad, AdCatalog, AdRequest, ConsentScope, PricingModel
from theraflo_ads.events.bus import Bus, Message, MessageType
from theraflo_ads.pricing import CategoryPriors

FEATURE_SCHEMA_VERSION = "features_v2"
DAYPARTS = ("night", "morning", "afternoon", "evening")
PLACEMENTS = ("home_feed", "content_library")
DEVICES = ("mobile", "web")


@dataclass(frozen=True)
class UserInteractions:
    impressions: Mapping[str, float] = field(default_factory=dict)
    clicks: Mapping[str, float] = field(default_factory=dict)


class InteractionStore(Protocol):
    """Online feature store for ad-interaction counts (Redis in production)."""

    def get(self, user_hash: str) -> UserInteractions: ...

    def record_impression(self, user_hash: str, category: str) -> None: ...

    def record_click(self, user_hash: str, category: str) -> None: ...


class InMemoryInteractionStore:
    def __init__(self) -> None:
        self._impressions: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._clicks: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def get(self, user_hash: str) -> UserInteractions:
        return UserInteractions(
            dict(self._impressions.get(user_hash, {})), dict(self._clicks.get(user_hash, {}))
        )

    def record_impression(self, user_hash: str, category: str) -> None:
        self._impressions[user_hash][category] += 1.0

    def record_click(self, user_hash: str, category: str) -> None:
        self._clicks[user_hash][category] += 1.0

    def forget(self, user_hash: str) -> int:
        found = user_hash in self._impressions or user_hash in self._clicks
        self._impressions.pop(user_hash, None)
        self._clicks.pop(user_hash, None)
        return int(found)


class InteractionAggregator:
    """Feature aggregator consumer (design doc: Event Infrastructure): turns decision and
    click messages into interaction counts, for personalized-consent users only."""

    def __init__(self, store: InteractionStore, catalog: AdCatalog) -> None:
        self._store = store
        self._catalog = catalog
        self._impressions: dict[str, tuple[str, str]] = {}

    def subscribe(self, bus: Bus) -> None:
        bus.subscribe(MessageType.DECISION, self.on_decision)
        bus.subscribe(MessageType.AD_CLICKED, self.on_click)

    def on_decision(self, message: Message) -> None:
        payload = message.payload
        if not payload.get("impression_ids"):
            return
        if payload.get("consent_scope") != ConsentScope.PERSONALIZED.value:
            return  # no behavioral data is collected without consent
        ad = self._catalog.ads.get(payload["chosen_actions"][0]["ad_id"])
        if ad is None:
            return
        self._impressions[payload["impression_ids"][0]] = (payload["user_hash"], ad.category)
        self._store.record_impression(payload["user_hash"], ad.category)

    def on_click(self, message: Message) -> None:
        known = self._impressions.get(message.payload["impression_id"])
        if known is not None:
            self._store.record_click(*known)


def daypart(when: datetime) -> str:
    return DAYPARTS[when.hour // 6]


class FeatureBuilder:
    def __init__(
        self,
        catalog: AdCatalog,
        priors: CategoryPriors,
        *,
        interactions: InteractionStore | None = None,
        excluded_categories: frozenset[str] = frozenset(),
        prior_strength: float = 20.0,
    ) -> None:
        self._catalog = catalog
        self._priors = priors
        self._interactions = interactions
        self._excluded = excluded_categories
        self._strength = prior_strength
        self.categories = tuple(sorted({ad.category for ad in catalog.ads.values()}))
        self.version = FEATURE_SCHEMA_VERSION
        self.names = (
            *(f"cat_{c}" for c in self.categories),
            *(f"placement_{p}" for p in PLACEMENTS),
            *(f"pricing_{m.value}" for m in PricingModel),
            *(f"daypart_{d}" for d in DAYPARTS),
            *(f"device_{d}" for d in DEVICES),
            "log_price",
            "prior_ctr",
            "prior_cvr",
            "ad_exposures_24h",
            "user_exposures_24h",
            "personalized",
            "user_cat_ctr",
            "user_cat_impressions",
            "user_total_clicks",
        )

    def context(self, request: AdRequest) -> dict[str, Any]:
        """The per-request inputs, logged with the decision so features can be rebuilt for
        any eligible ad later. Interaction counts are read only with personalization consent."""
        personalized = request.user.consent is ConsentScope.PERSONALIZED
        interactions = (
            self._interactions.get(request.user.user_hash)
            if personalized and self._interactions is not None
            else UserInteractions()
        )
        snapshot = request.frequency
        return {
            "version": self.version,
            "device": request.user.device,
            "user_exposures": snapshot.user_count if snapshot is not None else 0,
            "ad_exposures": dict(snapshot.ad_counts) if snapshot is not None else {},
            "impressions": dict(interactions.impressions),
            "clicks": dict(interactions.clicks),
        }

    def build(self, request: AdRequest, ads: Sequence[Ad]) -> list[dict[str, float]]:
        ctx = request.feature_context or self.context(request)
        personalized = request.user.consent is ConsentScope.PERSONALIZED
        impressions_by_cat = ctx.get("impressions", {}) if personalized else {}
        clicks_by_cat = ctx.get("clicks", {}) if personalized else {}
        total_clicks = sum(v for c, v in clicks_by_cat.items() if c not in self._excluded)
        ad_exposures = ctx.get("ad_exposures", {})
        device = ctx.get("device")
        part = daypart(request.now)
        rows = []
        for ad in ads:
            campaign = self._catalog.campaign_for(ad)
            prior_ctr = self._priors.p_ctr(ad.category)
            use_history = personalized and ad.category not in self._excluded
            impressions = impressions_by_cat.get(ad.category, 0.0) if use_history else 0.0
            clicks = clicks_by_cat.get(ad.category, 0.0) if use_history else 0.0
            row = dict.fromkeys(self.names, 0.0)
            row[f"cat_{ad.category}"] = 1.0
            if request.placement_id in PLACEMENTS:
                row[f"placement_{request.placement_id}"] = 1.0
            row[f"pricing_{campaign.pricing_model.value}"] = 1.0
            row[f"daypart_{part}"] = 1.0
            if device in DEVICES:
                row[f"device_{device}"] = 1.0
            row["log_price"] = math.log1p(campaign.price)
            row["prior_ctr"] = prior_ctr
            row["prior_cvr"] = self._priors.p_cvr(ad.category)
            row["ad_exposures_24h"] = float(ad_exposures.get(ad.ad_id, 0))
            row["user_exposures_24h"] = float(ctx.get("user_exposures", 0))
            row["personalized"] = 1.0 if personalized else 0.0
            # Shrunk toward the category prior, so a few clicks cannot dominate.
            row["user_cat_ctr"] = (clicks + self._strength * prior_ctr) / (
                impressions + self._strength
            )
            row["user_cat_impressions"] = math.log1p(impressions)
            row["user_total_clicks"] = math.log1p(total_clicks)
            rows.append(row)
        return rows

    def matrix(self, rows: Sequence[Mapping[str, float]]) -> list[list[float]]:
        return [[float(row.get(name, 0.0)) for name in self.names] for row in rows]
