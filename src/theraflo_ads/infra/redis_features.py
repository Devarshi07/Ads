"""Online feature store and bandit counts in Redis (design doc: Feature Store → Online Features,
Event Infrastructure → Feature Aggregator).

Written by the `features` worker from decision and click messages; read on the
serving path. Interaction history is recorded only for users with
personalization consent, and expires after 30 days (data-retention window).
Every value here can be rebuilt from the lake, so Redis needs no persistence.
"""

from __future__ import annotations

from collections.abc import Mapping

import redis

from theraflo_ads.domain import AdCatalog, ConsentScope
from theraflo_ads.events.bus import Message, MessageType
from theraflo_ads.ml.features import UserInteractions

RETENTION_SECONDS = 30 * 24 * 3600
PENDING_SECONDS = 2 * 24 * 3600


class RedisInteractionStore:
    def __init__(self, client: redis.Redis) -> None:
        self._redis = client

    def get(self, user_hash: str) -> UserInteractions:
        try:
            raw = self._redis.hgetall(f"feat:{user_hash}")
        except redis.RedisError:
            # Features are optional for serving: fall back to the non-personalized vector.
            return UserInteractions()
        impressions, clicks = {}, {}
        for key, value in raw.items():
            kind, _, category = key.decode().partition(":")
            (impressions if kind == "imp" else clicks)[category] = float(value)
        return UserInteractions(impressions, clicks)

    def record_impression(self, user_hash: str, category: str) -> None:
        self._incr(user_hash, f"imp:{category}")

    def record_click(self, user_hash: str, category: str) -> None:
        self._incr(user_hash, f"clk:{category}")

    def forget(self, user_hash: str) -> int:
        return int(self._redis.delete(f"feat:{user_hash}"))

    def _incr(self, user_hash: str, field: str) -> None:
        pipe = self._redis.pipeline()
        pipe.hincrbyfloat(f"feat:{user_hash}", field, 1.0)
        pipe.expire(f"feat:{user_hash}", RETENTION_SECONDS)
        pipe.execute()


class RedisClickCounts:
    """Bandit counts shared by every API replica."""

    def __init__(self, client: redis.Redis, namespace: str = "bandit:beta_ts_v1") -> None:
        self._redis = client
        self._ns = namespace

    def snapshot(self) -> tuple[Mapping[str, float], Mapping[str, float]]:
        pipe = self._redis.pipeline()
        pipe.hgetall(f"{self._ns}:imp")
        pipe.hgetall(f"{self._ns}:clk")
        impressions, clicks = pipe.execute()
        decode = lambda raw: {k.decode(): float(v) for k, v in raw.items()}  # noqa: E731
        return decode(impressions), decode(clicks)

    def add(self, ad_id: str, *, impressions: float = 0.0, clicks: float = 0.0) -> None:
        pipe = self._redis.pipeline()
        if impressions:
            pipe.hincrbyfloat(f"{self._ns}:imp", ad_id, impressions)
        if clicks:
            pipe.hincrbyfloat(f"{self._ns}:clk", ad_id, clicks)
        pipe.execute()


class FeatureAggregatorWorker:
    """Consumer for the `features` queue. Remembers which (user, ad) an impression was in
    Redis (with a TTL), so clicks can be attributed by any worker replica.
    Duplicate deliveries are deduplicated on message_id."""

    def __init__(
        self,
        client: redis.Redis,
        catalog: AdCatalog,
        interactions: RedisInteractionStore,
        counts: RedisClickCounts,
    ) -> None:
        self._redis = client
        self._catalog = catalog
        self._interactions = interactions
        self._counts = counts

    def __call__(self, message: Message) -> None:
        if not self._redis.set(f"seen:{message.message_id}", 1, nx=True, ex=PENDING_SECONDS):
            return  # already processed (at-least-once delivery)
        if message.type is MessageType.DECISION:
            self._on_decision(message.payload)
        elif message.type is MessageType.AD_CLICKED:
            self._on_click(message.payload)

    def _on_decision(self, payload) -> None:
        if not payload.get("impression_ids") or payload.get("degraded"):
            return
        ad = self._catalog.ads.get(payload["chosen_actions"][0]["ad_id"])
        if ad is None:
            return
        self._counts.add(ad.ad_id, impressions=1.0)
        consented = payload.get("consent_scope") == ConsentScope.PERSONALIZED.value
        if consented:
            self._interactions.record_impression(payload["user_hash"], ad.category)
        user = payload["user_hash"] if consented else ""
        self._redis.set(
            f"pending:{payload['impression_ids'][0]}", f"{ad.ad_id}|{user}", ex=PENDING_SECONDS
        )

    def _on_click(self, payload) -> None:
        value = self._redis.get(f"pending:{payload['impression_id']}")
        if value is None:
            return
        ad_id, user = value.decode().split("|")
        self._counts.add(ad_id, clicks=1.0)
        ad = self._catalog.ads.get(ad_id)
        if user and ad is not None:
            self._interactions.record_click(user, ad.category)
