"""Postgres-backed catalog, priors, ads-safe profiles, the shared kill switch, and the audit log."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import DBAPIError

from theraflo_ads.domain import (
    Ad,
    AdCatalog,
    AgeBand,
    Campaign,
    ConsentScope,
    PricingModel,
    ReviewStatus,
    UserContext,
)
from theraflo_ads.infra import schema
from theraflo_ads.infra.postgres import transaction
from theraflo_ads.pricing import CategoryPriors
from theraflo_ads.safety.policy import SafetyPolicy
from theraflo_ads.security.identifiers import UserHasher

log = logging.getLogger(__name__)


# ----- catalog -----


def save_catalog(engine: Engine, catalog: AdCatalog, priors: CategoryPriors | None = None) -> None:
    with transaction(engine) as conn:
        for c in catalog.campaigns.values():
            values = {
                "campaign_id": c.campaign_id,
                "advertiser_id": c.advertiser_id,
                "pricing_model": c.pricing_model.value,
                "price": c.price,
                "daily_budget": c.daily_budget,
                "active": c.active,
                "placements": sorted(c.placements) if c.placements is not None else None,
                "priority": c.priority,
            }
            stmt = insert(schema.campaigns).values(**values)
            conn.execute(stmt.on_conflict_do_update(index_elements=["campaign_id"], set_=values))
        for a in catalog.ads.values():
            values = {
                "ad_id": a.ad_id,
                "campaign_id": a.campaign_id,
                "category": a.category,
                "landing_url": a.landing_url,
                "review_status": a.review_status.value,
            }
            stmt = insert(schema.ads).values(**values)
            conn.execute(stmt.on_conflict_do_update(index_elements=["ad_id"], set_=values))
        if priors is not None:
            for category in set(priors.ctr) | set(priors.cvr):
                values = {
                    "category": category,
                    "ctr": priors.p_ctr(category),
                    "cvr": priors.p_cvr(category),
                }
                stmt = insert(schema.category_priors).values(**values)
                conn.execute(stmt.on_conflict_do_update(index_elements=["category"], set_=values))


def load_catalog(engine: Engine) -> AdCatalog:
    with transaction(engine) as conn:
        campaigns = {
            r["campaign_id"]: Campaign(
                campaign_id=r["campaign_id"],
                advertiser_id=r["advertiser_id"],
                pricing_model=PricingModel(r["pricing_model"]),
                price=float(r["price"]),
                daily_budget=float(r["daily_budget"]),
                active=r["active"],
                placements=frozenset(r["placements"]) if r["placements"] is not None else None,
                priority=r["priority"],
            )
            for r in conn.execute(select(schema.campaigns)).mappings()
        }
        ads = {
            r["ad_id"]: Ad(
                ad_id=r["ad_id"],
                campaign_id=r["campaign_id"],
                category=r["category"],
                landing_url=r["landing_url"],
                review_status=ReviewStatus(r["review_status"]),
            )
            for r in conn.execute(select(schema.ads)).mappings()
        }
    return AdCatalog(campaigns, ads)


def load_priors(engine: Engine) -> CategoryPriors:
    with transaction(engine) as conn:
        rows = list(conn.execute(select(schema.category_priors)).mappings())
    return CategoryPriors(
        ctr={r["category"]: float(r["ctr"]) for r in rows},
        cvr={r["category"]: float(r["cvr"]) for r in rows},
    )


# ----- ads-safe user profiles -----


class PgProfileStore:
    """Profiles are keyed by the keyed user hash, never by the raw Theraflo user ID."""

    def __init__(self, engine: Engine, hasher: UserHasher) -> None:
        self._engine = engine
        self._hasher = hasher

    def context(self, user_id: str, device: str | None = None) -> UserContext:
        user_hash = self._hasher.hash(user_id)
        with transaction(self._engine) as conn:
            row = (
                conn.execute(
                    select(schema.user_profiles).where(
                        schema.user_profiles.c.user_hash == user_hash
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return UserContext(user_hash=user_hash, device=device)
        return UserContext(
            user_hash=user_hash,
            consent=ConsentScope(row["consent"]),
            age_band=AgeBand(row["age_band"]),
            region=row["region"],
            device=device,
            suppressed_until=row["suppressed_until"],
            hidden_ads=frozenset(row["hidden_ads"]),
            hidden_advertisers=frozenset(row["hidden_advertisers"]),
        )

    def upsert(
        self,
        user_id: str,
        *,
        consent: ConsentScope = ConsentScope.CONTEXTUAL_ONLY,
        age_band: AgeBand = AgeBand.UNKNOWN,
        region: str | None = None,
    ) -> None:
        self._write(user_id, consent=consent.value, age_band=age_band.value, region=region)

    def hide_ad(self, user_id: str, ad_id: str) -> None:
        current = self.context(user_id)
        self._write(user_id, hidden_ads=sorted(current.hidden_ads | {ad_id}))

    def suppress(self, user_id: str, until: datetime) -> None:
        self._write(user_id, suppressed_until=until)

    def set_consent(self, user_id: str, consent: ConsentScope) -> None:
        self._write(user_id, consent=consent.value)

    def forget(self, user_id: str) -> None:
        user_hash = self._hasher.hash(user_id)
        with transaction(self._engine) as conn:
            conn.execute(
                delete(schema.user_profiles).where(schema.user_profiles.c.user_hash == user_hash)
            )

    def _write(self, user_id: str, **fields: Any) -> None:
        user_hash = self._hasher.hash(user_id)
        base = {"consent": ConsentScope.CONTEXTUAL_ONLY.value, "age_band": AgeBand.UNKNOWN.value}
        values = {"user_hash": user_hash, **base, **fields, "updated_at": datetime.now(UTC)}
        stmt = insert(schema.user_profiles).values(**values)
        update_fields = {k: v for k, v in values.items() if k in fields or k == "updated_at"}
        with transaction(self._engine) as conn:
            conn.execute(
                stmt.on_conflict_do_update(index_elements=["user_hash"], set_=update_fields)
            )


# ----- shared kill switch -----


class PgSafetyPolicyStore:
    """The kill switch must reach every API process, so it lives in Postgres. Reads are
    cached briefly; if the setting cannot be read, the switch is treated as ON."""

    def __init__(self, engine: Engine, base: SafetyPolicy, *, cache_seconds: float = 1.0) -> None:
        self._engine = engine
        self._base = base
        self._cache_seconds = cache_seconds
        self._cached: tuple[float, SafetyPolicy] | None = None
        self._lock = threading.Lock()

    def current(self) -> SafetyPolicy:
        with self._lock:
            if self._cached and time.monotonic() - self._cached[0] < self._cache_seconds:
                return self._cached[1]
            try:
                with self._engine.connect() as conn:
                    value = conn.execute(
                        select(schema.platform_settings.c.value).where(
                            schema.platform_settings.c.key == "kill_switch"
                        )
                    ).scalar_one_or_none()
                policy = replace(self._base, kill_switch=bool(value and value.get("enabled")))
            except DBAPIError as exc:
                log.error("kill switch unreadable, failing closed: %s", exc)
                policy = replace(self._base, kill_switch=True)
            self._cached = (time.monotonic(), policy)
            return policy

    def set_kill_switch(self, enabled: bool) -> SafetyPolicy:
        values = {
            "key": "kill_switch",
            "value": {"enabled": enabled},
            "updated_at": datetime.now(UTC),
        }
        stmt = insert(schema.platform_settings).values(**values)
        with transaction(self._engine) as conn:
            conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=["key"],
                    set_={"value": values["value"], "updated_at": values["updated_at"]},
                )
            )
        with self._lock:
            self._cached = None
        return self.current()


def record_audit(engine: Engine, event: str, actor: str, details: dict[str, Any]) -> None:
    with transaction(engine) as conn:
        conn.execute(
            insert(schema.audit_log).values(
                event=event, actor=actor, details=details, at=datetime.now(UTC)
            )
        )
