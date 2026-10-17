"""Postgres schema (design doc: Database Reliability → Key tables).

Money columns are NUMERIC (exact) and are converted to float only at the
Python boundary. Alembic migrations in infra/migrations are generated from
this metadata.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Identity,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=NAMING)


def Money(name: str, **kwargs) -> Column:  # noqa: N802 - column factory
    return Column(name, Numeric(14, 6, asdecimal=False), **kwargs)


def Timestamp(name: str, **kwargs) -> Column:  # noqa: N802 - column factory
    return Column(name, DateTime(timezone=True), **kwargs)


campaigns = Table(
    "campaigns",
    metadata,
    Column("campaign_id", String(64), primary_key=True),
    Column("advertiser_id", String(64), nullable=False, index=True),
    Column("pricing_model", String(8), nullable=False),
    Money("price", nullable=False),
    Money("daily_budget", nullable=False),
    Column("active", Boolean, nullable=False, server_default=text("true")),
    Column("placements", ARRAY(String(64)), nullable=True),
    Column("priority", Integer, nullable=False, server_default=text("0")),
)

ads = Table(
    "ads",
    metadata,
    Column("ad_id", String(64), primary_key=True),
    Column("campaign_id", String(64), nullable=False, index=True),
    Column("category", String(64), nullable=False),
    Column("landing_url", String(2048), nullable=False),
    Column("review_status", String(16), nullable=False),
)

category_priors = Table(
    "category_priors",
    metadata,
    Column("category", String(64), primary_key=True),
    Column("ctr", Numeric(10, 8, asdecimal=False), nullable=False),
    Column("cvr", Numeric(10, 8, asdecimal=False), nullable=False),
)

impressions = Table(
    "impressions",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("impression_id", String(64), nullable=False, unique=True),
    Column("user_hash", String(64), nullable=False),
    Column("ad_id", String(64), nullable=False),
    Column("advertiser_id", String(64), nullable=False),
    Timestamp("served_at", nullable=False),
)
Index("ix_impressions_user_time", impressions.c.user_hash, impressions.c.served_at)

decisions = Table(
    "decisions",
    metadata,
    Column("request_id", String(64), primary_key=True),
    Column("user_hash", String(64), nullable=False),
    Column("placement", String(64), nullable=False),
    Column("served_by", String(32), nullable=False),
    Column("chosen_action", String(64), nullable=False),
    Column("impression_id", String(64), nullable=True),
    Column("ope_eligible", Boolean, nullable=False),
    Column("degraded", Boolean, nullable=False),
    Timestamp("decided_at", nullable=False, index=True),
    Column("record", JSONB, nullable=False),
)
# "Why am I seeing this?" looks a decision up by the impression it served.
Index("ix_decisions_impression_id", decisions.c.impression_id)

impression_labels = Table(
    "impression_labels",
    metadata,
    Column("impression_id", String(64), primary_key=True),
    Column("request_id", String(64), nullable=False),
    Column("ad_id", String(64), nullable=False),
    Column("campaign_id", String(64), nullable=False),
    Column("advertiser_id", String(64), nullable=False),
    Column("pricing_model", String(8), nullable=False),
    Money("price", nullable=False),
    Column("budget_key", String(160), nullable=False),
    Money("budget_limit", nullable=False),
    Column("reservation_id", String(64), nullable=True),
    Timestamp("served_at", nullable=False, index=True),
    Column("viewable", Boolean, nullable=False, server_default=text("false")),
    Column("click_id", String(64), nullable=True, unique=True),
    Timestamp("clicked_at", nullable=True),
    Column("click_accounted", Boolean, nullable=False, server_default=text("false")),
    Column("conversions", Integer, nullable=False, server_default=text("0")),
    Money("conversion_value", nullable=False, server_default=text("0")),
    Money("platform_revenue", nullable=False, server_default=text("0")),
    Column("state", String(24), nullable=False, index=True),
)

conversions = Table(
    "conversions",
    metadata,
    Column("advertiser_id", String(64), primary_key=True),
    Column("order_id", String(128), primary_key=True),
    Column("click_id", String(64), nullable=False),
    Column("impression_id", String(64), nullable=True),
    Money("value", nullable=False),
    Money("payout", nullable=False),
    Timestamp("occurred_at", nullable=False),
    Timestamp("received_at", nullable=False),
    Column("outcome", String(24), nullable=False),
    Column("accounted", Boolean, nullable=False, server_default=text("false")),
    Column("reversal_requested", Boolean, nullable=False, server_default=text("false")),
    Column("reversed", Boolean, nullable=False, server_default=text("false")),
)

events = Table(
    "events",
    metadata,
    Column("event_id", String(64), primary_key=True),
    Column("kind", String(32), nullable=False),
    Column("payload", JSONB, nullable=False),
    Timestamp("accepted_at", nullable=False, server_default=text("now()")),
)

rejected_events = Table(
    "rejected_events",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("event_id", String(64), nullable=True),
    Column("reason", String(64), nullable=False),
    Timestamp("received_at", nullable=False),
)

outbox = Table(
    "outbox",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("message_id", String(200), nullable=False, unique=True),
    Column("type", String(32), nullable=False),
    Column("body", JSONB, nullable=False),
    Timestamp("created_at", nullable=False),
    Timestamp("published_at", nullable=True),
)
Index(
    "ix_outbox_unpublished",
    outbox.c.id,
    postgresql_where=outbox.c.published_at.is_(None),
)

budget_days = Table(
    "budget_days",
    metadata,
    Column("budget_key", String(160), primary_key=True),
    Column("day", Date, primary_key=True),
    Money("billed", nullable=False, server_default=text("0")),
    Money("overdelivery", nullable=False, server_default=text("0")),
    Money("credited", nullable=False, server_default=text("0")),
)

billing_ledger = Table(
    "billing_ledger",
    metadata,
    Column("idempotency_key", String(200), primary_key=True),
    Column("budget_key", String(160), nullable=False),
    Column("day", Date, nullable=False),
    Column("kind", String(8), nullable=False),
    Money("requested", nullable=False),
    Money("billed", nullable=False),
    Timestamp("created_at", nullable=False, server_default=text("now()")),
)

user_profiles = Table(
    "user_profiles",
    metadata,
    Column("user_hash", String(64), primary_key=True),
    Column("consent", String(24), nullable=False),
    Column("age_band", String(16), nullable=False),
    Column("region", String(16), nullable=True),
    Timestamp("suppressed_until", nullable=True),
    Column("hidden_ads", ARRAY(String(64)), nullable=False, server_default=text("'{}'")),
    Column("hidden_advertisers", ARRAY(String(64)), nullable=False, server_default=text("'{}'")),
    Timestamp("updated_at", nullable=False, server_default=text("now()")),
)

platform_settings = Table(
    "platform_settings",
    metadata,
    Column("key", String(64), primary_key=True),
    Column("value", JSONB, nullable=False),
    Timestamp("updated_at", nullable=False, server_default=text("now()")),
)

audit_log = Table(
    "audit_log",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("event", String(64), nullable=False),
    Column("actor", String(64), nullable=False),
    Column("details", JSONB, nullable=False),
    Timestamp("at", nullable=False),
)

# Crypto-shredding (design doc: DVC Edge Cases): per-user keys live only here,
# outside every dataset snapshot. Deleting the row makes snapshots unlinkable.
user_keys = Table(
    "user_keys",
    metadata,
    Column("user_hash", String(64), primary_key=True),
    Column("key", String(64), nullable=False),
    Timestamp("created_at", nullable=False, server_default=text("now()")),
)

# ----- creative review (design doc: Phase 8 — LLM-Assisted Ad Review) -----

creatives = Table(
    "creatives",
    metadata,
    Column("ad_id", String(64), primary_key=True),
    Column("advertiser_id", String(64), nullable=False, index=True),
    Column("campaign_id", String(64), nullable=False),
    Column("payload", JSONB, nullable=False),
    Timestamp("submitted_at", nullable=False),
    # True from submission until the machine review has run for this version.
    Column("pending", Boolean, nullable=False, server_default=text("true")),
)
Index("ix_creatives_pending", creatives.c.submitted_at, postgresql_where=creatives.c.pending)

creative_reviews = Table(
    "creative_reviews",
    metadata,
    Column("review_id", String(64), primary_key=True),
    Column("ad_id", String(64), nullable=False, index=True),
    Column("outcome", String(24), nullable=False),
    Column("prompt_version", String(64), nullable=True),
    Column("model", String(64), nullable=True),
    Column("record", JSONB, nullable=False),
    Timestamp("reviewed_at", nullable=False),
)

review_queue = Table(
    "review_queue",
    metadata,
    Column("ad_id", String(64), primary_key=True),
    Column("kind", String(16), primary_key=True),
    Column("item", JSONB, nullable=False),
    Column("priority", Integer, nullable=False, server_default=text("0")),
    Timestamp("enqueued_at", nullable=False),
)

review_decisions = Table(
    "review_decisions",
    metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("ad_id", String(64), nullable=False, index=True),
    Column("kind", String(16), nullable=False),
    Column("decision", JSONB, nullable=False),
    Timestamp("decided_at", nullable=False),
)

substantiated_claims = Table(
    "substantiated_claims",
    metadata,
    Column("claim_id", String(64), primary_key=True),
    Column("advertiser_id", String(64), nullable=False, index=True),
    Column("text", String(1000), nullable=False),
    Column("evidence", String(500), nullable=False),
)

deletion_requests = Table(
    "deletion_requests",
    metadata,
    Column("request_id", String(64), primary_key=True),
    Column("user_hash", String(64), nullable=False),
    Timestamp("requested_at", nullable=False),
    Timestamp("completed_at", nullable=True),
    Column("steps", JSONB, nullable=True),
)
Index(
    "ix_deletion_requests_pending",
    deletion_requests.c.requested_at,
    postgresql_where=deletion_requests.c.completed_at.is_(None),
)
