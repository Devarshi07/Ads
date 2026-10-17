"""Wiring for the API process.

The container holds a Platform (memory or infrastructure adapters) plus the
API-only pieces: authenticators, the ads-safe profile store, the client event
ingestor, and the audit trail.

- `build_dev_container`: in-memory platform, demo inventory, dev credentials.
- `build_infra_container`: Postgres/Redis/outbox platform (THERAFLO_ADS_BACKEND=infra).
Dev credentials are refused when THERAFLO_ADS_ENV=production.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import random
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from theraflo_ads.api.lab import PolicyLab
from theraflo_ads.domain import (
    Ad,
    AdCatalog,
    AgeBand,
    Campaign,
    Clock,
    ConsentScope,
    PricingModel,
    UserContext,
    utc_now,
)
from theraflo_ads.events.bus import Message, MessageType
from theraflo_ads.events.ingestion import ClientEventIngestor
from theraflo_ads.experiments.assignment import HoldoutConfig
from theraflo_ads.ml.features import FeatureBuilder, InMemoryInteractionStore, InteractionAggregator
from theraflo_ads.observability.metrics import Metrics
from theraflo_ads.observability.ratelimit import InProcessRateLimiter, NoRateLimit
from theraflo_ads.platform import Platform, build_memory_platform
from theraflo_ads.policy.bandits import BanditUpdater, ClickCounts, LinearBanditState
from theraflo_ads.policy.policies import EpsilonGreedyPolicy, Policy, PopularityPolicy
from theraflo_ads.policy.scoring import HeuristicScorer
from theraflo_ads.pricing import CategoryPriors, PriorCostEstimator
from theraflo_ads.privacy.shredding import (
    DeletionService,
    InMemoryDeletionRequests,
    InMemoryUserKeyStore,
    memory_erasers,
)
from theraflo_ads.review.llm import NarratorModel
from theraflo_ads.review.store import InMemoryReviewRepository, ReviewService
from theraflo_ads.review.taxonomy import Taxonomy, default_taxonomy
from theraflo_ads.review.wiring import (
    DEMO_CLAIMS,
    ReportSource,
    build_review_service,
    memory_report_source,
    narrator_from_env,
    reviewer_from_env,
)
from theraflo_ads.safety.policy import default_safety_policy
from theraflo_ads.security.identifiers import UserHasher
from theraflo_ads.security.sessions import SessionTokens
from theraflo_ads.security.tokens import ImpressionTokenSigner

# Dev-only credentials for local testing. Never valid outside development.
DEV_USER_TOKENS = {
    "dev-user-alice": "user-alice",
    "dev-user-bob": "user-bob",
    "dev-user-teen": "user-teen",
}
DEV_ADVERTISER_KEYS = {
    "dev-adv-sleepwell": "adv_sleepwell",
    "dev-adv-trailco": "adv_trailco",
    "dev-adv-bookly": "adv_bookly",
    "dev-adv-fitco": "adv_fitco",
}
DEV_ADMIN_TOKENS = {"dev-admin": "dev-admin"}
DEV_SERVICE_TOKENS = {"dev-theraflo-backend": "theraflo-backend"}

Authenticator = Callable[[str | None], str | None]


class ProfileStore(Protocol):
    def context(self, user_id: str, device: str | None = None) -> UserContext: ...

    def upsert(
        self,
        user_id: str,
        *,
        consent: ConsentScope = ConsentScope.CONTEXTUAL_ONLY,
        age_band: AgeBand = AgeBand.UNKNOWN,
        region: str | None = None,
    ) -> None: ...

    def hide_ad(self, user_id: str, ad_id: str) -> None: ...

    def suppress(self, user_id: str, until: datetime) -> None: ...

    def set_consent(self, user_id: str, consent: ConsentScope) -> None: ...

    def forget(self, user_id: str) -> None: ...


@dataclass
class UserProfile:
    consent: ConsentScope = ConsentScope.CONTEXTUAL_ONLY
    age_band: AgeBand = AgeBand.UNKNOWN
    region: str | None = None
    suppressed_until: datetime | None = None
    hidden_ads: set[str] = field(default_factory=set)
    hidden_advertisers: set[str] = field(default_factory=set)


class MemoryProfileStore:
    """Ads-safe user state. Suppression windows are set by the Theraflo backend when a
    user opens crisis resources; this service never sees why."""

    def __init__(self, hasher: UserHasher, profiles: Mapping[str, UserProfile] | None = None):
        self._hasher = hasher
        # Deep copy: profiles are mutable, and seed data must not change between stores.
        self._profiles = copy.deepcopy(dict(profiles or {}))

    def get(self, user_id: str) -> UserProfile:
        return self._profiles.setdefault(user_id, UserProfile())

    def context(self, user_id: str, device: str | None = None) -> UserContext:
        profile = self.get(user_id)
        return UserContext(
            user_hash=self._hasher.hash(user_id),
            consent=profile.consent,
            age_band=profile.age_band,
            region=profile.region,
            device=device,
            suppressed_until=profile.suppressed_until,
            hidden_ads=frozenset(profile.hidden_ads),
            hidden_advertisers=frozenset(profile.hidden_advertisers),
        )

    def upsert(
        self,
        user_id: str,
        *,
        consent: ConsentScope = ConsentScope.CONTEXTUAL_ONLY,
        age_band: AgeBand = AgeBand.UNKNOWN,
        region: str | None = None,
    ) -> None:
        profile = self.get(user_id)
        profile.consent, profile.age_band, profile.region = consent, age_band, region

    def hide_ad(self, user_id: str, ad_id: str) -> None:
        self.get(user_id).hidden_ads.add(ad_id)

    def suppress(self, user_id: str, until: datetime) -> None:
        self.get(user_id).suppressed_until = until

    def set_consent(self, user_id: str, consent: ConsentScope) -> None:
        self.get(user_id).consent = consent

    def forget(self, user_id: str) -> None:
        self._profiles.pop(user_id, None)


class JsonlDecisionSink:
    """Bus subscriber that appends decision messages to a JSONL file (dev analytics sink)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: Message) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(message.payload)) + "\n")


@dataclass
class Container:
    clock: Clock
    platform: Platform
    hasher: UserHasher
    ingestor: ClientEventIngestor
    profiles: ProfileStore
    authenticate_user: Authenticator
    authenticate_advertiser: Authenticator
    authenticate_admin: Authenticator
    authenticate_service: Authenticator
    # In-memory stores are not thread-safe; real stores are, so the lock is a no-op there.
    lock: AbstractContextManager = field(default_factory=threading.Lock)
    reports: list[tuple[str, str]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    audit_sink: Callable[[str, str, dict[str, Any]], None] | None = None
    # Creative review (Phase 8). `review_inline`: run the machine review as a background task
    # after the submission response (memory backend); otherwise the review worker runs it.
    review: ReviewService | None = None
    review_inline: bool = False
    narrator: NarratorModel | None = None
    report_source: ReportSource | None = None
    taxonomy: Taxonomy = field(default_factory=default_taxonomy)
    # Right to deletion: run inline after the response (memory backend) or by the deletion job.
    deletions: DeletionService | None = None
    deletion_inline: bool = False
    metrics: Metrics = field(default_factory=Metrics)
    # THERAFLO_ADS_RATE_LIMITS=off disables limiting (load tests).
    limiter: Any = field(default_factory=lambda: _limiter_from_env())
    # Dev console (/ui and /v1/dev): mounted only with dev credentials, never in production.
    backend: str = "memory"
    dev_mode: bool = False
    dev_users: DevUserTokens | None = None
    lab: PolicyLab | None = None
    # Newest first; Postgres for the infra backend, `audit` otherwise.
    audit_source: Callable[[int], list[dict[str, Any]]] | None = None

    @property
    def catalog(self) -> AdCatalog:
        return self.platform.catalog

    def recent_audit(self, limit: int) -> list[dict[str, Any]]:
        if self.audit_source is not None:
            return self.audit_source(limit)
        return self.audit[-limit:][::-1]

    def review_event(self, event: str, details: dict[str, Any]) -> None:
        if event == "creative_reviewed":
            self.metrics.review_outcomes.labels(details["outcome"]).inc()
        self.record_audit(event, "review", **details)

    def record_audit(self, event: str, actor: str, **details: Any) -> None:
        entry = {"event": event, "actor": actor, "at": self.clock(), **details}
        self.audit.append(entry)
        if self.audit_sink is not None:
            self.audit_sink(event, actor, {k: str(v) for k, v in details.items()})


def _limiter_from_env():
    if os.environ.get("THERAFLO_ADS_RATE_LIMITS", "on").lower() == "off":
        return NoRateLimit()
    return InProcessRateLimiter()


class DevUserTokens:
    """Bearer tokens for test users made in the dev console: session tokens when
    THERAFLO_ADS_SESSION_KEY is set (they survive restarts and work on every replica),
    otherwise random tokens that last until this process exits."""

    def __init__(self, sessions: SessionTokens | None, clock: Clock) -> None:
        self._sessions = sessions
        self._clock = clock
        self._by_digest: dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def kind(self) -> str:
        return "session" if self._sessions is not None else "process"

    def issue(self, user_id: str) -> str:
        if self._sessions is not None:
            return self._sessions.issue(user_id, self._clock())
        token = "dev-test-" + secrets.token_urlsafe(24)
        with self._lock:
            self._by_digest[_digest(token)] = user_id
        return token

    def __call__(self, presented: str | None) -> str | None:
        # Session tokens are verified by the session authenticator, with the same key.
        if not presented or self._sessions is not None:
            return None
        return self._by_digest.get(_digest(presented))


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def dev_user_tokens(clock: Clock) -> DevUserTokens:
    key = os.environ.get("THERAFLO_ADS_SESSION_KEY")
    sessions = SessionTokens(bytes.fromhex(key), ttl=timedelta(days=7)) if key else None
    return DevUserTokens(sessions, clock)


def table_authenticator(table: Mapping[str, str]) -> Authenticator:
    """Constant-time lookup over a credential table."""
    entries = [(k.encode(), v) for k, v in table.items()]

    def authenticate(presented: str | None) -> str | None:
        if not presented:
            return None
        match = None
        for key, value in entries:
            if hmac.compare_digest(key, presented.encode()):
                match = value
        return match

    return authenticate


def hashed_key_authenticator(hashed: Mapping[str, str]) -> Authenticator:
    """API keys stored as SHA-256 digests (env THERAFLO_ADS_ADVERTISER_KEYS={digest: id})."""
    table = table_authenticator(hashed)
    return lambda presented: (
        table(hashlib.sha256(presented.encode()).hexdigest()) if presented else None
    )


def chain_authenticators(*authenticators: Authenticator) -> Authenticator:
    def authenticate(presented: str | None) -> str | None:
        for authenticator in authenticators:
            result = authenticator(presented)
            if result is not None:
                return result
        return None

    return authenticate


def session_authenticator(tokens: SessionTokens, clock: Clock) -> Authenticator:
    return lambda presented: tokens.verify(presented, clock()) if presented else None


def demo_catalog() -> AdCatalog:
    campaigns = [
        Campaign("cmp_sleepwell", "adv_sleepwell", PricingModel.CPM, 8.0, 50.0, priority=5),
        Campaign("cmp_trailco", "adv_trailco", PricingModel.CPC, 0.8, 40.0, priority=3),
        Campaign("cmp_bookly", "adv_bookly", PricingModel.CPA, 12.0, 120.0, priority=2),
        Campaign(
            "cmp_fitco",
            "adv_fitco",
            PricingModel.CPM,
            6.0,
            30.0,
            placements=frozenset({"home_feed"}),
            priority=4,
        ),
        # Restricted category: present so the constraint engine has something to block.
        Campaign("cmp_luckyspin", "adv_luckyspin", PricingModel.CPM, 20.0, 500.0, priority=9),
    ]
    ads = [
        Ad("ad_sleepwell", "cmp_sleepwell", "sleep_products", "https://sleepwell.example/rest"),
        Ad("ad_trailco", "cmp_trailco", "outdoor_gear", "https://trailco.example/boots"),
        Ad("ad_bookly", "cmp_bookly", "books", "https://bookly.example/start?ref=theraflo"),
        Ad("ad_fitco", "cmp_fitco", "fitness", "https://fitco.example/classes"),
        Ad("ad_luckyspin", "cmp_luckyspin", "gambling", "https://luckyspin.example/"),
    ]
    return AdCatalog({c.campaign_id: c for c in campaigns}, {a.ad_id: a for a in ads})


DEMO_PRIORS = CategoryPriors(
    ctr={"sleep_products": 0.025, "outdoor_gear": 0.015, "books": 0.02, "fitness": 0.02},
    cvr={"sleep_products": 0.05, "outdoor_gear": 0.03, "books": 0.06, "fitness": 0.04},
)

DEMO_PROFILES = {
    "user-alice": UserProfile(ConsentScope.PERSONALIZED, AgeBand.ADULT, "MA"),
    "user-bob": UserProfile(ConsentScope.CONTEXTUAL_ONLY, AgeBand.ADULT, "NY"),
    "user-teen": UserProfile(ConsentScope.PERSONALIZED, AgeBand.TEEN, "CA"),
}


def heuristic_chain(catalog: AdCatalog, priors: CategoryPriors) -> list[Policy]:
    """Phase 1: heuristic with ε-exploration, then popularity, then NO_AD."""
    ranked = sorted(catalog.ads, key=lambda a: -catalog.campaign_for(catalog.ads[a]).priority)
    return [
        EpsilonGreedyPolicy(HeuristicScorer(catalog, priors), epsilon=0.05),
        PopularityPolicy(ranked),
    ]


def build_dev_container(
    *,
    clock: Clock = utc_now,
    user_hash_key: bytes | None = None,
    token_key: bytes | None = None,
    holdouts: HoldoutConfig | None = None,
    decision_log_path: Path | None = None,
    seed: int | None = None,
    chain: Callable[[AdCatalog], Sequence[Policy]] | None = None,
    reviewer=None,
) -> Container:
    _refuse_production()
    catalog = demo_catalog()
    interactions = InMemoryInteractionStore()
    # The same feature builder as the infra platform: decisions log features, the ML ranker
    # and the linear bandit read them, and personalized users get interaction history.
    features = FeatureBuilder(
        catalog,
        DEMO_PRIORS,
        interactions=interactions,
        excluded_categories=default_safety_policy().personalization_excluded_categories,
    )
    platform = build_memory_platform(
        catalog=catalog,
        cost=PriorCostEstimator(DEMO_PRIORS),
        chain=(chain or (lambda c: heuristic_chain(c, DEMO_PRIORS)))(catalog),
        signer=ImpressionTokenSigner(token_key or key_from_env("THERAFLO_ADS_TOKEN_KEY")),
        holdouts=holdouts,
        rng=random.Random(seed),
        features=features,
        # A long-running dev server: nothing reads the published-message history.
        keep_published=False,
    )
    InteractionAggregator(interactions, catalog).subscribe(platform.bus)
    hasher = UserHasher(
        "ads", {"k1": user_hash_key or key_from_env("THERAFLO_ADS_USER_HASH_KEY")}, "k1"
    )
    if decision_log_path is None and os.environ.get("THERAFLO_ADS_DECISION_LOG"):
        decision_log_path = Path(os.environ["THERAFLO_ADS_DECISION_LOG"])
    if decision_log_path is not None:
        platform.bus.subscribe(MessageType.DECISION, JsonlDecisionSink(decision_log_path))
    dev_users = dev_user_tokens(clock)
    container = assemble_container(
        platform,
        clock=clock,
        hasher=hasher,
        profiles=MemoryProfileStore(hasher, DEMO_PROFILES),
        lock=threading.Lock(),
        **dev_authenticators(clock, dev_users),
    )
    container.dev_mode = True
    container.dev_users = dev_users
    linear = LinearBanditState(features.names)
    container.lab = PolicyLab(
        platform=platform,
        features=features,
        counts=ClickCounts(),
        linear=linear,
        model_dir=Path(os.environ.get("THERAFLO_ADS_MODEL_DIR", "models")),
        serving="heuristic" if chain is None else "custom",
    )
    # Both bandits learn from every served impression and click, whichever policy serves.
    BanditUpdater(linear=linear, beta=container.lab.beta).subscribe(platform.bus)
    repository = InMemoryReviewRepository()
    for claim in DEMO_CLAIMS:
        repository.add_claim(claim)
    container.review = build_review_service(
        catalog=catalog,
        repository=repository,
        clock=clock,
        lock=container.lock,
        model=reviewer_from_env(container.taxonomy) if reviewer is None else reviewer,
        taxonomy=container.taxonomy,
        on_event=container.review_event,
    )
    container.review_inline = True
    container.narrator = narrator_from_env()
    container.report_source = memory_report_source(platform)
    container.deletions = DeletionService(
        requests=InMemoryDeletionRequests(),
        keys=InMemoryUserKeyStore(),
        erasers=memory_erasers(platform),
        clock=clock,
        on_event=lambda event, details: container.record_audit(event, "deletion_job", **details),
    )
    container.deletion_inline = True
    return container


def infra_chain(name: str):
    """THERAFLO_ADS_POLICY: heuristic | bandit | ranker (production models in the registry)."""

    def build(catalog: AdCatalog, priors: CategoryPriors, ctx) -> list[Policy]:
        fallback = heuristic_chain(catalog, priors)
        if name == "bandit":
            from theraflo_ads.infra.redis_features import RedisClickCounts
            from theraflo_ads.policy.bandits import BetaThompsonPolicy
            from theraflo_ads.policy.guard import GuardedPolicy

            bandit = BetaThompsonPolicy(catalog, priors, RedisClickCounts(ctx.redis))
            return [GuardedPolicy(bandit), *fallback]
        if name == "ranker":
            from theraflo_ads.ml.serving import build_serving_chain

            return build_serving_chain(
                catalog, priors, ctx.features, ctx.settings.model_dir / "registry"
            ).chain
        return fallback

    return build


def build_infra_container(
    settings=None,
    *,
    clock: Clock = utc_now,
    chain_factory=None,
    refresh_catalog: bool = True,
) -> Container:
    """Postgres/Redis/outbox platform. Keys and credentials come from the environment."""
    from theraflo_ads.infra.platform import build_infra_platform
    from theraflo_ads.infra.privacy_store import PgDeletionRequests, PgUserKeyStore, pg_erasers
    from theraflo_ads.infra.review_store import (
        CatalogRefresher,
        PgReviewRepository,
        ad_writer,
        pg_report_source,
        recent_audit,
    )
    from theraflo_ads.infra.settings import Settings
    from theraflo_ads.infra.stores import PgProfileStore, record_audit
    from theraflo_ads.observability.ratelimit import RedisRateLimiter

    settings = settings or Settings()
    # Several processes (API replicas, workers, seeding) must share these keys.
    token_key = key_from_env("THERAFLO_ADS_TOKEN_KEY", required=True)
    hash_key = key_from_env("THERAFLO_ADS_USER_HASH_KEY", required=True)
    policy_name = os.environ.get("THERAFLO_ADS_POLICY", "heuristic")
    platform, handles = build_infra_platform(
        settings,
        chain_factory=chain_factory or infra_chain(policy_name),
        signer=ImpressionTokenSigner(token_key),
    )
    hasher = UserHasher("ads", {"k1": hash_key}, "k1")
    dev_users = None
    if settings.production:
        authenticators = production_authenticators(clock)
    else:
        dev_users = dev_user_tokens(clock)
        authenticators = dev_authenticators(clock, dev_users)
    container = assemble_container(
        platform,
        clock=clock,
        hasher=hasher,
        profiles=PgProfileStore(handles.engine, hasher),
        lock=nullcontext(),
        audit_sink=lambda event, actor, details: record_audit(
            handles.engine, event, actor, details
        ),
        **authenticators,
    )
    # The API never calls a model: `theraflo-ads review worker` runs the machine review.
    container.review = build_review_service(
        catalog=platform.catalog,
        repository=PgReviewRepository(handles.engine),
        clock=clock,
        lock=threading.Lock(),
        model=None,
        taxonomy=container.taxonomy,
        persist_ad=ad_writer(handles.engine),
        on_event=container.review_event,
    )
    container.narrator = narrator_from_env()
    container.report_source = pg_report_source(handles.engine)
    if not isinstance(container.limiter, NoRateLimit):
        # One budget across every API replica.
        container.limiter = RedisRateLimiter(
            handles.redis, on_error=container.metrics.rate_limiter_errors.inc
        )
    # Requests are queued here; `theraflo-ads privacy delete-pending` (a cron job) runs them.
    container.deletions = DeletionService(
        requests=PgDeletionRequests(handles.engine),
        keys=PgUserKeyStore(handles.engine),
        erasers=pg_erasers(handles.engine, handles.redis),
        clock=clock,
        on_event=lambda event, details: container.record_audit(event, "deletion_job", **details),
    )
    container.backend = "infra"
    container.audit_source = recent_audit(handles.engine)
    features = platform.decisions.features
    if dev_users is not None and isinstance(features, FeatureBuilder):
        from theraflo_ads.infra.redis_features import RedisClickCounts

        container.dev_mode = True
        container.dev_users = dev_users
        container.lab = PolicyLab(
            platform=platform,
            features=features,
            # The Beta counts the features worker keeps in Redis (shared by every replica).
            counts=RedisClickCounts(handles.redis),
            linear=LinearBanditState(features.names),
            model_dir=settings.model_dir,
            serving={"bandit": "beta_ts"}.get(policy_name, policy_name),
            linear_learns=False,
        )
    if refresh_catalog:
        interval = float(os.environ.get("THERAFLO_ADS_CATALOG_REFRESH_S", "5"))
        CatalogRefresher(
            handles.engine,
            platform.catalog,
            threading.Lock(),
            priors=getattr(platform.cost, "priors", None),
            interval_s=interval,
        ).start()
    return container


def build_container_from_env() -> Container:
    backend = os.environ.get("THERAFLO_ADS_BACKEND", "memory").lower()
    return build_infra_container() if backend == "infra" else build_dev_container()


def assemble_container(
    platform: Platform,
    *,
    clock: Clock,
    hasher: UserHasher,
    profiles: ProfileStore,
    lock: AbstractContextManager,
    authenticate_user: Authenticator,
    authenticate_advertiser: Authenticator,
    authenticate_admin: Authenticator,
    authenticate_service: Authenticator,
    audit_sink: Callable[[str, str, dict[str, Any]], None] | None = None,
) -> Container:
    reports: list[tuple[str, str]] = []
    ingestor = ClientEventIngestor(
        signer=platform.signer,
        hasher=hasher,
        events=platform.events,
        catalog=platform.catalog,
        on_hidden=profiles.hide_ad,
        on_reported=lambda impression_id, ad_id: reports.append((impression_id, ad_id)),
    )
    return Container(
        clock=clock,
        platform=platform,
        hasher=hasher,
        ingestor=ingestor,
        profiles=profiles,
        authenticate_user=authenticate_user,
        authenticate_advertiser=authenticate_advertiser,
        authenticate_admin=authenticate_admin,
        authenticate_service=authenticate_service,
        lock=lock,
        reports=reports,
        audit_sink=audit_sink,
    )


def dev_authenticators(
    clock: Clock, test_users: Authenticator | None = None
) -> dict[str, Authenticator]:
    _refuse_production()
    users = table_authenticator(DEV_USER_TOKENS)
    if test_users is not None:
        users = chain_authenticators(users, test_users)
    session_key = os.environ.get("THERAFLO_ADS_SESSION_KEY")
    if session_key:
        users = chain_authenticators(
            users, session_authenticator(SessionTokens(bytes.fromhex(session_key)), clock)
        )
    return {
        "authenticate_user": users,
        "authenticate_advertiser": table_authenticator(DEV_ADVERTISER_KEYS),
        "authenticate_admin": table_authenticator(DEV_ADMIN_TOKENS),
        "authenticate_service": table_authenticator(DEV_SERVICE_TOKENS),
    }


def production_authenticators(clock: Clock) -> dict[str, Authenticator]:
    """Sessions verified with the shared session key; API keys stored only as digests."""
    session_key = key_from_env("THERAFLO_ADS_SESSION_KEY", required=True)

    def digests(name: str) -> Mapping[str, str]:
        return json.loads(os.environ.get(name, "{}"))

    return {
        "authenticate_user": session_authenticator(SessionTokens(session_key), clock),
        "authenticate_advertiser": hashed_key_authenticator(
            digests("THERAFLO_ADS_ADVERTISER_KEYS")
        ),
        "authenticate_admin": hashed_key_authenticator(digests("THERAFLO_ADS_ADMIN_KEYS")),
        "authenticate_service": hashed_key_authenticator(digests("THERAFLO_ADS_SERVICE_KEYS")),
    }


def key_from_env(name: str, *, required: bool = False) -> bytes:
    value = os.environ.get(name)
    if value:
        return bytes.fromhex(value)
    if required:
        raise RuntimeError(f"{name} must be set (hex); generate one with `theraflo-ads gen-keys`")
    # Dev fallback: a fresh random key per process, so tokens do not survive restarts.
    return secrets.token_bytes(32)


def _refuse_production() -> None:
    if os.environ.get("THERAFLO_ADS_ENV", "development").lower() == "production":
        raise RuntimeError("dev credentials and in-memory stores cannot run in production")
