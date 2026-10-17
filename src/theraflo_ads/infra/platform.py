"""Production assembly: the same Platform, wired to Postgres, Redis, and the outbox."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import redis
from sqlalchemy import Engine

from theraflo_ads.attribution.service import AttributionPolicy
from theraflo_ads.counters.frequency import FrequencyCaps
from theraflo_ads.domain import AdCatalog
from theraflo_ads.events.accounting import AccountingHandler
from theraflo_ads.events.bus import Handler, Message, MessageType
from theraflo_ads.events.service import EventService
from theraflo_ads.experiments.assignment import Experiment, HoldoutConfig
from theraflo_ads.infra.postgres import (
    PgBillingLedger,
    PgImpressionLog,
    make_engine,
    pg_uow_factory,
    ping,
)
from theraflo_ads.infra.redis_features import RedisInteractionStore
from theraflo_ads.infra.redis_stores import RedisBudgetStore, RedisFrequencyStore
from theraflo_ads.infra.review_store import decision_lookup, recent_lookup, request_lookup
from theraflo_ads.infra.settings import Settings
from theraflo_ads.infra.stores import PgSafetyPolicyStore, load_catalog, load_priors
from theraflo_ads.ml.features import FeatureBuilder
from theraflo_ads.platform import Platform
from theraflo_ads.policy.policies import Policy, UniformPolicy
from theraflo_ads.policy.service import DecisionService
from theraflo_ads.pricing import CategoryPriors, PriorCostEstimator
from theraflo_ads.safety.engine import ConstraintEngine
from theraflo_ads.safety.policy import default_safety_policy
from theraflo_ads.security.tokens import ImpressionTokenSigner
from theraflo_ads.storage.recorder import DecisionRecorder


@dataclass
class ChainContext:
    """What a policy chain may use on the serving path."""

    features: FeatureBuilder
    redis: redis.Redis
    settings: Settings


ChainFactory = Callable[[AdCatalog, CategoryPriors, ChainContext], Sequence[Policy]]


class OutboxBus:
    """Messages leave this process only through the transactional outbox; consumers run in
    worker processes (theraflo-ads worker <queue>)."""

    def publish(self, message: Message) -> None:
        raise RuntimeError("infra platform: messages are written to the outbox, not published")

    def subscribe(self, message_type: MessageType, handler: Handler) -> None:
        raise RuntimeError("infra platform: subscribe in a worker process")


@dataclass
class InfraHandles:
    engine: Engine
    redis: redis.Redis


def make_redis(settings: Settings) -> redis.Redis:
    return redis.Redis.from_url(
        settings.redis_url,
        socket_timeout=settings.redis_timeout_s,
        socket_connect_timeout=max(settings.redis_timeout_s, 0.2),
        health_check_interval=30,
        retry_on_timeout=False,  # fail fast; the request serves NO_AD instead of waiting
    )


def build_infra_platform(
    settings: Settings,
    *,
    chain_factory: ChainFactory,
    signer: ImpressionTokenSigner,
    holdouts: HoldoutConfig | None = None,
    caps: FrequencyCaps | None = None,
    rng: random.Random | None = None,
    experiment: Experiment | None = None,
    arm_chains: dict[str, Sequence[Policy]] | None = None,
    attribution_policy: AttributionPolicy | None = None,
    engine: Engine | None = None,
    redis_client: redis.Redis | None = None,
) -> tuple[Platform, InfraHandles]:
    engine = engine or make_engine(settings)
    redis_client = redis_client or make_redis(settings)
    attribution_policy = attribution_policy or AttributionPolicy()
    catalog = load_catalog(engine)
    priors = load_priors(engine)
    impression_log = PgImpressionLog(engine)
    frequency = RedisFrequencyStore(redis_client, impression_log, caps)
    budget = RedisBudgetStore(redis_client, PgBillingLedger(engine))
    safety = PgSafetyPolicyStore(engine, default_safety_policy())
    uow = pg_uow_factory(engine)
    cost = PriorCostEstimator(priors)
    features = FeatureBuilder(
        catalog,
        priors,
        interactions=RedisInteractionStore(redis_client),
        excluded_categories=safety.current().personalization_excluded_categories,
    )
    constraint_engine = ConstraintEngine(
        catalog=catalog, policies=safety, frequency=frequency, budget=budget, cost_estimator=cost
    )
    decisions = DecisionService(
        catalog=catalog,
        engine=constraint_engine,
        chain=chain_factory(catalog, priors, ChainContext(features, redis_client, settings)),
        uniform_policy=UniformPolicy(),
        holdouts=holdouts or HoldoutConfig(),
        frequency=frequency,
        budget=budget,
        cost_estimator=cost,
        signer=signer,
        sink=DecisionRecorder(uow, attribution_policy),
        rng=rng,
        experiment=experiment,
        arm_chains=arm_chains,
        features=features,
    )
    platform = Platform(
        catalog=catalog,
        safety=safety,
        impression_log=impression_log,
        frequency=frequency,
        budget=budget,
        engine=constraint_engine,
        decisions=decisions,
        uow=uow,
        bus=OutboxBus(),
        events=EventService(uow_factory=uow, policy=attribution_policy),
        accounting=AccountingHandler(uow_factory=uow, budget=budget, policy=attribution_policy),
        attribution_policy=attribution_policy,
        signer=signer,
        cost=cost,
        extra_health={"postgres": lambda: ping(engine)},
        decision_lookup=decision_lookup(engine),
        request_lookup=request_lookup(engine),
        recent_lookup=recent_lookup(engine),
    )
    return platform, InfraHandles(engine, redis_client)
