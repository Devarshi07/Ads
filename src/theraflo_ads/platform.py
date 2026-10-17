"""Assembly of the whole decision platform (design doc: Suggested First Production Architecture).

`build_memory_platform` wires every component with in-memory adapters, for
tests, the simulator, and the dev API. The production build
(`theraflo_ads.infra.build_infra_platform`) wires Postgres, Redis, and
RabbitMQ adapters behind the same interfaces.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from theraflo_ads.attribution.service import (
    AttributionPolicy,
    AttributionService,
    ImpressionLabel,
)
from theraflo_ads.counters.budget import BudgetLedger, BudgetStore
from theraflo_ads.counters.frequency import (
    DurableImpressionLog,
    FrequencyCaps,
    FrequencyStore,
    ImpressionLog,
    InMemoryFrequencyStore,
)
from theraflo_ads.domain import AdCatalog
from theraflo_ads.events.accounting import AccountingHandler
from theraflo_ads.events.bus import Bus, InlineBus
from theraflo_ads.events.service import EventService
from theraflo_ads.experiments.assignment import Experiment, HoldoutConfig
from theraflo_ads.policy.decision import Decision
from theraflo_ads.policy.policies import Policy, UniformPolicy
from theraflo_ads.policy.service import DecisionConfig, DecisionService, FeatureProvider
from theraflo_ads.safety.engine import ConstraintEngine, CostEstimator
from theraflo_ads.safety.policy import SafetyPolicyStore, default_safety_policy
from theraflo_ads.security.tokens import ImpressionTokenSigner
from theraflo_ads.storage.memory import MemoryStore, memory_uow_factory
from theraflo_ads.storage.recorder import DecisionRecorder
from theraflo_ads.storage.uow import UnitOfWorkFactory


@dataclass
class Platform:
    catalog: AdCatalog
    safety: SafetyPolicyStore
    impression_log: DurableImpressionLog
    frequency: FrequencyStore
    budget: BudgetStore
    engine: ConstraintEngine
    decisions: DecisionService
    uow: UnitOfWorkFactory
    bus: Bus
    events: EventService
    accounting: AccountingHandler
    attribution_policy: AttributionPolicy
    signer: ImpressionTokenSigner
    cost: CostEstimator
    store: MemoryStore | None = None
    extra_health: Mapping[str, Callable[[], bool]] = field(default_factory=dict)
    # impression_id -> the decision that served it (Postgres for the infra platform).
    decision_lookup: Callable[[str], Decision | None] | None = None
    # request_id -> its decision, and the latest decisions (the dev console; Postgres too).
    request_lookup: Callable[[str], Decision | None] | None = None
    recent_lookup: Callable[[int], list[Decision]] | None = None

    @contextmanager
    def read_attribution(self) -> Iterator[AttributionService]:
        with self.uow() as uow:
            yield AttributionService(uow.attribution, self.attribution_policy)

    def attribution(self) -> AttributionService:
        """Direct read access to in-memory labels (tests, simulator)."""
        if self.store is None:
            raise RuntimeError("use read_attribution() with an external store")
        return AttributionService(self.store.attribution, self.attribution_policy)

    def attribution_labels(self) -> list[ImpressionLabel]:
        if self.store is None:
            raise RuntimeError("labels live in Postgres for this platform")
        return self.store.attribution.labels()

    def decision_log(self) -> list[Decision]:
        if self.store is None:
            raise RuntimeError("the decision log lives in Postgres for this platform")
        return self.store.decisions.records

    def find_decision(self, impression_id: str) -> Decision | None:
        if self.decision_lookup is not None:
            return self.decision_lookup(impression_id)
        if self.store is None:
            raise RuntimeError("no decision lookup configured")
        return self.store.decisions.by_impression(impression_id)

    def find_request(self, request_id: str) -> Decision | None:
        if self.request_lookup is not None:
            return self.request_lookup(request_id)
        if self.store is None:
            raise RuntimeError("no decision lookup configured")
        return self.store.decisions.by_request(request_id)

    def recent_decisions(self, limit: int = 20) -> list[Decision]:
        """Newest first."""
        if self.recent_lookup is not None:
            return self.recent_lookup(limit)
        if self.store is None:
            raise RuntimeError("no decision lookup configured")
        return self.store.decisions.recent(limit)

    def health(self) -> Mapping[str, bool]:
        checks = {
            "frequency_store": self.frequency.healthy(),
            "impression_log": self.impression_log.healthy(),
            "budget_store": self.budget.healthy(),
        }
        if self.store is not None:
            checks["system_of_record"] = self.store.healthy()
        for name, check in self.extra_health.items():
            checks[name] = check()
        return checks


def build_memory_platform(
    *,
    catalog: AdCatalog,
    cost: CostEstimator,
    chain: Sequence[Policy],
    signer: ImpressionTokenSigner,
    uniform_policy: Policy | None = None,
    holdouts: HoldoutConfig | None = None,
    caps: FrequencyCaps | None = None,
    rng: random.Random | None = None,
    experiment: Experiment | None = None,
    arm_chains: Mapping[str, Sequence[Policy]] | None = None,
    attribution_policy: AttributionPolicy | None = None,
    decision_config: DecisionConfig | None = None,
    keep_published: bool = True,
    features: FeatureProvider | None = None,
    shadow_policies: Sequence[Policy] = (),
) -> Platform:
    attribution_policy = attribution_policy or AttributionPolicy()
    safety = SafetyPolicyStore(default_safety_policy())
    impression_log = ImpressionLog()
    frequency = InMemoryFrequencyStore(impression_log, caps or FrequencyCaps())
    budget = BudgetLedger()
    store = MemoryStore()
    bus = InlineBus(keep_published=keep_published)
    uow = memory_uow_factory(store, bus)
    engine = ConstraintEngine(
        catalog=catalog, policies=safety, frequency=frequency, budget=budget, cost_estimator=cost
    )
    decisions = DecisionService(
        catalog=catalog,
        engine=engine,
        chain=chain,
        uniform_policy=uniform_policy or UniformPolicy(),
        holdouts=holdouts or HoldoutConfig(),
        frequency=frequency,
        budget=budget,
        cost_estimator=cost,
        signer=signer,
        sink=DecisionRecorder(uow, attribution_policy),
        rng=rng,
        config=decision_config,
        experiment=experiment,
        arm_chains=arm_chains,
        features=features,
        shadow_policies=shadow_policies,
    )
    accounting = AccountingHandler(uow_factory=uow, budget=budget, policy=attribution_policy)
    accounting.subscribe(bus)
    return Platform(
        catalog=catalog,
        safety=safety,
        impression_log=impression_log,
        frequency=frequency,
        budget=budget,
        engine=engine,
        decisions=decisions,
        uow=uow,
        bus=bus,
        events=EventService(uow_factory=uow, policy=attribution_policy),
        accounting=accounting,
        attribution_policy=attribution_policy,
        signer=signer,
        cost=cost,
        store=store,
    )
