"""Simulation loop. Drives the real platform (constraint engine, policies, counters, budget,
event service, accounting worker, attribution) and checks every served ad against the
safety rules independently of the constraint engine (an oracle check)."""

from __future__ import annotations

import heapq
import itertools
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from theraflo_ads.attribution.service import AttributionPolicy
from theraflo_ads.counters.budget import BudgetStore
from theraflo_ads.counters.frequency import FrequencyCaps
from theraflo_ads.domain import Ad, AdRequest, AgeBand, Campaign, ReviewStatus
from theraflo_ads.events.bus import Bus
from theraflo_ads.events.service import EventService
from theraflo_ads.experiments.assignment import Experiment, Holdout, HoldoutConfig
from theraflo_ads.ml.features import FeatureBuilder, InMemoryInteractionStore, InteractionAggregator
from theraflo_ads.platform import Platform, build_memory_platform
from theraflo_ads.policy.decision import Decision, ServedBy
from theraflo_ads.policy.policies import (
    EpsilonGreedyPolicy,
    Policy,
    PopularityPolicy,
    UniformPolicy,
)
from theraflo_ads.policy.scoring import HeuristicScorer
from theraflo_ads.pricing import PriorCostEstimator
from theraflo_ads.safety.engine import restricted_categories
from theraflo_ads.safety.policy import AD_FREE_PLACEMENTS, SafetyPolicy, default_safety_policy
from theraflo_ads.security.tokens import ImpressionTokenSigner
from theraflo_ads.simulator.world import (
    IMPULSE_CATEGORIES,
    PLACEMENT_MIX,
    SIM_TOKEN_KEY,
    SimConfig,
    SimUser,
    World,
    build_world,
)

STATELESS_CAPS = FrequencyCaps(
    per_ad_24h=10**9,
    per_advertiser_24h=10**9,
    per_user_24h=10**9,
    min_gap_same_ad=timedelta(0),
)


@dataclass(frozen=True)
class PolicySetup:
    chain: Sequence[Policy]
    uniform: Policy = field(default_factory=UniformPolicy)
    experiment: Experiment | None = None
    arm_chains: Mapping[str, Sequence[Policy]] | None = None
    shadow: Sequence[Policy] = ()
    # Consumers to attach to the platform bus (e.g. a bandit updater).
    subscribers: Sequence[Callable[[Bus], None]] = ()


PolicyFactory = Callable[[World], PolicySetup | tuple[Sequence[Policy], Policy]]


def default_policies(world: World) -> PolicySetup:
    """Phase 1 chain: heuristic with ε-exploration, then popularity, then NO_AD."""
    scorer = HeuristicScorer(world.catalog, world.priors)
    heuristic = EpsilonGreedyPolicy(scorer, epsilon=world.config.epsilon)
    return PolicySetup((heuristic, popularity_fallback(world)))


def popularity_fallback(world: World) -> PopularityPolicy:
    by_priority = sorted(
        world.catalog.ads.values(),
        key=lambda ad: (-world.catalog.campaign_for(ad).priority, ad.ad_id),
    )
    return PopularityPolicy([ad.ad_id for ad in by_priority])


@dataclass
class CohortStats:
    opportunities: int = 0
    served: int = 0
    impulse_served: int = 0

    @property
    def ad_load(self) -> float:
        return self.served / self.opportunities if self.opportunities else 0.0

    @property
    def impulse_share(self) -> float:
        return self.impulse_served / self.served if self.served else 0.0


@dataclass
class SimResult:
    world: World
    platform: Platform = field(repr=False)
    decisions: list[Decision]
    rewards: dict[str, float]
    clicks: dict[str, int]
    violations: Counter[str]
    cohorts: dict[str, CohortStats]
    retention: dict[str, tuple[int, int]]
    conversion_outcomes: Counter[str]
    served_by: Counter[str]
    blocked: Counter[str]
    platform_revenue: float
    billed: float
    overdelivery: float
    final_time: datetime

    @property
    def budget(self) -> BudgetStore:
        return self.platform.budget

    def policy_value(self, decisions: Sequence[Decision] | None = None) -> float:
        """Mean reward over decisions where a serving policy ran (blocked requests and the
        no-ads holdout excluded). This is the estimand OPE targets."""
        ran = [d for d in (decisions or self.decisions) if policy_ran(d)]
        return sum(self.rewards.get(d.request_id, 0.0) for d in ran) / len(ran) if ran else 0.0


def run_simulation(config: SimConfig, policies: PolicyFactory | None = None) -> SimResult:
    world = build_world(config)
    run_seed = config.seed if config.run_seed is None else config.run_seed
    rng = random.Random(run_seed + 1)
    caps = STATELESS_CAPS if config.stateless else FrequencyCaps()
    interactions = InMemoryInteractionStore()
    excluded = (
        default_safety_policy().personalization_excluded_categories
        if config.personalization_exclusions
        else frozenset()
    )
    world.features = FeatureBuilder(
        world.catalog, world.priors, interactions=interactions, excluded_categories=excluded
    )
    setup = (policies or default_policies)(world)
    if not isinstance(setup, PolicySetup):
        setup = PolicySetup(*setup)
    holdouts = HoldoutConfig(
        uniform_exploration_share=config.uniform_holdout_share,
        no_ads_share=config.no_ads_holdout_share,
    )
    platform = build_memory_platform(
        catalog=world.catalog,
        cost=PriorCostEstimator(world.priors),
        chain=setup.chain,
        uniform_policy=setup.uniform,
        signer=ImpressionTokenSigner(SIM_TOKEN_KEY),
        holdouts=holdouts,
        caps=caps,
        rng=random.Random(run_seed + 2),
        experiment=setup.experiment,
        arm_chains=setup.arm_chains,
        attribution_policy=AttributionPolicy(click_window=config.click_window),
        keep_published=False,
        features=world.features,
        shadow_policies=setup.shadow,
    )
    InteractionAggregator(interactions, world.catalog).subscribe(platform.bus)
    for subscribe in setup.subscribers:
        subscribe(platform.bus)
    events = platform.events

    pending: list[tuple[datetime, int, Callable[[], None]]] = []
    sequence = itertools.count()

    def schedule(at: datetime, action: Callable[[], None]) -> None:
        heapq.heappush(pending, (at, next(sequence), action))

    def drain(until: datetime) -> None:
        while pending and pending[0][0] <= until:
            heapq.heappop(pending)[2]()

    outcomes: Counter[str] = Counter()
    clicks: dict[str, int] = {}
    violations: Counter[str] = Counter()
    cohorts = {"vulnerable": CohortStats(), "other": CohortStats()}
    served_log: dict[str, list[tuple[datetime, str, str]]] = defaultdict(list)
    policy = platform.safety.current()
    request_ids = itertools.count()
    order_ids = itertools.count()
    placements = [p for p, _ in PLACEMENT_MIX]
    weights = [w for _, w in PLACEMENT_MIX]

    for day in range(config.n_days):
        day_start = config.start + timedelta(days=day)
        opportunities: list[tuple[datetime, SimUser, str, str]] = []
        for user in world.users:
            if user.churned or rng.random() >= config.visit_prob:
                continue
            start = day_start + timedelta(seconds=rng.uniform(0, 86_000))
            visit = rng.choices(placements, weights=weights, k=config.opportunities_per_visit)
            if rng.random() < config.crisis_prob:
                visit.insert(rng.randrange(len(visit) + 1), "crisis_resources")
            session_id = f"s{user.index}-{day}"
            for k, placement in enumerate(visit):
                opportunities.append(
                    (start + timedelta(minutes=2 * k), user, placement, session_id)
                )
        opportunities.sort(key=lambda o: (o[0], o[1].index))

        ads_today: Counter[int] = Counter()
        for when, user, placement, session_id in opportunities:
            drain(when)
            request = AdRequest(
                f"r{next(request_ids)}", session_id, placement, user.context(), when
            )
            outcome = platform.decisions.decide(request)
            decision = outcome.decision
            if placement == "crisis_resources":
                user.suppressed_until = when + policy.suppression_window

            cohort = cohorts["vulnerable" if user.vulnerable else "other"]
            if _ad_opportunity(user, placement, when, policy.ad_placements, holdouts):
                cohort.opportunities += 1

            served = outcome.served
            if served is None:
                continue
            ad, campaign = served.ad, served.campaign
            cohort.served += 1
            cohort.impulse_served += ad.category in IMPULSE_CATEGORIES
            ads_today[user.index] += 1
            violations.update(
                _oracle_violations(
                    user, ad, campaign, placement, when, policy, caps, served_log[user.user_hash]
                )
            )
            served_log[user.user_hash].append((when, ad.ad_id, campaign.advertiser_id))

            exposures = user.exposures(when)
            user.recent.append(when)
            if rng.random() >= config.viewable_rate:
                continue
            events.viewable(served.impression_id, ad.ad_id, when + timedelta(seconds=2))
            if rng.random() >= world.p_click(user, ad, exposures, when):
                continue
            clicked_at = when + timedelta(seconds=30)
            click_id = events.click(served.impression_id, clicked_at)
            clicks[decision.request_id] = 1
            if rng.random() >= world.p_convert(user, ad):
                continue
            occurred = clicked_at + timedelta(
                days=rng.expovariate(1.0 / config.mean_conversion_delay_days)
            )
            order_id = f"o{next(order_ids)}"
            postback = _postback(events, outcomes, campaign.advertiser_id, click_id, order_id)
            value = rng.lognormvariate(3.5, 0.6)
            schedule(occurred + timedelta(minutes=5), postback(value, occurred))
            if rng.random() < 0.02:  # advertisers occasionally resend postbacks
                schedule(occurred + timedelta(hours=1), postback(value, occurred))
            if rng.random() < 0.05:  # refunds
                refunded_at = occurred + timedelta(days=3)
                schedule(
                    refunded_at,
                    _reversal(events, outcomes, campaign.advertiser_id, order_id, refunded_at),
                )

        if not config.stateless:
            for user in world.users:
                churn = 0.004 + 0.003 * ads_today[user.index]
                if not user.churned and rng.random() < churn:
                    user.churned = True

    attribution = platform.attribution()
    end = config.start + timedelta(days=config.n_days)
    final = end + platform.attribution_policy.maturity_delay + timedelta(days=30)
    drain(final)
    attribution.finalize(final)
    rewards = attribution.rewards_by_request(final)

    retention: dict[str, tuple[int, int]] = {}
    for holdout in Holdout:
        members = [u for u in world.users if holdouts.assign(u.user_hash) is holdout]
        retention[holdout.value] = (sum(not u.churned for u in members), len(members))

    decisions = platform.decision_log()
    budget = platform.budget
    keys = set(world.catalog.campaigns)
    if platform.store is not None:
        keys |= {label.budget_key for label in platform.store.attribution.labels()}
    days = [(config.start + timedelta(days=d)).date() for d in range(config.n_days + 40)]
    return SimResult(
        world=world,
        platform=platform,
        decisions=decisions,
        rewards=rewards,
        clicks=clicks,
        violations=violations,
        cohorts=cohorts,
        retention=retention,
        conversion_outcomes=outcomes,
        served_by=Counter(d.served_by.value for d in decisions),
        blocked=Counter(d.blocked_reason for d in decisions if d.blocked_reason),
        platform_revenue=sum(rewards.values()),
        billed=sum(budget.billed(k, day) for k in keys for day in days),
        overdelivery=sum(budget.overdelivery(k, day) for k in keys for day in days),
        final_time=final,
    )


def policy_ran(decision: Decision) -> bool:
    return decision.blocked_reason is None and decision.served_by is not ServedBy.NO_ADS_HOLDOUT


def _ad_opportunity(
    user: SimUser,
    placement: str,
    when: datetime,
    ad_placements: frozenset[str],
    holdouts: HoldoutConfig,
) -> bool:
    """An opportunity where serving an ad is allowed at all, so cohorts compare like for like."""
    suppressed = user.suppressed_until is not None and when < user.suppressed_until
    return (
        placement in ad_placements
        and user.age_band is not AgeBand.UNDER_13
        and not suppressed
        and holdouts.assign(user.user_hash) is not Holdout.NO_ADS
    )


def _oracle_violations(
    user: SimUser,
    ad: Ad,
    campaign: Campaign,
    placement: str,
    when: datetime,
    policy: SafetyPolicy,
    caps: FrequencyCaps,
    history: list[tuple[datetime, str, str]],
) -> list[str]:
    found = []
    if placement in AD_FREE_PLACEMENTS or placement not in policy.ad_placements:
        found.append("ad_free_placement")
    if user.age_band is AgeBand.UNDER_13:
        found.append("under_13")
    if user.suppressed_until is not None and when < user.suppressed_until:
        found.append("suppression_window")
    if ad.category in restricted_categories(policy, user.age_band):
        found.append("restricted_category")
    if ad.review_status is not ReviewStatus.APPROVED:
        found.append("not_approved")
    if not campaign.active:
        found.append("inactive_campaign")
    window = [h for h in history if h[0] > when - caps.window]
    if sum(h[1] == ad.ad_id for h in window) >= caps.per_ad_24h:
        found.append("per_ad_cap")
    if sum(h[2] == campaign.advertiser_id for h in window) >= caps.per_advertiser_24h:
        found.append("per_advertiser_cap")
    if len(window) >= caps.per_user_24h:
        found.append("per_user_cap")
    if any(h[1] == ad.ad_id and when - h[0] < caps.min_gap_same_ad for h in window):
        found.append("min_gap")
    return found


def _postback(
    events: EventService,
    outcomes: Counter[str],
    advertiser_id: str,
    click_id: str,
    order_id: str,
) -> Callable[[float, datetime], Callable[[], None]]:
    def make(value: float, occurred: datetime) -> Callable[[], None]:
        def deliver() -> None:
            outcome = events.conversion(
                advertiser_id=advertiser_id,
                click_id=click_id,
                order_id=order_id,
                value=value,
                occurred_at=occurred,
                received_at=occurred + timedelta(minutes=5),
            )
            outcomes[outcome.value] += 1

        return deliver

    return make


def _reversal(
    events: EventService,
    outcomes: Counter[str],
    advertiser_id: str,
    order_id: str,
    at: datetime,
) -> Callable[[], None]:
    def deliver() -> None:
        if events.reversal(advertiser_id=advertiser_id, order_id=order_id, at=at):
            outcomes["reversed"] += 1

    return deliver
