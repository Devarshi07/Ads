"""The online decision path (design doc: Online Request Flow, Failure Hierarchy).

Order matters: hard constraints → policy samples over the eligible set →
budget reservation and frequency increment → token → decision log. Every
fallback level sees the same pre-filtered eligible set, and anything that
goes wrong after the policy samples serves NO_AD and marks the decision
degraded so it never reaches OPE or training.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any, Protocol
from uuid import uuid4

from theraflo_ads.counters.budget import BudgetStore
from theraflo_ads.counters.frequency import CounterUnavailable, FrequencyStore, ImpressionRecord
from theraflo_ads.domain import NO_AD, Ad, AdCatalog, AdRequest, ConsentScope
from theraflo_ads.experiments.assignment import Experiment, Holdout, HoldoutConfig
from theraflo_ads.policy.decision import Decision, DecisionOutcome, ServedAd, ServedBy
from theraflo_ads.policy.policies import (
    ActionDistribution,
    Policy,
    sample,
    validate_distribution,
)
from theraflo_ads.pricing import budget_scope, max_event_cost
from theraflo_ads.safety.engine import FAILURE_BLOCKS, ConstraintEngine, CostEstimator
from theraflo_ads.security.tokens import ImpressionClaims, ImpressionTokenSigner

# Persists the decision (and, when served, its reward label) atomically.
DecisionSink = Callable[[DecisionOutcome], None]


class FeatureProvider(Protocol):
    """The same FeatureBuilder the ML scorer uses online (see theraflo_ads.ml.features)."""

    version: str

    def context(self, request: AdRequest) -> dict[str, Any]: ...

    def build(self, request: AdRequest, ads: Sequence[Ad]) -> list[dict[str, float]]: ...


@dataclass(frozen=True)
class DecisionConfig:
    token_ttl: timedelta = timedelta(minutes=10)
    feature_schema_version: str = "features_v1"


@dataclass
class _Draft:
    request: AdRequest
    holdout: Holdout
    cohort: str | None
    consent: ConsentScope = ConsentScope.CONTEXTUAL_ONLY
    safety_version: str = ""
    shadow: dict[str, dict[str, float]] = field(default_factory=dict)


class DecisionService:
    def __init__(
        self,
        *,
        catalog: AdCatalog,
        engine: ConstraintEngine,
        chain: Sequence[Policy],
        uniform_policy: Policy,
        holdouts: HoldoutConfig,
        frequency: FrequencyStore,
        budget: BudgetStore,
        cost_estimator: CostEstimator,
        signer: ImpressionTokenSigner,
        sink: DecisionSink,
        rng: random.Random | None = None,
        config: DecisionConfig | None = None,
        experiment: Experiment | None = None,
        arm_chains: Mapping[str, Sequence[Policy]] | None = None,
        id_factory: Callable[[], str] | None = None,
        features: FeatureProvider | None = None,
        shadow_policies: Sequence[Policy] = (),
    ) -> None:
        if not chain:
            raise ValueError("the fallback chain needs at least one policy")
        if experiment is not None:
            missing = {name for name, _ in experiment.arms} - set(arm_chains or {})
            if missing:
                raise ValueError(f"no policy chain for experiment arms: {sorted(missing)}")
        self._catalog = catalog
        self._engine = engine
        self._chain = tuple(chain)
        self._uniform = uniform_policy
        self._holdouts = holdouts
        self._frequency = frequency
        self._budget = budget
        self._cost = cost_estimator
        self._signer = signer
        self._sink = sink
        self._rng = rng or random.Random()
        self._config = config or DecisionConfig()
        self._experiment = experiment
        self._arm_chains = {arm: tuple(c) for arm, c in (arm_chains or {}).items()}
        self._new_id = id_factory or (lambda: uuid4().hex)
        self._features = features
        self._shadow = tuple(shadow_policies)
        self.sink_failures = 0
        self.shadow_failures = 0

    @property
    def chain(self) -> tuple[Policy, ...]:
        return self._chain

    @property
    def features(self) -> FeatureProvider | None:
        return self._features

    @property
    def holdouts(self) -> HoldoutConfig:
        return self._holdouts

    def set_chain(self, chain: Sequence[Policy]) -> None:
        """Swap the serving chain at runtime (the dev console's policy switch). A request
        already in flight keeps the chain it started with: `decide` reads it once."""
        if not chain:
            raise ValueError("the fallback chain needs at least one policy")
        self._chain = tuple(chain)

    def decide(self, request: AdRequest) -> DecisionOutcome:
        user_hash = request.user.user_hash
        holdout = self._holdouts.assign(user_hash)
        chain = self._chain
        cohort = None
        arm = None
        experiment = self._experiment
        if experiment is not None and not experiment.active(request.now):
            experiment = None
        if holdout is Holdout.NONE and experiment is not None:
            arm = experiment.assign(user_hash)
            cohort = f"{experiment.experiment_id}:{arm}"
            chain = self._arm_chains[arm]
        if experiment is not None and experiment.budget_split:
            request = replace(request, budget_slice=self._budget_slice(holdout, arm, cohort))
        draft = _Draft(request, holdout, cohort)

        eligibility = self._engine.evaluate(request)
        request = replace(request, frequency=eligibility.snapshot)
        if self._features is not None and eligibility.blocked is None:
            request = self._with_feature_context(request)
        draft.request = request
        draft.consent = eligibility.effective_consent
        draft.safety_version = eligibility.safety_policy_version
        if eligibility.blocked is not None:
            failure = eligibility.blocked in FAILURE_BLOCKS
            return self._finish(
                draft,
                served_by=ServedBy.NO_AD,
                blocked=eligibility.blocked.value,
                degraded=(f"blocked:{eligibility.blocked.value}",) if failure else (),
            )
        if holdout is Holdout.NO_ADS:
            return self._finish(draft, served_by=ServedBy.NO_ADS_HOLDOUT)
        if holdout is Holdout.UNIFORM_EXPLORATION:
            chain = (self._uniform,)

        eligible_ids = {ad.ad_id for ad in eligibility.eligible}
        draft.shadow = self._score_shadow(request, eligibility.eligible)
        degraded: list[str] = []
        for level in chain:
            try:
                dist = level.distribution(request, eligibility.eligible)
                validate_distribution(dist, eligible_ids)
            except Exception as exc:  # any failing level degrades to the next one
                degraded.append(f"{level.served_by.value}_failed:{type(exc).__name__}")
                continue
            return self._serve(draft, level, dist, sample(dist, self._rng), tuple(degraded))
        return self._finish(
            draft, served_by=ServedBy.NO_AD, degraded=(*degraded, "all_levels_failed")
        )

    def _serve(
        self,
        draft: _Draft,
        level: Policy,
        dist: ActionDistribution,
        action: str,
        degraded: tuple[str, ...],
    ) -> DecisionOutcome:
        policy = {"dist": dist, "policy_action": action, "policy_version": level.version}
        if action == NO_AD:
            return self._finish(draft, served_by=level.served_by, degraded=degraded, **policy)

        def dropped(reason: str) -> DecisionOutcome:
            return self._finish(
                draft, served_by=ServedBy.NO_AD, degraded=(*degraded, reason), **policy
            )

        request = draft.request
        ad = self._catalog.ads[action]
        campaign = self._catalog.campaign_for(ad)
        budget_key, budget_limit = budget_scope(campaign, request.budget_slice)
        try:
            reservation_id = self._budget.reserve(
                budget_key,
                budget_limit,
                self._cost(ad, campaign),
                request.now,
                min_headroom=max_event_cost(campaign),
            )
        except CounterUnavailable:
            return dropped("budget_unavailable")
        if reservation_id is None:
            return dropped("budget_race")
        impression_id = self._new_id()
        try:
            self._frequency.record(
                ImpressionRecord(
                    request.user.user_hash,
                    ad.ad_id,
                    campaign.advertiser_id,
                    request.now,
                    impression_id,
                )
            )
        except CounterUnavailable:
            self._release_quietly(reservation_id)
            return dropped("frequency_unavailable")

        expires_at = request.now + self._config.token_ttl
        token = self._signer.sign(
            ImpressionClaims(impression_id, ad.ad_id, request.user.user_hash, expires_at)
        )
        served = ServedAd(
            impression_id,
            ad,
            campaign,
            1,
            token,
            expires_at,
            reservation_id,
            budget_key,
            budget_limit,
        )
        return self._finish(
            draft,
            served_by=level.served_by,
            chosen=action,
            served=served,
            degraded=degraded,
            **policy,
        )

    def _finish(
        self,
        draft: _Draft,
        *,
        served_by: ServedBy,
        chosen: str = NO_AD,
        dist: ActionDistribution | None = None,
        policy_action: str | None = None,
        policy_version: str | None = None,
        blocked: str | None = None,
        degraded: tuple[str, ...] = (),
        served: ServedAd | None = None,
    ) -> DecisionOutcome:
        if dist is None:
            probabilities: Mapping[str, float] = {NO_AD: 1.0}
            stochastic = False
            predicted: Mapping[str, float] = {}
        else:
            probabilities = dict(dist.probabilities)
            stochastic = dist.stochastic
            predicted = dict(dist.predictions.get(chosen, {}))
        request = draft.request
        decision = Decision(
            request_id=request.request_id,
            session_id=request.session_id,
            user_hash=request.user.user_hash,
            placement=request.placement_id,
            consent_scope=draft.consent,
            holdout=draft.holdout,
            experiment_cohort=draft.cohort,
            eligible_actions=tuple(probabilities),
            action_probabilities=probabilities,
            policy_action=policy_action if dist is not None else None,
            chosen_action=chosen,
            position=served.position if served else None,
            served_by=served_by,
            stochastic=stochastic,
            degraded_reasons=degraded,
            blocked_reason=blocked,
            policy_version=policy_version,
            safety_policy_version=draft.safety_version,
            # The schema of `features` below: the provider's, when one builds them.
            feature_schema_version=(
                self._features.version
                if self._features is not None
                else self._config.feature_schema_version
            ),
            impression_id=served.impression_id if served else None,
            timestamp=request.now,
            predicted=predicted,
            features=self._chosen_features(request, served),
            feature_context=request.feature_context or {},
            shadow=draft.shadow,
        )
        outcome = DecisionOutcome(decision, served)
        try:
            self._sink(outcome)
        except Exception:
            # Without a decision record an impression cannot be attributed or
            # billed, so nothing is served.
            self.sink_failures += 1
            if served is not None:
                self._release_quietly(served.reservation_id)
            return DecisionOutcome(
                replace(
                    decision,
                    chosen_action=NO_AD,
                    impression_id=None,
                    position=None,
                    served_by=ServedBy.NO_AD,
                    degraded_reasons=(*decision.degraded_reasons, "decision_log_unavailable"),
                ),
                None,
            )
        return outcome

    def _with_feature_context(self, request: AdRequest) -> AdRequest:
        if self._features is None:
            raise RuntimeError("feature context requested without a feature provider")
        try:
            return replace(request, feature_context=self._features.context(request))
        except Exception:  # feature stores are optional for serving: policies fall back
            return request

    def _chosen_features(self, request: AdRequest, served: ServedAd | None) -> dict[str, float]:
        if served is None or self._features is None or request.feature_context is None:
            return {}
        try:
            return dict(self._features.build(request, [served.ad])[0])
        except Exception:  # features are for training; never fail a served ad over them
            return {}

    def _score_shadow(
        self, request: AdRequest, eligible: Sequence[Ad]
    ) -> dict[str, dict[str, float]]:
        """Candidate policies score the same eligible set; their output is logged, never served."""
        scores: dict[str, dict[str, float]] = {}
        for policy in self._shadow:
            try:
                scores[policy.version] = dict(policy.distribution(request, eligible).probabilities)
            except Exception:
                self.shadow_failures += 1
        return scores

    def _budget_slice(
        self, holdout: Holdout, arm: str | None, cohort: str | None
    ) -> tuple[str, float]:
        """Arms split what the uniform-exploration holdout does not use; the no-ads holdout
        spends nothing, so it needs no slice."""
        if self._experiment is None:
            raise RuntimeError("budget slices requested without an experiment")
        holdout_share = self._holdouts.uniform_exploration_share
        if arm is None or cohort is None:
            return ("holdouts", holdout_share)
        return (cohort, self._experiment.share(arm) * (1.0 - holdout_share))

    def _release_quietly(self, reservation_id: str) -> None:
        with contextlib.suppress(CounterUnavailable):  # the reservation expires on its own
            self._budget.release(reservation_id)
