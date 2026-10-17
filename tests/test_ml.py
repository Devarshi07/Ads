"""ML layer: features, bundles, calibration, gates, registry, serving chain, guards."""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from tests.helpers import NOW, PRIORS, make_catalog, make_request, make_stack, make_user
from theraflo_ads.counters.frequency import FrequencySnapshot
from theraflo_ads.domain import ConsentScope
from theraflo_ads.ml import dataset
from theraflo_ads.ml.bundle import BundleError, load_bundle, save_bundle
from theraflo_ads.ml.evaluation import (
    GateConfig,
    classification_metrics,
    evaluate_gates,
    slice_regressions,
)
from theraflo_ads.ml.features import (
    FeatureBuilder,
    InMemoryInteractionStore,
    InteractionAggregator,
)
from theraflo_ads.ml.models import CalibratedModel, LightGBMModel, LogisticModel
from theraflo_ads.ml.post_training import export_parity, fit_calibrator
from theraflo_ads.ml.registry import ModelRegistry, RegistryError, Stage
from theraflo_ads.ml.scorer import MLScorer
from theraflo_ads.ml.serving import build_serving_chain
from theraflo_ads.ope.estimators import LoggedSample, snips, top_weight_events
from theraflo_ads.policy.decision import ServedBy
from theraflo_ads.policy.guard import CircuitBreaker, CircuitOpen, GuardedPolicy
from theraflo_ads.policy.policies import PopularityPolicy

CATALOG = make_catalog()
SAFE = [CATALOG.ads[a] for a in ("ad_cpm", "ad_cpc", "ad_cpa")]


def builder(interactions=None, excluded=frozenset()):
    return FeatureBuilder(CATALOG, PRIORS, interactions=interactions, excluded_categories=excluded)


def synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    p = 1 / (1 + np.exp(-(x @ np.array([1.0, -0.5, 0.25, 0.0]) - 3)))
    return x, (rng.random(n) < p).astype(float)


# ----- features -----


def test_contextual_users_get_no_behavioral_features():
    store = InMemoryInteractionStore()
    for _ in range(5):
        store.record_click("k1:user", "fitness")
        store.record_impression("k1:user", "fitness")
    features = builder(store)
    personalized = features.build(make_request(user=make_user()), SAFE)[0]
    contextual = features.build(
        make_request(user=make_user(consent=ConsentScope.CONTEXTUAL_ONLY)), SAFE
    )[0]
    assert personalized["user_total_clicks"] > 0
    assert contextual["user_total_clicks"] == 0
    assert contextual["user_cat_ctr"] == pytest.approx(PRIORS.p_ctr("fitness"))
    assert contextual["personalized"] == 0


def test_excluded_categories_are_never_personalized():
    store = InMemoryInteractionStore()
    for _ in range(10):
        store.record_click("k1:user", "fitness")
        store.record_impression("k1:user", "fitness")
    row = builder(store, excluded=frozenset({"fitness"})).build(make_request(), SAFE)[0]
    assert row["user_cat_impressions"] == 0
    assert row["user_total_clicks"] == 0  # excluded clicks do not leak through totals


def test_logged_context_rebuilds_identical_features():
    store = InMemoryInteractionStore()
    store.record_click("k1:user", "books")
    features = builder(store)
    snapshot = FrequencySnapshot({"ad_cpm": 2}, {"adv_a": 2}, 2, {})
    live = make_request()
    live = type(live)(**{**live.__dict__, "frequency": snapshot})
    context = features.context(live)
    replayed = type(live)(**{**live.__dict__, "frequency": None, "feature_context": context})
    store.record_click("k1:user", "books")  # the live store moves on; the log does not
    assert features.build(replayed, SAFE) == builder(InMemoryInteractionStore()).build(
        replayed, SAFE
    )
    assert features.build(replayed, SAFE)[0]["ad_exposures_24h"] == 2


def test_aggregator_records_only_consented_users():
    stack = make_stack(chain=lambda c: [PopularityPolicy(["ad_cpc"])])
    store = InMemoryInteractionStore()
    InteractionAggregator(store, stack.catalog).subscribe(stack.bus)
    personalized = stack.service.decide(make_request(request_id="a")).served
    stack.service.decide(
        make_request(
            user=make_user("k1:other", consent=ConsentScope.CONTEXTUAL_ONLY), request_id="b"
        )
    )
    stack.events.click(personalized.impression_id, NOW)
    assert store.get("k1:user").clicks == {"books": 1.0}
    assert store.get("k1:other") == type(store.get("k1:other"))()


def test_decisions_log_features_and_context():
    stack = make_stack(features=builder(InMemoryInteractionStore()))
    decision = stack.service.decide(make_request()).decision
    assert decision.features["personalized"] == 1.0
    assert decision.feature_context["version"] == builder().version
    json.dumps(decision.to_log())


# ----- models, bundles, calibration -----


@pytest.mark.parametrize("algorithm", ["logistic", "lightgbm"])
def test_bundle_round_trip_and_parity(tmp_path, algorithm):
    x, y = synthetic()
    names = ("a", "b", "c", "d")
    model = (
        LogisticModel.fit(x, y, names)
        if algorithm == "logistic"
        else LightGBMModel.fit(x, y, names, num_boost_round=50)
    )
    calibrated = fit_calibrator(CalibratedModel(model), x, y)
    save_bundle(tmp_path, name="ctr", version="t1", model=calibrated, feature_schema_version="s1")
    loaded = load_bundle(tmp_path, feature_schema_version="s1")
    assert np.allclose(loaded.model.predict(x[:50]), calibrated.predict(x[:50]))
    report = export_parity(calibrated, x[:100], x, y, feature_schema_version="s1")
    assert report.passed, report.reasons


def test_tampered_or_mismatched_bundles_are_refused(tmp_path):
    x, y = synthetic(500)
    model = CalibratedModel(LogisticModel.fit(x, y, ("a", "b", "c", "d")))
    key = b"m" * 32
    save_bundle(
        tmp_path,
        name="ctr",
        version="t1",
        model=model,
        feature_schema_version="s1",
        signing_key=key,
    )
    with pytest.raises(BundleError, match="feature schema"):
        load_bundle(tmp_path, feature_schema_version="s2")
    with pytest.raises(BundleError, match="bad signature"):
        load_bundle(tmp_path, signing_key=b"x" * 32)
    (tmp_path / "model.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BundleError, match="checksum"):
        load_bundle(tmp_path, signing_key=key)


def test_unsigned_bundle_refused_when_signatures_are_required(tmp_path):
    x, y = synthetic(500)
    save_bundle(
        tmp_path,
        name="ctr",
        version="t1",
        model=CalibratedModel(LogisticModel.fit(x, y, ("a", "b", "c", "d"))),
        feature_schema_version="s1",
    )
    with pytest.raises(BundleError, match="unsigned"):
        load_bundle(tmp_path, require_signature=True)


def test_calibration_fixes_a_miscalibrated_model():
    x, y = synthetic(20000)
    raw = CalibratedModel(LogisticModel.fit(x, y, ("a", "b", "c", "d"), c=0.0005))
    calibrated = fit_calibrator(raw, x[:10000], y[:10000])
    before = classification_metrics(y[10000:], raw.predict(x[10000:]))["ece"]
    after = classification_metrics(y[10000:], calibrated.predict(x[10000:]))["ece"]
    assert after < before


# ----- gates -----


def test_slice_gate_ignores_noise_and_catches_confident_regressions():
    rng = np.random.default_rng(1)
    n = 6000
    y = (rng.random(n) < 0.05).astype(float)
    frame = pd.DataFrame(
        {"segment": np.where(np.arange(n) < 5000, "big", "tiny"), "clicked": y, "baseline": 0.05}
    )
    good = frame.assign(candidate=np.where(y == 1, 0.06, 0.049))
    regressions = slice_regressions(good, "clicked", ["segment"], min_positives=30)
    assert evaluate_gates(
        classification_metrics(y, good.candidate),
        classification_metrics(y, good.baseline),
        regressions,
    ).passed
    bad = frame.assign(candidate=np.where(frame.segment == "big", 0.2, 0.05))
    regressions = slice_regressions(bad, "clicked", ["segment"], min_positives=30)
    result = evaluate_gates(
        classification_metrics(y, bad.candidate),
        classification_metrics(y, bad.baseline),
        regressions,
        GateConfig(max_ece=1.0, max_log_loss_regression=1.0),
    )
    assert not result.passed
    assert any("segment=big" in r for r in result.reasons)


# ----- registry -----


def test_registry_lifecycle_and_rollback(tmp_path):
    x, y = synthetic(500)
    bundle = save_bundle(
        tmp_path / "b",
        name="ctr",
        version="t",
        model=CalibratedModel(LogisticModel.fit(x, y, ("a", "b", "c", "d"))),
        feature_schema_version="s1",
    )
    registry = ModelRegistry(tmp_path / "registry")
    v1, v2 = registry.register("ctr", bundle), registry.register("ctr", bundle)
    with pytest.raises(RegistryError, match="not allowed"):
        registry.transition("ctr", v1, Stage.PRODUCTION, "skip the line")
    for version in (v1, v2):
        registry.transition("ctr", version, Stage.OFFLINE_VALIDATED, "gates")
        registry.promote_to_production("ctr", version, "test")
    assert registry.production("ctr").version == v2
    assert registry.versions("ctr")[v1] is Stage.RETIRED
    assert registry.rollback("ctr", "bad canary") == v1
    assert registry.production("ctr").version == v1
    with pytest.raises(RegistryError, match="no version to roll back"):
        registry.rollback("ctr", "again")


# ----- dataset -----


def test_examples_respect_label_maturity_and_degradation():
    stack = make_stack(
        chain=lambda c: [PopularityPolicy(["ad_cpm"])], features=builder(InMemoryInteractionStore())
    )
    early = stack.service.decide(make_request(request_id="early")).decision
    late_time = NOW + timedelta(days=5)
    stack.service.decide(make_request(request_id="late", now=late_time))
    labels = {lbl.impression_id: lbl for lbl in stack.platform.store.attribution.labels()}
    frame = dataset.examples(
        [d.to_log() for d in stack.decisions],
        labels,
        categories={a.ad_id: a.category for a in CATALOG.ads.values()},
        pricing={
            a.ad_id: CATALOG.campaign_for(a).pricing_model.value for a in CATALOG.ads.values()
        },
        cutoff=NOW + timedelta(days=8),
        maturity=timedelta(days=8),
    )
    assert list(frame.request_id) == [early.request_id]  # the late one is not mature yet
    assert "f_prior_ctr" in frame.columns


# ----- serving -----


def test_serving_chain_uses_production_models_and_falls_back(tmp_path):
    features = builder(InMemoryInteractionStore())
    rows = features.build(make_request(), SAFE * 200)
    x = np.array(features.matrix(rows))
    y = (np.random.default_rng(0).random(len(x)) < 0.05).astype(float)
    model = CalibratedModel(LogisticModel.fit(x, y, features.names))
    bundle = save_bundle(
        tmp_path / "b",
        name="ctr",
        version="t",
        model=model,
        feature_schema_version=features.version,
    )
    registry = ModelRegistry(tmp_path / "registry")
    version = registry.register("ctr", bundle)
    registry.transition("ctr", version, Stage.OFFLINE_VALIDATED, "gates")
    registry.promote_to_production("ctr", version, "test")

    serving = build_serving_chain(CATALOG, PRIORS, features, tmp_path / "registry")
    assert serving.chain[0].served_by is ServedBy.RANKER
    assert serving.model_versions == {"ctr": "t", "cvr": "prior"}
    stack = make_stack(chain=lambda c: serving.chain, features=features)
    assert stack.service.decide(make_request()).decision.served_by is ServedBy.RANKER

    (registry.production("ctr").path / "calibrator.json").write_text("{}", encoding="utf-8")
    broken = build_serving_chain(CATALOG, PRIORS, features, tmp_path / "registry")
    assert broken.chain[0].served_by is ServedBy.HEURISTIC
    assert "checksum" in broken.problems[0]


def test_mlscorer_hybrid_uses_prior_cvr():
    features = builder()
    x = np.array(features.matrix(features.build(make_request(), SAFE * 100)))
    y = (np.random.default_rng(0).random(len(x)) < 0.05).astype(float)
    ctr = type(
        "B", (), {"model": CalibratedModel(LogisticModel.fit(x, y, features.names)), "version": "t"}
    )()
    scored = MLScorer(CATALOG, features, ctr, None, priors=PRIORS).score(make_request(), SAFE)
    assert {s.ad.ad_id: s.p_cvr for s in scored}["ad_cpa"] == PRIORS.p_cvr("sleep_products")


# ----- guards -----


class Flaky:
    served_by = ServedBy.RANKER
    version = "flaky"

    def __init__(self):
        self.fail = True

    def distribution(self, request, eligible):
        if self.fail:
            raise TimeoutError("model server")
        return PopularityPolicy(["ad_cpm"]).distribution(request, eligible)


def test_circuit_breaker_opens_and_recovers():
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=3, reset_after_s=10, clock=lambda: now[0])
    flaky = Flaky()
    guarded = GuardedPolicy(flaky, breaker=breaker)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            guarded.distribution(make_request(), SAFE)
    with pytest.raises(CircuitOpen):
        guarded.distribution(make_request(), SAFE)
    now[0] = 11.0  # cooldown passed: one probe goes through
    flaky.fail = False
    guarded.distribution(make_request(), SAFE)
    assert breaker.state == "closed"


def test_latency_budget_breaches_count_as_failures():
    ticks = iter([0.0, 0.5])
    guarded = GuardedPolicy(
        PopularityPolicy(["ad_cpm"]), latency_budget_ms=40, clock=lambda: next(ticks)
    )
    with pytest.raises(Exception, match="ms >"):
        guarded.distribution(make_request(), SAFE)


def test_shadow_policies_are_logged_never_served():
    stack = make_stack(
        chain=lambda c: [PopularityPolicy(["ad_cpm"])],
        shadow_policies=[PopularityPolicy(["ad_cpc"], version="shadow_pop")],
    )
    decision = stack.service.decide(make_request()).decision
    assert decision.chosen_action == "ad_cpm"
    assert decision.shadow["shadow_pop"]["ad_cpc"] == 1.0


# ----- OPE reliability -----


def test_heavy_weights_with_no_reward_events_are_inconclusive():
    actions = ("a", "b")
    samples = [LoggedSample(actions, "a", 0.99, 0.01) for _ in range(5000)]
    samples += [LoggedSample(actions, "b", 0.01, 0.0) for _ in range(50)]  # rarely explored
    target = lambda s: {"a": 0.0, "b": 1.0}  # noqa: E731
    estimate = snips(samples, target, min_ess=10)
    assert not estimate.conclusive
    assert "reward events" in estimate.reason
    assert top_weight_events(np.array([1.0, 100.0]), np.array([1.0, 0.0])) == 0
