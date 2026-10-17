"""Review operations: the service and human queue, advertiser reports, narratives,
explanations, the eval suite, post-training export and gates, and the Claude adapter."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from tests.helpers import NOW, make_catalog, make_request, make_stack, make_user
from tests.test_review import StubModel, creative, payload
from theraflo_ads.attribution.service import ImpressionLabel, LabelState
from theraflo_ads.domain import ConsentScope, PricingModel, ReviewStatus
from theraflo_ads.experiments.assignment import Holdout
from theraflo_ads.ml.registry import ModelRegistry, Stage
from theraflo_ads.policy.decision import Decision, ServedBy
from theraflo_ads.review.creative import (
    InvalidReviewOutput,
    ReviewerUnavailable,
    SubstantiatedClaim,
    Verdict,
    Violation,
)
from theraflo_ads.review.evals import run_review_eval
from theraflo_ads.review.evalset import build_evalset, split_of
from theraflo_ads.review.explanations import TEMPLATES, explain
from theraflo_ads.review.metrics import cohens_kappa, wilson_interval
from theraflo_ads.review.narrator import (
    check_narrative,
    facts_table,
    narrate,
    number_tokens,
    unfaithful_numbers,
)
from theraflo_ads.review.pipeline import Outcome, ReviewConfig, ReviewPipeline
from theraflo_ads.review.post_training import (
    assert_no_user_data,
    export_post_training,
    labels_from_evalset,
    post_training_gate,
    register_reviewer,
)
from theraflo_ads.review.reports import ReportThresholds, build_advertiser_report
from theraflo_ads.review.simulated import SimulatedReviewerConfig, SimulatedReviewModel
from theraflo_ads.review.store import (
    InMemoryReviewRepository,
    QueueKind,
    ReviewService,
    SubmissionError,
)
from theraflo_ads.review.taxonomy import default_taxonomy

TAX = default_taxonomy()

# ----- service and human queue -----


def service_for(model, *, audit_rate=0.0, catalog=None):
    catalog = catalog or make_catalog()
    repo = InMemoryReviewRepository()
    pipeline = ReviewPipeline(
        model=model,
        claims=repo.claims_for,
        config=ReviewConfig(audit_rate=audit_rate),
        clock=lambda: NOW,
    )
    return ReviewService(
        repository=repo, pipeline=pipeline, catalog=catalog, clock=lambda: NOW
    ), catalog


def submission(ad_id="ad_new", **overrides):
    values = {
        "advertiser_id": "adv_b",
        "campaign_id": "c_cpc",
        "declared_category": "books",
        "body": "Cozy mysteries for rainy evenings.",
    } | overrides
    return creative(ad_id=ad_id, **values)


def test_submitted_creative_is_never_served_until_reviewed():
    service, catalog = service_for(StubModel(payload(category="books")))
    ad = service.submit(submission())
    assert ad.review_status is ReviewStatus.PENDING
    stack = make_stack(catalog=catalog)
    eligibility = stack.engine.evaluate(make_request())
    assert eligibility.rejections["ad_new"].value == "not_approved"
    [record] = service.process_pending()
    assert record.outcome is Outcome.AUTO_APPROVED
    assert catalog.ads["ad_new"].review_status is ReviewStatus.APPROVED
    assert "ad_new" in {a.ad_id for a in stack.engine.evaluate(make_request()).eligible}
    assert service.process_pending() == []


def test_review_assigns_the_reviewed_category_for_serving():
    service, catalog = service_for(StubModel())
    service.submit(submission(body="Spin to win at the casino."))
    service.process_pending()
    ad = catalog.ads["ad_new"]
    assert ad.review_status is ReviewStatus.REJECTED and ad.category == "gambling"


def test_submissions_are_checked_against_the_advertisers_campaigns():
    service, _ = service_for(StubModel())
    with pytest.raises(SubmissionError):
        service.submit(submission(campaign_id="c_cpm"))  # adv_a's campaign
    with pytest.raises(SubmissionError):
        service.submit(submission(ad_id="ad_cpm"))  # an existing ad of another campaign
    service.submit(submission())
    assert service.status("ad_new", "adv_b")["review_status"] == "pending"
    assert service.status("ad_new", "adv_a") is None  # no probing other advertisers' ads


def test_reviewer_outage_queues_for_humans_and_a_human_decides():
    service, catalog = service_for(StubModel(error=ReviewerUnavailable("down")))
    service.submit(submission())
    [record] = service.process_pending()
    assert record.outcome is Outcome.HUMAN_REVIEW
    [item] = service.queue(QueueKind.REVIEW)
    assert item.priority == 1 and item.ad_id == "ad_new"
    with pytest.raises(ValueError):
        service.decide(
            "ad_new",
            kind=QueueKind.REVIEW,
            verdict=Verdict.APPROVE,
            reviewer="h",
            category="gambling",
        )
    decision = service.decide(
        "ad_new", kind=QueueKind.REVIEW, verdict=Verdict.APPROVE, reviewer="human_1"
    )
    assert catalog.ads["ad_new"].review_status is ReviewStatus.APPROVED
    assert decision.machine_outcome == "human_review" and not decision.false_approval
    assert service.queue() == []
    with pytest.raises(KeyError):
        service.decide("ad_new", kind=QueueKind.REVIEW, verdict=Verdict.REJECT, reviewer="h")


def test_resubmission_takes_the_ad_out_of_rotation_and_drops_the_stale_queue_item():
    service, catalog = service_for(StubModel(payload(category="books", confidence="low")))
    service.submit(submission())
    service.process_pending()
    assert len(service.queue(QueueKind.REVIEW)) == 1
    catalog.ads["ad_new"] = replace(catalog.ads["ad_new"], review_status=ReviewStatus.APPROVED)
    service.submit(submission(headline="Edited"))
    assert catalog.ads["ad_new"].review_status is ReviewStatus.PENDING
    assert service.queue(QueueKind.REVIEW) == []
    assert [c.headline for c in service.repository.pending_review()] == ["Edited"]


def test_audit_that_overturns_an_auto_approval_takes_the_ad_down():
    service, catalog = service_for(StubModel(payload(category="books")), audit_rate=1.0)
    service.submit(submission())
    [record] = service.process_pending()
    assert record.outcome is Outcome.AUTO_APPROVED and record.audited
    assert catalog.ads["ad_new"].review_status is ReviewStatus.APPROVED
    decision = service.decide(
        "ad_new",
        kind=QueueKind.AUDIT,
        verdict=Verdict.REJECT,
        reviewer="auditor",
        violations=[Violation.MISLEADING],
    )
    assert decision.false_approval
    assert catalog.ads["ad_new"].review_status is ReviewStatus.REJECTED
    summary = service.audit_summary()
    assert summary["audited"] == 1 and summary["overturned"] == 1
    assert summary["ci95"][0] > 0


# ----- reports -----


def _report_inputs(n_users: int, conversions_per_user: int, placement_split: float = 0.5):
    labels, decisions = [], {}
    for i in range(n_users):
        placement = "home_feed" if i < n_users * placement_split else "content_library"
        rid = f"r{i}"
        labels.append(
            ImpressionLabel(
                f"i{i}",
                rid,
                "ad_cpc",
                "c_cpc",
                "adv_b",
                PricingModel.CPC,
                1.0,
                "c_cpc",
                50.0,
                None,
                NOW,
                click_id=f"c{i}",
                conversions=conversions_per_user,
                platform_revenue=1.0,
                state=LabelState.FINALIZED,
            )
        )
        decisions[rid] = Decision(
            rid,
            "s",
            f"user{i}",
            placement,
            ConsentScope.CONTEXTUAL_ONLY,
            Holdout.NONE,
            None,
            ("ad_cpc",),
            {"ad_cpc": 1.0},
            "ad_cpc",
            "ad_cpc",
            0,
            ServedBy.HEURISTIC,
            False,
            (),
            None,
            "h",
            "safety_v1",
            "features_v2",
            f"i{i}",
            NOW,
            feature_context={"device": "web"},
        )
    return labels, decisions


def test_report_hides_small_segments_and_their_complement():
    labels, decisions = _report_inputs(140, 1, placement_split=0.7)  # 98 / 42 users
    # Another advertiser's and reversed labels never count.
    labels.append(replace(labels[0], impression_id="x", advertiser_id="adv_a"))
    labels.append(replace(labels[1], impression_id="y", state=LabelState.REVERSED))
    report = build_advertiser_report(
        "adv_b", labels, decisions, start=NOW - timedelta(days=1), end=NOW + timedelta(days=1)
    )
    assert report.impressions == 140 and report.conversions == 140
    placements = [s for s in report.segments if s.dimension == "placement"]
    # content_library (42 users) is below 50; home_feed would equal total minus it: both hidden.
    assert placements == [] and report.suppressed["placement"] == 2
    assert [s.value for s in report.segments if s.dimension == "device"] == ["web"]


def test_report_shows_segments_above_both_thresholds():
    labels, decisions = _report_inputs(200, 1)
    report = build_advertiser_report(
        "adv_b", labels, decisions, start=NOW - timedelta(days=1), end=NOW + timedelta(days=1)
    )
    placements = {s.value: s for s in report.segments if s.dimension == "placement"}
    assert set(placements) == {"home_feed", "content_library"}
    strict = build_advertiser_report(
        "adv_b",
        labels,
        decisions,
        start=NOW - timedelta(days=1),
        end=NOW + timedelta(days=1),
        thresholds=ReportThresholds(min_users=50, min_conversions=150),
    )
    assert [s for s in strict.segments if s.dimension == "placement"] == []
    assert "user" not in json.dumps(report.to_dict()).replace("users", "")


# ----- narrator -----


class ScriptedNarrator:
    name = "scripted"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = 0

    def write(self, facts, guidance=""):
        self.calls += 1
        text = self.texts.pop(0)
        if isinstance(text, Exception):
            raise text
        return text


def _report():
    labels, decisions = _report_inputs(200, 1)
    return build_advertiser_report(
        "adv_b", labels, decisions, start=NOW - timedelta(days=1), end=NOW + timedelta(days=1)
    )


def test_number_tokens_compare_numbers_as_displayed():
    assert number_tokens("$1,234.50 and 12.5% on 2026-10-01") == ["1234.50", "12.5%"]
    facts = "Click-through rate: 1.52%\nSpend: $1,234.50"
    assert unfaithful_numbers("CTR was 1.5% on $1,234.50", facts) == ["1.5%"]


def test_faithful_model_text_is_used():
    report = _report()
    facts = facts_table(report)
    text = f"Your ads reached {report.impressions} impressions."
    assert check_narrative(text, facts) == []
    narrative = narrate(report, ScriptedNarrator(text))
    assert narrative.source == "model" and narrative.attempts == 1


def test_unfaithful_text_is_regenerated_once_then_templated():
    report = _report()
    narrator = ScriptedNarrator("CTR up 12% vs last month.", "Users who feel anxious clicked most.")
    narrative = narrate(report, narrator)
    assert narrator.calls == 2 and narrative.source == "template"
    assert unfaithful_numbers(narrative.text, facts_table(report)) == []
    assert any("not in source" in r for r in narrative.rejected)
    assert any("describes users" in r for r in narrative.rejected)
    fixed = narrate(report, ScriptedNarrator("CTR up 12%.", f"{report.clicks} clicks."))
    assert fixed.source == "model" and fixed.attempts == 2


def test_narrator_outage_falls_back_to_the_template():
    narrative = narrate(_report(), ScriptedNarrator(ReviewerUnavailable("down")))
    assert narrative.source == "template"
    assert narrate(_report(), None).source == "template"


# ----- explanations -----


def _served(stack, user, request_id="r1"):
    outcome = stack.service.decide(make_request(user=user, request_id=request_id))
    assert outcome.served is not None
    return outcome.decision


FORBIDDEN_WORDS = ("mood", "journal", "crisis", "suppress", "age", "wellness", "therapy", "health")


def test_explanation_is_templated_and_ad_safe():
    stack = make_stack()
    decision = _served(stack, make_user(consent=ConsentScope.CONTEXTUAL_ONLY))
    explanation = explain(decision, stack.catalog)
    assert explanation.factor_keys[0] == "placement"
    assert "interest" not in explanation.factor_keys
    text = " ".join(explanation.factors).lower()
    assert not any(word in text.split() for word in FORBIDDEN_WORDS)
    assert set(explanation.factors) <= {
        t.format(placement="your home feed", category=c)
        for t in TEMPLATES.values()
        for c in ("fitness", "books", "sleep products", "travel")
    }


def test_interest_is_only_claimed_when_a_personalizing_policy_chose_the_ad():
    stack = make_stack()
    base = _served(stack, make_user())
    features = {"user_cat_impressions": 1.0, "user_cat_ctr": 0.5, "prior_ctr": 0.02}
    ranked = replace(
        base,
        served_by=ServedBy.RANKER,
        features=features,
        action_probabilities={base.chosen_action: 1.0},
    )
    assert "interest" in explain(ranked, stack.catalog).factor_keys
    assert (
        "interest"
        not in explain(replace(ranked, served_by=ServedBy.HEURISTIC), stack.catalog).factor_keys
    )
    assert (
        "interest"
        not in explain(
            replace(ranked, consent_scope=ConsentScope.CONTEXTUAL_ONLY), stack.catalog
        ).factor_keys
    )
    rotated = replace(ranked, served_by=ServedBy.UNIFORM_HOLDOUT)
    keys = explain(rotated, stack.catalog).factor_keys
    assert "rotation" in keys and "interest" not in keys
    with pytest.raises(ValueError):
        explain(replace(base, impression_id=None), stack.catalog)


# ----- metrics and evals -----


def test_wilson_interval_and_kappa():
    low, high = wilson_interval(0, 100)
    assert low == 0.0 and 0.03 < high < 0.04
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0


def test_evalset_is_split_by_advertiser():
    evalset = build_evalset()
    by_split: dict[str, set[str]] = {}
    for case in evalset.cases:
        by_split.setdefault(split_of(case.creative.advertiser_id), set()).add(
            case.creative.advertiser_id
        )
    splits = list(by_split.values())
    assert all(not (a & b) for i, a in enumerate(splits) for b in splits[i + 1 :])
    assert {c.slice for c in evalset.cases} >= {"injection", "restricted_euphemism", "clean"}


def test_eval_suite_controls_hold_with_a_simulated_reviewer():
    evalset = build_evalset()
    report = run_review_eval(evalset, SimulatedReviewModel(), narrator_reports=10)
    assert report.false_approval.count == 0
    assert report.injection_approvals == 0 and report.health_adjacent_auto == 0
    assert report.claim_grounding_rate == 1.0
    assert report.numeric_faithfulness["faithful_rate"] == 1.0
    assert report.passed, report.gate_reasons
    # The ablation: the same model without the controls approves violating creatives.
    assert report.model_only["false_approval"]["count"] > 0
    rules = run_review_eval(evalset, None, narrator_reports=0)
    assert rules.false_approval.count == 0
    assert report.automation_rate > rules.automation_rate + 0.2


def test_controls_survive_a_reviewer_that_obeys_injected_instructions():
    naive = SimulatedReviewModel(SimulatedReviewerConfig(follows_injections=True))
    report = run_review_eval(build_evalset(), naive, narrator_reports=0)
    assert report.model_only["injection_approvals"] >= 15
    assert report.injection_approvals == 0 and report.false_approval.count == 0


def test_gates_fail_on_false_approvals():
    sloppy = SimulatedReviewModel(SimulatedReviewerConfig(miss_rate=1.0, mislabel_rate=1.0))
    report = run_review_eval(build_evalset(), sloppy, narrator_reports=0)
    assert report.false_approval.count > 0 and not report.passed
    assert report.false_approval_examples


# ----- post-training -----


def _judgements(evalset, model):
    p = ReviewPipeline(
        model=model,
        claims=evalset.claims_for,
        config=ReviewConfig(audit_rate=0.0, always_consult_model=True),
    )
    out = {}
    for case in evalset.cases:
        primary = [j for j in p.review(case.creative).judgements if j.variant == "primary"]
        if primary:
            out[case.creative.ad_id] = primary[0]
    return out


def test_export_writes_advertiser_split_sft_and_weighted_preference_pairs(tmp_path):
    evalset = build_evalset()
    naive = SimulatedReviewModel(SimulatedReviewerConfig(follows_injections=True))
    result = export_post_training(
        labels_from_evalset(evalset.cases), _judgements(evalset, naive), tmp_path
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contains_user_data"] is False and manifest["training_data_hash"]
    assert sum(c["sft"] for c in result.counts.values()) == len(evalset.cases)
    assert result.corrections["false_approval"] > 0
    pairs = [
        json.loads(line)
        for split in result.counts
        for line in (tmp_path / f"{split}.dpo.jsonl").read_text().splitlines()
    ]
    weights = {p["correction"]: p["weight"] for p in pairs}
    assert weights["false_approval"] == 3.0
    assert all(p["chosen"] != p["rejected"] for p in pairs)
    seen: dict[str, str] = {}
    for split in result.counts:
        for line in (tmp_path / f"{split}.sft.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert seen.setdefault(row["advertiser_id"], split) == split


def test_training_records_reject_user_data():
    assert_no_user_data({"id": "a", "input": "x", "target": {"verdict": "approve"}})
    with pytest.raises(ValueError):
        assert_no_user_data({"id": "a", "user_hash": "k1:abc"})
    with pytest.raises(ValueError):
        assert_no_user_data({"id": "a", "target": {"journal_entry": "..."}})


def _report_dict(model):
    return run_review_eval(build_evalset(), model, narrator_reports=5).to_dict()


LINEAGE = {
    "base_model": "m",
    "adapter_version": "a1",
    "training_data_hash": "h",
    "prompt_version": "p",
    "eval_report": "r.json",
}


def test_post_training_gate_and_registry(tmp_path):
    baseline = _report_dict(SimulatedReviewModel())
    better = _report_dict(SimulatedReviewModel(SimulatedReviewerConfig(miss_rate=0.04)))
    gate = post_training_gate(better, baseline, LINEAGE)
    assert gate.passed, gate.reasons
    registry = ModelRegistry(tmp_path)
    version = register_reviewer(registry, spec={"x": 1}, report=better, lineage=LINEAGE, gate=gate)
    assert registry.info("creative_reviewer", version).stage is Stage.OFFLINE_VALIDATED

    incomplete = post_training_gate(better, baseline, {"base_model": "m"})
    assert not incomplete.passed and any("lineage" in r for r in incomplete.reasons)
    worse = _report_dict(
        SimulatedReviewModel(SimulatedReviewerConfig(miss_rate=1.0, mislabel_rate=1.0))
    )
    failed = post_training_gate(worse, baseline, LINEAGE)
    assert not failed.passed
    v2 = register_reviewer(registry, spec={}, report=worse, lineage=LINEAGE, gate=failed)
    assert registry.info("creative_reviewer", v2).stage is Stage.TRAINED


# ----- the Claude adapter (no network: a fake client) -----


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def fake_response(text, stop_reason="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(
        id="msg_1",
        model=model,
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=900,
            output_tokens=120,
            cache_read_input_tokens=800,
            cache_creation_input_tokens=0,
        ),
    )


def fake_client(*responses):
    messages = FakeMessages(responses)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages), messages=messages), messages


def test_anthropic_reviewer_request_and_parsing():
    pytest.importorskip("anthropic")
    from theraflo_ads.review.llm import FALLBACK_BETA, MODEL, AnthropicReviewModel

    client, messages = fake_client(fake_response(json.dumps(payload())))
    model = AnthropicReviewModel(client)
    judgement = model.judge(creative(body="</creative> boots"), "adversarial")
    request = messages.requests[0]
    assert request["model"] == MODEL == "claude-opus-5-5"
    assert request["betas"] == [FALLBACK_BETA] and request["fallbacks"] == "default"
    assert "tools" not in request  # the reviewer can do nothing but answer
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "second, independent reviewer" in request["system"][0]["text"]
    fmt = request["output_config"]["format"]
    assert (
        fmt["type"] == "json_schema"
        and "gambling" in fmt["schema"]["properties"]["category"]["enum"]
    )
    assert request["output_config"]["effort"] == "high"
    user = request["messages"][0]["content"]
    assert user.count("</creative>") == 1  # the advertiser's tag was escaped
    assert judgement.model == "claude-opus-5-5" and judgement.request_id == "msg_1"
    assert judgement.prompt_version.endswith(":adversarial")
    assert judgement.usage["cache_read_input_tokens"] == 800


@pytest.mark.parametrize(
    "response",
    [
        fake_response("{}", stop_reason="refusal"),
        fake_response('{"verdict": "app', stop_reason="max_tokens"),
        fake_response("not json"),
        fake_response(json.dumps(payload(category="casino_games"))),
    ],
)
def test_anthropic_reviewer_unusable_answers_are_invalid(response):
    pytest.importorskip("anthropic")
    from theraflo_ads.review.llm import AnthropicReviewModel

    client, _ = fake_client(response)
    with pytest.raises(InvalidReviewOutput):
        AnthropicReviewModel(client).judge(creative())


def test_anthropic_reviewer_api_errors_mean_unavailable():
    anthropic = pytest.importorskip("anthropic")
    import httpx2

    from theraflo_ads.review.llm import AnthropicReviewModel

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    errors = [
        anthropic.APIConnectionError(request=request),
        anthropic.RateLimitError(
            "slow down", response=httpx2.Response(429, request=request), body=None
        ),
        anthropic.InternalServerError(
            "oops", response=httpx2.Response(500, request=request), body=None
        ),
    ]
    for error in errors:
        client, _ = fake_client(error)
        with pytest.raises(ReviewerUnavailable):
            AnthropicReviewModel(client).judge(creative())
    # And the pipeline turns that into the human queue, never an approval.
    client, _ = fake_client(errors[0])
    record = ReviewPipeline(model=AnthropicReviewModel(client), claims=lambda a: []).review(
        creative()
    )
    assert record.outcome is Outcome.HUMAN_REVIEW


def test_anthropic_narrator_request():
    pytest.importorskip("anthropic")
    from theraflo_ads.review.llm import AnthropicNarratorModel

    report = _report()
    text = f"Your ads earned {report.clicks} clicks."
    client, messages = fake_client(fake_response(text))
    narrative = narrate(report, AnthropicNarratorModel(client))
    assert narrative.source == "model"
    request = messages.requests[0]
    assert "<metrics>" in request["messages"][0]["content"]
    assert request["output_config"] == {"effort": "medium"}


def test_claims_registry_is_per_advertiser():
    repo = InMemoryReviewRepository()
    repo.add_claim(SubstantiatedClaim("c1", "adv_x", "text", "doc"))
    assert repo.claims_for("adv_x")[0].claim_id == "c1" and repo.claims_for("adv_y") == []
    assert TAX.display("buy_now_pay_later") == "buy now, pay later"
