"""The creative review pipeline (design doc: Controls 1-10, Phase 8 — LLM-Assisted Ad Review).

Goal: reduce human review load without increasing the false-approval rate. A
false approval is the critical error, so every step can only move a creative
toward "reject" or "human", and auto-approval needs every check to pass:

    1. injection scan          addresses the reviewer → reject
    2. deterministic rules     restricted term, vulnerability targeting, health-data request
                               → reject; weaker signals block auto-approval
    3. model review            structured output; unavailable / invalid → human (fail closed)
    4. claim grounding         every claim matched to substantiation; an unmatched health
                               claim → reject; other unmatched / partial claims → human
    5. dual check              high-risk categories and rule signals need a second,
                               independent review that agrees; disagreement → human
    6. abstention              anything short of a confident approve → human; health-adjacent
                               categories always → human before going live
    7. audit sampling          a fixed share of auto-approvals is also queued for a human audit

Every record carries the pipeline, prompt, model, taxonomy, and rules versions.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from theraflo_ads.domain import Clock, ReviewStatus, utc_now
from theraflo_ads.review.creative import (
    HARD_VIOLATIONS,
    Confidence,
    Creative,
    InvalidReviewOutput,
    Judgement,
    ReviewerUnavailable,
    SubstantiatedClaim,
    Verdict,
    Violation,
    judgement_from_dict,
)
from theraflo_ads.review.grounding import ClaimCheck, ClaimMatcher, ClaimStatus, ground_claims
from theraflo_ads.review.injection import scan_injection
from theraflo_ads.review.llm import ReviewModel
from theraflo_ads.review.rules import RULES_VERSION, rule_review
from theraflo_ads.review.taxonomy import OTHER, Taxonomy, default_taxonomy

PIPELINE_VERSION = "review_pipeline_v1"


class Outcome(StrEnum):
    AUTO_APPROVED = "auto_approved"
    AUTO_REJECTED = "auto_rejected"
    HUMAN_REVIEW = "human_review"


OUTCOME_STATUS = {
    Outcome.AUTO_APPROVED: ReviewStatus.APPROVED,
    Outcome.AUTO_REJECTED: ReviewStatus.REJECTED,
    Outcome.HUMAN_REVIEW: ReviewStatus.PENDING,
}


@dataclass(frozen=True)
class ReviewConfig:
    audit_rate: float = 0.05
    audit_salt: str = "audit_v1"
    dual_check: bool = True
    # Evals only: ask the model even when a deterministic check already rejected the creative,
    # so the model-alone ablation covers every case. The outcome is the same either way.
    always_consult_model: bool = False
    version: str = PIPELINE_VERSION


@dataclass(frozen=True)
class ReviewRecord:
    review_id: str
    ad_id: str
    advertiser_id: str
    outcome: Outcome
    category: str
    reasons: tuple[str, ...]
    violations: tuple[Violation, ...]
    claims: tuple[ClaimCheck, ...]
    injection_signals: tuple[str, ...]
    rule_signals: tuple[str, ...]
    judgements: tuple[Judgement, ...]
    audited: bool
    reviewed_at: datetime
    error: str | None = None
    pipeline_version: str = PIPELINE_VERSION
    taxonomy_version: str = ""
    rules_version: str = RULES_VERSION
    prompt_version: str | None = None
    model: str | None = None

    @property
    def status(self) -> ReviewStatus:
        return OUTCOME_STATUS[self.outcome]

    @property
    def hallucinated_claims(self) -> int:
        return sum(1 for c in self.claims if c.status is ClaimStatus.NOT_IN_CREATIVE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "ad_id": self.ad_id,
            "advertiser_id": self.advertiser_id,
            "outcome": self.outcome.value,
            "category": self.category,
            "reasons": list(self.reasons),
            "violations": [v.value for v in self.violations],
            "claims": [c.to_dict() for c in self.claims],
            "injection_signals": list(self.injection_signals),
            "rule_signals": list(self.rule_signals),
            "judgements": [j.to_dict() for j in self.judgements],
            "audited": self.audited,
            "reviewed_at": self.reviewed_at.isoformat(),
            "error": self.error,
            "pipeline_version": self.pipeline_version,
            "taxonomy_version": self.taxonomy_version,
            "rules_version": self.rules_version,
            "prompt_version": self.prompt_version,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewRecord:
        return cls(
            review_id=data["review_id"],
            ad_id=data["ad_id"],
            advertiser_id=data["advertiser_id"],
            outcome=Outcome(data["outcome"]),
            category=data["category"],
            reasons=tuple(data["reasons"]),
            violations=tuple(Violation(v) for v in data["violations"]),
            claims=tuple(
                ClaimCheck(
                    c["text"],
                    c["source"],
                    ClaimStatus(c["status"]),
                    c["health"],
                    c.get("claim_id"),
                    c.get("coverage", 0.0),
                )
                for c in data["claims"]
            ),
            injection_signals=tuple(data["injection_signals"]),
            rule_signals=tuple(data["rule_signals"]),
            judgements=tuple(judgement_from_dict(j) for j in data["judgements"]),
            audited=data["audited"],
            reviewed_at=datetime.fromisoformat(data["reviewed_at"]),
            error=data.get("error"),
            pipeline_version=data["pipeline_version"],
            taxonomy_version=data["taxonomy_version"],
            rules_version=data["rules_version"],
            prompt_version=data.get("prompt_version"),
            model=data.get("model"),
        )


@dataclass
class _Draft:
    reasons: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    judgements: list[Judgement] = field(default_factory=list)
    claims: list[ClaimCheck] = field(default_factory=list)
    reject: bool = False
    human: bool = False
    error: str | None = None

    def violate(self, violation: Violation, reason: str) -> None:
        if violation not in self.violations:
            self.violations.append(violation)
        self.reasons.append(reason)
        if violation in HARD_VIOLATIONS:
            self.reject = True
        else:
            self.human = True

    def escalate(self, reason: str) -> None:
        self.reasons.append(reason)
        self.human = True


ClaimSource = Callable[[str], Sequence[SubstantiatedClaim]]

# Rule signals a second review can clear (a weak restricted term in an otherwise clean ad).
DUAL_CHECK_SIGNALS = ("weak_restricted_term",)


class ReviewPipeline:
    def __init__(
        self,
        *,
        model: ReviewModel | None,
        claims: ClaimSource,
        taxonomy: Taxonomy | None = None,
        config: ReviewConfig | None = None,
        clock: Clock = utc_now,
        matcher: ClaimMatcher | None = None,
    ) -> None:
        self._model = model
        self._claims = claims
        self._taxonomy = taxonomy or default_taxonomy()
        self._config = config or ReviewConfig()
        self._clock = clock
        self._matcher = matcher or ClaimMatcher()

    @property
    def taxonomy(self) -> Taxonomy:
        return self._taxonomy

    def review(self, creative: Creative) -> ReviewRecord:
        draft = _Draft()
        tax = self._taxonomy

        scan = scan_injection(creative)
        if scan.flagged:
            draft.violate(Violation.PROMPT_INJECTION, "creative addresses the reviewer")
        elif scan.suspicious:
            draft.escalate("possible injection signal")

        rules = rule_review(creative, tax)
        for violation in sorted(rules.violations):
            draft.violate(violation, f"rules: {violation.value}")
        category = rules.implied_category or (
            creative.declared_category if tax.is_known(creative.declared_category) else OTHER
        )
        blocking_signals = [s for s in rules.signals if not s.startswith(DUAL_CHECK_SIGNALS)]
        for signal in blocking_signals:
            draft.escalate(f"rules: {signal}")
        needs_dual = bool(set(rules.signals) - set(blocking_signals))

        # A deterministic reject needs no model call: the outcome cannot become an approval.
        consult = not draft.reject or self._config.always_consult_model
        primary = self._judge(creative, "primary", draft) if consult else None
        if primary is not None:
            category = self._apply_judgement(primary, creative, category, rules, draft)

        claims = ground_claims(
            creative,
            self._claims(creative.advertiser_id),
            model_claims=primary.claims if primary else (),
            matcher=self._matcher,
        )
        draft.claims = claims
        for check in claims:
            if check.status is ClaimStatus.UNGROUNDED and check.health:
                draft.violate(Violation.UNSUBSTANTIATED_CLAIM, f"unsubstantiated: {check.text!r}")
            elif check.status in (ClaimStatus.UNGROUNDED, ClaimStatus.PARTIAL):
                draft.escalate(f"claim needs a human: {check.text!r} ({check.status.value})")

        high_risk = category in tax.high_risk or category == OTHER or needs_dual
        if primary is not None and not draft.reject and self._config.dual_check and high_risk:
            second = self._judge(creative, "adversarial", draft)
            if second is not None:
                self._apply_second(primary, second, draft)

        if not draft.reject and category in tax.health_adjacent:
            draft.escalate(f"health-adjacent category {category}: human sign-off required")
        if not draft.reject and not tax.is_known(category):
            draft.escalate(f"unknown category {category}")

        if draft.reject:
            outcome = Outcome.AUTO_REJECTED
        elif draft.human or primary is None:
            outcome = Outcome.HUMAN_REVIEW
        else:
            outcome = Outcome.AUTO_APPROVED
        audited = outcome is Outcome.AUTO_APPROVED and self._audit_sample(creative.ad_id)
        return ReviewRecord(
            review_id=uuid.uuid4().hex,
            ad_id=creative.ad_id,
            advertiser_id=creative.advertiser_id,
            outcome=outcome,
            category=category,
            reasons=tuple(draft.reasons),
            violations=tuple(draft.violations),
            claims=tuple(claims),
            injection_signals=scan.signals,
            rule_signals=rules.signals,
            judgements=tuple(draft.judgements),
            audited=audited,
            reviewed_at=self._clock(),
            error=draft.error,
            pipeline_version=self._config.version,
            taxonomy_version=tax.version,
            rules_version=RULES_VERSION,
            prompt_version=primary.prompt_version if primary else None,
            model=primary.model if primary else None,
        )

    def _judge(self, creative: Creative, variant: str, draft: _Draft) -> Judgement | None:
        if self._model is None:
            draft.escalate("no model reviewer configured")
            return None
        try:
            judgement = self._model.judge(creative, variant)
        except ReviewerUnavailable as exc:
            draft.error = f"reviewer_unavailable:{exc}"
            draft.escalate(f"{variant} reviewer unavailable ({exc}): fail closed")
            return None
        except InvalidReviewOutput as exc:
            draft.error = f"invalid_output:{exc}"
            draft.escalate(f"{variant} reviewer output invalid ({exc})")
            return None
        draft.judgements.append(judgement)
        return judgement

    def _apply_judgement(
        self, j: Judgement, creative: Creative, category: str, rules, draft: _Draft
    ) -> str:
        tax = self._taxonomy
        if j.addresses_reviewer:
            draft.violate(Violation.PROMPT_INJECTION, "model: creative addresses the reviewer")
        for violation in j.violations:
            draft.violate(violation, f"model: {violation.value}")
        if tax.is_restricted(j.category):
            draft.violate(Violation.RESTRICTED_CATEGORY, f"model labels it {j.category}")
        # The more restrictive label wins: rules can never be overridden toward "allowed".
        if rules.implied_category is None:
            if j.category != creative.declared_category:
                draft.escalate(
                    f"category mismatch: declared {creative.declared_category}, model {j.category}"
                )
            category = j.category
        if j.verdict is Verdict.REJECT and not j.violations:
            draft.escalate("model rejected without a policy label")
        elif j.verdict is Verdict.ESCALATE:
            draft.escalate("model escalated")
        if j.confidence is not Confidence.HIGH:
            draft.escalate(f"model confidence {j.confidence.value}")
        return category

    def _apply_second(self, first: Judgement, second: Judgement, draft: _Draft) -> None:
        if second.addresses_reviewer:
            draft.violate(Violation.PROMPT_INJECTION, "second review: addresses the reviewer")
        for violation in second.violations:
            if violation not in draft.violations:
                # One reviewer found a violation the other did not: a human decides.
                draft.escalate(f"second review found {violation.value}")
                draft.violations.append(violation)
        if self._taxonomy.is_restricted(second.category):
            draft.violate(
                Violation.RESTRICTED_CATEGORY, f"second review labels it {second.category}"
            )
        if second.verdict is not first.verdict or second.category != first.category:
            draft.escalate(
                f"reviewers disagree: {first.verdict.value}/{first.category} vs "
                f"{second.verdict.value}/{second.category}"
            )
        if second.confidence is not Confidence.HIGH:
            draft.escalate(f"second review confidence {second.confidence.value}")

    def _audit_sample(self, ad_id: str) -> bool:
        digest = hashlib.sha256(f"{self._config.audit_salt}:{ad_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self._config.audit_rate
