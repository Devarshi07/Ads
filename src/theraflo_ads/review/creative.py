"""Creative review types (design doc: Creative Review, LLM Components & Hallucination Controls).

A reviewer's output is a `Judgement`: enumerated labels only, validated against
a schema. Anything that does not validate (unknown label, wrong type, extra
field) raises `InvalidReviewOutput` and the creative goes to the human queue.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from theraflo_ads.review.taxonomy import Taxonomy

MAX_CLAIMS = 20
MAX_CLAIM_CHARS = 300
MAX_RATIONALE_CHARS = 1000


@dataclass(frozen=True)
class Creative:
    """Advertiser-supplied content. Every text field is untrusted data."""

    ad_id: str
    advertiser_id: str
    campaign_id: str
    declared_category: str
    headline: str
    body: str
    landing_url: str
    landing_text: str = ""
    submitted_at: datetime | None = None

    def texts(self) -> dict[str, str]:
        return {"headline": self.headline, "body": self.body, "landing_page": self.landing_text}

    def full_text(self) -> str:
        return "\n".join(t for t in self.texts().values() if t)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["submitted_at"] = self.submitted_at.isoformat() if self.submitted_at else None
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Creative:
        values = dict(data)
        if values.get("submitted_at"):
            values["submitted_at"] = datetime.fromisoformat(values["submitted_at"])
        return cls(**{k: values[k] for k in cls.__dataclass_fields__ if k in values})


@dataclass(frozen=True)
class SubstantiatedClaim:
    """A claim the advertiser has documented evidence for, reviewed by policy owners."""

    claim_id: str
    advertiser_id: str
    text: str
    evidence: str


class Verdict(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


class Violation(StrEnum):
    RESTRICTED_CATEGORY = "restricted_category"
    UNSUBSTANTIATED_CLAIM = "unsubstantiated_claim"
    VULNERABILITY_TARGETING = "vulnerability_targeting"
    MANIPULATIVE_URGENCY = "manipulative_urgency"
    MISLEADING = "misleading"
    PERSONAL_DATA_REQUEST = "personal_data_request"
    PROMPT_INJECTION = "prompt_injection"
    LANDING_MISMATCH = "landing_mismatch"


# Found by any reviewer, these reject the creative outright. The others escalate.
HARD_VIOLATIONS = frozenset(
    {
        Violation.RESTRICTED_CATEGORY,
        Violation.UNSUBSTANTIATED_CLAIM,
        Violation.VULNERABILITY_TARGETING,
        Violation.PERSONAL_DATA_REQUEST,
        Violation.PROMPT_INJECTION,
    }
)


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InvalidReviewOutput(Exception):
    """The reviewer answered, but not with a valid judgement (→ human queue)."""


class ReviewerUnavailable(Exception):
    """The reviewer could not be reached (→ human queue; nothing is auto-approved)."""


@dataclass(frozen=True)
class Judgement:
    verdict: Verdict
    category: str
    violations: tuple[Violation, ...]
    claims: tuple[str, ...]
    addresses_reviewer: bool
    confidence: Confidence
    rationale: str
    # Provenance (design doc: Controls → 10. Versioning).
    model: str
    prompt_version: str
    variant: str = "primary"
    request_id: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        data["violations"] = [v.value for v in self.violations]
        data["confidence"] = self.confidence.value
        return data

    def payload(self) -> dict[str, Any]:
        """The schema fields only: what a reviewer is asked to produce (post-training targets)."""
        return {
            "verdict": self.verdict.value,
            "category": self.category,
            "violations": [v.value for v in self.violations],
            "claims": list(self.claims),
            "addresses_reviewer": self.addresses_reviewer,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
        }


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    category: str
    violations: list[Violation]
    claims: list[str] = Field(max_length=MAX_CLAIMS)
    addresses_reviewer: bool
    confidence: Confidence
    rationale: str = Field(max_length=MAX_RATIONALE_CHARS)


def judgement_schema(taxonomy: Taxonomy) -> dict[str, Any]:
    """JSON schema for structured outputs. Only enum/type keywords: range checks happen in
    `parse_judgement`, which runs on every response regardless of what the API enforced."""
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": [v.value for v in Verdict]},
            "category": {"type": "string", "enum": sorted(taxonomy.labels)},
            "violations": {
                "type": "array",
                "items": {"type": "string", "enum": [v.value for v in Violation]},
            },
            "claims": {"type": "array", "items": {"type": "string"}},
            "addresses_reviewer": {"type": "boolean"},
            "confidence": {"type": "string", "enum": [c.value for c in Confidence]},
            "rationale": {"type": "string"},
        },
        "required": [
            "verdict",
            "category",
            "violations",
            "claims",
            "addresses_reviewer",
            "confidence",
            "rationale",
        ],
        "additionalProperties": False,
    }


def parse_judgement(
    data: Any,
    taxonomy: Taxonomy,
    *,
    model: str,
    prompt_version: str,
    variant: str = "primary",
    request_id: str | None = None,
    usage: Mapping[str, int] | None = None,
) -> Judgement:
    try:
        payload = _Payload.model_validate(data)
    except ValidationError as exc:
        raise InvalidReviewOutput(f"schema: {exc.error_count()} errors") from exc
    if not taxonomy.is_known(payload.category):
        raise InvalidReviewOutput(f"unknown category label {payload.category!r}")
    claims = tuple(c.strip()[:MAX_CLAIM_CHARS] for c in payload.claims if c.strip())
    if payload.verdict is Verdict.APPROVE and payload.violations:
        raise InvalidReviewOutput("approve verdict with violations")
    return Judgement(
        verdict=payload.verdict,
        category=payload.category,
        violations=tuple(dict.fromkeys(payload.violations)),
        claims=claims,
        addresses_reviewer=payload.addresses_reviewer,
        confidence=payload.confidence,
        rationale=payload.rationale,
        model=model,
        prompt_version=prompt_version,
        variant=variant,
        request_id=request_id,
        usage=dict(usage or {}),
    )


def judgement_from_dict(data: Mapping[str, Any]) -> Judgement:
    return Judgement(
        verdict=Verdict(data["verdict"]),
        category=data["category"],
        violations=tuple(Violation(v) for v in data["violations"]),
        claims=tuple(data["claims"]),
        addresses_reviewer=data["addresses_reviewer"],
        confidence=Confidence(data["confidence"]),
        rationale=data["rationale"],
        model=data["model"],
        prompt_version=data["prompt_version"],
        variant=data.get("variant", "primary"),
        request_id=data.get("request_id"),
        usage=dict(data.get("usage", {})),
    )
