"""Claim grounding (design doc: Controls → 2. Grounding).

Every claim in a creative is matched against the advertiser's substantiated
claims. Matching is deterministic code, never a model: "the model never decides
on its own that a health claim is true".

A claim is grounded when its numbers all appear in one substantiated claim and
at least `match` of its content words are covered by it (a creative may say less
than the evidence, never more). Strengthening words ("clinically", "cure",
"guaranteed") must be in the substantiation too. Claims a model reports that do
not occur in the creative are dropped and counted: that is a hallucination.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from theraflo_ads.review.creative import Creative, SubstantiatedClaim
from theraflo_ads.review.text import content_words, normalize, numbers, sentences

HEALTH_TERMS = frozenset(
    {
        "anxiety",
        "anxious",
        "depression",
        "depressed",
        "stress",
        "stressed",
        "sleep",
        "insomnia",
        "mood",
        "focus",
        "adhd",
        "trauma",
        "panic",
        "pain",
        "weight",
        "immunity",
        "immune",
        "cortisol",
        "burnout",
        "calm",
        "relax",
        "mental",
        "wellbeing",
        "well-being",
        "energy",
        "fatigue",
        "memory",
        "brain",
        "heal",
        "symptom",
        "disorder",
        "blood",
        "heart",
        "cholesterol",
        "metabolism",
        "recovery",
        "rest",
        "restful",
    }
)
EFFICACY_TERMS = frozenset(
    {
        "reduce",
        "reduces",
        "improve",
        "improves",
        "boost",
        "boosts",
        "cure",
        "cures",
        "treat",
        "treats",
        "heal",
        "heals",
        "prevent",
        "prevents",
        "relieve",
        "relieves",
        "eliminate",
        "eliminates",
        "lower",
        "lowers",
        "increase",
        "increases",
        "fight",
        "fights",
        "beat",
        "help",
        "helps",
        "fix",
        "fixes",
        "restore",
        "restores",
        "calm",
        "calms",
        "ease",
        "eases",
    }
)
EVIDENCE_TERMS = frozenset(
    {
        "clinically",
        "clinical",
        "proven",
        "scientifically",
        "studies",
        "study",
        "research",
        "doctor",
        "doctors",
        "dermatologist",
        "tested",
        "certified",
        "recommended",
        "evidence",
    }
)
STRENGTHENERS = frozenset(
    {
        "clinically",
        "proven",
        "cure",
        "cures",
        "guaranteed",
        "guarantee",
        "permanently",
        "instantly",
        "doctor",
        "doctors",
        "scientifically",
        "fda",
    }
)
_SUPERLATIVE = re.compile(r"(#\s?1\b|\bnumber one\b|\bbest\b|\b\d+x\b|\d+\s?%|\bpercent\b)")
# Ratings and audience statistics are claims too: "Rated 4.9 stars by 80,000 readers".
_STATISTIC = re.compile(
    r"\b(rated|rating|stars?|reviews?|survey|on average|average)\b"
    r"|\b\d[\d,.]*\+?\s*(stars?|reviews?|users|customers|members|readers|learners|riders|people|"
    r"adults|patients|hours|calories|miles)\b"
)


class ClaimStatus(StrEnum):
    GROUNDED = "grounded"
    PARTIAL = "partial"  # some overlap with a substantiated claim: a human decides
    UNGROUNDED = "ungrounded"
    NOT_IN_CREATIVE = "not_in_creative"  # reported by a model, absent from the text


@dataclass(frozen=True)
class ClaimCheck:
    text: str
    source: str  # rules | model | both
    status: ClaimStatus
    health: bool
    claim_id: str | None = None
    coverage: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "status": self.status.value,
            "health": self.health,
            "claim_id": self.claim_id,
            "coverage": round(self.coverage, 3),
        }


def _tokens(text: str) -> set[str]:
    return set(normalize(text).replace("-", " ").split()) | set(
        re.findall(r"[a-z]+", normalize(text))
    )


def is_health_claim(text: str) -> bool:
    tokens = _tokens(text)
    return bool(tokens & HEALTH_TERMS)


def is_claim(sentence: str) -> bool:
    tokens = _tokens(sentence)
    efficacy = bool(tokens & EFFICACY_TERMS)
    evidence = bool(tokens & EVIDENCE_TERMS)
    health = bool(tokens & HEALTH_TERMS)
    text = normalize(sentence)
    quantified = _SUPERLATIVE.search(text) is not None
    statistic = _STATISTIC.search(text) is not None
    return evidence or statistic or (efficacy and (health or quantified)) or (health and quantified)


def extract_claims(creative: Creative) -> list[str]:
    """Rule-based extraction: sentences that assert an effect, evidence, or a number."""
    found = []
    for text in creative.texts().values():
        found.extend(s for s in sentences(text) if is_claim(s))
    return list(dict.fromkeys(found))


def occurs_in(claim: str, creative: Creative, *, coverage: float = 0.7) -> bool:
    """Whether a model-reported claim is really in the creative (not invented)."""
    words = content_words(claim)
    if not words:
        return False
    text = creative.full_text()
    if not numbers(claim) <= numbers(text):
        return False
    return len(words & content_words(text)) / len(words) >= coverage


@dataclass(frozen=True)
class ClaimMatcher:
    match: float = 0.8
    partial: float = 0.5

    def check(
        self, claim: str, substantiated: Sequence[SubstantiatedClaim], *, source: str
    ) -> ClaimCheck:
        words = content_words(claim)
        claim_numbers = numbers(claim)
        strengtheners = _tokens(claim) & STRENGTHENERS
        health = is_health_claim(claim)
        best: tuple[float, SubstantiatedClaim | None] = (0.0, None)
        for candidate in substantiated:
            if not claim_numbers <= numbers(candidate.text):
                continue
            if not strengtheners <= _tokens(candidate.text):
                continue
            covered = len(words & content_words(candidate.text)) / len(words) if words else 0.0
            if covered > best[0]:
                best = (covered, candidate)
        coverage, candidate = best
        if candidate is not None and coverage >= self.match:
            return ClaimCheck(
                claim, source, ClaimStatus.GROUNDED, health, candidate.claim_id, coverage
            )
        if candidate is not None and coverage >= self.partial:
            return ClaimCheck(
                claim, source, ClaimStatus.PARTIAL, health, candidate.claim_id, coverage
            )
        return ClaimCheck(claim, source, ClaimStatus.UNGROUNDED, health, None, coverage)


def ground_claims(
    creative: Creative,
    substantiated: Sequence[SubstantiatedClaim],
    *,
    model_claims: Iterable[str] = (),
    matcher: ClaimMatcher | None = None,
) -> list[ClaimCheck]:
    matcher = matcher or ClaimMatcher()
    rule_claims = extract_claims(creative)
    checks: list[ClaimCheck] = []
    seen: dict[str, str] = {normalize(c): "rules" for c in rule_claims}
    model_only: list[str] = []
    for claim in model_claims:
        key = normalize(claim)
        if key in seen:
            seen[key] = "both"
        elif any(_same_claim(claim, existing) for existing in rule_claims):
            continue  # a paraphrase of a sentence the rules already extracted
        elif not occurs_in(claim, creative):
            checks.append(
                ClaimCheck(claim, "model", ClaimStatus.NOT_IN_CREATIVE, is_health_claim(claim))
            )
        else:
            model_only.append(claim)
    for claim in rule_claims:
        checks.append(matcher.check(claim, substantiated, source=seen[normalize(claim)]))
    for claim in model_only:
        checks.append(matcher.check(claim, substantiated, source="model"))
    return checks


def _same_claim(a: str, b: str) -> bool:
    wa, wb = content_words(a), content_words(b)
    return bool(wa) and len(wa & wb) / len(wa) >= 0.8 and numbers(a) <= numbers(b)


def claims_by_advertiser(
    claims: Iterable[SubstantiatedClaim],
) -> Mapping[str, list[SubstantiatedClaim]]:
    out: dict[str, list[SubstantiatedClaim]] = {}
    for claim in claims:
        out.setdefault(claim.advertiser_id, []).append(claim)
    return out
