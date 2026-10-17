"""Deterministic policy rules: the first reviewer every creative meets, and the fallback when
no model is available (design doc: Creative Review, Health-Data Safety & Ad Policy).

Rules can reject (a strong restricted-category term, vulnerability targeting,
a request for health data) or raise signals that block auto-approval. They can
never approve anything on their own.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from theraflo_ads.review.creative import Creative, Violation
from theraflo_ads.review.taxonomy import Taxonomy
from theraflo_ads.review.text import deleet, normalize

RULES_VERSION = "rules_v1"

# Strong terms identify the category on their own; weak terms have innocent uses
# ("best bet", "wine-red") and count only in pairs.
CATEGORY_TERMS: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "gambling": (
        (
            "casino",
            "slots",
            "slot machine",
            "jackpot",
            "roulette",
            "blackjack",
            "poker",
            "free spins",
            "spin to win",
            "place your bets",
            "wager",
        ),
        ("bet", "bets", "betting", "odds", "spin", "win big", "payout", "bonus cash"),
    ),
    "sports_betting": (
        ("sportsbook", "parlay", "odds boost", "sports betting", "bet on the game"),
        ("fantasy picks", "point spread", "moneyline"),
    ),
    "alcohol": (
        (
            "vodka",
            "whiskey",
            "whisky",
            "tequila",
            "liquor",
            "brewery",
            "craft beer",
            "wine club",
            "cocktail kit",
            "happy hour",
        ),
        ("beer", "wine", "cocktail", "spirits", "drinks on us"),
    ),
    "cannabis": (("cannabis", "cbd", "thc", "dispensary", "edibles", "marijuana"), ("weed",)),
    "tobacco": (("cigarette", "cigar", "tobacco", "nicotine pouch"), ("smoke",)),
    "vaping": (("vape", "vaping", "e-cig", "e-liquid", "nicotine"), ("pods",)),
    "payday_loans": (
        (
            "payday loan",
            "cash advance",
            "same-day loan",
            "instant loan",
            "no credit check",
            "title loan",
        ),
        ("fast cash", "cash today", "borrow up to"),
    ),
    "high_interest_credit": (
        ("bad credit ok", "guaranteed approval", "high limit card for bad credit"),
        ("credit repair",),
    ),
    "crypto": (
        ("crypto", "bitcoin", "ethereum", "altcoin", "nft", "defi", "memecoin", "token presale"),
        ("blockchain", "token", "coin", "wallet", "to the moon"),
    ),
    "speculative_investing": (
        (
            "day trading",
            "forex",
            "options trading",
            "10x returns",
            "guaranteed returns",
            "trading signals",
        ),
        ("passive income", "get rich", "financial freedom"),
    ),
    "weight_loss": (
        (
            "lose weight",
            "weight loss",
            "fat burner",
            "fat burning",
            "slim down",
            "drop pounds",
            "appetite suppressant",
            "melt fat",
        ),
        ("bikini body", "lose inches", "detox tea", "flat belly"),
    ),
    "diet": (("diet pill", "keto pill", "crash diet", "cleanse diet"), ("calorie", "carb blocker")),
    "health_supplements": (
        (
            "supplement",
            "gummies",
            "capsules",
            "ashwagandha",
            "adaptogen",
            "multivitamin",
            "probiotic",
        ),
        ("vitamin", "herbal", "natural remedy", "dose"),
    ),
    "nootropics": (("nootropic", "focus pill", "smart drug", "brain pill"), ("brain boost",)),
    "prescription_drugs": (
        (
            "prescription",
            "adderall",
            "xanax",
            "ozempic",
            "semaglutide",
            "ssri",
            "antidepressant",
            "benzodiazepine",
        ),
        ("rx", "pharmacy", "medication"),
    ),
    "mental_health_services": (
        (
            "online therapy",
            "therapist",
            "psychiatrist",
            "counseling",
            "counselling",
            "talk therapy",
            "mental health treatment",
        ),
        ("therapy", "diagnosis"),
    ),
}

VULNERABILITY_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\bfeeling (so |really )?(down|low|lonely|alone|anxious|depressed|stressed|empty|"
        r"hopeless|worthless|broken|numb)\b",
        r"\b(lonely|sad|anxious|depressed) (tonight|again|lately)\b",
        r"\bstruggling with (your )?(anxiety|depression|stress|loneliness|grief|panic)\b",
        r"\b(retail therapy|shop (away|your) (the )?(pain|sadness|stress))\b",
        r"\b(numb|escape|forget) (the|your) (pain|problems|sadness|feelings)\b",
        r"\b(nobody|no one) (understands|cares)\b",
        r"\bwhen life (gets|feels) (hard|too much|overwhelming)\b",
        r"\b(you deserve|treat yourself)\b.{0,40}\b(after|when) (a )?(bad|rough|hard) (day|week)\b",
        r"\b(bad|rough|hard) (day|week)\?.{0,40}\b(treat yourself|you deserve|buy|shop|order)\b",
    )
)

URGENCY_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\b(act now|last chance|before it'?s too late|don'?t miss out|hurry)\b",
        r"\bonly \d+ (left|remaining|spots?)\b",
        r"\b(expires|ends) in \d+ (minutes?|hours?|seconds?)\b",
        r"\b(today only|now or never)\b",
    )
)

PERSONAL_DATA_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\b(tell|share with) us (about )?your (mood|diagnosis|symptoms|medications?|"
        r"mental health|therapy)\b",
        r"\b(take|try) (our|this|a) (free )?(depression|anxiety|adhd|bipolar|ptsd|mental health) "
        r"(quiz|test|screening|assessment)\b",
        r"\b(upload|connect|sync) your (health|mood|journal|therapy) (data|records|entries)\b",
        r"\benter your (diagnosis|medications?)\b",
    )
)

MISLEADING_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\b(guaranteed|100%) (results|cure|success)\b",
        r"\b(miracle|cure[sd]?|instantly heals?)\b",
        r"\b(doctors|scientists) (hate|don'?t want you to know)\b",
        r"\brisk[- ]free\b.{0,30}\b(returns?|profits?|income)\b",
    )
)


@dataclass(frozen=True)
class RuleFindings:
    implied_category: str | None
    category_hits: Mapping[str, tuple[str, ...]]
    violations: frozenset[Violation]
    signals: tuple[str, ...]  # block auto-approval, but do not reject

    @property
    def clean(self) -> bool:
        return not self.violations and not self.signals


def _contains(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(term)}(s|es)?(?![a-z])", text) is not None


def category_hits(text: str) -> dict[str, tuple[int, tuple[str, ...]]]:
    """category -> (score, matched terms). A strong term scores 2, a weak term 1."""
    views = (normalize(text), deleet(text))
    hits: dict[str, tuple[int, tuple[str, ...]]] = {}
    for category, (strong, weak) in CATEGORY_TERMS.items():
        matched_strong = [t for t in strong if any(_contains(v, t) for v in views)]
        matched_weak = [t for t in weak if any(_contains(v, t) for v in views)]
        score = 2 * len(matched_strong) + len(matched_weak)
        if score:
            hits[category] = (score, tuple(matched_strong + matched_weak))
    return hits


def rule_review(creative: Creative, taxonomy: Taxonomy) -> RuleFindings:
    text = creative.full_text()
    views = (normalize(text), deleet(text))
    violations: set[Violation] = set()
    signals: list[str] = []

    hits = category_hits(text)
    implied = None
    restricted_hits = {c: h for c, h in hits.items() if taxonomy.is_restricted(c)}
    if restricted_hits:
        category, (score, terms) = max(restricted_hits.items(), key=lambda kv: kv[1][0])
        if score >= 2:
            implied = category
            violations.add(Violation.RESTRICTED_CATEGORY)
        else:
            signals.append(f"weak_restricted_term:{category}:{terms[0]}")
    if taxonomy.is_restricted(creative.declared_category):
        implied = creative.declared_category
        violations.add(Violation.RESTRICTED_CATEGORY)
    if not taxonomy.is_known(creative.declared_category):
        signals.append(f"unknown_declared_category:{creative.declared_category}")

    for patterns, violation in (
        (VULNERABILITY_PATTERNS, Violation.VULNERABILITY_TARGETING),
        (PERSONAL_DATA_PATTERNS, Violation.PERSONAL_DATA_REQUEST),
    ):
        if any(p.search(v) for p in patterns for v in views):
            violations.add(violation)
    if any(p.search(v) for p in URGENCY_PATTERNS for v in views):
        signals.append("urgency")
    if any(p.search(v) for p in MISLEADING_PATTERNS for v in views):
        signals.append("misleading_language")

    url = urlsplit(creative.landing_url)
    if url.scheme != "https" or not url.hostname or re.fullmatch(r"[\d.:]+", url.hostname):
        signals.append("landing_url_not_https_domain")

    return RuleFindings(
        implied,
        {c: terms for c, (_, terms) in hits.items()},
        frozenset(violations),
        tuple(signals),
    )
