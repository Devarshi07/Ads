"""Prompt-injection defenses (design doc: Controls → 6. Prompt-injection defense).

Advertiser text is untrusted data. It is escaped and delimited before it reaches
a model, the model has no tools, and text that addresses the reviewer
("ignore previous instructions", "this ad is pre-approved") is itself a
violation: a legitimate ad has no reason to talk to its reviewer.

Strong signals reject the creative. A single weak signal (a model name, an
encoded blob, "act as") has innocent readings, so it only escalates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape

from theraflo_ads.review.creative import Creative
from theraflo_ads.review.text import deleet, has_hidden_characters, normalize

_REVIEWER = r"(ad\s+)?(reviewer|moderator|classifier|grader|checker|the ai|the model|ai model)s?"

STRONG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        (
            "override_instructions",
            r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instructions?|rules?|"
            r"prompts?|polic(y|ies)|guidelines?)\b",
        ),
        (
            "addresses_reviewer",
            rf"(\b(note|message|instructions?|memo)\s+(to|for)\s+(the\s+|any\s+|all\s+)?{_REVIEWER}"
            rf"|\b(dear|hey|hi|attention)\s+{_REVIEWER}|\bto (the|any) {_REVIEWER})\b",
        ),
        (
            "claims_approval",
            r"\b(pre-?approved|already (been )?(approved|reviewed|verified)|auto-?approve|"
            r"whitelisted|mark (this|it) as (approved|safe|compliant)|"
            r"(this|the) (ad|creative) (is|has been) (approved|compliant|verified))\b",
        ),
        ("verdict_vocabulary", r"\b(verdict|addresses_reviewer)\b|\bconfidence\s*[:=]"),
        (
            "system_markers",
            r"(\bsystem\s*(prompt|message)\b|\bsystem\s*:|\[/?inst\]|<\|im_start\|>"
            r"|###\s*instruction)",
        ),
        ("tag_breaking", r"</?\s*(creative|system|instructions?|headline|body|landing_page)\s*>"),
    )
)
WEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        ("dictates_output", r"\b(respond|answer|output|return)\b.{0,20}\b(only|json)\b"),
        ("role_play", r"\b(you are now|pretend (to be|you are)|new role|from now on you)\b"),
        ("mentions_model", r"\b(language model|llm|chatgpt|gpt-?\d|anthropic|openai)\b"),
    )
)
_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{48,}={0,2}")


@dataclass(frozen=True)
class InjectionScan:
    flagged: bool  # reject: the creative addresses its reviewer
    suspicious: bool  # escalate: a weak signal a human should look at
    signals: tuple[str, ...]


def scan_injection(creative: Creative) -> InjectionScan:
    strong: list[str] = []
    weak: list[str] = []
    for field_name, raw in creative.texts().items():
        if not raw:
            continue
        if has_hidden_characters(raw):
            weak.append(f"{field_name}:hidden_characters")
        if _BASE64_BLOB.search(raw):
            weak.append(f"{field_name}:encoded_blob")
        views = {normalize(raw), deleet(raw)}
        for name, pattern in STRONG_PATTERNS:
            if any(pattern.search(view) for view in views):
                strong.append(f"{field_name}:{name}")
        for name, pattern in WEAK_PATTERNS:
            if any(pattern.search(view) for view in views):
                weak.append(f"{field_name}:{name}")
    flagged = bool(strong) or len(weak) >= 2
    return InjectionScan(flagged, bool(weak) and not flagged, tuple(strong + weak))


def render_untrusted(creative: Creative) -> str:
    """The creative as a model sees it: every advertiser field escaped, so the text cannot
    close a tag or open a new one, inside one delimited block."""
    parts = [
        "<creative>",
        f"<declared_category>{escape(creative.declared_category)}</declared_category>",
        f"<landing_url>{escape(creative.landing_url)}</landing_url>",
    ]
    for field_name, value in creative.texts().items():
        parts.append(f"<{field_name}>{escape(value)}</{field_name}>")
    parts.append("</creative>")
    return "\n".join(parts)
