"""Report narratives with a numeric-faithfulness check (design doc: Controls → 7. Numbers are
never generated).

The model receives precomputed metrics only. Every number in its text is
compared with the numbers in the source table, as displayed; any mismatch (or
any speculation about who the users are) → regenerate once, then fall back to a
template. The template is always correct, so a report never waits on a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from theraflo_ads.review.creative import InvalidReviewOutput, ReviewerUnavailable
from theraflo_ads.review.llm import NarratorModel
from theraflo_ads.review.reports import AdvertiserReport

_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# The narrative may describe the ads; never the people who saw them.
FORBIDDEN = re.compile(
    r"\b(anxious|anxiety|depress\w*|lonely|stressed|struggling|vulnerable|patients?|"
    r"mental[- ]health|diagnos\w*|therapy|mood|wellness (activity|data)|"
    r"users who (feel|are feeling|struggle))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Narrative:
    text: str
    source: str  # model | template
    attempts: int
    rejected: tuple[str, ...] = ()  # why each model attempt was discarded
    model: str | None = None


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def facts_table(report: AdvertiserReport) -> str:
    lines = [
        f"Impressions: {_fmt_int(report.impressions)}",
        f"Clicks: {_fmt_int(report.clicks)}",
        f"Click-through rate: {_fmt_pct(report.ctr)}",
        f"Conversions: {_fmt_int(report.conversions)}",
        f"Spend: {_fmt_money(report.spend)}",
    ]
    cpa = report.cost_per_conversion
    lines.append(f"Cost per conversion: {_fmt_money(cpa) if cpa is not None else 'n/a'}")
    for row in report.segments:
        lines.append(
            f"Segment {row.dimension}={row.value}: impressions {_fmt_int(row.impressions)}, "
            f"clicks {_fmt_int(row.clicks)}, click-through rate {_fmt_pct(row.ctr)}, "
            f"conversions {_fmt_int(row.conversions)}, spend {_fmt_money(row.spend)}"
        )
    for dimension, hidden in sorted(report.suppressed.items()):
        lines.append(f"Segments by {dimension}: {hidden} suppressed for privacy")
    return "\n".join(lines)


def _canonical(token: str) -> str:
    return token.replace("$", "").replace(",", "")


def number_tokens(text: str) -> list[str]:
    return [_canonical(t) for t in _NUMBER.findall(_ISO_DATE.sub(" ", text))]


def unfaithful_numbers(text: str, facts: str) -> list[str]:
    """Numbers in `text` that do not appear, as displayed, in `facts`."""
    allowed = set(number_tokens(facts))
    return [t for t in number_tokens(text) if t not in allowed]


def template_narrative(report: AdvertiserReport) -> str:
    cpa = report.cost_per_conversion
    text = (
        f"Your ads received {_fmt_int(report.impressions)} impressions and "
        f"{_fmt_int(report.clicks)} clicks (click-through rate {_fmt_pct(report.ctr)}), "
        f"with {_fmt_int(report.conversions)} conversions and {_fmt_money(report.spend)} in spend"
    )
    text += f" ({_fmt_money(cpa)} per conversion)." if cpa is not None else "."
    if report.suppressed:
        text += " Some segments are hidden because they are too small to report privately."
    return text


def check_narrative(text: str, facts: str) -> list[str]:
    problems = [f"number not in source: {n}" for n in unfaithful_numbers(text, facts)]
    if FORBIDDEN.search(text):
        problems.append("describes users rather than ads")
    return problems


def narrate(
    report: AdvertiserReport, model: NarratorModel | None, *, attempts: int = 2
) -> Narrative:
    facts = facts_table(report)
    rejected: list[str] = []
    if model is not None:
        guidance = ""
        for attempt in range(1, attempts + 1):
            try:
                text = model.write(facts, guidance)
            except (ReviewerUnavailable, InvalidReviewOutput) as exc:
                rejected.append(f"attempt {attempt}: unavailable ({exc})")
                break
            problems = check_narrative(text, facts)
            if not problems:
                return Narrative(text, "model", attempt, tuple(rejected), model.name)
            rejected.append(f"attempt {attempt}: {'; '.join(problems)}")
            guidance = (
                "Your previous draft was rejected: "
                + "; ".join(problems)
                + ". Use only numbers exactly as they appear in the metrics."
            )
    return Narrative(
        template_narrative(report),
        "template",
        len(rejected),
        tuple(rejected),
        None,
    )
