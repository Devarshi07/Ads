"""Small-sample statistics for review evals (design doc: LLM Evals → "CI reported")."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval. Unlike the normal approximation it is sensible at 0/n:
    zero false approvals in 100 trials still means "up to ~3.7%", not "0%"."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Agreement beyond chance between two raters (e.g. an LLM judge and human labels)."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)
