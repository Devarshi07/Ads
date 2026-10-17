"""Stable assignment (design doc: Experiment Isolation, Permanent Holdouts).

Units are bucketed by hashing "layer:unit". The delimiter matters: without it,
"u1" + "23" and "u12" + "3" would collide. Permanent holdouts are assigned
first and excluded from every experiment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


def bucket(layer: str, unit_id: str) -> float:
    """Deterministic, roughly uniform value in [0, 1)."""
    digest = hashlib.sha256(f"{layer}:{unit_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class Holdout(StrEnum):
    NONE = "none"
    UNIFORM_EXPLORATION = "uniform_exploration"
    NO_ADS = "no_ads"


@dataclass(frozen=True)
class HoldoutConfig:
    layer: str = "holdouts:v1"
    uniform_exploration_share: float = 0.02
    no_ads_share: float = 0.02

    def __post_init__(self) -> None:
        shares = (self.uniform_exploration_share, self.no_ads_share)
        if any(s < 0 for s in shares) or sum(shares) >= 1:
            raise ValueError("holdout shares must be non-negative and sum to less than 1")

    def assign(self, user_hash: str) -> Holdout:
        b = bucket(self.layer, user_hash)
        if b < self.uniform_exploration_share:
            return Holdout.UNIFORM_EXPLORATION
        if b < self.uniform_exploration_share + self.no_ads_share:
            return Holdout.NO_ADS
        return Holdout.NONE


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    arms: tuple[tuple[str, float], ...]
    # Give each arm its own slice of every campaign's budget, so one arm spending
    # faster cannot starve the other (design doc: Marketplace Interference).
    budget_split: bool = False
    # Outside [starts_at, ends_at) nobody is in the experiment; traffic before the start
    # is the pre-period that CUPED uses as its covariate.
    starts_at: datetime | None = None
    ends_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.arms or any(weight <= 0 for _, weight in self.arms):
            raise ValueError("an experiment needs arms with positive weights")
        names = [name for name, _ in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")

    def assign(self, user_hash: str) -> str:
        total = sum(weight for _, weight in self.arms)
        point = bucket(f"exp:{self.experiment_id}", user_hash) * total
        cumulative = 0.0
        for name, weight in self.arms:
            cumulative += weight
            if point < cumulative:
                return name
        return self.arms[-1][0]

    def cohort(self, user_hash: str) -> str:
        return f"{self.experiment_id}:{self.assign(user_hash)}"

    def active(self, now: datetime) -> bool:
        if self.starts_at is not None and now < self.starts_at:
            return False
        return self.ends_at is None or now < self.ends_at

    def share(self, arm: str) -> float:
        total = sum(weight for _, weight in self.arms)
        return dict(self.arms)[arm] / total
