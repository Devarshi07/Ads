"""Health-data safety policy (design doc: Health-Data Safety & Ad Policy).

Every rule here is a hard constraint applied before the policy samples, never
a penalty in the reward.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from threading import Lock

DEFAULT_AD_PLACEMENTS = frozenset({"home_feed", "content_library"})

# Placements that must never show ads. Kept explicit so a misconfigured
# allowlist fails at construction time instead of in production.
AD_FREE_PLACEMENTS = frozenset(
    {
        "crisis",
        "crisis_resources",
        "therapy_chat",
        "journal",
        "mood_checkin",
        "assessment",
        "onboarding",
    }
)

DEFAULT_RESTRICTED_CATEGORIES = frozenset(
    {
        "gambling",
        "sports_betting",
        "alcohol",
        "cannabis",
        "tobacco",
        "vaping",
        "payday_loans",
        "high_interest_credit",
        "crypto",
        "speculative_investing",
        "weight_loss",
        "diet",
        "health_supplements",
        "nootropics",
        "prescription_drugs",
        "mental_health_services",
    }
)

# Added on top of the restricted list for teens and users of unknown age.
DEFAULT_MINOR_RESTRICTED_CATEGORIES = frozenset({"dating", "cosmetic_procedures", "energy_drinks"})

# Allowed, but impulse-prone: contextual targeting only, never behavioral
# personalization. A model trained on clicks would otherwise learn to target the
# users who over-respond to them (design doc: Why Penalties Are Not Enough).
DEFAULT_PERSONALIZATION_EXCLUDED = frozenset(
    {"streaming", "travel", "gaming", "fast_fashion", "luxury_goods", "buy_now_pay_later"}
)


@dataclass(frozen=True)
class SafetyPolicy:
    version: str
    ad_placements: frozenset[str]
    restricted_categories: frozenset[str]
    minor_restricted_categories: frozenset[str]
    suppression_window: timedelta
    kill_switch: bool = False
    personalization_excluded_categories: frozenset[str] = DEFAULT_PERSONALIZATION_EXCLUDED

    def __post_init__(self) -> None:
        overlap = self.ad_placements & AD_FREE_PLACEMENTS
        if overlap:
            raise ValueError(f"ad-free placements cannot be on the allowlist: {sorted(overlap)}")


def default_safety_policy() -> SafetyPolicy:
    return SafetyPolicy(
        version="safety_v1",
        ad_placements=DEFAULT_AD_PLACEMENTS,
        restricted_categories=DEFAULT_RESTRICTED_CATEGORIES,
        minor_restricted_categories=DEFAULT_MINOR_RESTRICTED_CATEGORIES,
        suppression_window=timedelta(hours=24),
    )


class SafetyPolicyStore:
    """Holds the live policy. The kill switch swaps it atomically."""

    def __init__(self, policy: SafetyPolicy) -> None:
        self._policy = policy
        self._lock = Lock()

    def current(self) -> SafetyPolicy:
        return self._policy

    def set_kill_switch(self, enabled: bool) -> SafetyPolicy:
        with self._lock:
            self._policy = replace(self._policy, kill_switch=enabled)
            return self._policy
