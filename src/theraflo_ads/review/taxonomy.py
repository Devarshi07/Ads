"""Ad category taxonomy (design doc: Ad Representations, LLM Components → taxonomy).

The taxonomy is the closed set of labels a reviewer (model or human) may assign.
A label outside it is an invalid output and goes to the human queue. The
restricted lists come from the live SafetyPolicy, so review and serving can never
disagree about what is restricted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from theraflo_ads.safety.policy import SafetyPolicy, default_safety_policy

TAXONOMY_VERSION = "taxonomy_v1"
OTHER = "other"

ALLOWED_CATEGORIES = frozenset(
    {
        "fitness",
        "sleep_products",
        "books",
        "meal_kits",
        "outdoor_gear",
        "online_courses",
        "travel",
        "budgeting_apps",
        "streaming",
        "home_goods",
        "gaming",
        "fast_fashion",
        "luxury_goods",
        "buy_now_pay_later",
    }
)

# Allowed, but close enough to health that a human signs off before they go live
# (design doc: Controls → 3. Abstention).
HEALTH_ADJACENT = frozenset({"fitness", "sleep_products", "meal_kits"})

# Allowed categories that sit next to a restricted one (gaming / gambling,
# budgeting / lending, BNPL / credit, meal kits / diet). Two independent reviews
# must agree before one of these is auto-approved (Controls → 5. Dual check).
HIGH_RISK = frozenset({"gaming", "budgeting_apps", "buy_now_pay_later", "meal_kits", "fitness"})


@dataclass(frozen=True)
class Taxonomy:
    version: str
    allowed: frozenset[str]
    restricted: frozenset[str]
    minor_restricted: frozenset[str]
    health_adjacent: frozenset[str] = HEALTH_ADJACENT
    high_risk: frozenset[str] = HIGH_RISK
    display_names: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlap = self.allowed & self.restricted
        if overlap:
            raise ValueError(f"categories both allowed and restricted: {sorted(overlap)}")

    @property
    def labels(self) -> frozenset[str]:
        return self.allowed | self.restricted | self.minor_restricted | {OTHER}

    def is_known(self, label: str) -> bool:
        return label in self.labels

    def is_restricted(self, label: str) -> bool:
        return label in self.restricted

    def display(self, label: str) -> str:
        return self.display_names.get(label) or label.replace("_", " ")


def default_taxonomy(policy: SafetyPolicy | None = None) -> Taxonomy:
    policy = policy or default_safety_policy()
    return Taxonomy(
        version=TAXONOMY_VERSION,
        allowed=ALLOWED_CATEGORIES - policy.restricted_categories,
        restricted=policy.restricted_categories,
        # Allowed for adults; the constraint engine blocks them for teens and unknown ages.
        minor_restricted=policy.minor_restricted_categories - policy.restricted_categories,
        display_names={
            "budgeting_apps": "budgeting apps",
            "buy_now_pay_later": "buy now, pay later",
            "online_courses": "online courses",
        },
    )
