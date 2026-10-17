"""'Why am I seeing this?' (design doc: Explainability, Controls → 8. Explanations are templated).

Rendered from the logged decision with fixed templates, never free-generated.
Only ad-safe factors can appear: where the ad was shown, rotation/exploration,
and, for users who opted into personalized ads, their past interactions with
ads in that category. Suppression windows, age, consent history, and anything
about wellness activity are never mentioned, because no template exists for
them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from theraflo_ads.domain import NO_AD, AdCatalog, ConsentScope
from theraflo_ads.policy.decision import Decision, ServedBy
from theraflo_ads.review.taxonomy import Taxonomy, default_taxonomy
from theraflo_ads.safety.policy import DEFAULT_PERSONALIZATION_EXCLUDED

TEMPLATE_VERSION = "why_v1"

PLACEMENT_NAMES = {"home_feed": "your home feed", "content_library": "the content library"}

TEMPLATES: Mapping[str, str] = {
    "placement": "This ad was shown in {placement}. Ads there are chosen for the space, "
    "not from your activity in Theraflo.",
    "rotation": "Some ads are shown in rotation so we can learn which ads people find useful.",
    "interest": "You turned on personalized ads, and you have interacted with {category} ads "
    "on Theraflo before.",
    "running": "This is one of the ads currently running on Theraflo.",
}
CONTROLS = (
    "Hide this ad",
    "Turn personalized ads on or off in Settings",
    "Ad-free with Theraflo Plus",
)


@dataclass(frozen=True)
class Explanation:
    impression_id: str
    factors: tuple[str, ...]
    factor_keys: tuple[str, ...]
    controls: tuple[str, ...] = CONTROLS
    template_version: str = TEMPLATE_VERSION

    def to_dict(self) -> dict:
        return {
            "impression_id": self.impression_id,
            "factors": list(self.factors),
            "controls": list(self.controls),
            "template_version": self.template_version,
        }


def factor_keys(
    decision: Decision,
    catalog: AdCatalog,
    excluded: frozenset[str] = DEFAULT_PERSONALIZATION_EXCLUDED,
) -> list[str]:
    keys = ["placement"]
    chosen = decision.chosen_action
    probabilities = decision.action_probabilities
    if decision.served_by is ServedBy.UNIFORM_HOLDOUT or (
        probabilities and probabilities.get(chosen, 0.0) < max(probabilities.values())
    ):
        keys.append("rotation")
    ad = catalog.ads.get(chosen)
    features = decision.features or {}
    if (
        ad is not None
        # Only when a policy that reads interaction features chose the ad on purpose:
        # otherwise "because of your interests" would be an invented reason.
        and "rotation" not in keys
        and _personalizing(decision)
        and decision.consent_scope is ConsentScope.PERSONALIZED
        and ad.category not in excluded
        and features.get("user_cat_impressions", 0.0) > 0.0
        and features.get("user_cat_ctr", 0.0) > features.get("prior_ctr", 0.0)
    ):
        keys.append("interest")
    if decision.served_by in (ServedBy.POPULARITY, ServedBy.HOUSE_AD):
        keys.append("running")
    return keys


def _personalizing(decision: Decision) -> bool:
    if decision.served_by in (ServedBy.RANKER, ServedBy.RL):
        return True
    return decision.served_by is ServedBy.BANDIT and (decision.policy_version or "").startswith(
        "lin_"
    )


def explain(
    decision: Decision,
    catalog: AdCatalog,
    taxonomy: Taxonomy | None = None,
    excluded: frozenset[str] = DEFAULT_PERSONALIZATION_EXCLUDED,
) -> Explanation:
    if decision.impression_id is None or decision.chosen_action == NO_AD:
        raise ValueError("no ad was served for this decision")
    taxonomy = taxonomy or default_taxonomy()
    ad = catalog.ads.get(decision.chosen_action)
    values = {
        "placement": PLACEMENT_NAMES.get(decision.placement, "this part of the app"),
        "category": taxonomy.display(ad.category) if ad else "similar",
    }
    keys = factor_keys(decision, catalog, excluded)
    return Explanation(
        decision.impression_id,
        tuple(TEMPLATES[k].format(**values) for k in keys),
        tuple(keys),
    )
