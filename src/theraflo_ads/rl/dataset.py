"""Offline RL dataset (design doc: Offline RL Dataset).

One transition per non-blocked, non-degraded decision:
    (state, action, propensity, reward, next state, done)
grouped into per-user episodes in time order. The state is the logged feature
context of the request; actions are the logged eligible ads plus NO_AD, so the
features of every action, now and at the next step, can be rebuilt exactly.

Reward = platform revenue of the impression (0 for NO_AD) minus a small cost per
ad shown, standing in for the UX cost the design doc puts in guardrails. An
episode ends at the user's last decision or after `episode_gap` without a visit.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from theraflo_ads.attribution.service import ImpressionLabel
from theraflo_ads.domain import NO_AD
from theraflo_ads.policy.decision import Decision, decision_from_record

REWARD_VERSION = "revenue_minus_ad_cost_v1"


@dataclass(frozen=True)
class Transition:
    decision: Decision
    reward: float
    next_index: int | None  # index of the next transition in the same episode
    done: bool

    @property
    def action(self) -> str:
        return self.decision.chosen_action

    @property
    def propensity(self) -> float:
        return self.decision.propensity


def build_transitions(
    records: Iterable[Mapping[str, Any]],
    labels: Mapping[str, ImpressionLabel],
    *,
    ad_cost: float = 0.0,
    episode_gap: timedelta = timedelta(days=7),
) -> list[Transition]:
    by_user: dict[str, list[Decision]] = defaultdict(list)
    for record in records:
        if record.get("blocked_reason") or record.get("degraded"):
            continue
        if record.get("served_by") == "no_ads_holdout":
            continue
        by_user[record["user_hash"]].append(decision_from_record(record))
    transitions: list[Transition] = []
    for decisions in by_user.values():
        decisions.sort(key=lambda d: d.timestamp)
        start = len(transitions)
        for k, decision in enumerate(decisions):
            reward = 0.0
            if decision.impression_id is not None:
                label = labels.get(decision.impression_id)
                reward = (label.platform_revenue if label else 0.0) - ad_cost
            nxt = decisions[k + 1] if k + 1 < len(decisions) else None
            done = nxt is None or nxt.timestamp - decision.timestamp > episode_gap
            transitions.append(Transition(decision, reward, None if done else start + k + 1, done))
    return transitions


def window(transitions: list[Transition], start: datetime, end: datetime) -> list[int]:
    return [i for i, t in enumerate(transitions) if start <= t.decision.timestamp < end]


def actions_of(transition: Transition) -> list[str]:
    return [a for a in transition.decision.eligible_actions] or [NO_AD]
