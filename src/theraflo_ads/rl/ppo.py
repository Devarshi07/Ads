"""PPO in the simulator (design doc: Reinforcement Learning → "PPO / actor-critic only inside
the simulator; on-policy methods need live interaction").

A linear-softmax policy over φ(s, a) (the same featurizer as FQI), mixed with an
exploration floor. Each training round runs the simulator with the current
policy, computes discounted returns per user trajectory, and takes clipped-PPO
gradient steps on the logged (φ, action, old probability) tuples:

    L = E[min(ρ·Â, clip(ρ, 1-ε, 1+ε)·Â)],   ρ = π_θ(a|s) / π_old(a|s)
    ∇ log π(a|s) = φ_a - Σ_b π(b|s) φ_b     (linear softmax)
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from theraflo_ads.domain import NO_AD, Ad, AdRequest
from theraflo_ads.policy.decision import ServedBy
from theraflo_ads.policy.policies import ActionDistribution
from theraflo_ads.rl.fqi import ActionFeaturizer


@dataclass
class _Step:
    request_id: str
    user_hash: str
    timestamp: float
    phi: np.ndarray
    actions: list[str]
    old_probs: np.ndarray


class LinearSoftmaxPolicy:
    served_by = ServedBy.RL

    def __init__(
        self,
        featurizer: ActionFeaturizer,
        *,
        floor: float = 0.05,
        temperature: float = 1.0,
        rng: random.Random | None = None,
        version: str = "ppo_v1",
    ) -> None:
        self.featurizer = featurizer
        self.theta = np.zeros(len(featurizer.names))
        self._floor = floor
        self._temperature = temperature
        self.version = version
        self.buffer: dict[str, _Step] = {}
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def standardize(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self._mean, self._scale = mean, np.where(scale > 0, scale, 1.0)

    def _phi(self, request: AdRequest, actions: Sequence[str]) -> np.ndarray:
        phi = self.featurizer.matrix(request, actions)
        if self._mean is not None:
            phi = (phi - self._mean) / self._scale
        return phi

    def probabilities(self, phi: np.ndarray) -> np.ndarray:
        logits = phi @ self.theta / self._temperature
        logits -= logits.max()
        soft = np.exp(logits)
        soft /= soft.sum()
        return (1 - self._floor) * soft + self._floor / len(soft)

    def distribution(self, request: AdRequest, eligible: Sequence[Ad]) -> ActionDistribution:
        actions = [ad.ad_id for ad in eligible] + [NO_AD]
        phi = self._phi(request, actions)
        probs = self.probabilities(phi)
        self.buffer[request.request_id] = _Step(
            request.request_id,
            request.user.user_hash,
            request.now.timestamp(),
            phi,
            actions,
            probs,
        )
        return ActionDistribution(dict(zip(actions, probs.tolist(), strict=True)), len(actions) > 1)


@dataclass
class PPOConfig:
    gamma: float = 0.9
    clip: float = 0.2
    learning_rate: float = 0.05
    epochs: int = 4
    l2: float = 1e-3
    history: list[dict[str, float]] = field(default_factory=list)


def ppo_update(
    policy: LinearSoftmaxPolicy,
    chosen: dict[str, str],
    rewards: dict[str, float],
    config: PPOConfig,
) -> dict[str, float]:
    """One PPO round on the policy's buffer. `chosen` maps request_id → action actually served
    by this policy; `rewards` maps request_id → reward."""
    steps = [s for s in policy.buffer.values() if s.request_id in chosen]
    by_user: dict[str, list[_Step]] = defaultdict(list)
    for step in steps:
        by_user[step.user_hash].append(step)
    returns: dict[str, float] = {}
    for trajectory in by_user.values():
        trajectory.sort(key=lambda s: s.timestamp)
        running = 0.0
        for step in reversed(trajectory):
            running = rewards.get(step.request_id, 0.0) + config.gamma * running
            returns[step.request_id] = running
    values = np.array([returns[s.request_id] for s in steps])
    if len(values) == 0:
        return {"steps": 0}
    advantages = (values - values.mean()) / (values.std() + 1e-9)
    for _ in range(config.epochs):
        gradient = np.zeros_like(policy.theta)
        for step, advantage in zip(steps, advantages, strict=True):
            a = step.actions.index(chosen[step.request_id])
            probs = policy.probabilities(step.phi)
            ratio = probs[a] / step.old_probs[a]
            clipped = (ratio > 1 + config.clip and advantage > 0) or (
                ratio < 1 - config.clip and advantage < 0
            )
            if clipped:
                continue  # the clipped surrogate has zero gradient here
            grad_log = step.phi[a] - probs @ step.phi
            gradient += ratio * advantage * grad_log
        gradient = gradient / len(steps) - config.l2 * policy.theta
        policy.theta += config.learning_rate * gradient
    policy.buffer.clear()
    summary = {
        "steps": float(len(steps)),
        "mean_return": float(values.mean()),
        "theta_norm": float(np.linalg.norm(policy.theta)),
    }
    config.history.append(summary)
    return summary
