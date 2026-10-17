"""Off-policy evaluation (design doc: Counterfactual Logging, Evals → OPE Evals).

Each estimate reports its effective sample size and a bootstrap confidence
interval. An estimate with too little effective data, or an interval wider
than the effect being tested, is inconclusive, which is not a pass.

Sparse rewards need a second check. When a target's preferred action was rarely
explored, a handful of heavily weighted samples decide the estimate; if they
happen to contain no clicks, every bootstrap resample agrees and the interval
looks narrow but is wrong. So an estimate also needs enough reward events among
the samples that carry the top half of the total importance weight.

A third check catches wrong logged propensities and missing support: importance
weights must average 1 (E[π_target / π_logging] = 1 under the logging policy). An
estimate is inconclusive when the mean weight is more than 5% from 1 and its 99%
bootstrap interval excludes 1. The deviation bounds the normalization error, so
smaller, real deviations (a "uniform" logger that is 2% off) do not block an estimate.
Found on the Open Bandit Dataset, where Monte Carlo propensities floored at 1e-6
put 12% of the weight on 159 rows, passed the first two checks, and moved SNIPS 7%.

Rows are rarely independent: decisions of one user, or impressions in one hour,
share effects. Pass `clusters` (one id per sample) to resample whole clusters in the
bootstrap. On the Open Bandit Dataset, hour-level intervals were about 1.5x wider
than row-level ones, and only they covered the truth in every campaign.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from theraflo_ads.policy.decision import Decision


@dataclass(frozen=True)
class LoggedSample:
    eligible_actions: tuple[str, ...]
    action: str
    propensity: float
    reward: float
    context: object = None


TargetPolicy = Callable[[LoggedSample], Mapping[str, float]]
MIN_CLUSTERS = 20
RewardModel = Callable[[LoggedSample, str], float]


@dataclass(frozen=True)
class OPEEstimate:
    estimator: str
    value: float
    ci_low: float
    ci_high: float
    n: int
    ess: float
    conclusive: bool
    reason: str | None = None
    top_weight_events: int = 0  # nonzero rewards among samples carrying half the total weight
    weight_mean: float = float("nan")  # should be ≈ 1


def samples_from_decisions(
    decisions: Iterable[Decision], rewards: Mapping[str, float]
) -> list[LoggedSample]:
    """OPE-eligible decisions only; rewards are keyed by request_id (missing = 0)."""
    return [
        LoggedSample(
            eligible_actions=d.eligible_actions,
            action=d.chosen_action,
            propensity=d.propensity,
            reward=rewards.get(d.request_id, 0.0),
            context=d,
        )
        for d in decisions
        if d.ope_eligible
    ]


def user_clusters(samples: Sequence[LoggedSample]) -> list[str]:
    """One cluster per user, for samples built from decisions: a user's decisions share
    the user's tastes, so the bootstrap must resample users, not decisions."""
    return [s.context.user_hash for s in samples]


def importance_weights(samples: Sequence[LoggedSample], target: TargetPolicy) -> np.ndarray:
    weights = np.empty(len(samples))
    for i, s in enumerate(samples):
        if s.propensity <= 0:
            raise ValueError("logged propensities must be positive")
        weights[i] = _target_probabilities(s, target).get(s.action, 0.0) / s.propensity
    return weights


def top_weight_events(weights: np.ndarray, rewards: np.ndarray, mass: float = 0.5) -> int:
    """Nonzero rewards among the heaviest samples that together carry `mass` of the weight."""
    total = weights.sum()
    if total <= 0:
        return 0
    order = np.argsort(-weights)
    covered = np.cumsum(weights[order]) / total
    heavy = order[: int(np.searchsorted(covered, mass)) + 1]
    return int(np.count_nonzero(rewards[heavy]))


def effective_sample_size(weights: np.ndarray) -> float:
    total = float(weights.sum())
    squares = float((weights**2).sum())
    return total * total / squares if squares > 0 else 0.0


def ips(samples: Sequence[LoggedSample], target: TargetPolicy, **options) -> OPEEstimate:
    return ips_arrays(importance_weights(samples, target), _rewards(samples), **options)


def snips(samples: Sequence[LoggedSample], target: TargetPolicy, **options) -> OPEEstimate:
    return snips_arrays(importance_weights(samples, target), _rewards(samples), **options)


def doubly_robust(
    samples: Sequence[LoggedSample],
    target: TargetPolicy,
    reward_model: RewardModel,
    **options,
) -> OPEEstimate:
    direct = np.empty(len(samples))
    baseline = np.empty(len(samples))
    for i, s in enumerate(samples):
        probs = _target_probabilities(s, target)
        direct[i] = sum(p * reward_model(s, a) for a, p in probs.items() if p > 0)
        baseline[i] = reward_model(s, s.action)
    return dr_arrays(
        importance_weights(samples, target), _rewards(samples), direct, baseline, **options
    )


# ----- array core (shared by the per-sample API above and large public-dataset logs) -----


def ips_arrays(weights: np.ndarray, rewards: np.ndarray, **options) -> OPEEstimate:
    """weights[i] = π_target(a_i | x_i) / π_logging(a_i | x_i)."""
    contributions = weights * rewards
    return _estimate("ips", contributions, weights, rewards, _mean_of(contributions), **options)


def snips_arrays(weights: np.ndarray, rewards: np.ndarray, **options) -> OPEEstimate:
    weighted = weights * rewards

    def statistic(idx: np.ndarray) -> float:
        total = weights[idx].sum()
        return float(weighted[idx].sum() / total) if total > 0 else 0.0

    return _estimate("snips", weighted, weights, rewards, statistic, **options)


def dr_arrays(
    weights: np.ndarray,
    rewards: np.ndarray,
    direct: np.ndarray,
    baseline: np.ndarray,
    **options,
) -> OPEEstimate:
    """direct[i] = Σ_a π_target(a | x_i) q(x_i, a); baseline[i] = q(x_i, a_i)."""
    contributions = direct + weights * (rewards - baseline)
    return _estimate("dr", contributions, weights, rewards, _mean_of(contributions), **options)


def _mean_of(values: np.ndarray) -> Callable[[np.ndarray], float]:
    return lambda idx: float(values[idx].mean())


def _estimate(
    name: str,
    contributions: np.ndarray,
    weights: np.ndarray,
    rewards: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    min_ess: float = 100.0,
    min_top_weight_events: int = 30,
    max_ci_width: float | None = None,
    max_weight_mean_deviation: float = 0.05,
    clusters: Sequence | np.ndarray | None = None,
    seed: int = 0,
) -> OPEEstimate:
    n = len(contributions)
    if n == 0:
        return OPEEstimate(name, float("nan"), float("nan"), float("nan"), 0, 0.0, False, "no data")
    value = float(statistic(np.arange(n)))
    rng = np.random.default_rng(seed)
    draw = _cluster_sampler(clusters, n, rng) if clusters is not None else None
    boot = np.empty(n_bootstrap)
    weight_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = draw() if draw is not None else rng.integers(0, n, n)
        boot[b] = statistic(idx)
        weight_means[b] = weights[idx].mean()
    low, high = (float(q) for q in np.quantile(boot, [alpha / 2, 1 - alpha / 2]))
    w_low, w_high = (float(q) for q in np.quantile(weight_means, [0.005, 0.995]))
    weight_mean = float(weights.mean())
    ess = effective_sample_size(weights)
    events = top_weight_events(weights, rewards)
    reason = None
    if ess < min_ess:
        reason = f"effective sample size {ess:.0f} < {min_ess:.0f}"
    elif events < min_top_weight_events:
        reason = (
            f"only {events} reward events among the samples carrying half the weight "
            f"(< {min_top_weight_events})"
        )
    elif abs(weight_mean - 1.0) > max_weight_mean_deviation and not w_low <= 1.0 <= w_high:
        reason = (
            f"importance weights average {weight_mean:.3f} (99% CI [{w_low:.3f}, {w_high:.3f}]), "
            f"more than {max_weight_mean_deviation:.0%} from 1: logged propensities or the "
            "target's support are wrong"
        )
    elif max_ci_width is not None and high - low > max_ci_width:
        reason = f"confidence interval width {high - low:.4g} > {max_ci_width:.4g}"
    return OPEEstimate(name, value, low, high, n, ess, reason is None, reason, events, weight_mean)


def _cluster_sampler(
    clusters: Sequence | np.ndarray, n: int, rng: np.random.Generator
) -> Callable[[], np.ndarray]:
    """Row indices of whole clusters, drawn with replacement (as many clusters as there are)."""
    ids = np.unique(np.asarray(clusters), return_inverse=True)[1].ravel()
    if len(ids) != n:
        raise ValueError("clusters must have one id per sample")
    order = np.argsort(ids, kind="stable")
    sizes = np.bincount(ids)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    k = len(sizes)
    if k < MIN_CLUSTERS:
        # Too few clusters for a bootstrap: refuse rather than report a too-narrow interval.
        raise ValueError(f"a cluster bootstrap needs at least {MIN_CLUSTERS} clusters, got {k}")

    def draw() -> np.ndarray:
        chosen = rng.integers(0, k, k)
        lengths = sizes[chosen]
        offsets = np.repeat(
            starts[chosen] - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths
        )
        return order[np.arange(lengths.sum()) + offsets]

    return draw


def _target_probabilities(sample: LoggedSample, target: TargetPolicy) -> Mapping[str, float]:
    probs = target(sample)
    outside = [a for a, p in probs.items() if p > 0 and a not in sample.eligible_actions]
    if outside:
        raise ValueError(f"target policy puts mass outside the logged eligible set: {outside}")
    return probs


def _rewards(samples: Sequence[LoggedSample]) -> np.ndarray:
    return np.array([s.reward for s in samples], dtype=float)
