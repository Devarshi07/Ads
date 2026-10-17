"""Peeking-safe tests (design doc: Sequential Testing).

Checking an ordinary p-value at every look and stopping on the first p < 0.05
inflates the false-positive rate several-fold. Use always-valid mSPRT p-values
for continuous monitoring, or pre-planned looks with alpha-spending boundaries.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# Odd so Simpson's rule applies; spacing stays well below the unit-variance step kernel width.
_GRID_POINTS = 801


def _cumulative_mean_and_variance(
    values: np.ndarray, sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Shifting by the first value avoids cancellation in sum(x²) − n·mean² for large-mean metrics.
    shift = values[0]
    shifted = values - shift
    sums = np.cumsum(shifted)[sizes - 1]
    squares = np.cumsum(shifted**2)[sizes - 1]
    means = sums / sizes
    variances = np.maximum(squares - sizes * means**2, 0.0) / (sizes - 1)
    return means + shift, variances


def msprt_p_values(
    control: np.ndarray, treatment: np.ndarray, *, tau: float, look_every: int
) -> np.ndarray:
    """Always-valid p-values for H0: equal means, one per look of ``look_every`` units per arm.

    Normal-mixture SPRT (Johari, Pekelis & Walsh, "Always Valid Inference")
    with mixing distribution N(0, tau²) over the difference in means and the
    per-arm variances replaced by their running sample estimates. ``tau``
    should be on the scale of plausible effects. The p-value at a look is
    min(1, 1 / Λ) carried forward as a running minimum, so it never rises.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    if look_every < 2:
        raise ValueError("look_every must be at least 2 so variances can be estimated")
    control_array = np.asarray(control, dtype=float)
    treatment_array = np.asarray(treatment, dtype=float)
    if control_array.ndim != 1 or treatment_array.ndim != 1:
        raise ValueError("control and treatment must be 1-D arrays")
    n_looks = min(control_array.size, treatment_array.size) // look_every
    if n_looks == 0:
        raise ValueError("not enough units for a single look")
    sizes = look_every * np.arange(1, n_looks + 1)
    mean_c, var_c = _cumulative_mean_and_variance(control_array, sizes)
    mean_t, var_t = _cumulative_mean_and_variance(treatment_array, sizes)
    diff = mean_t - mean_c
    v = (var_c + var_t) / sizes
    tau2 = tau**2
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lambda = 0.5 * np.log(v / (v + tau2)) + tau2 * diff**2 / (2.0 * v * (v + tau2))
    # Zero plug-in variance is the limit of a perfectly precise estimate.
    log_lambda = np.where(v > 0, log_lambda, np.where(diff == 0, -np.inf, np.inf))
    p_values = np.minimum(1.0, np.exp(-log_lambda))
    return np.minimum.accumulate(p_values)


def _simpson_weights(points: np.ndarray) -> np.ndarray:
    weights = np.ones(points.size)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights * (points[1] - points[0]) / 3.0


def _excess_crossing(bound: float, grid: np.ndarray, mass: np.ndarray, target: float) -> float:
    """P(first crossing of ±bound at the next look) − target, from the surviving sub-density."""
    return float(np.dot(mass, norm.sf(bound - grid) + norm.cdf(-bound - grid))) - target


def obrien_fleming_boundaries(n_looks: int, *, alpha: float = 0.05) -> list[float]:
    """Two-sided z boundaries for ``n_looks`` equally spaced looks.

    Uses the Lan–DeMets O'Brien-Fleming-type spending function
    α(t) = 2 − 2Φ(z_{1−α/2} / √t). Each boundary is solved so that the
    probability of first crossing at that look, under H0, equals the alpha spent
    since the previous look. That probability comes from recursive numerical
    integration of the score process (Armitage, McPherson & Rowe) with
    Simpson's rule on an 801-point grid, rather than the exact multivariate
    normal, so boundaries are accurate to roughly 1e-4 in z. Stop at look k
    when |Z_k| ≥ boundary[k].
    """
    if n_looks < 1:
        raise ValueError("n_looks must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    times = np.arange(1, n_looks + 1) / n_looks
    spent = 2.0 * norm.sf(norm.isf(alpha / 2) / np.sqrt(times))
    increments = np.diff(spent, prepend=0.0)

    # Score process S_k = √k · Z_k has independent N(0, 1) steps between equally spaced looks.
    first = float(norm.isf(increments[0] / 2))
    boundaries = [first]
    grid = np.linspace(-first, first, _GRID_POINTS)
    mass = _simpson_weights(grid) * norm.pdf(grid)
    for look in range(2, n_looks + 1):
        bound = brentq(
            _excess_crossing, 0.0, 60.0, args=(grid, mass, increments[look - 1]), xtol=1e-10
        )
        boundaries.append(bound / np.sqrt(look))
        new_grid = np.linspace(-bound, bound, _GRID_POINTS)
        density = norm.pdf(new_grid[:, None] - grid[None, :]) @ mass
        grid, mass = new_grid, _simpson_weights(new_grid) * density
    return [float(b) for b in boundaries]
