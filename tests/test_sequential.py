import numpy as np
import pytest
from scipy.stats import norm

from theraflo_ads.experiments.sequential import msprt_p_values, obrien_fleming_boundaries

LOOK_EVERY = 100
PEEKS = 20
TAU = 0.1


def _naive_p_values(control, treatment, look_every):
    """Fixed-horizon z-test p-values recomputed at every peek: the practice to avoid."""
    sizes = look_every * np.arange(1, control.size // look_every + 1)
    mean_c = np.cumsum(control)[sizes - 1] / sizes
    mean_t = np.cumsum(treatment)[sizes - 1] / sizes
    var_c = (np.cumsum(control**2)[sizes - 1] - sizes * mean_c**2) / (sizes - 1)
    var_t = (np.cumsum(treatment**2)[sizes - 1] - sizes * mean_t**2) / (sizes - 1)
    z = (mean_t - mean_c) / np.sqrt((var_c + var_t) / sizes)
    return 2 * norm.sf(np.abs(z))


def _simulate(rng, effect, experiments=300):
    n = LOOK_EVERY * PEEKS
    for _ in range(experiments):
        yield rng.standard_normal(n), rng.standard_normal(n) + effect


def test_peeking_inflates_type_one_error_but_msprt_does_not():
    rng = np.random.default_rng(11)
    naive_stops = msprt_stops = 0
    experiments = 300
    for control, treatment in _simulate(rng, 0.0, experiments):
        naive_stops += bool((_naive_p_values(control, treatment, LOOK_EVERY) < 0.05).any())
        msprt = msprt_p_values(control, treatment, tau=TAU, look_every=LOOK_EVERY)
        msprt_stops += bool((msprt < 0.05).any())
    assert naive_stops / experiments > 0.10
    assert msprt_stops / experiments <= 0.07


def test_msprt_detects_a_real_effect():
    rng = np.random.default_rng(12)
    detections = []
    for control, treatment in _simulate(rng, 0.2, experiments=100):
        p_values = msprt_p_values(control, treatment, tau=TAU, look_every=LOOK_EVERY)
        detections.append(p_values[-1] < 0.05)
    assert np.mean(detections) >= 0.9


def test_msprt_p_values_are_monotone_and_bounded():
    rng = np.random.default_rng(13)
    control, treatment = rng.standard_normal(2000), rng.standard_normal(2000) + 0.05
    p_values = msprt_p_values(control, treatment, tau=TAU, look_every=LOOK_EVERY)
    assert p_values.shape == (PEEKS,)
    assert np.all(np.diff(p_values) <= 0)
    assert np.all((p_values > 0) & (p_values <= 1))


def test_msprt_uses_shortest_arm_and_handles_large_means():
    rng = np.random.default_rng(14)
    control = 1e6 + rng.standard_normal(1050)
    treatment = 1e6 + rng.standard_normal(990)
    p_values = msprt_p_values(control, treatment, tau=TAU, look_every=LOOK_EVERY)
    assert p_values.shape == (9,)
    assert p_values[-1] > 0.05


def test_msprt_constant_data():
    assert np.all(msprt_p_values(np.zeros(10), np.zeros(10), tau=1.0, look_every=5) == 1.0)
    assert np.all(msprt_p_values(np.zeros(10), np.ones(10), tau=1.0, look_every=5) == 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau": 0.0, "look_every": 10},
        {"tau": 1.0, "look_every": 1},
        {"tau": 1.0, "look_every": 1000},
    ],
)
def test_msprt_validation(kwargs):
    with pytest.raises(ValueError):
        msprt_p_values(np.zeros(100), np.zeros(100), **kwargs)


def test_single_look_boundary_is_fixed_horizon():
    assert obrien_fleming_boundaries(1) == [pytest.approx(norm.isf(0.025))]
    assert obrien_fleming_boundaries(1, alpha=0.01) == [pytest.approx(norm.isf(0.005))]


@pytest.mark.parametrize("looks", [2, 5, 20])
def test_obrien_fleming_shape(looks):
    boundaries = obrien_fleming_boundaries(looks)
    assert len(boundaries) == looks
    assert np.all(np.diff(boundaries) < 0)
    # Spending at t = 1/K is exactly 2 − 2Φ(z / √t), so the first boundary is z·√K.
    assert boundaries[0] == pytest.approx(norm.isf(0.025) * np.sqrt(looks))
    assert norm.isf(0.025) < boundaries[-1] < norm.isf(0.025) + 0.25


@pytest.mark.parametrize("looks", [5, 10])
def test_obrien_fleming_spends_alpha_as_planned(looks):
    rng = np.random.default_rng(15)
    boundaries = np.array(obrien_fleming_boundaries(looks))
    paths = np.cumsum(rng.standard_normal((200_000, looks)), axis=1)
    z = paths / np.sqrt(np.arange(1, looks + 1))
    crossed_by = np.maximum.accumulate(np.abs(z) >= boundaries, axis=1).mean(axis=0)
    times = np.arange(1, looks + 1) / looks
    planned = 2 * norm.sf(norm.isf(0.025) / np.sqrt(times))
    np.testing.assert_allclose(crossed_by, planned, atol=0.002)


def test_obrien_fleming_validation():
    with pytest.raises(ValueError):
        obrien_fleming_boundaries(0)
    with pytest.raises(ValueError):
        obrien_fleming_boundaries(3, alpha=1.0)
