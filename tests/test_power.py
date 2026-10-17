import pytest

from theraflo_ads.experiments.power import (
    MDERow,
    days_to_detect,
    mde_table,
    mde_two_proportions,
    sample_size_mean,
    sample_size_two_proportions,
)

P_BASELINE = 0.0003

# Design doc table: (lift, units per arm, days @ 500K/day, days @ 112K/day, tolerance).
# The doc used 2·p(1−p) for both arms; separate arm variances add p·lift to the
# variance, which pushes the 20% row 11–15% above the doc. It stays a lower bound.
DOC_TABLE = [
    (0.02, 130e6, 520, 2300, 0.10),
    (0.05, 21e6, 84, 370, 0.10),
    (0.10, 5.2e6, 21, 93, 0.10),
    (0.20, 1.3e6, 5, 23, 0.16),
]


@pytest.mark.parametrize(("lift", "units", "days_500k", "days_112k", "tol"), DOC_TABLE)
def test_reproduces_design_doc_table(lift, units, days_500k, days_112k, tol):
    n = sample_size_two_proportions(P_BASELINE, lift)
    assert n == pytest.approx(units, rel=tol)
    assert n >= units * 0.99
    assert days_to_detect(P_BASELINE, lift, 500_000) == pytest.approx(days_500k, rel=tol)
    assert days_to_detect(P_BASELINE, lift, 112_000) == pytest.approx(days_112k, rel=tol)


def test_sample_size_matches_hand_computation():
    # (z_0.975 + z_0.8)² · (0.1·0.9 + 0.12·0.88) / 0.02² = 3,838.1
    assert sample_size_two_proportions(0.10, 0.20) == 3839


@pytest.mark.parametrize("n_per_arm", [100_000, 1_000_000, 5_200_000, 130_000_000])
def test_mde_inverts_sample_size(n_per_arm):
    mde = mde_two_proportions(P_BASELINE, n_per_arm)
    assert sample_size_two_proportions(P_BASELINE, mde) in (n_per_arm, n_per_arm + 1)


@pytest.mark.parametrize("lift", [0.02, 0.05, 0.10, 0.20])
def test_sample_size_inverts_mde(lift):
    n = sample_size_two_proportions(P_BASELINE, lift)
    assert mde_two_proportions(P_BASELINE, n) == pytest.approx(lift, rel=1e-4)


def test_mde_shrinks_with_more_units_and_grows_with_stricter_alpha():
    assert mde_two_proportions(P_BASELINE, 4e6) < mde_two_proportions(P_BASELINE, 1e6)
    assert mde_two_proportions(P_BASELINE, 1e6, alpha=0.01) > mde_two_proportions(P_BASELINE, 1e6)
    assert mde_two_proportions(P_BASELINE, 1e6, power=0.9) > mde_two_proportions(P_BASELINE, 1e6)


def test_undetectable_with_tiny_sample():
    with pytest.raises(ValueError, match="no lift"):
        mde_two_proportions(0.5, 1)


def test_unequal_ratio_needs_more_total_units():
    balanced = sample_size_two_proportions(P_BASELINE, 0.10)
    control = sample_size_two_proportions(P_BASELINE, 0.10, ratio=2.0)
    assert control < balanced
    assert control * 3 > balanced * 2


def test_even_split_is_fastest():
    even = days_to_detect(P_BASELINE, 0.10, 500_000)
    assert days_to_detect(P_BASELINE, 0.10, 500_000, split=0.2) > even
    assert days_to_detect(P_BASELINE, 0.10, 500_000, split=0.8) > even


def test_negative_lift_is_supported():
    assert sample_size_two_proportions(P_BASELINE, -0.10) < sample_size_two_proportions(
        P_BASELINE, 0.10
    )


def test_sample_size_mean():
    # 2 · (z_0.975 + z_0.8)² · 1² / 0.1² = 1,569.8
    assert sample_size_mean(1.0, 0.1) == 1570
    assert sample_size_mean(2.0, 0.1) == pytest.approx(4 * 1570, abs=4)


def test_mde_table():
    rows = mde_table(P_BASELINE, [0.05, 0.10], [500_000, 112_000])
    assert [row.relative_lift for row in rows] == [0.05, 0.10]
    assert all(isinstance(row, MDERow) for row in rows)
    assert rows[1].units_per_arm == sample_size_two_proportions(P_BASELINE, 0.10)
    assert rows[1].days == (
        days_to_detect(P_BASELINE, 0.10, 500_000),
        days_to_detect(P_BASELINE, 0.10, 112_000),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_baseline": 0.0, "relative_lift": 0.1},
        {"p_baseline": 1.0, "relative_lift": 0.1},
        {"p_baseline": 0.5, "relative_lift": 0.0},
        {"p_baseline": 0.5, "relative_lift": 1.0},
        {"p_baseline": 0.1, "relative_lift": 0.1, "alpha": 0.0},
        {"p_baseline": 0.1, "relative_lift": 0.1, "power": 1.0},
        {"p_baseline": 0.1, "relative_lift": 0.1, "ratio": 0.0},
    ],
)
def test_sample_size_validation(kwargs):
    with pytest.raises(ValueError):
        sample_size_two_proportions(**kwargs)


@pytest.mark.parametrize(
    "call",
    [
        lambda: mde_two_proportions(P_BASELINE, 0),
        lambda: days_to_detect(P_BASELINE, 0.1, 0),
        lambda: days_to_detect(P_BASELINE, 0.1, 1000, split=1.0),
        lambda: sample_size_mean(0.0, 0.1),
        lambda: sample_size_mean(1.0, 0.0),
        lambda: mde_table(P_BASELINE, [0.1], [1000], split=0.0),
    ],
)
def test_other_validation(call):
    with pytest.raises(ValueError):
        call()
