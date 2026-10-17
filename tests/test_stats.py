import math

import numpy as np
import pytest
from scipy.stats import chi2_contingency, chisquare, norm, ttest_ind

from theraflo_ads.experiments.stats import (
    TestResult,
    cuped_adjust,
    cuped_ttest,
    holm_bonferroni,
    non_inferiority_test,
    ratio_metric_delta,
    srm_check,
    two_proportion_ztest,
    welch_ttest,
    winsorize,
)


def test_ztest_matches_uncorrected_chi_square():
    result = two_proportion_ztest(300, 1_000_000, 360, 1_000_000)
    table = [[300, 1_000_000 - 300], [360, 1_000_000 - 360]]
    chi2_stat, chi2_p, _, _ = chi2_contingency(table, correction=False)
    assert result.statistic**2 == pytest.approx(chi2_stat, rel=1e-9)
    assert result.p_value == pytest.approx(chi2_p, rel=1e-9)
    assert result.statistic > 0
    assert result.method == "two_proportion_z"


def test_ztest_interval_uses_unpooled_se():
    result = two_proportion_ztest(100, 1000, 130, 1000)
    se = math.sqrt(0.1 * 0.9 / 1000 + 0.13 * 0.87 / 1000)
    assert result.estimate == pytest.approx(0.03)
    assert result.relative_lift == pytest.approx(0.3)
    assert result.ci_low == pytest.approx(0.03 - norm.isf(0.025) * se)
    assert result.ci_high == pytest.approx(0.03 + norm.isf(0.025) * se)
    assert result.relative_ci_low < 0.3 < result.relative_ci_high


def test_ztest_alpha_widens_interval():
    narrow = two_proportion_ztest(100, 1000, 130, 1000, alpha=0.10)
    wide = two_proportion_ztest(100, 1000, 130, 1000, alpha=0.01)
    assert wide.ci_high - wide.ci_low > narrow.ci_high - narrow.ci_low


def test_ztest_with_no_conversions_anywhere():
    result = two_proportion_ztest(0, 5000, 0, 5000)
    assert result.p_value == 1.0
    assert result.statistic == 0.0
    assert result.relative_lift is None
    assert result.relative_ci_low is None


@pytest.mark.parametrize("args", [(1, 0, 1, 10), (11, 10, 1, 10), (-1, 10, 1, 10)])
def test_ztest_validation(args):
    with pytest.raises(ValueError):
        two_proportion_ztest(*args)


def test_welch_matches_scipy():
    rng = np.random.default_rng(1)
    control = rng.normal(10.0, 2.0, 400)
    treatment = rng.normal(10.4, 3.5, 250)
    result = welch_ttest(control, treatment)
    reference = ttest_ind(treatment, control, equal_var=False)
    interval = reference.confidence_interval(0.95)
    assert result.statistic == pytest.approx(reference.statistic, rel=1e-9)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-9)
    assert result.ci_low == pytest.approx(interval.low, rel=1e-9)
    assert result.ci_high == pytest.approx(interval.high, rel=1e-9)
    assert result.estimate == pytest.approx(treatment.mean() - control.mean())
    assert result.relative_lift == pytest.approx(result.estimate / control.mean())


def test_relative_interval_accounts_for_control_noise():
    rng = np.random.default_rng(2)
    control = rng.normal(5.0, 2.0, 1000)
    treatment = rng.normal(5.5, 2.0, 1000)
    result = welch_ttest(control, treatment)
    fixed_denominator_half = (result.ci_high - result.ci_low) / 2 / control.mean()
    relative_half = (result.relative_ci_high - result.relative_ci_low) / 2
    assert relative_half > fixed_denominator_half
    assert result.relative_ci_low < result.relative_lift < result.relative_ci_high


def test_welch_rejects_bad_input():
    with pytest.raises(ValueError):
        welch_ttest(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        welch_ttest(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_welch_constant_arms():
    result = welch_ttest(np.zeros(10), np.zeros(10))
    assert result.p_value == 1.0
    assert result.ci_low == result.ci_high == 0.0


def _user_level_clicks(rng, n, ctr_alpha):
    impressions = 1 + rng.negative_binomial(2, 0.1, n)
    ctr = rng.beta(ctr_alpha, 30.0, n)
    return rng.binomial(impressions, ctr).astype(float), impressions.astype(float)


def _bootstrap_ratio_se(rng, num_c, den_c, num_t, den_t, draws=1000):
    def ratios(num, den):
        idx = rng.integers(0, num.size, (draws, num.size))
        return num[idx].sum(axis=1) / den[idx].sum(axis=1)

    return float(np.std(ratios(num_t, den_t) - ratios(num_c, den_c), ddof=1))


def test_ratio_delta_matches_bootstrap():
    rng = np.random.default_rng(3)
    num_c, den_c = _user_level_clicks(rng, 4000, 1.0)
    num_t, den_t = _user_level_clicks(rng, 4000, 1.1)
    result = ratio_metric_delta(num_c, den_c, num_t, den_t)
    assert result.estimate == pytest.approx(num_t.sum() / den_t.sum() - num_c.sum() / den_c.sum())
    delta_se = (result.ci_high - result.ci_low) / (2 * norm.isf(0.025))
    bootstrap_se = _bootstrap_ratio_se(rng, num_c, den_c, num_t, den_t)
    assert delta_se == pytest.approx(bootstrap_se, rel=0.15)
    assert result.method == "ratio_delta"


def test_ratio_delta_is_wider_than_impression_level_binomial():
    # Users differ in CTR, so impressions are not independent; the naive SE is too small.
    rng = np.random.default_rng(4)
    num_c, den_c = _user_level_clicks(rng, 4000, 1.0)
    num_t, den_t = _user_level_clicks(rng, 4000, 1.0)
    delta = ratio_metric_delta(num_c, den_c, num_t, den_t)
    naive = two_proportion_ztest(
        int(num_c.sum()), int(den_c.sum()), int(num_t.sum()), int(den_t.sum())
    )
    assert delta.ci_high - delta.ci_low > 1.2 * (naive.ci_high - naive.ci_low)


def test_ratio_delta_validation():
    ones = np.ones(5)
    with pytest.raises(ValueError):
        ratio_metric_delta(ones, np.ones(4), ones, ones)
    with pytest.raises(ValueError):
        ratio_metric_delta(ones, np.zeros(5), ones, ones)


def _pre_post(rng, n, effect):
    x = rng.normal(20.0, 5.0, n)
    y = 3.0 + 0.8 * x + rng.normal(0.0, 1.5, n) + effect
    return y, x


def test_cuped_adjust_recovers_slope_and_keeps_mean():
    rng = np.random.default_rng(5)
    y, x = _pre_post(rng, 5000, 0.0)
    adjusted, theta = cuped_adjust(y, x)
    assert theta == pytest.approx(0.8, abs=0.02)
    assert adjusted.mean() == pytest.approx(y.mean())
    assert adjusted.var() < 0.15 * y.var()


def test_cuped_adjust_with_constant_covariate_is_identity():
    y = np.array([1.0, 2.0, 3.0])
    adjusted, theta = cuped_adjust(y, np.full(3, 7.0))
    assert theta == 0.0
    np.testing.assert_allclose(adjusted, y)


def test_cuped_narrows_interval_without_bias():
    rng = np.random.default_rng(6)
    effect = 0.5
    cuped_estimates = []
    for _ in range(200):
        y_c, x_c = _pre_post(rng, 1000, 0.0)
        y_t, x_t = _pre_post(rng, 1000, effect)
        cuped = cuped_ttest(y_c, x_c, y_t, x_t)
        cuped_estimates.append(cuped.estimate)
    plain = welch_ttest(y_c, y_t)
    assert cuped.ci_high - cuped.ci_low < 0.5 * (plain.ci_high - plain.ci_low)
    assert cuped.ci_low < effect < cuped.ci_high
    assert cuped.method == "cuped_welch_t"
    # 200 replicates with per-replicate SE ~0.07 put the mean within ~0.005 of the truth.
    assert np.mean(cuped_estimates) == pytest.approx(effect, abs=0.02)


def test_cuped_with_useless_covariate_matches_welch():
    rng = np.random.default_rng(7)
    y_c, y_t = rng.normal(0, 1, 2000), rng.normal(0.1, 1, 2000)
    x_c, x_t = rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)
    cuped = cuped_ttest(y_c, x_c, y_t, x_t)
    plain = welch_ttest(y_c, y_t)
    assert cuped.ci_high - cuped.ci_low == pytest.approx(plain.ci_high - plain.ci_low, rel=0.02)


def test_cuped_validation():
    with pytest.raises(ValueError):
        cuped_ttest(np.ones(5), np.ones(4), np.ones(5), np.ones(5))


def test_winsorize_upper_tail():
    values = np.arange(1.0, 101.0)
    clipped = winsorize(values)
    assert clipped.max() == pytest.approx(np.quantile(values, 0.99))
    assert clipped.min() == 1.0
    assert values.max() == 100.0


def test_winsorize_both_tails():
    values = np.concatenate([[-1000.0], np.zeros(98), [1000.0]])
    clipped = winsorize(values, upper_quantile=0.95, lower_quantile=0.05)
    assert clipped.max() == 0.0
    assert clipped.min() == 0.0


@pytest.mark.parametrize(("upper", "lower"), [(0.0, None), (1.5, None), (0.9, 0.95)])
def test_winsorize_validation(upper, lower):
    with pytest.raises(ValueError):
        winsorize(np.arange(10.0), upper_quantile=upper, lower_quantile=lower)


def _result(ci_low, ci_high):
    return TestResult(
        estimate=(ci_low + ci_high) / 2,
        relative_lift=None,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=0.5,
        statistic=0.0,
        method="fixed",
    )


def test_non_inferiority():
    passed = non_inferiority_test(_result(-0.004, 0.01), margin=0.005)
    assert passed.passed
    assert passed.lower_bound == -0.004
    assert not non_inferiority_test(_result(-0.006, 0.01), margin=0.005).passed
    assert not non_inferiority_test(_result(-0.005, 0.01), margin=0.005).passed
    with pytest.raises(ValueError):
        non_inferiority_test(_result(0.0, 1.0), margin=-0.1)


def test_srm_balanced_passes():
    result = srm_check(50_000, 50_120)
    assert result.passed
    assert result.observed_ratio == pytest.approx(50_120 / 50_000)


def test_srm_mismatch_fails():
    result = srm_check(50_000, 48_500)
    assert not result.passed
    reference = chisquare([50_000, 48_500])
    assert result.statistic == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.pvalue)


def test_srm_respects_expected_ratio():
    assert srm_check(33_400, 66_600, expected_ratio=2.0).passed
    assert not srm_check(33_400, 66_600).passed


@pytest.mark.parametrize(
    "kwargs",
    [{"n_control": 0, "n_treatment": 0}, {"n_control": 1, "n_treatment": 1, "expected_ratio": 0}],
)
def test_srm_validation(kwargs):
    with pytest.raises(ValueError):
        srm_check(**kwargs)


def test_holm_bonferroni():
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert list(adjusted) == ["a", "b", "c", "d"]
    expected = {"a": (0.03, True), "b": (0.06, False), "c": (0.06, False), "d": (0.02, True)}
    for name, (p, reject) in expected.items():
        assert adjusted[name][0] == pytest.approx(p)
        assert adjusted[name][1] is reject


def test_holm_caps_at_one_and_handles_empty():
    assert holm_bonferroni({"x": 0.6, "y": 0.7}) == {"x": (1.0, False), "y": (1.0, False)}
    assert holm_bonferroni({}) == {}
    with pytest.raises(ValueError):
        holm_bonferroni({"x": 1.5})
