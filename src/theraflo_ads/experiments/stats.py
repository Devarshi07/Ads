"""Fixed-horizon tests (design doc: Statistical Testing; Experiment Hygiene).

Every test reports treatment − control with a confidence interval, plus a
relative lift and its delta-method interval so guardrails can be judged on a
relative scale. Units are users: ratio metrics use the delta method because
impressions from one user are not independent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm
from scipy.stats import t as t_dist


@dataclass(frozen=True)
class TestResult:
    # Stops pytest from trying to collect this as a test class when it is imported into tests.
    __test__ = False

    estimate: float
    relative_lift: float | None
    ci_low: float
    ci_high: float
    p_value: float
    statistic: float
    method: str
    relative_ci_low: float | None = None
    relative_ci_high: float | None = None


@dataclass(frozen=True)
class NonInferiorityResult:
    passed: bool
    lower_bound: float
    margin: float


@dataclass(frozen=True)
class SRMResult:
    passed: bool
    p_value: float
    statistic: float
    observed_ratio: float
    expected_ratio: float


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _as_sample(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a 1-D array with at least two values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _result(
    *,
    mean_control: float,
    var_control: float,
    mean_treatment: float,
    var_treatment: float,
    critical: float,
    statistic: float,
    p_value: float,
    method: str,
) -> TestResult:
    """Assemble a result from each arm's mean and the variance of that mean."""
    estimate = mean_treatment - mean_control
    half_width = critical * math.sqrt(var_control + var_treatment)
    relative_lift = relative_low = relative_high = None
    if mean_control != 0.0:
        relative_lift = estimate / mean_control
        # Delta method for mean_t / mean_c, so the control mean's own noise widens the interval.
        relative_var = (
            var_treatment / mean_control**2 + mean_treatment**2 * var_control / mean_control**4
        )
        relative_half = critical * math.sqrt(relative_var)
        relative_low = relative_lift - relative_half
        relative_high = relative_lift + relative_half
    return TestResult(
        estimate=estimate,
        relative_lift=relative_lift,
        ci_low=estimate - half_width,
        ci_high=estimate + half_width,
        p_value=p_value,
        statistic=statistic,
        method=method,
        relative_ci_low=relative_low,
        relative_ci_high=relative_high,
    )


def _two_sided(estimate: float, se: float, df: float | None = None) -> tuple[float, float]:
    """Test statistic and two-sided p-value; a zero SE is the limit of an exact result."""
    if se == 0.0:
        return (0.0, 1.0) if estimate == 0.0 else (math.copysign(math.inf, estimate), 0.0)
    statistic = estimate / se
    tail = norm.sf(abs(statistic)) if df is None else t_dist.sf(abs(statistic), df)
    return statistic, float(min(1.0, 2.0 * tail))


def two_proportion_ztest(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    *,
    alpha: float = 0.05,
) -> TestResult:
    """Pooled SE for the test statistic (exact under H0), unpooled SE for the interval."""
    _check_alpha(alpha)
    for successes, n in ((successes_control, n_control), (successes_treatment, n_treatment)):
        if n <= 0 or not 0 <= successes <= n:
            raise ValueError("each arm needs n > 0 and 0 <= successes <= n")
    p_c = successes_control / n_control
    p_t = successes_treatment / n_treatment
    pooled = (successes_control + successes_treatment) / (n_control + n_treatment)
    pooled_se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_control + 1.0 / n_treatment))
    statistic, p_value = _two_sided(p_t - p_c, pooled_se)
    return _result(
        mean_control=p_c,
        var_control=p_c * (1.0 - p_c) / n_control,
        mean_treatment=p_t,
        var_treatment=p_t * (1.0 - p_t) / n_treatment,
        critical=float(norm.isf(alpha / 2)),
        statistic=statistic,
        p_value=p_value,
        method="two_proportion_z",
    )


def _welch(control: np.ndarray, treatment: np.ndarray, alpha: float, method: str) -> TestResult:
    var_c = control.var(ddof=1) / control.size
    var_t = treatment.var(ddof=1) / treatment.size
    total = var_c + var_t
    if total == 0.0:
        df = math.inf
    else:
        df = total**2 / (var_c**2 / (control.size - 1) + var_t**2 / (treatment.size - 1))
    mean_c = float(control.mean())
    mean_t = float(treatment.mean())
    statistic, p_value = _two_sided(mean_t - mean_c, math.sqrt(total), df)
    return _result(
        mean_control=mean_c,
        var_control=float(var_c),
        mean_treatment=mean_t,
        var_treatment=float(var_t),
        critical=float(t_dist.isf(alpha / 2, df)),
        statistic=statistic,
        p_value=p_value,
        method=method,
    )


def welch_ttest(control: np.ndarray, treatment: np.ndarray, *, alpha: float = 0.05) -> TestResult:
    """Unequal-variance t-test with Welch–Satterthwaite degrees of freedom."""
    _check_alpha(alpha)
    return _welch(
        _as_sample(control, "control"), _as_sample(treatment, "treatment"), alpha, "welch_t"
    )


def _ratio_and_variance(numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, float]:
    n = numerator.size
    mean_num = numerator.mean()
    mean_den = denominator.mean()
    if mean_den <= 0.0:
        raise ValueError("ratio denominators must have a positive sum")
    ratio = mean_num / mean_den
    (var_num, cov), (_, var_den) = np.cov(numerator, denominator, ddof=1)
    variance = (var_num - 2.0 * ratio * cov + ratio**2 * var_den) / (n * mean_den**2)
    return float(ratio), float(max(variance, 0.0))


def ratio_metric_delta(
    num_control: np.ndarray,
    den_control: np.ndarray,
    num_treatment: np.ndarray,
    den_treatment: np.ndarray,
    *,
    alpha: float = 0.05,
) -> TestResult:
    """Difference of ratio-of-sums metrics, one array element per randomized user."""
    _check_alpha(alpha)
    arms = []
    for num, den, name in (
        (num_control, den_control, "control"),
        (num_treatment, den_treatment, "treatment"),
    ):
        num_array = _as_sample(num, f"{name} numerator")
        den_array = _as_sample(den, f"{name} denominator")
        if num_array.size != den_array.size:
            raise ValueError(f"{name} numerator and denominator lengths differ")
        arms.append(_ratio_and_variance(num_array, den_array))
    (ratio_c, var_c), (ratio_t, var_t) = arms
    statistic, p_value = _two_sided(ratio_t - ratio_c, math.sqrt(var_c + var_t))
    return _result(
        mean_control=ratio_c,
        var_control=var_c,
        mean_treatment=ratio_t,
        var_treatment=var_t,
        critical=float(norm.isf(alpha / 2)),
        statistic=statistic,
        p_value=p_value,
        method="ratio_delta",
    )


def _theta(x_centered: np.ndarray, y_centered: np.ndarray) -> float:
    denominator = float(np.dot(x_centered, x_centered))
    return float(np.dot(x_centered, y_centered)) / denominator if denominator > 0 else 0.0


def cuped_adjust(y: np.ndarray, x_pre: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED-adjusted outcome y − θ(x − x̄) with the variance-minimizing θ; the mean is kept."""
    y_array = _as_sample(y, "y")
    x_array = _as_sample(x_pre, "x_pre")
    if y_array.size != x_array.size:
        raise ValueError("y and x_pre lengths differ")
    x_centered = x_array - x_array.mean()
    theta = _theta(x_centered, y_array - y_array.mean())
    return y_array - theta * x_centered, theta


def cuped_ttest(
    y_control: np.ndarray,
    x_control: np.ndarray,
    y_treatment: np.ndarray,
    x_treatment: np.ndarray,
    *,
    alpha: float = 0.05,
) -> TestResult:
    """Welch test on CUPED-adjusted outcomes with one θ shared by both arms.

    θ comes from within-arm deviations, so the treatment effect itself does not
    leak into it, and both arms are centered on the same pooled covariate mean,
    which keeps the difference in adjusted means unbiased.
    """
    _check_alpha(alpha)
    yc, xc = _as_sample(y_control, "y_control"), _as_sample(x_control, "x_control")
    yt, xt = _as_sample(y_treatment, "y_treatment"), _as_sample(x_treatment, "x_treatment")
    if yc.size != xc.size or yt.size != xt.size:
        raise ValueError("outcome and covariate lengths differ")
    theta = _theta(
        np.concatenate([xc - xc.mean(), xt - xt.mean()]),
        np.concatenate([yc - yc.mean(), yt - yt.mean()]),
    )
    x_mean = np.concatenate([xc, xt]).mean()
    return _welch(yc - theta * (xc - x_mean), yt - theta * (xt - x_mean), alpha, "cuped_welch_t")


def winsorize(
    values: np.ndarray, *, upper_quantile: float = 0.99, lower_quantile: float | None = None
) -> np.ndarray:
    """Clip heavy tails at the given quantiles of ``values`` (lower tail untouched by default)."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array.copy()
    if not 0.0 < upper_quantile <= 1.0:
        raise ValueError("upper_quantile must be in (0, 1]")
    if lower_quantile is not None and not 0.0 <= lower_quantile < upper_quantile:
        raise ValueError("lower_quantile must be in [0, upper_quantile)")
    low = -np.inf if lower_quantile is None else np.quantile(array, lower_quantile)
    return np.clip(array, low, np.quantile(array, upper_quantile))


def non_inferiority_test(result: TestResult, margin: float) -> NonInferiorityResult:
    """Passes when the CI lower bound on treatment − control is above −margin."""
    if margin < 0:
        raise ValueError("margin must be non-negative")
    return NonInferiorityResult(
        passed=result.ci_low > -margin, lower_bound=result.ci_low, margin=margin
    )


def srm_check(
    n_control: int, n_treatment: int, *, expected_ratio: float = 1.0, alpha: float = 0.001
) -> SRMResult:
    """Chi-square goodness of fit on arm sizes; ``expected_ratio`` is treatment / control.

    The strict default alpha is deliberate: SRM runs on every experiment, and a
    failure means assignment or logging is broken, not that the effect is real.
    """
    _check_alpha(alpha)
    if n_control < 0 or n_treatment < 0 or n_control + n_treatment == 0:
        raise ValueError("arm sizes must be non-negative with a positive total")
    if expected_ratio <= 0:
        raise ValueError("expected_ratio must be positive")
    total = n_control + n_treatment
    expected_control = total / (1.0 + expected_ratio)
    expected_treatment = total - expected_control
    statistic = (n_control - expected_control) ** 2 / expected_control + (
        n_treatment - expected_treatment
    ) ** 2 / expected_treatment
    p_value = float(chi2.sf(statistic, df=1))
    return SRMResult(
        passed=p_value >= alpha,
        p_value=p_value,
        statistic=float(statistic),
        observed_ratio=n_treatment / n_control if n_control > 0 else math.inf,
        expected_ratio=expected_ratio,
    )


def holm_bonferroni(
    p_values: Mapping[str, float], *, alpha: float = 0.05
) -> dict[str, tuple[float, bool]]:
    """Holm step-down adjusted p-values and reject decisions, in the input order."""
    _check_alpha(alpha)
    if any(not 0.0 <= p <= 1.0 for p in p_values.values()):
        raise ValueError("p-values must be in [0, 1]")
    m = len(p_values)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(sorted(p_values.items(), key=lambda item: item[1])):
        running_max = max(running_max, min(1.0, (m - rank) * p))
        adjusted[name] = running_max
    return {name: (adjusted[name], adjusted[name] <= alpha) for name in p_values}
