"""Learning-curve / scaling research (design doc: Phase 9 — Learning-Curve / Scaling Research).

Controlled experiments on a fixed, time-split dataset:

    data volume     quality vs training rows (user-grouped subsamples, several seeds),
                    with a saturating power law  NE(n) = a + b·n^(-c)  fitted to the means
    model size      quality, training time, and serving latency vs ensemble size
    features        cumulative feature-group ablation
    business value  simulated revenue vs model complexity, against serving cost

Quality is normalized entropy (NE = log loss / entropy of the base rate; < 1 beats
predicting the average), plus AUC and ECE. Every number comes from simulated data:
the curves show how to run the study and read it, not what production would show.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from theraflo_ads.ml import dataset
from theraflo_ads.ml.evaluation import classification_metrics
from theraflo_ads.ml.models import CalibratedModel, LightGBMModel, LogisticModel, PriorModel
from theraflo_ads.ml.post_training import fit_calibrator

FEATURE_GROUPS: dict[str, Callable[[str], bool]] = {
    "ad": lambda n: (
        n.startswith(("cat_", "pricing_")) or n in ("log_price", "prior_ctr", "prior_cvr")
    ),
    "context": lambda n: n.startswith(("placement_", "daypart_", "device_")),
    "exposure": lambda n: n in ("ad_exposures_24h", "user_exposures_24h"),
    "personalization": lambda n: (
        n in ("personalized", "user_cat_ctr", "user_cat_impressions", "user_total_clicks")
    ),
}
# Assumption for the cost axis, stated in the report: on-demand vCPU price.
VCPU_HOUR_USD = 0.04


@dataclass
class Data:
    names: list[str]
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def xy(self, frame: pd.DataFrame, names: Sequence[str] | None = None):
        cols = [f"{dataset.FEATURE_PREFIX}{n}" for n in (names or self.names)]
        return frame[cols].to_numpy(float), frame["clicked"].to_numpy(float)


def load(features: Path) -> Data:
    info = json.loads((features / "dataset.json").read_text(encoding="utf-8"))
    frames = [pd.read_parquet(features / f"{s}.parquet") for s in ("train", "validation", "test")]
    return Data(info["feature_names"], *frames)


def normalized_entropy(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    log_loss = -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    base = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    entropy = -(base * math.log(base) + (1 - base) * math.log(1 - base))
    return log_loss / entropy


@dataclass
class Fit:
    model: CalibratedModel
    seconds: float
    leaves: int


def fit_model(
    data: Data,
    train: pd.DataFrame,
    *,
    algorithm: str = "lightgbm",
    num_leaves: int = 15,
    names: Sequence[str] | None = None,
    rounds: int = 400,
) -> Fit:
    names = list(names or data.names)
    x, y = data.xy(train, names)
    xv, yv = data.xy(data.validation, names)
    started = time.perf_counter()
    if algorithm == "prior":
        model = PriorModel(tuple(names))  # the heuristic: category prior CTR, calibrated
        leaves = 0
    elif algorithm == "logistic":
        model = LogisticModel.fit(x, y, names, c=0.5)
        leaves = len(names) + 1
    else:
        model = LightGBMModel.fit(
            x,
            y,
            names,
            valid=(xv, yv),
            num_boost_round=rounds,
            params={"num_leaves": num_leaves, "min_data_in_leaf": 20 if num_leaves <= 4 else 50},
        )
        leaves = (model.booster.best_iteration or model.booster.current_iteration()) * num_leaves
    calibrated = fit_calibrator(CalibratedModel(model), xv, yv, kind="isotonic")
    return Fit(calibrated, time.perf_counter() - started, leaves)


def ne_interval(y: np.ndarray, p: np.ndarray, *, n_boot: int = 200, seed: int = 0):
    """95% bootstrap interval of NE over test rows. With a few hundred test clicks,
    differences of a few thousandths of NE are inside it."""
    rng = np.random.default_rng(seed)
    values = [
        normalized_entropy(y[idx], p[idx])
        for idx in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
    ]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def score(data: Data, fit: Fit, names: Sequence[str] | None = None) -> dict[str, Any]:
    x, y = data.xy(data.test, names)
    p = fit.model.predict(x)
    metrics = classification_metrics(y, p)
    return {
        "ne": normalized_entropy(y, p),
        "ne_ci95": ne_interval(y, p),
        "auc": metrics["auc"],
        "ece": metrics["ece"],
        "log_loss": metrics["log_loss"],
    }


def latency_ms(
    data: Data,
    fit: Fit,
    *,
    candidates: int = 10,
    trials: int = 300,
    names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per-request scoring latency: one call scoring `candidates` ads, as on the serving path."""
    x, _ = data.xy(data.test.head(candidates * 20), names)
    samples = []
    for i in range(trials):
        batch = x[(i % 20) * candidates : (i % 20 + 1) * candidates]
        started = time.perf_counter()
        fit.model.predict(batch)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return {"p50": samples[len(samples) // 2], "p99": samples[int(len(samples) * 0.99) - 1]}


def user_subsample(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Subsample whole users, so a smaller dataset looks like fewer users, not thinner logs."""
    users = frame["user_hash"].unique()
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(users, size=max(1, int(len(users) * fraction)), replace=False))
    return frame[frame["user_hash"].isin(keep)]


# ----- experiments -----


@dataclass
class CurvePoint:
    fraction: float
    rows: float
    clicks: float
    ne_mean: float
    ne_sd: float
    auc_mean: float
    ece_mean: float


def data_volume(
    data: Data, fractions: Sequence[float], seeds: Sequence[int], algorithm: str = "lightgbm"
) -> list[CurvePoint]:
    points = []
    for fraction in fractions:
        runs = []
        for seed in seeds if fraction < 1.0 else seeds[:1]:
            train = user_subsample(data.train, fraction, seed) if fraction < 1.0 else data.train
            fit = fit_model(data, train, algorithm=algorithm)
            runs.append((len(train), float(train.clicked.sum()), score(data, fit)))
        ne = [r[2]["ne"] for r in runs]
        points.append(
            CurvePoint(
                fraction,
                float(np.mean([r[0] for r in runs])),
                float(np.mean([r[1] for r in runs])),
                float(np.mean(ne)),
                float(np.std(ne)),
                float(np.mean([r[2]["auc"] for r in runs])),
                float(np.mean([r[2]["ece"] for r in runs])),
            )
        )
    return points


@dataclass
class PowerLaw:
    a: float  # irreducible NE at infinite data
    b: float
    c: float  # how fast extra data helps
    fitted: bool
    rows_for_half_gap: float | None = None  # rows needed to close half the gap to `a`

    def predict(self, n: float) -> float:
        return self.a + self.b * n ** (-self.c)


def fit_power_law(points: Sequence[CurvePoint]) -> PowerLaw:
    n = np.array([p.rows for p in points])
    ne = np.array([p.ne_mean for p in points])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (a, b, c), _ = curve_fit(
                lambda x, a, b, c: a + b * np.power(x, -c),
                n,
                ne,
                p0=(ne.min() * 0.95, 1.0, 0.5),
                bounds=([0, 0, 0.01], [2, 1e6, 3]),
                maxfev=20000,
            )
    except (RuntimeError, ValueError):
        return PowerLaw(float(ne.min()), 0.0, 0.0, False)
    law = PowerLaw(float(a), float(b), float(c), True)
    gap_now = law.predict(n.max()) - a
    if gap_now > 0 and c > 0:
        law.rows_for_half_gap = float((b / (gap_now / 2)) ** (1 / c))
    return law


@dataclass
class SizePoint:
    model: str
    leaves: int
    ne: float
    ne_ci95: tuple[float, float]
    auc: float
    ece: float
    train_seconds: float
    latency_p50_ms: float
    latency_p99_ms: float


def model_size(data: Data, leaves: Sequence[int]) -> list[SizePoint]:
    points = []
    fits = [
        ("category prior (heuristic)", fit_model(data, data.train, algorithm="prior")),
        ("logistic", fit_model(data, data.train, algorithm="logistic")),
    ]
    fits += [(f"lightgbm_{n}", fit_model(data, data.train, num_leaves=n)) for n in leaves]
    for name, fit in fits:
        quality, latency = score(data, fit), latency_ms(data, fit)
        points.append(
            SizePoint(
                name,
                fit.leaves,
                quality["ne"],
                tuple(quality["ne_ci95"]),
                quality["auc"],
                quality["ece"],
                fit.seconds,
                latency["p50"],
                latency["p99"],
            )
        )
    return points


@dataclass
class AblationPoint:
    features: str
    n_features: int
    ne: float
    ne_ci95: tuple[float, float]
    auc: float
    ece: float


def feature_ablation(data: Data) -> list[AblationPoint]:
    points = []
    included: list[str] = []
    for group, member in FEATURE_GROUPS.items():
        included += [n for n in data.names if member(n)]
        fit = fit_model(data, data.train, names=included)
        quality = score(data, fit, included)
        points.append(
            AblationPoint(
                f"+{group}",
                len(included),
                quality["ne"],
                tuple(quality["ne_ci95"]),
                quality["auc"],
                quality["ece"],
            )
        )
    return points


@dataclass
class ValuePoint:
    variant: str
    leaves: int
    revenue_per_1k_requests: float
    lift_vs_heuristic: float
    lift_per_seed: tuple[float, ...]
    serving_cost_per_1m_requests_usd: float
    net_gain_per_1m_requests_usd: float


def business_value(
    data: Data,
    variants: Sequence[int],
    *,
    users: int,
    days: int,
    out: Path,
    seeds: Sequence[int] = (21, 22),
) -> list[ValuePoint]:
    """Simulated revenue of the ML ranker at several model sizes, each run on the same world
    and random stream as the heuristic (common random numbers), for each seed, against
    serving cost."""
    from theraflo_ads.ml.bundle import save_bundle
    from theraflo_ads.ml.features import FEATURE_SCHEMA_VERSION
    from theraflo_ads.simulator.ab import heuristic_arm, ml_arm
    from theraflo_ads.simulator.runner import PolicySetup, run_simulation
    from theraflo_ads.simulator.world import SimConfig

    def revenue(factory) -> list[float]:
        per_seed = []
        for seed in seeds:
            config = SimConfig(n_users=users, n_days=days, seed=7, run_seed=seed)
            result = run_simulation(config, lambda world: PolicySetup(list(factory(world))))
            per_seed.append(result.platform_revenue / max(len(result.decisions), 1) * 1000)
        return per_seed

    base = revenue(heuristic_arm)
    baseline = float(np.mean(base))
    points = [ValuePoint("heuristic", 0, baseline, 0.0, (0.0,) * len(seeds), 0.0, 0.0)]
    for leaves in variants:
        fit = fit_model(data, data.train, num_leaves=leaves)
        bundle = out / f"ctr_leaves_{leaves}"
        save_bundle(
            bundle,
            name="ctr",
            version=f"leaves{leaves}",
            model=fit.model,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )
        per_seed = revenue(ml_arm(bundle))
        value = float(np.mean(per_seed))
        lifts = tuple(v / b - 1 if b else 0.0 for v, b in zip(per_seed, base, strict=True))
        cpu_seconds = latency_ms(data, fit)["p50"] / 1000
        cost = cpu_seconds * 1e6 * VCPU_HOUR_USD / 3600
        gain = (value - baseline) * 1000  # per 1M requests
        points.append(
            ValuePoint(
                f"lightgbm_{leaves}",
                fit.leaves,
                value,
                value / baseline - 1 if baseline else 0.0,
                lifts,
                cost,
                gain - cost,
            )
        )
    return points


@dataclass
class ScalingReport:
    dataset: dict[str, Any]
    data_volume: list[CurvePoint]
    power_law: PowerLaw
    data_volume_logistic: list[CurvePoint]
    model_size: list[SizePoint]
    features: list[AblationPoint]
    business_value: list[ValuePoint] = field(default_factory=list)
    assumptions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run(
    features: Path,
    out: Path,
    *,
    fractions: Sequence[float] = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
    seeds: Sequence[int] = (1, 2, 3),
    leaves: Sequence[int] = (2, 4, 8, 16, 32, 64, 128),
    value_variants: Sequence[int] = (4, 16, 64),
    sim_users: int = 3000,
    sim_days: int = 14,
    progress: Callable[[str], None] = lambda message: None,
) -> ScalingReport:
    data = load(features)
    out.mkdir(parents=True, exist_ok=True)
    progress("data volume (lightgbm)")
    curve = data_volume(data, fractions, seeds)
    progress("data volume (logistic)")
    curve_lr = data_volume(data, fractions, seeds[:1], algorithm="logistic")
    progress("model size")
    sizes = model_size(data, leaves)
    progress("feature ablation")
    ablation = feature_ablation(data)
    value: list[ValuePoint] = []
    if value_variants:
        progress("business value (simulator)")
        value = business_value(
            data, value_variants, users=sim_users, days=sim_days, out=out / "bundles"
        )
    report = ScalingReport(
        dataset={
            "train_rows": len(data.train),
            "train_clicks": int(data.train.clicked.sum()),
            "test_rows": len(data.test),
            "test_clicks": int(data.test.clicked.sum()),
            "features": len(data.names),
        },
        data_volume=curve,
        power_law=fit_power_law(curve),
        data_volume_logistic=curve_lr,
        model_size=sizes,
        features=ablation,
        business_value=value,
        assumptions={
            "vcpu_hour_usd": VCPU_HOUR_USD,
            "latency": "single-thread, 10 candidates",
            "simulation": {"users": sim_users, "days": sim_days},
        },
    )
    (out / "scaling.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (out / "scaling.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: ScalingReport) -> str:
    d = report.dataset
    law = report.power_law
    lines = [
        "# Scaling study",
        "",
        f"Dataset: {d['train_rows']:,} training rows ({d['train_clicks']:,} clicks), "
        f"{d['test_rows']:,} test rows ({d['test_clicks']:,} clicks), {d['features']} features. "
        "Simulated data: read the shapes, not the absolute values.",
        "",
        "## Quality vs data (LightGBM, user-grouped subsamples)",
        "",
        "| share | rows | clicks | NE (mean ± sd) | AUC | ECE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for p in report.data_volume:
        lines.append(
            f"| {p.fraction:.0%} | {p.rows:,.0f} | {p.clicks:,.0f} | "
            f"{p.ne_mean:.4f} ± {p.ne_sd:.4f} | {p.auc_mean:.3f} | {p.ece_mean:.4f} |"
        )
    if law.fitted:
        half = f"{law.rows_for_half_gap:,.0f}" if law.rows_for_half_gap else "n/a"
        lines += [
            "",
            f"Power law: NE(n) = {law.a:.4f} + {law.b:.3g}·n^(-{law.c:.3f}). "
            f"Irreducible NE ≈ {law.a:.4f}; closing half the remaining gap needs ≈ "
            f"{half} rows.",
        ]
    lines += [
        "",
        "Logistic regression on the same subsamples:",
        "",
        "| share | rows | NE | AUC |",
        "|---:|---:|---:|---:|",
    ]
    for p in report.data_volume_logistic:
        lines.append(f"| {p.fraction:.0%} | {p.rows:,.0f} | {p.ne_mean:.4f} | {p.auc_mean:.3f} |")
    lines += [
        "",
        "## Quality, training time, and latency vs model size",
        "",
        "| model | leaves | NE [95% CI] | AUC | ECE | train s | p50 ms | p99 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in report.model_size:
        lines.append(
            f"| {p.model} | {p.leaves:,} | {p.ne:.4f} [{p.ne_ci95[0]:.4f}, "
            f"{p.ne_ci95[1]:.4f}] | {p.auc:.3f} | {p.ece:.4f} | {p.train_seconds:.1f} | "
            f"{p.latency_p50_ms:.3f} | {p.latency_p99_ms:.3f} |"
        )
    lines += [
        "",
        "## Feature groups (cumulative)",
        "",
        "| features | n | NE [95% CI] | AUC | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in report.features:
        lines.append(
            f"| {p.features} | {p.n_features} | {p.ne:.4f} [{p.ne_ci95[0]:.4f}, "
            f"{p.ne_ci95[1]:.4f}] | {p.auc:.3f} | {p.ece:.4f} |"
        )
    if report.business_value:
        lines += [
            "",
            "## Revenue vs complexity vs serving cost (simulator)",
            "",
            "| ranker | leaves | revenue / 1k req | lift (per seed) | serving $ / 1M req | "
            "net $ / 1M req |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for p in report.business_value:
            seeds = ", ".join(f"{x:+.1%}" for x in p.lift_per_seed)
            lines.append(
                f"| {p.variant} | {p.leaves:,} | ${p.revenue_per_1k_requests:.3f} | "
                f"{p.lift_vs_heuristic:+.1%} ({seeds}) | "
                f"${p.serving_cost_per_1m_requests_usd:.4f} | "
                f"${p.net_gain_per_1m_requests_usd:,.2f} |"
            )
        lines += [
            "",
            f"Serving cost assumes ${VCPU_HOUR_USD}/vCPU-hour and the measured p50 "
            "scoring time. Each seed pairs the ranker with the heuristic on the same world "
            "and random stream; the spread across seeds is the noise to compare lifts with.",
        ]
    return "\n".join(lines) + "\n"
