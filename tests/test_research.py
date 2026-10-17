"""Scaling-study building blocks on small synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from theraflo_ads.research import scaling


def synthetic(n_users=400, rows_per_user=30, seed=0) -> scaling.Data:
    rng = np.random.default_rng(seed)
    names = ["cat_a", "cat_b", "prior_ctr", "placement_home_feed", "user_cat_ctr"]
    rows = []
    for u in range(n_users):
        affinity = rng.gamma(2.0, 0.5)
        for _ in range(rows_per_user):
            cat_a = float(rng.random() < 0.5)
            prior = 0.04 if cat_a else 0.01
            p = min(prior * affinity, 0.5)
            rows.append(
                {
                    "user_hash": f"u{u}",
                    "day": rng.integers(0, 10),
                    "clicked": float(rng.random() < p),
                    "f_cat_a": cat_a,
                    "f_cat_b": 1 - cat_a,
                    "f_prior_ctr": prior,
                    "f_placement_home_feed": float(rng.random() < 0.5),
                    "f_user_cat_ctr": prior * affinity + rng.normal(0, 0.002),
                }
            )
    frame = pd.DataFrame(rows)
    users = np.array([f"u{u}" for u in range(n_users)])
    split = {u: i % 5 for i, u in enumerate(users)}
    part = frame["user_hash"].map(split)
    return scaling.Data(names, frame[part <= 2], frame[part == 3], frame[part == 4])


def test_normalized_entropy_is_one_for_the_base_rate():
    y = np.array([0, 0, 0, 1] * 50, dtype=float)
    assert scaling.normalized_entropy(y, np.full(len(y), y.mean())) == pytest.approx(1.0)
    better = np.where(y == 1, 0.6, 0.15)
    assert scaling.normalized_entropy(y, better) < 1.0
    low, high = scaling.ne_interval(y, better, n_boot=50)
    assert low <= scaling.normalized_entropy(y, better) <= high


def test_power_law_fit_recovers_a_known_curve():
    points = [
        scaling.CurvePoint(f, n, 0, 0.9 + 3.0 * n**-0.5, 0.0, 0.6, 0.01)
        for f, n in ((0.1, 1e3), (0.2, 2e3), (0.5, 5e3), (1.0, 1e4), (2.0, 4e4))
    ]
    law = scaling.fit_power_law(points)
    assert law.fitted
    assert law.a == pytest.approx(0.9, abs=0.01) and law.c == pytest.approx(0.5, abs=0.05)
    assert law.rows_for_half_gap > 4e4


def test_subsamples_keep_whole_users():
    data = synthetic()
    sample = scaling.user_subsample(data.train, 0.25, seed=1)
    counts = sample.groupby("user_hash").size()
    assert (counts == 30).all()
    assert 0.2 < sample.user_hash.nunique() / data.train.user_hash.nunique() < 0.3


def test_study_runs_end_to_end(tmp_path):
    pytest.importorskip("lightgbm")
    data = synthetic()
    curve = scaling.data_volume(data, (0.3, 1.0), seeds=(1, 2))
    assert [p.fraction for p in curve] == [0.3, 1.0] and curve[0].rows < curve[1].rows
    sizes = scaling.model_size(data, leaves=(2, 8))
    assert [s.model for s in sizes] == [
        "category prior (heuristic)",
        "logistic",
        "lightgbm_2",
        "lightgbm_8",
    ]
    assert all(s.latency_p50_ms >= 0 and s.ne_ci95[0] <= s.ne_ci95[1] for s in sizes)
    ablation = scaling.feature_ablation(data)
    assert [a.features for a in ablation] == ["+ad", "+context", "+exposure", "+personalization"]
    report = scaling.ScalingReport(
        dataset={
            "train_rows": 1,
            "train_clicks": 1,
            "test_rows": 1,
            "test_clicks": 1,
            "features": 5,
        },
        data_volume=curve,
        power_law=scaling.fit_power_law(curve),
        data_volume_logistic=curve,
        model_size=sizes,
        features=ablation,
    )
    markdown = scaling.render_markdown(report)
    assert "## Quality vs data" in markdown and "lightgbm_8" in markdown
