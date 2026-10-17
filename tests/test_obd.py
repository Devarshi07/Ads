"""Open Bandit Dataset validation on a synthetic dataset with the same layout and a known
answer: a uniform logger and a drifting Thompson-sampling-like policy."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from theraflo_ads.research import obd

N_ITEMS = 5
START = pd.Timestamp("2019-11-24")


def ctr(items: np.ndarray, day: np.ndarray) -> np.ndarray:
    # Items 3 and 4 get better on day 1, and the drifting policy moves toward them.
    base = np.array([0.02, 0.02, 0.03, 0.02, 0.03])[items]
    return np.where((day == 1) & (items >= 3), base * 2.5, base)


def drifting_policy(day: np.ndarray) -> np.ndarray:
    early = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    late = np.array([0.1, 0.1, 0.1, 0.35, 0.35])
    return np.where(day[:, None] == 0, early, late)


def logs(n: int, policy: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seconds = rng.integers(0, 2 * 86_400, n)
    day = seconds // 86_400
    probs = np.full((n, N_ITEMS), 1 / N_ITEMS) if policy == "random" else drifting_policy(day)
    cumulative = probs.cumsum(axis=1)
    items = (rng.random(n)[:, None] > cumulative).sum(axis=1)
    return pd.DataFrame(
        {
            "timestamp": START + pd.to_timedelta(seconds, unit="s"),
            "item_id": items.astype(np.int16),
            "position": rng.integers(1, 4, n).astype(np.int8),
            "click": (rng.random(n) < ctr(items, day)).astype(np.int8),
            "propensity_score": probs[np.arange(n), items],
        }
    )


@pytest.fixture(scope="module")
def data():
    return logs(300_000, "random", 1), logs(600_000, "bts", 2)


def test_hourly_policy_recovers_the_drift(data):
    _, bts = data
    hourly = obd.bts_policy(bts, N_ITEMS)
    assert hourly.probabilities.shape == (48, obd.POSITIONS, N_ITEMS)
    np.testing.assert_allclose(
        hourly.probabilities[:24].mean(axis=0)[0], [0.4, 0.3, 0.1, 0.1, 0.1], atol=0.02
    )
    drift = obd.nonstationarity(hourly, days=2)
    # 0.5 · (|0.4-0.1| + |0.3-0.1| + 0 + |0.1-0.35| + |0.1-0.35|) = 0.5
    assert drift["tv_first_vs_last_day"] == pytest.approx(0.5, abs=0.03)
    static = obd.bts_policy(bts, N_ITEMS, bucket=None)
    assert static.probabilities.shape == (1, obd.POSITIONS, N_ITEMS)


def test_hourly_target_is_accurate_and_a_static_one_is_biased(data):
    random, bts = data
    truth = obd.ground_truth(bts)
    hourly = obd.estimate_all(random, obd.bts_policy(bts, N_ITEMS), N_ITEMS, n_bootstrap=200)
    for estimate in hourly:
        result = obd.check("synthetic", "random", "bts", estimate, truth, "hourly")
        assert result.covered and result.conclusive, result
    static = obd.estimate_all(
        random, obd.bts_policy(bts, N_ITEMS, bucket=None), N_ITEMS, n_bootstrap=200
    )
    # The policy moved toward items exactly when they got better: averaging it away
    # evaluates a different, worse policy.
    assert all(e.value < truth.value * 0.92 for e in static)


def test_logged_propensities_reproduce_the_logger(data):
    random, bts = data
    truth = obd.ground_truth(random)
    for estimate in obd.estimate_all(bts, obd.uniform_policy(N_ITEMS), N_ITEMS, n_bootstrap=200):
        assert obd.check("synthetic", "bts", "random", estimate, truth).covered
        assert estimate.weight_mean == pytest.approx(1.0, abs=0.03)


def test_wrong_propensities_are_flagged(data):
    _, bts = data
    # Understated propensities on 1% of rows: the weights average about 1.08, and nothing
    # else looks wrong (plenty of effective data and clicks among the heavy rows).
    understated = bts.copy()
    rows = understated.index[understated.item_id == 2][:6_000]
    understated.loc[rows, "propensity_score"] = 0.02  # logged 0.02, really 0.1
    for estimate in obd.estimate_all(
        understated, obd.uniform_policy(N_ITEMS), N_ITEMS, n_bootstrap=100
    ):
        assert estimate.weight_mean == pytest.approx(1.08, abs=0.02)
        assert not estimate.conclusive and "importance weights" in estimate.reason
    # The Monte Carlo floor itself (1e-6) is caught earlier: a few rows carry the weight.
    floored = bts.copy()
    floored.loc[understated.index[understated.item_id == 2][:400], "propensity_score"] = 1e-6
    for estimate in obd.estimate_all(
        floored, obd.uniform_policy(N_ITEMS), N_ITEMS, n_bootstrap=100
    ):
        assert not estimate.conclusive


def test_reward_invariance_detects_policy_dependent_rewards(data):
    random, bts = data
    same = obd.reward_invariance(random, bts, N_ITEMS, "synthetic")
    # Unbiased across seeds (mean 1.007 ± 0.004 over 16); one draw is within a few percent.
    assert same.ratio == pytest.approx(1.0, abs=0.05)
    # The same items, positions, and hours, but clicks lost when the drifting policy shows
    # them (fatigue from repetition, say): off-policy estimates would be biased by it.
    shifted = bts.copy()
    lost = np.random.default_rng(5).random(len(shifted)) < 0.3
    shifted.loc[lost, "click"] = 0
    result = obd.reward_invariance(random, shifted, N_ITEMS, "synthetic")
    assert result.ratio / same.ratio == pytest.approx(0.7, abs=0.02) and result.ci[1] < 1.0


def test_extract_reads_only_the_ope_columns(tmp_path, data):
    random, bts = data
    archive = tmp_path / "open_bandit_dataset.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for policy, frame in (("random", random.head(500)), ("bts", bts.head(800))):
            csv = frame.assign(user_feature_0="abc", **{"user-item_affinity_0": 0.0})
            csv["timestamp"] = csv["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f+00:00")
            buffer = io.StringIO()
            csv.to_csv(buffer)
            z.writestr(f"open_bandit_dataset/{policy}/men/men.csv", buffer.getvalue())
    rows = obd.extract(archive, tmp_path / "parquet", campaigns=("men",))
    assert rows == {"random_men": 500, "bts_men": 800}
    loaded = obd.load(tmp_path / "parquet", "bts", "men")
    assert list(loaded.columns) == list(obd.COLUMNS)
    assert loaded["timestamp"].iloc[0] == bts["timestamp"].iloc[0].floor("s")
    np.testing.assert_allclose(loaded["propensity_score"], bts["propensity_score"].head(800))


def test_report_renders(tmp_path, data):
    random, bts = data
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    random.to_parquet(parquet / "random_men.parquet")
    bts.to_parquet(parquet / "bts_men.parquet")
    report = obd.run(parquet, tmp_path / "out", campaigns=("men",), sizes=(20_000,), repetitions=3)
    assert {(c.target, c.target_spec) for c in report.checks} == {
        ("bts", "hourly"),
        ("bts", "week-static"),
        ("random", "logged propensities"),
        ("random", "hourly frequencies as propensities"),
    }
    assert [p.n for p in report.subsamples] == [20_000] * 3
    markdown = (tmp_path / "out" / "obd_ope.md").read_text(encoding="utf-8")
    assert "CC BY 4.0" in markdown and "## Smaller logs" in markdown
