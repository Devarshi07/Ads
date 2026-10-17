from collections import Counter

import pytest

from theraflo_ads.experiments import Experiment, Holdout, HoldoutConfig, bucket

UNITS = [f"k1:user{i}" for i in range(20_000)]


def test_bucket_is_stable_and_in_range():
    assert bucket("layer", "u1") == bucket("layer", "u1")
    assert all(0.0 <= bucket("layer", u) < 1.0 for u in UNITS[:1000])


def test_delimiter_prevents_concatenation_collisions():
    assert bucket("u1", "23") != bucket("u12", "3")


def test_holdout_shares():
    config = HoldoutConfig(uniform_exploration_share=0.02, no_ads_share=0.03)
    counts = Counter(config.assign(u) for u in UNITS)
    assert counts[Holdout.UNIFORM_EXPLORATION] / len(UNITS) == pytest.approx(0.02, abs=0.005)
    assert counts[Holdout.NO_ADS] / len(UNITS) == pytest.approx(0.03, abs=0.005)


@pytest.mark.parametrize("shares", [(-0.1, 0.0), (0.6, 0.5)])
def test_invalid_holdout_config(shares):
    with pytest.raises(ValueError):
        HoldoutConfig(uniform_exploration_share=shares[0], no_ads_share=shares[1])


def test_experiment_weights():
    experiment = Experiment("exp_1", (("control", 1.0), ("treatment", 3.0)))
    counts = Counter(experiment.assign(u) for u in UNITS)
    assert counts["treatment"] / len(UNITS) == pytest.approx(0.75, abs=0.01)
    assert experiment.cohort(UNITS[0]) in {"exp_1:control", "exp_1:treatment"}


def test_experiment_layers_are_independent_of_holdouts():
    holdouts = HoldoutConfig(uniform_exploration_share=0.5, no_ads_share=0.0)
    experiment = Experiment("exp_1", (("a", 1.0), ("b", 1.0)))
    in_holdout = [u for u in UNITS if holdouts.assign(u) is Holdout.UNIFORM_EXPLORATION]
    share_b = sum(experiment.assign(u) == "b" for u in in_holdout) / len(in_holdout)
    assert share_b == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("arms", [(), (("a", 0.0),), (("a", 1.0), ("a", 1.0))])
def test_invalid_experiments(arms):
    with pytest.raises(ValueError):
        Experiment("exp", arms)
