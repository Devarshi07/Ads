import random
from collections import Counter

import pytest

from tests.helpers import PRIORS, make_catalog, make_request
from theraflo_ads.domain import NO_AD
from theraflo_ads.policy import (
    ActionDistribution,
    EpsilonGreedyPolicy,
    HeuristicScorer,
    InvalidDistribution,
    PopularityPolicy,
    UniformPolicy,
)
from theraflo_ads.policy.policies import sample, validate_distribution

CATALOG = make_catalog()
SAFE = [CATALOG.ads[a] for a in ("ad_cpm", "ad_cpc", "ad_cpa")]


def scorer(**options):
    return HeuristicScorer(CATALOG, PRIORS, **options)


def test_heuristic_scores_expected_platform_revenue():
    scored = {s.ad.ad_id: s.expected_revenue for s in scorer().score(make_request(), SAFE)}
    assert scored["ad_cpm"] == pytest.approx(10.0 / 1000)
    assert scored["ad_cpc"] == pytest.approx(1.0 * 0.03)
    assert scored["ad_cpa"] == pytest.approx(20.0 * 0.02 * 0.05)


def test_quality_floor_removes_low_cvr_ads_before_sampling():
    scored = scorer(min_p_cvr=0.045).score(make_request(), SAFE)
    assert {s.ad.ad_id for s in scored} == {"ad_cpm", "ad_cpa"}


def test_epsilon_greedy_distribution():
    dist = EpsilonGreedyPolicy(scorer(), epsilon=0.2).distribution(make_request(), SAFE)
    probs = dist.probabilities
    assert set(probs) == {"ad_cpm", "ad_cpc", "ad_cpa", NO_AD}
    assert sum(probs.values()) == pytest.approx(1.0)
    assert min(probs.values()) == pytest.approx(0.2 / 4)  # exploration floor
    assert probs["ad_cpc"] == pytest.approx(0.8 + 0.05)  # greedy: highest expected revenue
    assert dist.stochastic
    assert dist.predictions["ad_cpc"]["p_ctr"] == pytest.approx(0.03)


def test_zero_epsilon_is_deterministic():
    dist = EpsilonGreedyPolicy(scorer(), epsilon=0.0).distribution(make_request(), SAFE)
    assert dist.probabilities["ad_cpc"] == 1.0
    assert not dist.stochastic


def test_no_ad_wins_when_nothing_clears_the_floor():
    policy = EpsilonGreedyPolicy(scorer(), epsilon=0.0, no_ad_floor=1.0)
    assert policy.distribution(make_request(), SAFE).probabilities[NO_AD] == 1.0


def test_empty_eligible_set_is_no_ad_only():
    dist = EpsilonGreedyPolicy(scorer(), epsilon=0.2).distribution(make_request(), [])
    assert dist.probabilities == {NO_AD: 1.0}
    assert not dist.stochastic


def test_invalid_epsilon():
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(scorer(), epsilon=1.5)


def test_uniform_includes_no_ad():
    dist = UniformPolicy().distribution(make_request(), SAFE)
    assert dist.probabilities == {a: 0.25 for a in ("ad_cpm", "ad_cpc", "ad_cpa", NO_AD)}


def test_popularity_picks_first_eligible_then_no_ad():
    policy = PopularityPolicy(["ad_gamble", "ad_cpa", "ad_cpm"])
    assert policy.distribution(make_request(), SAFE).probabilities["ad_cpa"] == 1.0
    empty = PopularityPolicy(["ad_gamble"]).distribution(make_request(), SAFE)
    assert empty.probabilities[NO_AD] == 1.0
    assert not empty.stochastic


@pytest.mark.parametrize(
    "probabilities",
    [
        {"ad_gamble": 1.0},
        {"ad_cpm": 0.5, NO_AD: 0.4},
        {"ad_cpm": 1.2, NO_AD: -0.2},
        {"ad_cpm": float("nan"), NO_AD: 1.0},
    ],
)
def test_validation_rejects_bad_distributions(probabilities):
    with pytest.raises(InvalidDistribution):
        validate_distribution(ActionDistribution(probabilities, True), {"ad_cpm"})


def test_sampling_matches_probabilities():
    dist = ActionDistribution({"a": 0.7, "b": 0.2, NO_AD: 0.1}, True)
    rng = random.Random(1)
    counts = Counter(sample(dist, rng) for _ in range(20_000))
    assert counts["a"] / 20_000 == pytest.approx(0.7, abs=0.015)
    assert counts[NO_AD] / 20_000 == pytest.approx(0.1, abs=0.01)


def test_sampling_never_returns_zero_probability_actions():
    dist = ActionDistribution({"a": 0.0, "b": 1.0, NO_AD: 0.0}, False)
    rng = random.Random(1)
    assert {sample(dist, rng) for _ in range(100)} == {"b"}
