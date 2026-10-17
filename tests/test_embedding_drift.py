"""Embedding drift: aggregate statistics plus nearest-neighbor overlap."""

from __future__ import annotations

import numpy as np
import pytest

from theraflo_ads.monitoring import embedding_drift


def embeddings(n=1200, dim=16, seed=0):
    # n > the internal chunk size, so the chunked neighbor search is exercised.
    return np.random.default_rng(seed).normal(1.0, 1.0, size=(n, dim))


def test_identical_embeddings_show_no_drift():
    x = embeddings()
    result = embedding_drift(x, x.copy())
    assert result.mean_norm_ratio == pytest.approx(1.0)
    assert result.centroid_cosine_distance == pytest.approx(0.0, abs=1e-12)
    assert result.variance_ratio == pytest.approx(1.0)
    assert result.neighbor_overlap == pytest.approx(1.0)
    assert result.k == 10


def test_rotation_moves_the_centroid_but_keeps_the_geometry():
    x = embeddings()
    rotation, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(16, 16)))
    result = embedding_drift(x, x @ rotation)
    assert result.centroid_cosine_distance > 0.5
    assert result.mean_norm_ratio == pytest.approx(1.0)
    assert result.variance_ratio == pytest.approx(1.0)
    assert result.neighbor_overlap > 0.99


def test_shift_changes_norms_centroid_and_neighbors():
    x = embeddings()
    offset = np.zeros(16)
    offset[0] = 4.0
    result = embedding_drift(x, x + offset)
    assert result.mean_norm_ratio > 1.2
    assert result.centroid_cosine_distance > 0.1
    assert result.neighbor_overlap < 0.8


def test_rescaling_shows_in_norm_and_variance_ratios():
    x = embeddings()
    result = embedding_drift(x, 2.0 * x)
    assert result.mean_norm_ratio == pytest.approx(2.0)
    assert result.variance_ratio == pytest.approx(4.0)
    assert result.neighbor_overlap == pytest.approx(1.0)


def test_reshuffled_encoder_is_caught_only_by_neighbor_overlap():
    # Same rows, assigned to the wrong entities: every aggregate is unchanged.
    x = embeddings()
    result = embedding_drift(x, x[np.random.default_rng(2).permutation(len(x))])
    assert result.mean_norm_ratio == pytest.approx(1.0)
    assert result.centroid_cosine_distance == pytest.approx(0.0, abs=1e-9)
    assert result.variance_ratio == pytest.approx(1.0)
    assert result.neighbor_overlap < 0.05


@pytest.mark.parametrize(
    "reference,current,k",
    [
        (np.ones((10, 4)), np.ones((10, 5)), 3),
        (np.ones((10, 4)), np.ones((9, 4)), 3),
        (np.ones(10), np.ones(10), 3),
        (np.ones((10, 4)), np.ones((10, 4)), 10),
        (np.ones((10, 4)), np.ones((10, 4)), 0),
    ],
)
def test_invalid_inputs_are_rejected(reference, current, k):
    with pytest.raises(ValueError):
        embedding_drift(reference, current, k=k)
