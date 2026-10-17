"""Embedding drift (design doc: Embedding Drift).

Aggregate statistics (norms, centroid, variance) catch shifts and rescaling;
nearest-neighbor overlap catches an upstream encoder change that reshuffles
which entities look alike while leaving every aggregate untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Rows of the n x n similarity matrix processed at once, so n = 5,000 never
# materializes the full matrix (200 MB in float64).
_CHUNK_ROWS = 512


@dataclass(frozen=True)
class EmbeddingDrift:
    mean_norm_ratio: float
    centroid_cosine_distance: float
    variance_ratio: float
    neighbor_overlap: float
    k: int


def embedding_drift(reference: np.ndarray, current: np.ndarray, *, k: int = 10) -> EmbeddingDrift:
    """Compare two embedding tables where row i is the same entity in both."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.ndim != 2 or ref.shape != cur.shape:
        raise ValueError(
            f"reference and current must be 2-D with the same shape, got {ref.shape}, {cur.shape}"
        )
    if not 1 <= k < ref.shape[0]:
        raise ValueError(f"k must be in [1, n_rows), got k={k} with {ref.shape[0]} rows")
    if not (np.isfinite(ref).all() and np.isfinite(cur).all()):
        raise ValueError("embeddings must be finite")
    return EmbeddingDrift(
        mean_norm_ratio=_ratio(
            float(np.linalg.norm(cur, axis=1).mean()), float(np.linalg.norm(ref, axis=1).mean())
        ),
        centroid_cosine_distance=_cosine_distance(ref.mean(axis=0), cur.mean(axis=0)),
        variance_ratio=_ratio(float(cur.var(axis=0).sum()), float(ref.var(axis=0).sum())),
        neighbor_overlap=_neighbor_overlap(ref, cur, k),
        k=k,
    )


def _neighbor_overlap(reference: np.ndarray, current: np.ndarray, k: int) -> float:
    ref = _unit_rows(reference)
    cur = _unit_rows(current)
    jaccard = np.empty(ref.shape[0])
    for start in range(0, ref.shape[0], _CHUNK_ROWS):
        rows = slice(start, min(start + _CHUNK_ROWS, ref.shape[0]))
        ref_nn = _top_k_neighbors(ref, rows, k)
        cur_nn = _top_k_neighbors(cur, rows, k)
        shared = (ref_nn[:, :, None] == cur_nn[:, None, :]).sum(axis=(1, 2))
        jaccard[rows] = shared / (2 * k - shared)
    return float(jaccard.mean())


def _top_k_neighbors(unit: np.ndarray, rows: slice, k: int) -> np.ndarray:
    similarity = unit[rows] @ unit.T
    own = np.arange(rows.start, rows.stop)
    similarity[own - rows.start, own] = -np.inf
    return np.argpartition(-similarity, k - 1, axis=1)[:, :k]


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        # Direction is undefined for a zero centroid; only "both zero" counts as unmoved.
        return 0.0 if norm_a == norm_b else 1.0
    return float(np.clip(1.0 - np.dot(a, b) / (norm_a * norm_b), 0.0, 2.0))


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else float("inf")
    return numerator / denominator
