"""Colour nodes by k-mer composition clusters.

The cluster labels are colours. They are not genome labels and they are not
an evaluation target. Fitting uses only sequence composition.
"""

from __future__ import annotations

import numpy as np

from metametro.contracts.colouring import paint_namespace
from metametro.errors import ContractError
from metametro.formats.cfa.model import CfaGraph

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
_BASES = "ACGT"


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def canonical_kmer_index(k: int) -> dict[str, int]:
    """Map each canonical k-mer to a column index."""
    if k < 1:
        raise ContractError(["composition k must be >= 1"])
    found: dict[str, int] = {}
    for index in range(4**k):
        raw = []
        value = index
        for _ in range(k):
            raw.append(_BASES[value % 4])
            value //= 4
        kmer = "".join(reversed(raw))
        canonical = min(kmer, _reverse_complement(kmer))
        if canonical not in found:
            found[canonical] = len(found)
    return found


def kmer_vector(sequence: str, k: int, index: dict[str, int]) -> np.ndarray:
    """L2-normalised canonical k-mer counts. Short sequences are a zero vector."""
    vector = np.zeros(len(index), dtype=np.float64)
    text = sequence.upper()
    if len(text) < k:
        return vector
    for start in range(len(text) - k + 1):
        kmer = text[start : start + k]
        if any(base not in _BASES for base in kmer):
            continue
        canonical = min(kmer, _reverse_complement(kmer))
        vector[index[canonical]] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def kmeans_labels(matrix: np.ndarray, n_clusters: int, seed: int = 0, *, n_init: int = 5, max_iter: int = 50) -> np.ndarray:
    """Integer cluster labels from a numpy k-means. No scikit-learn dependency."""
    data = np.asarray(matrix, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ContractError(["k-means needs a non-empty 2-d matrix"])
    clusters = int(min(n_clusters, max(2, data.shape[0])))
    if clusters < 2:
        raise ContractError(["k-means needs at least two rows"])
    scale = data.std(axis=0)
    scale[scale < 1e-8] = 1.0
    scaled = (data - data.mean(axis=0)) / scale
    rng = np.random.default_rng(seed)
    best_inertia = np.inf
    best_labels = np.zeros(scaled.shape[0], dtype=np.int64)
    for _ in range(n_init):
        chosen = rng.choice(scaled.shape[0], size=clusters, replace=False)
        centers = scaled[chosen].copy()
        labels = np.zeros(scaled.shape[0], dtype=np.int64)
        for _step in range(max_iter):
            dist = ((scaled[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dist.argmin(axis=1)
            new_centers = np.array(
                [
                    scaled[labels == index].mean(axis=0) if np.any(labels == index) else centers[index]
                    for index in range(clusters)
                ]
            )
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        inertia = float(((scaled - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels


def kmeans_onehot(matrix: np.ndarray, n_clusters: int = 8, seed: int = 0) -> np.ndarray:
    """uint8 one-hot matrix of k-means labels. Fit ignores genome labels."""
    labels = kmeans_labels(matrix, n_clusters, seed)
    width = int(labels.max()) + 1
    colors = np.zeros((labels.shape[0], width), dtype=np.uint8)
    colors[np.arange(labels.shape[0]), labels] = 1
    return colors


def colour_by_composition(
    cfa: CfaGraph,
    *,
    k: int = 4,
    n_clusters: int = 8,
    seed: int = 0,
    operation: str = "merge",
) -> CfaGraph:
    """Paint a ``composition`` namespace from canonical k-mer k-means."""
    if not cfa.nodes:
        raise ContractError(["composition colouring needs at least one node"])
    index = canonical_kmer_index(k)
    matrix = np.stack([kmer_vector(cfa.sequences[row["node_id"]], k, index) for row in cfa.nodes])
    labels = kmeans_labels(matrix, n_clusters, seed)
    node_values = {row["node_id"]: [f"cluster_{int(label)}"] for row, label in zip(cfa.nodes, labels)}
    by_node = {row["node_id"]: f"cluster_{int(label)}" for row, label in zip(cfa.nodes, labels)}
    edge_values: dict[str, list[str]] = {}
    for row in cfa.edges:
        left = by_node[row["source"]]
        right = by_node[row["target"]]
        edge_values[row["edge_id"]] = [left] if left == right else []
    return paint_namespace(cfa, node_values, edge_values, namespace="composition", operation=operation)
