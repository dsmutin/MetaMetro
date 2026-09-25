"""Contig binning F1. Cluster ids are not a biological name."""

from __future__ import annotations

import numpy as np


def contig_f1(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Harmonic mean of precision and recall after majority-label matching.

    Each predicted cluster is assigned the majority ground-truth label. A
    ground-truth label is used at most once, largest overlap first.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    if true_labels.shape != pred_labels.shape:
        raise ValueError("true_labels and pred_labels must have the same shape")
    true_ids = [int(label) for label in np.unique(true_labels) if int(label) >= 0]
    pred_ids = np.unique(pred_labels)
    if not true_ids or pred_ids.size == 0:
        return 0.0
    ranked = []
    for pred in pred_ids:
        members = true_labels[pred_labels == pred]
        if members.size == 0:
            continue
        values, counts = np.unique(members, return_counts=True)
        truth = int(values[np.argmax(counts)])
        if truth < 0:
            continue
        ranked.append((int(counts.max()), int(pred), truth))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    true_sizes = {label: int(np.sum(true_labels == label)) for label in true_ids}
    matched = 0
    seen: set[int] = set()
    for hit, _pred, truth in ranked:
        if truth in seen:
            continue
        seen.add(truth)
        matched += hit
    precision = matched / max(len(pred_labels), 1)
    recall = matched / max(sum(true_sizes.values()), 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
