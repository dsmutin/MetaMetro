"""Profiling scores. Truth is passed in by the caller and is not read from the graph."""

from __future__ import annotations

import math


def l1(predicted: dict[int, float], truth: dict[int, float]) -> float:
    """L1 distance after ``truth`` is rescaled to sum to 1. ``predicted`` is used as given."""
    truth_sum = sum(truth.values())
    truth_rel = {key: value / truth_sum for key, value in truth.items()} if truth_sum else {}
    keys = set(truth_rel) | set(predicted)
    return sum(abs(predicted.get(key, 0.0) - truth_rel.get(key, 0.0)) for key in keys)


def bray_curtis(predicted: dict[int, float], truth: dict[int, float]) -> float:
    """Bray–Curtis distance on the same rescaling as ``l1``."""
    truth_sum = sum(truth.values())
    truth_rel = {key: value / truth_sum for key, value in truth.items()} if truth_sum else {}
    keys = set(truth_rel) | set(predicted)
    union = sum(predicted.get(key, 0.0) + truth_rel.get(key, 0.0) for key in keys)
    if union <= 0:
        return 0.0
    shared = sum(min(predicted.get(key, 0.0), truth_rel.get(key, 0.0)) for key in keys)
    return 1.0 - (2.0 * shared / union)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation. ``None`` when either side has no variance or fewer than two points."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def presence_f1(predicted: dict[int, float], truth: dict[int, float], *, minimum: float = 0.0) -> float:
    """Set F1 of taxa whose abundance is above ``minimum``. Taxon 0 is ignored."""
    truth_ids = {taxon_id for taxon_id, value in truth.items() if taxon_id != 0 and value > minimum}
    pred_ids = {taxon_id for taxon_id, value in predicted.items() if taxon_id != 0 and value > minimum}
    tp = len(truth_ids & pred_ids)
    fp = len(pred_ids - truth_ids)
    fn = len(truth_ids - pred_ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
