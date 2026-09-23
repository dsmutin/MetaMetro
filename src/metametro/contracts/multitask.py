"""Multi-class node GCN with an independent 0/1 head on every edge label.

One sparse convolution averages each node with its outgoing neighbors. A
linear head then predicts one node class. A second linear head reads the two
endpoint vectors and predicts each edge label with its own sigmoid. The input
CGT is not modified. Coordinates are the only features: colour bits are the
targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from metametro.errors import ContractError
from metametro.formats.cgt.model import Cgt
from metametro.formats.cgt.validator import validate_cgt


@dataclass
class MultitaskGcn:
    """Fitted heads and the coordinate scaling used at inference."""

    node_weight: np.ndarray
    node_bias: np.ndarray
    edge_weight: np.ndarray
    edge_bias: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    feature_index: np.ndarray
    node_class_names: list[str]
    edge_label_names: list[str]
    threshold: float
    seed: int
    epochs: int


def fit_multitask_gcn(
    cgt: Cgt,
    edge_targets: np.ndarray,
    *,
    node_class_names: list[str],
    edge_label_names: list[str],
    epochs: int = 80,
    seed: int = 0,
    learning_rate: float = 0.2,
    train_fraction: float = 0.8,
    threshold: float = 0.5,
) -> tuple[MultitaskGcn, dict[str, Any]]:
    """Fit both heads. Loss uses a seeded subset; inference covers every element.

    Node classes are weighted by inverse training frequency so a rare class is
    not ignored. Each edge label is then cut at the training score that marks
    as many edges positive as that label has in the training set. A label with
    no training positive stays at 0.5. Held-out metrics use that cut.
    """
    _check_fit_args(epochs, seed, learning_rate, train_fraction, threshold)
    validate_cgt(cgt)
    if cgt.node_labels is None:
        raise ContractError(["multitask GCN requires node labels"])
    features, feature_index = _geometry(cgt)
    labels = np.array(cgt.node_labels, dtype=np.int64, copy=True)
    targets = np.array(edge_targets, dtype=np.float64, copy=True)
    if targets.shape != (cgt.num_edges, len(edge_label_names)):
        raise ContractError(["edge targets must have shape (E, number of edge labels)"])
    if targets.size and (targets.min() < 0 or targets.max() > 1):
        raise ContractError(["edge targets must be 0 or 1"])
    if labels.size and len(node_class_names) != int(labels.max()) + 1:
        raise ContractError(["node class names must match the label ids"])
    before = np.array(cgt.node_features, copy=True)
    train_nodes, train_edges = _split(labels, targets, len(node_class_names), train_fraction=train_fraction, seed=seed)
    center = features[train_nodes].mean(axis=0)
    spread = features[train_nodes].std(axis=0)
    spread = np.where(spread < 1e-8, 1.0, spread)
    scaled = (features - center) / spread
    hidden = _aggregate(cgt.indptr, cgt.indices, scaled)
    sources = np.repeat(np.arange(cgt.num_nodes), np.diff(cgt.indptr))
    edge_input = np.concatenate([hidden[sources], hidden[cgt.indices]], axis=1) if cgt.num_edges else np.zeros((0, hidden.shape[1] * 2))
    model = _train(
        hidden,
        edge_input,
        labels,
        targets,
        train_nodes,
        train_edges,
        node_class_names=node_class_names,
        edge_label_names=edge_label_names,
        center=center,
        scale=spread,
        feature_index=feature_index,
        epochs=epochs,
        seed=seed,
        learning_rate=learning_rate,
        threshold=threshold,
    )
    _calibrate_thresholds(model, edge_input, targets, train_edges)
    if not np.array_equal(before, cgt.node_features):
        raise ContractError(["multitask GCN mutated CGT node features"])
    inference = infer_multitask_gcn(model, cgt)
    metrics = _metrics(labels, targets, inference, train_nodes, train_edges)
    metrics.update(
        {
            "seed": seed,
            "epochs": epochs,
            "train_nodes": int(train_nodes.size),
            "heldout_nodes": int(labels.size - train_nodes.size),
            "train_edges": int(train_edges.size),
            "heldout_edges": int(targets.shape[0] - train_edges.size),
            "node_classes": len(node_class_names),
            "edge_labels": len(edge_label_names),
        }
    )
    return model, metrics


def infer_multitask_gcn(model: MultitaskGcn, cgt: Cgt) -> dict[str, Any]:
    """Predict a node class and a 0/1 vector for every edge.

    Node probabilities are a softmax. Edge probabilities are independent
    sigmoids. Each edge label uses the cut stored on the model.
    """
    validate_cgt(cgt)
    features, _ = _geometry(cgt)
    if features.shape[1] != model.center.shape[0]:
        raise ContractError(["CGT geometry width does not match the fitted GCN"])
    scaled = (features - model.center) / model.scale
    hidden = _aggregate(cgt.indptr, cgt.indices, scaled)
    node_probability = _softmax(hidden @ model.node_weight + model.node_bias)
    node_class = node_probability.argmax(axis=1).astype(np.int64)
    if cgt.num_edges:
        sources = np.repeat(np.arange(cgt.num_nodes), np.diff(cgt.indptr))
        edge_input = np.concatenate([hidden[sources], hidden[cgt.indices]], axis=1)
        edge_probability = _sigmoid(edge_input @ model.edge_weight + model.edge_bias)
    else:
        edge_probability = np.zeros((0, len(model.edge_label_names)), dtype=np.float64)
    cut = np.asarray(model.threshold, dtype=np.float64)
    edge_binary = (edge_probability >= cut).astype(np.uint8)
    return {
        "node_class": node_class,
        "node_class_name": [model.node_class_names[int(label)] for label in node_class],
        "node_probability": node_probability,
        "edge_probability": edge_probability,
        "edge_binary": edge_binary,
    }


def _check_fit_args(epochs: int, seed: int, learning_rate: float, train_fraction: float, threshold: float) -> None:
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ContractError(["epochs must be a positive integer"])
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractError(["seed must be an integer"])
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ContractError(["learning rate must be a finite number > 0"])
    if not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ContractError(["train fraction must be strictly between 0 and 1"])
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ContractError(["decision threshold must be strictly between 0 and 1"])


def _geometry(cgt: Cgt) -> tuple[np.ndarray, np.ndarray]:
    names = list(cgt.metadata.get("node_feature_names") or [])
    missing = [name for name in ("longitude", "latitude") if name not in names]
    if missing:
        raise ContractError([f"GCN geometry is missing {name}" for name in missing])
    index = np.asarray([names.index(name) for name in ("longitude", "latitude")], dtype=np.int64)
    return np.array(cgt.node_features[:, index], dtype=np.float64, copy=True), index


def _aggregate(indptr: np.ndarray, indices: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Mean of a node and its outgoing neighbors."""
    count = features.shape[0]
    sources = np.repeat(np.arange(count), np.diff(indptr))
    summed = np.array(features, dtype=np.float64, copy=True)
    degree = np.ones(count, dtype=np.float64)
    if sources.size:
        np.add.at(summed, sources, features[indices])
        np.add.at(degree, sources, 1.0)
    return summed / degree[:, None]


def _split(
    labels: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    one_hot = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    if labels.size:
        one_hot[np.arange(labels.shape[0]), labels] = 1.0
    nodes = _cover_groups(one_hot, train_fraction, rng)
    edges = _cover_groups(targets, train_fraction, rng) if targets.shape[0] else np.zeros(0, dtype=np.int64)
    return nodes, edges


def _cover_groups(bits: np.ndarray, train_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Keep one positive row for every column, then fill up to the fraction."""
    count = bits.shape[0]
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    order = rng.permutation(count)
    chosen: list[int] = []
    seen: set[int] = set()
    for row in order:
        positive = np.flatnonzero(bits[int(row)] > 0)
        fresh = [int(column) for column in positive if int(column) not in seen]
        if fresh or positive.size == 0:
            if positive.size == 0 and seen:
                continue
            chosen.append(int(row))
            seen.update(int(column) for column in positive)
        if len(seen) == bits.shape[1] and bits.shape[1]:
            break
    # A row of all zeros still needs a chance to be the representative of an
    # all-zero label vector. Columns that are never positive are not forced.
    target = max(len(chosen), int(round(train_fraction * count)))
    target = min(count - 1 if count > 1 else 1, max(target, 1))
    for row in order:
        if len(chosen) >= target:
            break
        if int(row) not in chosen:
            chosen.append(int(row))
    return np.asarray(sorted(chosen), dtype=np.int64)


def _train(
    hidden: np.ndarray,
    edge_input: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    train_nodes: np.ndarray,
    train_edges: np.ndarray,
    *,
    node_class_names: list[str],
    edge_label_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    feature_index: np.ndarray,
    epochs: int,
    seed: int,
    learning_rate: float,
    threshold: float,
) -> MultitaskGcn:
    rng = np.random.default_rng(seed)
    n_classes = len(node_class_names)
    node_weight = rng.normal(0.0, 0.1, size=(hidden.shape[1], n_classes))
    node_bias = np.zeros(n_classes, dtype=np.float64)
    edge_weight = rng.normal(0.0, 0.1, size=(edge_input.shape[1], targets.shape[1]))
    edge_bias = np.zeros(targets.shape[1], dtype=np.float64)
    node_mask = np.zeros(hidden.shape[0], dtype=np.float64)
    node_mask[train_nodes] = 1.0
    edge_mask = np.zeros(targets.shape[0], dtype=np.float64)
    if train_edges.size:
        edge_mask[train_edges] = 1.0
    edge_scale = max(float(edge_mask.sum()), 1.0)
    one_hot = np.zeros((hidden.shape[0], n_classes), dtype=np.float64)
    one_hot[np.arange(hidden.shape[0]), labels] = 1.0
    class_count = np.bincount(labels[train_nodes], minlength=n_classes).astype(np.float64)
    class_count = np.maximum(class_count, 1.0)
    class_weight = class_count.sum() / (n_classes * class_count)
    node_weight_row = class_weight[labels] * node_mask
    node_weight_row /= max(float(node_weight_row.sum()), 1.0)
    for _ in range(epochs):
        node_probability = _softmax(hidden @ node_weight + node_bias)
        node_gradient = (node_probability - one_hot) * node_weight_row[:, None]
        node_weight -= learning_rate * (hidden.T @ node_gradient)
        node_bias -= learning_rate * node_gradient.sum(axis=0)
        if targets.shape[1] and targets.shape[0]:
            edge_probability = _sigmoid(edge_input @ edge_weight + edge_bias)
            edge_gradient = (edge_probability - targets) * edge_mask[:, None] / edge_scale
            edge_weight -= learning_rate * (edge_input.T @ edge_gradient)
            edge_bias -= learning_rate * edge_gradient.sum(axis=0)
    return MultitaskGcn(
        node_weight=node_weight,
        node_bias=node_bias,
        edge_weight=edge_weight,
        edge_bias=edge_bias,
        center=center,
        scale=scale,
        feature_index=feature_index,
        node_class_names=list(node_class_names),
        edge_label_names=list(edge_label_names),
        threshold=threshold,
        seed=seed,
        epochs=epochs,
    )


def _calibrate_thresholds(
    model: MultitaskGcn,
    edge_input: np.ndarray,
    targets: np.ndarray,
    train_edges: np.ndarray,
) -> None:
    """Cut each label where the training set has that many high scores."""
    width = targets.shape[1]
    cuts = np.full(width, 0.5, dtype=np.float64)
    if width and train_edges.size and edge_input.size:
        probability = _sigmoid(edge_input @ model.edge_weight + model.edge_bias)
        train_probability = probability[train_edges]
        train_targets = targets[train_edges]
        for column in range(width):
            count = int(train_targets[:, column].sum())
            if count <= 0:
                continue
            count = min(count, train_probability.shape[0])
            cuts[column] = float(np.partition(train_probability[:, column], -count)[-count])
    model.threshold = cuts


def _metrics(
    labels: np.ndarray,
    targets: np.ndarray,
    inference: dict[str, Any],
    train_nodes: np.ndarray,
    train_edges: np.ndarray,
) -> dict[str, Any]:
    held_nodes = np.ones(labels.shape[0], dtype=bool)
    held_nodes[train_nodes] = False
    held_edges = np.ones(targets.shape[0], dtype=bool)
    if train_edges.size:
        held_edges[train_edges] = False
    predicted = inference["node_class"]
    majority = int(np.bincount(labels[train_nodes]).argmax()) if train_nodes.size else 0
    return {
        "node_accuracy_train": _accuracy(labels, predicted, train_nodes),
        "node_accuracy_heldout": _accuracy(labels, predicted, np.flatnonzero(held_nodes)),
        "node_majority_accuracy_heldout": _accuracy(
            labels,
            np.full(labels.shape, majority, dtype=np.int64),
            np.flatnonzero(held_nodes),
        ),
        "edge_micro_f1_train": _micro_f1(targets, inference["edge_binary"], train_edges),
        "edge_micro_f1_heldout": _micro_f1(targets, inference["edge_binary"], np.flatnonzero(held_edges)),
    }


def _accuracy(labels: np.ndarray, predicted: np.ndarray, index: np.ndarray) -> float:
    if index.size == 0:
        return float("nan")
    return float(np.mean(labels[index] == predicted[index]))


def _micro_f1(targets: np.ndarray, binary: np.ndarray, index: np.ndarray) -> float:
    if index.size == 0 or targets.shape[1] == 0:
        return float("nan")
    truth = targets[index].astype(np.uint8)
    guess = binary[index]
    true_positive = int(np.sum((truth == 1) & (guess == 1)))
    false_positive = int(np.sum((truth == 0) & (guess == 1)))
    false_negative = int(np.sum((truth == 1) & (guess == 0)))
    precision_denom = true_positive + false_positive
    recall_denom = true_positive + false_negative
    precision = true_positive / precision_denom if precision_denom else 0.0
    recall = true_positive / recall_denom if recall_denom else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
