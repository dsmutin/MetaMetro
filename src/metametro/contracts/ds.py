"""Contract 7: a minimal GCN that reads a CGT and does not modify it.

The NumPy model is one normalized sparse graph convolution followed by a
linear classifier. It is deterministic for a fixed seed. PyTorch Geometric
is used when ``backend='pyg'`` and the package is installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metametro.errors import ContractError
from metametro.formats.cgt.model import Cgt
from metametro.formats.cgt.validator import validate_cgt


def _aggregate(indptr: np.ndarray, indices: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Mean of self plus outgoing neighbors. O(N + E) and sparse."""
    output = np.zeros_like(features)
    for node in range(features.shape[0]):
        total = features[node].copy()
        degree = 1
        for pointer in range(int(indptr[node]), int(indptr[node + 1])):
            total += features[int(indices[pointer])]
            degree += 1
        output[node] = total / degree
    return output


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_numpy_gcn(
    cgt: Cgt,
    *,
    epochs: int = 40,
    seed: int = 0,
    learning_rate: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Train the NumPy GCN. Returns predicted labels and class probabilities."""
    if cgt.node_labels is None:
        raise ContractError(["DS requires node labels"])
    features = np.array(cgt.node_features, dtype=np.float64, copy=True)
    if features.shape[1] == 0:
        features = np.ones((cgt.num_nodes, 1), dtype=np.float64)
    labels = np.array(cgt.node_labels, dtype=np.int64, copy=True)
    indptr = np.array(cgt.indptr, copy=True)
    indices = np.array(cgt.indices, copy=True)
    n_classes = int(labels.max()) + 1 if labels.size else 1
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.1, size=(features.shape[1], n_classes))
    bias = np.zeros(n_classes, dtype=np.float64)
    hidden = _aggregate(indptr, indices, features)
    for _ in range(epochs):
        probabilities = _softmax(hidden @ weights + bias)
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(len(labels)), labels] = 1.0
        gradient = (probabilities - one_hot) / max(len(labels), 1)
        weights -= learning_rate * (hidden.T @ gradient)
        bias -= learning_rate * gradient.sum(axis=0)
    probabilities = _softmax(hidden @ weights + bias)
    return probabilities.argmax(axis=1).astype(np.int64), probabilities


def train_pyg_gcn(
    cgt: Cgt,
    *,
    epochs: int = 40,
    seed: int = 0,
    learning_rate: float = 0.05,
    hidden: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a two-layer PyG GCN. Imports torch only when this backend is used."""
    if cgt.node_labels is None:
        raise ContractError(["DS requires node labels"])
    import torch
    from torch_geometric.nn import GCNConv

    from metametro.converters.cgt_to_pyg import to_pyg

    torch.manual_seed(seed)
    data = to_pyg(cgt)
    in_features = int(data.x.shape[1]) or 1
    if data.x.shape[1] == 0:
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)
    n_classes = int(data.y.max().item()) + 1

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = GCNConv(in_features, hidden)
            self.conv2 = GCNConv(hidden, n_classes)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(logits, data.y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return probabilities.argmax(axis=1).astype(np.int64), probabilities


def run_ds(
    cgt: Cgt,
    *,
    epochs: int = 40,
    seed: int = 0,
    backend: str = "numpy",
) -> dict[str, Any]:
    """Predict a label per dense node id. The input CGT arrays are not written."""
    validate_cgt(cgt)
    before = (
        np.array(cgt.node_features, copy=True),
        None if cgt.node_labels is None else np.array(cgt.node_labels, copy=True),
    )
    if backend == "numpy":
        predicted, probabilities = train_numpy_gcn(cgt, epochs=epochs, seed=seed)
    elif backend == "pyg":
        predicted, probabilities = train_pyg_gcn(cgt, epochs=epochs, seed=seed)
    else:
        raise ContractError([f"unknown DS backend: {backend}"])
    if not np.array_equal(before[0], cgt.node_features):
        raise ContractError(["DS mutated CGT node features"])
    if before[1] is not None and not np.array_equal(before[1], cgt.node_labels):
        raise ContractError(["DS mutated CGT node labels"])
    rows = []
    for row, label, probability in zip(cgt.mapping, predicted, probabilities):
        rows.append(
            {
                "dense_id": int(row["dense_id"]),
                "source_id": row["source_id"],
                "cfa_node_ids": list(row["cfa_node_ids"]),
                "predicted_label": int(label),
                "probability": float(probability[int(label)]),
            }
        )
    return {
        "result": rows,
        "mapping": [dict(row) for row in cgt.mapping],
        "metadata": {
            "contract": "ds_on_cgt",
            "contract_version": "1.0",
            "backend": backend,
            "seed": seed,
            "epochs": epochs,
            "num_predictions": len(rows),
            "graph_id": cgt.metadata.get("source", {}).get("graph_id"),
        },
    }
