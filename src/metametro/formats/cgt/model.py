"""In-memory coloured graph tensor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"


@dataclass
class Cgt:
    """CSR topology plus dense feature, label, and colour arrays.

    Node ``i`` and every node-aligned array row ``i`` are the same dense id.
    Edge slot ``j`` in ``indices`` / ``edge_features`` / ``edge_labels`` /
    ``edge_colors`` is the same directed adjacency entry.
    """

    metadata: dict[str, Any]
    indptr: np.ndarray
    indices: np.ndarray
    node_features: np.ndarray
    edge_features: np.ndarray
    node_colors: np.ndarray
    edge_colors: np.ndarray
    mapping: list[dict[str, Any]]
    node_labels: np.ndarray | None = None
    edge_labels: np.ndarray | None = None
    color_ids: list[int] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.indices.shape[0])
