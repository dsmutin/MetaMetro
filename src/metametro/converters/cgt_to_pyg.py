"""Thin PyTorch Geometric view of a CGT.

The CGT stays the canonical runtime object. This module only builds the
``edge_index`` layout and, when PyTorch Geometric is installed, a ``Data``
object.
"""

from __future__ import annotations

import numpy as np

from metametro.formats.cgt.model import Cgt
from metametro.formats.cgt.validator import validate_cgt


def edge_index_array(cgt: Cgt) -> np.ndarray:
    """Return ``edge_index`` with shape ``(2, E)`` in the same order as ``indices``."""
    validate_cgt(cgt)
    sources = np.empty(cgt.num_edges, dtype=np.int64)
    for node in range(cgt.num_nodes):
        start = int(cgt.indptr[node])
        stop = int(cgt.indptr[node + 1])
        sources[start:stop] = node
    if cgt.num_edges == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.vstack([sources, cgt.indices])


def to_pyg(cgt: Cgt):
    """Build ``torch_geometric.data.Data`` from a CGT.

    Raises ``ImportError`` when PyTorch Geometric is not installed. The array
    layout is available without that dependency through ``edge_index_array``.
    """
    validate_cgt(cgt)
    import torch
    from torch_geometric.data import Data

    edge_index = torch.from_numpy(edge_index_array(cgt).copy())
    kwargs = {
        "x": torch.from_numpy(np.array(cgt.node_features, copy=True)),
        "edge_index": edge_index,
    }
    if cgt.edge_features.shape[1] > 0:
        kwargs["edge_attr"] = torch.from_numpy(np.array(cgt.edge_features, copy=True))
    if cgt.node_labels is not None:
        kwargs["y"] = torch.from_numpy(np.array(cgt.node_labels, copy=True))
    return Data(**kwargs)
