"""Derived indexes over a CGT.

CSR is the canonical topology. ``indptr`` and ``indices`` are stored in
source-major order, and ``edge_features``, ``edge_labels``, and
``edge_colors`` are defined on that same CSR slot. CSC is an incoming-neighbor
index computed from those arrays. It is not written back into the CGT and it
is not a second copy of the edge features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metametro.formats.cgt.model import Cgt
from metametro.formats.cgt.validator import validate_cgt


@dataclass(frozen=True)
class CscAdjacency:
    """Incoming adjacency derived from CSR.

    ``indptr`` and ``indices`` address predecessors. ``indices[k]`` is the
    source dense id of one incoming edge. ``csr_slots[k]`` is the CSR slot of
    that same directed edge, so the feature row is ``edge_features[csr_slots[k]]``.
    Reading that row does not reorder ``edge_features``.
    """

    indptr: np.ndarray
    indices: np.ndarray
    csr_slots: np.ndarray


def csc_from_cgt(cgt: Cgt) -> CscAdjacency:
    """Return CSC indptr and indices for incoming edges.

    CSR stays canonical: edges are stored in source-major order, and every
    feature, label, and colour row is defined on that CSR slot. This function
    allocates new arrays. It does not write into ``cgt``, does not build a
    dense ``N×N`` matrix, and does not permute ``edge_features``.

    Within one target, incoming edges follow increasing CSR slot order.
    """
    validate_cgt(cgt)
    node_count = cgt.num_nodes
    edge_count = cgt.num_edges
    counts = np.zeros(node_count, dtype=np.int64)
    for slot in range(edge_count):
        counts[int(cgt.indices[slot])] += 1
    indptr = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    cursor = indptr.copy()
    indices = np.empty(edge_count, dtype=np.int64)
    csr_slots = np.empty(edge_count, dtype=np.int64)
    for source in range(node_count):
        start = int(cgt.indptr[source])
        stop = int(cgt.indptr[source + 1])
        for slot in range(start, stop):
            target = int(cgt.indices[slot])
            position = int(cursor[target])
            indices[position] = source
            csr_slots[position] = slot
            cursor[target] += 1
    return CscAdjacency(indptr=indptr, indices=indices, csr_slots=csr_slots)
