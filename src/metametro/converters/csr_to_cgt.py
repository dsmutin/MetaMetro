"""Build a CGT from an external CSR adjacency.

This is the layout ParaGVAE uses for a VAEGbin bundle: sparse adjacency,
node features, and an optional edge-weight column. It is not the CFA → CDBG
path. Colours stay on the uint8 mask. They are not copied into ``X``. An
optional float32 weight uses the same colour columns. Edge colours use that
same width; a missing edge colour matrix means every edge is uncoloured.

The function does not allocate a dense ``N×N`` matrix and does not invent
edge weights. Within each row, targets are sorted and the edge-aligned
arrays move with them, so slot ``j`` stays one directed edge.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from metametro.errors import ContractError
from metametro.formats.cgt.model import SCHEMA_VERSION, Cgt
from metametro.formats.cgt.registry import build_feature_registry
from metametro.formats.cgt.validator import validate_cgt


def cgt_from_csr(
    *,
    graph_id: str,
    indptr: np.ndarray,
    indices: np.ndarray,
    node_features: np.ndarray,
    source_ids: Sequence[str],
    cfa_node_ids: Sequence[Sequence[str]] | None = None,
    node_feature_names: Sequence[str] | None = None,
    edge_features: np.ndarray | None = None,
    edge_feature_names: Sequence[str] | None = None,
    node_colors: np.ndarray | None = None,
    edge_colors: np.ndarray | None = None,
    node_color_weights: np.ndarray | None = None,
    edge_color_weights: np.ndarray | None = None,
    color_ids: Sequence[int] | None = None,
    node_labels: np.ndarray | None = None,
    source: str,
) -> Cgt:
    """Return a validated CGT for one external CSR graph.

    ``source_ids[i]`` is the caller's node id for dense row ``i``. When
    ``cfa_node_ids`` is omitted, that same id is stored as the only CFA id.
    MetaMetro does not treat it as a genome or a sample. ``edge_features``
    may be a vector of length ``E`` or a matrix ``(E, F)``. ``None`` stores
    width 0. A 1-d vector is named ``weight`` unless names are passed.
    """
    if not isinstance(graph_id, str) or graph_id == "":
        raise ContractError(["graph_id is required"])
    if not isinstance(source, str) or source == "":
        raise ContractError(["source is required"])
    pointer = _csr(indptr, indices)
    n = pointer.size - 1
    if len(source_ids) != n:
        raise ContractError(["source_ids length must equal the node count"])
    if any(not isinstance(node_id, str) or node_id == "" for node_id in source_ids):
        raise ContractError(["source_ids must be non-empty strings"])
    if len(set(source_ids)) != n:
        raise ContractError(["source_ids must be unique"])
    members = _members(source_ids, cfa_node_ids, n)
    features = _node_features(node_features, n)
    names, positional = _names(node_feature_names, features.shape[1], default_one=None)
    edge_matrix, edge_names, edge_positional = _edge_features(edge_features, int(pointer[-1]), edge_feature_names)
    node_mask, edge_mask, ids = _colours(node_colors, edge_colors, color_ids, n, int(pointer[-1]))
    node_weights, edge_weights = _weights(
        node_color_weights, edge_color_weights, n, int(pointer[-1]), node_mask.shape[1]
    )
    labels = None if node_labels is None else np.array(node_labels, dtype=np.int64, copy=True)
    if labels is not None and labels.shape != (n,):
        raise ContractError(["node labels must have shape (N,)"])
    sorted_indices, order = _sort_rows(pointer, np.asarray(indices))
    graph = Cgt(
        metadata={
            "schema_version": SCHEMA_VERSION,
            "num_nodes": n,
            "num_edges": int(pointer[-1]),
            "node_feature_names": names,
            "edge_feature_names": edge_names,
            "node_feature_dtype": "float32",
            "edge_feature_dtype": "float32",
            "topology": "csr",
            "contract": "external_csr",
            "contract_version": "1.0",
            "source": {"format": "external_csr", "graph_id": graph_id, "note": source},
            "node_feature_registry": build_feature_registry(
                names, explicit_count=len(names), positional=positional, annotation_sources=()
            ),
            "edge_feature_registry": build_feature_registry(
                edge_names,
                explicit_count=len(edge_names),
                positional=edge_positional,
                annotation_sources=(),
            ),
        },
        indptr=np.array(pointer, dtype=np.int64, copy=True),
        indices=sorted_indices,
        node_features=features,
        edge_features=_permute_rows(edge_matrix, order),
        node_colors=node_mask,
        edge_colors=_permute_rows(edge_mask, order),
        mapping=[
            {"dense_id": index, "source_id": source_ids[index], "cfa_node_ids": list(members[index])}
            for index in range(n)
        ],
        node_labels=labels,
        color_ids=ids,
        node_color_weights=node_weights,
        edge_color_weights=None if edge_weights is None else _permute_rows(edge_weights, order),
    )
    validate_cgt(graph)
    return graph


def _csr(indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    pointer = np.asarray(indptr)
    targets = np.asarray(indices)
    if pointer.dtype != np.int64:
        pointer = pointer.astype(np.int64)
    if pointer.ndim != 1 or pointer.size < 1:
        raise ContractError(["indptr must be a 1-d array"])
    if int(pointer[0]) != 0:
        raise ContractError(["indptr must start at 0"])
    if np.any(np.diff(pointer) < 0):
        raise ContractError(["indptr is not monotonic"])
    nnz = int(pointer[-1])
    if targets.shape != (nnz,):
        raise ContractError(["indices length must equal indptr[-1]"])
    n = pointer.size - 1
    if nnz and (int(np.min(targets)) < 0 or int(np.max(targets)) >= n):
        raise ContractError(["indices reference a missing node"])
    return np.array(pointer, dtype=np.int64, copy=True)


def _members(
    source_ids: Sequence[str],
    cfa_node_ids: Sequence[Sequence[str]] | None,
    n: int,
) -> list[tuple[str, ...]]:
    if cfa_node_ids is None:
        return [(node_id,) for node_id in source_ids]
    if len(cfa_node_ids) != n:
        raise ContractError(["cfa_node_ids length must equal the node count"])
    rows: list[tuple[str, ...]] = []
    for group in cfa_node_ids:
        if not group or any(not isinstance(node_id, str) or node_id == "" for node_id in group):
            raise ContractError(["each dense row needs at least one CFA node id"])
        rows.append(tuple(group))
    return rows


def _node_features(values: np.ndarray, n: int) -> np.ndarray:
    matrix = np.array(values, dtype=np.float32, copy=True)
    if matrix.ndim == 1:
        matrix = matrix.reshape(n, 1)
    if matrix.ndim != 2 or matrix.shape[0] != n:
        raise ContractError(["node features must have one row per node"])
    if matrix.size and not np.isfinite(matrix).all():
        raise ContractError(["node features must be finite"])
    return matrix


def _names(
    names: Sequence[str] | None,
    width: int,
    *,
    default_one: str | None,
) -> tuple[list[str], bool]:
    if names is None:
        if width == 1 and default_one is not None:
            return [default_one], False
        return [f"f{index}" for index in range(width)], True
    resolved = [str(name) for name in names]
    if len(resolved) != width:
        raise ContractError(["feature names do not match the matrix width"])
    return resolved, False


def _edge_features(
    values: np.ndarray | None,
    nnz: int,
    names: Sequence[str] | None,
) -> tuple[np.ndarray, list[str], bool]:
    if values is None:
        if names not in (None, [], ()):
            raise ContractError(["edge feature names require edge features"])
        return np.zeros((nnz, 0), dtype=np.float32), [], False
    matrix = np.array(values, dtype=np.float32, copy=True)
    named_weight = matrix.ndim == 1
    if matrix.ndim == 1:
        if matrix.shape != (nnz,):
            raise ContractError(["edge feature length must equal the CSR edge count"])
        matrix = matrix.reshape(nnz, 1)
    if matrix.ndim != 2 or matrix.shape[0] != nnz:
        raise ContractError(["edge features must have one row per CSR edge"])
    if matrix.size and not np.isfinite(matrix).all():
        raise ContractError(["edge features must be finite"])
    default = "weight" if named_weight else None
    resolved, positional = _names(names, matrix.shape[1], default_one=default)
    return matrix, resolved, positional


def _colours(
    node_colors: np.ndarray | None,
    edge_colors: np.ndarray | None,
    color_ids: Sequence[int] | None,
    n: int,
    nnz: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if node_colors is None:
        node_mask = np.zeros((n, 0), dtype=np.uint8)
    else:
        node_mask = np.array(node_colors, dtype=np.uint8, copy=True)
        if node_mask.ndim != 2 or node_mask.shape[0] != n:
            raise ContractError(["node colours must have one row per node"])
    width = node_mask.shape[1]
    if edge_colors is None:
        edge_mask = np.zeros((nnz, width), dtype=np.uint8)
    else:
        edge_mask = np.array(edge_colors, dtype=np.uint8, copy=True)
        if edge_mask.shape != (nnz, width):
            raise ContractError(["edge colours must match the node colour width and the CSR edge count"])
    if color_ids is None:
        ids = list(range(width))
    else:
        ids = [int(color_id) for color_id in color_ids]
        if len(ids) != width:
            raise ContractError(["color_ids length must equal the colour width"])
    return node_mask, edge_mask, ids


def _weights(
    node_weights: np.ndarray | None,
    edge_weights: np.ndarray | None,
    n: int,
    nnz: int,
    width: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if node_weights is None and edge_weights is None:
        return None, None
    if node_weights is None:
        raise ContractError(["edge colour weights require node colour weights"])
    node = np.array(node_weights, dtype=np.float32, copy=True)
    if node.shape != (n, width):
        raise ContractError(["node colour weights must match the colour mask"])
    if edge_weights is None:
        edge = np.zeros((nnz, width), dtype=np.float32)
    else:
        edge = np.array(edge_weights, dtype=np.float32, copy=True)
        if edge.shape != (nnz, width):
            raise ContractError(["edge colour weights must match the colour mask and the CSR edge count"])
    return node, edge


def _sort_rows(indptr: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort targets inside each CSR row and return the slot permutation.

    A repeated target in one row raises ``ContractError``. Weights are not
    summed.
    """
    targets = np.array(indices, dtype=np.int64, copy=True)
    nnz = int(indptr[-1])
    order = np.empty(nnz, dtype=np.int64)
    cursor = 0
    for node in range(indptr.size - 1):
        start = int(indptr[node])
        end = int(indptr[node + 1])
        row = targets[start:end]
        if len(set(row.tolist())) != len(row):
            raise ContractError([f"duplicate CSR edge from node {node}"])
        slots = np.argsort(row, kind="mergesort")
        targets[start:end] = row[slots]
        order[cursor : cursor + len(slots)] = start + slots
        cursor += len(slots)
    return targets, order


def _permute_rows(matrix: np.ndarray, order: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.array(matrix, copy=True)
    return np.array(matrix[order], copy=True)
