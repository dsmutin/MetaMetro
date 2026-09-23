"""CDBG to coloured graph tensor.

Node features are aligned to unitig ids sorted lexicographically, which is
also the dense id order. Edge features are aligned to ``cdbg.links`` and then
moved with those links into CSR order, so ``indices[j]`` and
``edge_features[j]`` stay the same adjacency entry.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from metametro.errors import ContractError
from metametro.formats.cdbg.annotations import annotation_feature_block
from metametro.formats.cdbg.model import Cdbg
from metametro.formats.cdbg.validator import validate_cdbg
from metametro.formats.cgt.model import SCHEMA_VERSION, Cgt


def _as_matrix(
    values: np.ndarray | Mapping[str, Sequence[float]] | None,
    keys: list[str],
    width_name: str,
) -> np.ndarray:
    n = len(keys)
    if values is None:
        return np.zeros((n, 0), dtype=np.float32)
    if isinstance(values, np.ndarray):
        if values.shape[0] != n:
            raise ContractError([f"{width_name} row count must equal {n}"])
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(n, 1)
        return matrix
    rows = []
    for key in keys:
        if key not in values:
            raise ContractError([f"missing features for {key}"])
        rows.append(np.asarray(values[key], dtype=np.float32))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    width = rows[0].shape[0]
    if any(row.shape != (width,) for row in rows):
        raise ContractError([f"{width_name} vectors must share one width"])
    return np.stack(rows).astype(np.float32)


def csr_link_order(cdbg: Cdbg) -> list[int]:
    """Return ``cdbg.links`` positions in CSR order.

    The sort key is source dense id, target dense id, then link id. Dense ids
    follow unitig ids sorted lexicographically. A link whose endpoint is not a
    unitig raises ``ContractError``.
    """
    unitigs = sorted(cdbg.unitigs, key=lambda unitig: unitig.unitig_id)
    dense = {unitig.unitig_id: index for index, unitig in enumerate(unitigs)}
    missing = [
        link.link_id
        for link in cdbg.links
        if link.source not in dense or link.target not in dense
    ]
    if missing:
        raise ContractError([f"dangling link {link_id}" for link_id in missing])
    return sorted(
        range(len(cdbg.links)),
        key=lambda index: (
            dense[cdbg.links[index].source],
            dense[cdbg.links[index].target],
            cdbg.links[index].link_id,
        ),
    )


def _as_labels(
    values: np.ndarray | Mapping[str, int] | None,
    keys: list[str],
) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        if values.shape != (len(keys),):
            raise ContractError(["label vector length must match the object order"])
        return np.asarray(values, dtype=np.int64)
    return np.asarray([int(values[key]) for key in keys], dtype=np.int64)


def cdbg_to_cgt(
    cdbg: Cdbg,
    node_features: np.ndarray | Mapping[str, Sequence[float]] | None = None,
    edge_features: np.ndarray | Mapping[str, Sequence[float]] | None = None,
    node_labels: np.ndarray | Mapping[str, int] | None = None,
    edge_labels: np.ndarray | Mapping[str, int] | None = None,
    node_feature_names: Sequence[str] | None = None,
    edge_feature_names: Sequence[str] | None = None,
    node_annotation: Sequence[tuple[str, str]] | None = None,
    edge_annotation: Sequence[tuple[str, str]] | None = None,
) -> Cgt:
    """Materialize a CSR tensor from a CDBG in one pass over nodes, edges, and features.

    ``node_features`` and ``edge_features`` keep their previous meaning.
    ``node_annotation`` and ``edge_annotation`` are ``(namespace, feature)``
    pairs drawn from the CDBG sidecar. Node pairs must already be unitig-level
    (``target_type="node"``). Edge pairs must already be link-level
    (``target_type="edge"``). Internal unitig edges are not CSR edges and are
    not written into ``X_edge``. Per-CFA-node rows stay on the sidecar; join
    them with ``node_lineage`` after this conversion. Selected annotation
    columns are appended after the explicit feature columns. When feature
    names are omitted, explicit columns are named ``f0``, ``f1``, ... and
    annotation columns are named ``namespace:feature`` (or
    ``namespace:feature:i`` for a vector).
    """
    validate_cdbg(cdbg)
    unitigs = sorted(cdbg.unitigs, key=lambda unitig: unitig.unitig_id)
    source_ids = [unitig.unitig_id for unitig in unitigs]
    dense = {unitig_id: index for index, unitig_id in enumerate(source_ids)}
    members = {unitig.unitig_id: list(unitig.members) for unitig in unitigs}
    node_matrix = _as_matrix(node_features, source_ids, "node features")
    node_block, node_ann_names = annotation_feature_block(cdbg, node_annotation, source_ids, "node")
    if node_block.shape[1]:
        node_matrix = np.hstack([node_matrix, node_block])
    node_y = _as_labels(node_labels, source_ids)
    link_ids = [link.link_id for link in cdbg.links]
    edge_matrix = _as_matrix(edge_features, link_ids, "edge features")
    edge_block, edge_ann_names = annotation_feature_block(cdbg, edge_annotation, link_ids, "edge")
    if edge_block.shape[1]:
        edge_matrix = np.hstack([edge_matrix, edge_block])
    edge_y = _as_labels(edge_labels, link_ids)

    colored = cdbg.colors or []
    palette = sorted(int(row["color_id"]) for row in colored) if colored else sorted(
        {color_id for unitig in unitigs for color_id in unitig.color_ids}
        | {color_id for link in cdbg.links for color_id in link.color_ids}
    )
    column = {color_id: index for index, color_id in enumerate(palette)}
    node_colors = np.zeros((len(unitigs), len(palette)), dtype=np.uint8)
    for unitig in unitigs:
        row = dense[unitig.unitig_id]
        for color_id in unitig.color_ids:
            if color_id in column:
                node_colors[row, column[color_id]] = 1

    order = csr_link_order(cdbg)
    indices = np.asarray([dense[cdbg.links[index].target] for index in order], dtype=np.int64)
    counts = np.zeros(len(unitigs), dtype=np.int64)
    for index in order:
        counts[dense[cdbg.links[index].source]] += 1
    indptr = np.zeros(len(unitigs) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    edge_matrix = edge_matrix[np.asarray(order, dtype=np.int64)]
    if edge_y is not None:
        edge_y = edge_y[order]
    edge_colors = np.zeros((len(order), len(palette)), dtype=np.uint8)
    for slot, index in enumerate(order):
        for color_id in cdbg.links[index].color_ids:
            if color_id in column:
                edge_colors[slot, column[color_id]] = 1

    if node_feature_names is None:
        base_width = node_matrix.shape[1] - node_block.shape[1]
        node_feature_names = [f"f{i}" for i in range(base_width)] + node_ann_names
    if edge_feature_names is None:
        base_width = edge_matrix.shape[1] - edge_block.shape[1]
        edge_feature_names = [f"f{i}" for i in range(base_width)] + edge_ann_names
    if len(node_feature_names) != node_matrix.shape[1]:
        raise ContractError(["node_feature_names do not match the feature width"])
    if len(edge_feature_names) != edge_matrix.shape[1]:
        raise ContractError(["edge_feature_names do not match the feature width"])

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "num_nodes": len(unitigs),
        "num_edges": len(order),
        "node_feature_names": list(node_feature_names),
        "edge_feature_names": list(edge_feature_names),
        "node_feature_dtype": "float32",
        "edge_feature_dtype": "float32",
        "topology": "csr",
        "contract": "cdbg_to_cgt",
        "contract_version": "1.0",
        "source": {
            "format": "cdbg",
            "graph_id": cdbg.metadata.get("graph_id"),
        },
    }
    if node_ann_names:
        metadata["node_annotation_features"] = list(node_ann_names)
    if edge_ann_names:
        metadata["edge_annotation_features"] = list(edge_ann_names)
    mapping = [
        {
            "dense_id": dense[unitig.unitig_id],
            "source_id": unitig.unitig_id,
            "cfa_node_ids": members[unitig.unitig_id],
        }
        for unitig in unitigs
    ]
    return Cgt(
        metadata=metadata,
        indptr=indptr,
        indices=indices,
        node_features=node_matrix,
        edge_features=edge_matrix.astype(np.float32),
        node_labels=node_y,
        edge_labels=edge_y,
        node_colors=node_colors,
        edge_colors=edge_colors,
        mapping=mapping,
        color_ids=palette,
    )
