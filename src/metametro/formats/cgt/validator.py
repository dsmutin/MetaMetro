"""CGT invariants."""

from __future__ import annotations

import numpy as np

from metametro.errors import ContractError
from metametro.formats.cgt.model import SCHEMA_VERSION, Cgt


def _i64(array: np.ndarray, name: str, errors: list[str]) -> None:
    if array.dtype != np.int64:
        errors.append(f"wrong dtype for {name}: expected int64, found {array.dtype}")


def validate_cgt(graph: Cgt) -> None:
    """Reject tensors that break schema 1.0 alignment invariants."""
    errors: list[str] = []
    metadata = graph.metadata
    version = str(metadata.get("schema_version", ""))
    if version != SCHEMA_VERSION:
        errors.append(
            f"incompatible schema version: {version!r} (supported CGT schema is {SCHEMA_VERSION})"
        )
    if metadata.get("topology") != "csr":
        errors.append("topology must be csr")
    for key in (
        "num_nodes",
        "num_edges",
        "node_feature_names",
        "edge_feature_names",
        "node_feature_dtype",
        "edge_feature_dtype",
        "source",
    ):
        if key not in metadata:
            errors.append(f"missing required field: {key}")
    if metadata.get("node_feature_dtype") not in (None, "float32"):
        errors.append("wrong dtype: node_feature_dtype must be float32")
    if metadata.get("edge_feature_dtype") not in (None, "float32"):
        errors.append("wrong dtype: edge_feature_dtype must be float32")

    n = graph.num_nodes
    e = graph.num_edges
    if metadata.get("num_nodes") not in (None, n):
        errors.append("num_nodes does not match the feature matrix")
    if metadata.get("num_edges") not in (None, e):
        errors.append("num_edges does not match the topology")
    if graph.indptr.shape != (n + 1,):
        errors.append("invalid topology: indptr length must be num_nodes + 1")
    else:
        _i64(graph.indptr, "indptr", errors)
        if int(graph.indptr[0]) != 0 or int(graph.indptr[-1]) != e:
            errors.append("invalid topology: indptr must start at 0 and end at num_edges")
        if np.any(np.diff(graph.indptr) < 0):
            errors.append("invalid topology: indptr is not monotonic")
    if graph.indices.shape != (e,):
        errors.append("invalid topology: indices length must equal num_edges")
    else:
        _i64(graph.indices, "indices", errors)
        if e and (int(graph.indices.min()) < 0 or int(graph.indices.max()) >= n):
            errors.append("invalid topology: indices reference a missing node")
    if graph.node_features.dtype != np.float32 or graph.node_features.shape[0] != n:
        errors.append("wrong dtype or shape for node features")
    if graph.node_features.ndim != 2:
        errors.append("node features must be a dense 2-d matrix")
    names = metadata.get("node_feature_names")
    if isinstance(names, list) and graph.node_features.ndim == 2:
        if len(names) != graph.node_features.shape[1]:
            errors.append("node_feature_names do not match X_node width")
    if graph.edge_features.dtype != np.float32 or graph.edge_features.shape != (e, graph.edge_features.shape[1] if graph.edge_features.ndim == 2 else 0):
        errors.append("wrong dtype or shape for edge features")
    if graph.edge_features.ndim != 2 or graph.edge_features.shape[0] != e:
        errors.append("edge features must align with indices[i]")
    edge_names = metadata.get("edge_feature_names")
    if isinstance(edge_names, list) and graph.edge_features.ndim == 2:
        if len(edge_names) != graph.edge_features.shape[1]:
            errors.append("edge_feature_names do not match X_edge width")
    if graph.node_labels is not None:
        if graph.node_labels.shape != (n,) or graph.node_labels.dtype != np.int64:
            errors.append("node labels must have shape (N,) and dtype int64")
    if graph.edge_labels is not None:
        if graph.edge_labels.shape != (e,) or graph.edge_labels.dtype != np.int64:
            errors.append("edge labels must have shape (E,) and dtype int64")
    if graph.node_colors.ndim != 2 or graph.node_colors.shape[0] != n or graph.node_colors.dtype != np.uint8:
        errors.append("node colours must be a uint8 matrix with one row per node")
    if graph.edge_colors.ndim != 2 or graph.edge_colors.shape[0] != e or graph.edge_colors.dtype != np.uint8:
        errors.append("edge colours must align with indices[i]")
    elif (
        graph.node_colors.ndim == 2
        and graph.node_colors.shape[1] != graph.edge_colors.shape[1]
    ):
        errors.append("node and edge colour matrices have different widths")
    if graph.node_colors.ndim == 2 and len(graph.color_ids) != graph.node_colors.shape[1]:
        errors.append("color_ids do not match the colour matrix width")
    if len(graph.mapping) != n:
        errors.append("invalid mapping: dense_id table length must equal N")
    dense_ids = [row.get("dense_id") for row in graph.mapping]
    if dense_ids != list(range(n)):
        errors.append("invalid mapping: dense ids must be 0 .. N-1 in order")
    seen_source: set[str] = set()
    seen_cfa: set[str] = set()
    for row in graph.mapping:
        source_id = row.get("source_id")
        cfa_node_ids = row.get("cfa_node_ids")
        if not isinstance(source_id, str) or source_id == "":
            errors.append("invalid mapping: missing source_id")
        elif source_id in seen_source:
            errors.append(f"invalid mapping: duplicate source_id {source_id}")
        else:
            seen_source.add(source_id)
        if (
            not isinstance(cfa_node_ids, list)
            or not cfa_node_ids
            or any(not isinstance(node_id, str) or node_id == "" for node_id in cfa_node_ids)
        ):
            errors.append(f"invalid mapping: dense id {row.get('dense_id')} has no CFA node ids")
        else:
            for node_id in cfa_node_ids:
                if node_id in seen_cfa:
                    errors.append(f"invalid mapping: CFA node {node_id} is repeated")
                seen_cfa.add(node_id)
    for array in (
        graph.indptr,
        graph.indices,
        graph.node_features,
        graph.edge_features,
        graph.node_colors,
        graph.edge_colors,
    ):
        if array.dtype == object:
            errors.append("CGT must not contain Python objects")
    if errors:
        raise ContractError(errors)
