"""CFA invariants."""

from __future__ import annotations

from typing import Any

from metametro.errors import ContractError
from metametro.formats.cfa.model import (
    ALPHABET,
    FEATURE_TYPES,
    ORIENTATIONS,
    REJECTED_COLUMNS,
    SCHEMA_VERSION,
    CfaGraph,
)


def parse_color_set(text: str) -> list[int]:
    """Parse ``0,1`` into integer color ids. Empty text is an empty set."""
    raw = text.strip()
    if raw == "":
        return []
    values: list[int] = []
    for piece in raw.split(","):
        token = piece.strip()
        if token == "" or not _is_int(token):
            raise ContractError([f"malformed color_set: {text!r}"])
        values.append(int(token))
    return values


def _is_int(token: str) -> bool:
    if token.startswith("-"):
        return token[1:].isdigit() and token != "-"
    return token.isdigit()


def _feature_types(metadata: dict[str, Any], kind: str) -> dict[str, str]:
    features = metadata.get("features") or {}
    if not isinstance(features, dict):
        raise ContractError(["features must be a mapping of node/edge declarations"])
    declared = features.get(kind) or {}
    if declared is None:
        return {}
    if isinstance(declared, list):
        raise ContractError(
            [f"features.{kind} must declare a type for each column, not a bare name list"]
        )
    if not isinstance(declared, dict):
        raise ContractError([f"features.{kind} must be a mapping of column to type"])
    typed: dict[str, str] = {}
    for name, dtype in declared.items():
        if dtype not in FEATURE_TYPES:
            raise ContractError([f"unknown feature type for {kind}.{name}: {dtype!r}"])
        typed[str(name)] = str(dtype)
    return typed


def _check_cell(dtype: str, value: str, where: str, errors: list[str]) -> None:
    if dtype == "float":
        try:
            float(value)
        except ValueError:
            errors.append(f"wrong dtype at {where}: expected float, found {value!r}")
        return
    if dtype == "int":
        if not _is_int(value.strip()):
            errors.append(f"wrong dtype at {where}: expected int, found {value!r}")
        return
    if dtype == "color_set":
        try:
            parse_color_set(value)
        except ContractError as exc:
            errors.append(f"wrong dtype at {where}: {exc}")
        return
    if dtype == "orientation":
        if value not in ORIENTATIONS:
            errors.append(f"wrong dtype at {where}: expected orientation, found {value!r}")
        return
    if dtype == "str" and not isinstance(value, str):
        errors.append(f"wrong dtype at {where}: expected str")


def validate_cfa(graph: CfaGraph) -> None:
    """Reject CFA objects that break schema 1.0 invariants."""
    errors: list[str] = []
    metadata = graph.metadata
    if not isinstance(metadata, dict):
        raise ContractError(["metadata must be a mapping"])
    version = str(metadata.get("schema_version", ""))
    if version != SCHEMA_VERSION:
        errors.append(
            f"incompatible schema version: {version!r} (supported CFA schema is {SCHEMA_VERSION})"
        )
    for key in ("graph_id", "graph_type"):
        if not metadata.get(key):
            errors.append(f"missing required field: {key}")
    graph_type = str(metadata.get("graph_type", ""))
    if graph_type == "de_bruijn":
        k = metadata.get("k")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            errors.append("de Bruijn CFA requires integer k > 0")
    try:
        node_types = _feature_types(metadata, "node")
        edge_types = _feature_types(metadata, "edge")
    except ContractError as exc:
        errors.extend(exc.errors)
        node_types, edge_types = {}, {}

    if not graph.nodes:
        errors.append("CFA requires at least one node")
    if graph.node_header[:1] != ["node_id"]:
        errors.append("first nodes.tsv column must be node_id")
    if graph.edge_header[:3] != ["edge_id", "source", "target"]:
        errors.append("edges.tsv columns must start with edge_id, source, target")

    seen_nodes: set[str] = set()
    for row in graph.nodes:
        node_id = row.get("node_id", "")
        if node_id == "":
            errors.append("node row missing node_id")
            continue
        if node_id in seen_nodes:
            errors.append(f"duplicate node_id: {node_id}")
        seen_nodes.add(node_id)
        for column in row:
            if column in REJECTED_COLUMNS:
                errors.append(
                    f"column {column} is not allowed; sequence length is len(sequence) and is not stored"
                )
        if node_id not in graph.sequences:
            errors.append(f"node {node_id} has no sequence in nodes.fna")
        for column, dtype in node_types.items():
            if column not in row:
                errors.append(f"node {node_id} missing declared feature {column}")
                continue
            _check_cell(dtype, row[column], f"node {node_id}.{column}", errors)
        for column in row:
            if column == "node_id":
                continue
            if column not in node_types:
                errors.append(f"undeclared node feature column: {column}")

    for node_id, sequence in graph.sequences.items():
        if node_id not in seen_nodes:
            errors.append(f"sequence {node_id} has no nodes.tsv row")
        if sequence == "" or any(base not in ALPHABET for base in sequence):
            errors.append(f"malformed sequence for {node_id}")

    color_ids = _dictionary_ids(graph.colors, "color", errors)
    label_ids = _dictionary_ids(graph.labels, "label", errors)

    seen_edges: set[str] = set()
    for row in graph.edges:
        edge_id = row.get("edge_id", "")
        source = row.get("source", "")
        target = row.get("target", "")
        if edge_id == "":
            errors.append("edge row missing edge_id")
            continue
        if edge_id in seen_edges:
            errors.append(f"duplicate edge_id: {edge_id}")
        seen_edges.add(edge_id)
        if source not in seen_nodes or target not in seen_nodes:
            errors.append(f"dangling edge {edge_id}: {source} -> {target}")
        for column in row:
            if column in REJECTED_COLUMNS:
                errors.append(f"column {column} is not allowed on edges")
        for column, dtype in edge_types.items():
            if column not in row:
                errors.append(f"edge {edge_id} missing declared feature {column}")
                continue
            _check_cell(dtype, row[column], f"edge {edge_id}.{column}", errors)
        for column in row:
            if column in {"edge_id", "source", "target"}:
                continue
            if column not in edge_types:
                errors.append(f"undeclared edge feature column: {column}")
        _check_refs(row, "color_set", color_ids, f"edge {edge_id}", errors)
        _check_refs(row, "colors", color_ids, f"edge {edge_id}", errors)

    for row in graph.nodes:
        node_id = row.get("node_id", "")
        _check_refs(row, "color_set", color_ids, f"node {node_id}", errors)
        _check_refs(row, "colors", color_ids, f"node {node_id}", errors)
        if "label" in row and row["label"].strip() != "":
            if not _is_int(row["label"].strip()):
                errors.append(f"malformed label on node {node_id}")
            elif int(row["label"]) not in label_ids:
                errors.append(f"undefined label {row['label']} on node {node_id}")

    if errors:
        raise ContractError(errors)


def _dictionary_ids(
    rows: list[dict[str, str]] | None,
    kind: str,
    errors: list[str],
) -> set[int]:
    if rows is None:
        return set()
    found: set[int] = set()
    id_column = f"{kind}_id"
    for row in rows:
        token = row.get(id_column, "").strip()
        if not _is_int(token):
            errors.append(f"malformed {id_column}: {token!r}")
            continue
        value = int(token)
        if value in found:
            errors.append(f"duplicate {id_column}: {value}")
        found.add(value)
        if kind == "color":
            if "value" not in row and "name" not in row:
                errors.append(f"color {value} needs a name or value")
        else:
            if "namespace" not in row or "value" not in row:
                errors.append(f"label {value} needs namespace and value")
    return found


def _check_refs(
    row: dict[str, str],
    column: str,
    allowed: set[int],
    where: str,
    errors: list[str],
) -> None:
    if column not in row or row[column].strip() == "":
        return
    try:
        ids = parse_color_set(row[column])
    except ContractError:
        return
    for color_id in ids:
        if color_id not in allowed:
            errors.append(f"undefined color {color_id} on {where}")
