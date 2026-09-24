"""Columnar annotation sidecar for a CDBG (ToCUMG).

CFA is the semantic source of truth. This module stores annotations that
must stay attached to a compacted graph: one table per layer, not a Python
object per node and not a dense feature matrix. A CGT feature matrix is a
later, explicit selection of unitig rows and CDBG-link rows.

Target ids:

- ``node``: whole unitig (``unitig_id`` is not a biological name)
- ``internal_node``: one original CFA node
- ``internal_edge``: one CFA edge absorbed into a unitig
- ``edge``: one CDBG link (the CFA edge id stored as ``link_id``)

Compaction does not copy CFA columns. ``transfer_annotations`` does that
when the caller supplies provenance. ``aggregate_annotations`` collapses
member rows only when the caller names a policy. Missing values raise
``ContractError`` and are not imputed.
"""

from __future__ import annotations

import math
import re
import shutil
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metametro.errors import ContractError
from metametro.formats.cdbg.model import Cdbg
from metametro.tables import read_tsv, read_yaml, write_tsv, write_yaml


TARGET_TYPES = frozenset({"node", "edge", "internal_node", "internal_edge"})
NODE_TARGET_TYPES = frozenset({"node", "internal_node"})
EDGE_TARGET_TYPES = frozenset({"edge", "internal_edge"})
DTYPES = frozenset({"float64", "int64", "category"})
KINDS = frozenset({"scalar", "vector"})
AGGREGATION_POLICIES = frozenset(
    {"mean", "sum", "min", "max", "median", "weighted_mean", "union", "majority", "keep_per_member"}
)
NUMERIC_REDUCTIONS = frozenset({"mean", "sum", "min", "max", "median", "weighted_mean"})
SIDECAR_VERSION = "1.0"
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")
_PROVENANCE_FIELDS = (
    "source",
    "method",
    "version",
    "parameters",
    "parent_graph_id",
    "parent_schema_version",
    "parent_contract_version",
)


@dataclass(frozen=True)
class AnnotationProvenance:
    """Reproducible origin of one annotation layer.

    Every field is required. ``parameters`` is a plain mapping the caller
    can dump to YAML. ``parent_graph_id`` is the CDBG ``graph_id``, not a
    biological name invented for a unitig.
    """

    source: str
    method: str
    version: str
    parameters: dict[str, Any]
    parent_graph_id: str
    parent_schema_version: str
    parent_contract_version: str


@dataclass
class AnnotationLayer:
    """One columnar annotation table.

    ``target_ids[i]`` and ``values[i]`` are the same record. Scalar numeric
    values have shape ``(n,)``. Vector values have shape ``(n, width)``.
    Categories are a one-dimensional array of strings. There is no per-node
    object and no dense ``N × F`` matrix on the graph.
    """

    namespace: str
    feature: str
    target_type: str
    dtype: str
    kind: str
    provenance: AnnotationProvenance
    aggregation: str | None
    target_ids: np.ndarray
    values: np.ndarray


def annotate_cdbg(
    cdbg: Cdbg,
    *,
    namespace: str,
    feature: str,
    target_type: str,
    values: Mapping[str, Any],
    dtype: str,
    provenance: Mapping[str, Any],
    kind: str = "scalar",
    replace: bool = False,
) -> Cdbg:
    """Attach one annotation layer to ``cdbg``.

    Topology, colours, and the CFA mapping are left as they are. An existing
    layer with the same namespace, feature, and target type raises
    ``ContractError`` unless ``replace`` is true. Unknown targets, empty
    values, and mixed dtypes also raise. The function returns the same
    ``cdbg`` object.
    """
    from metametro.formats.cdbg.validator import validate_cdbg

    validate_cdbg(cdbg)
    _require_flag(replace, "replace")
    _require_name(namespace, "namespace")
    _require_name(feature, "feature")
    if target_type not in TARGET_TYPES:
        raise ContractError([f"unknown annotation target_type: {target_type}"])
    if dtype not in DTYPES or kind not in KINDS:
        raise ContractError([f"incompatible annotation dtype: {dtype} kind {kind}"])
    if kind == "vector" and dtype == "category":
        raise ContractError(["incompatible annotation dtype: category vectors are not supported"])
    record = _parse_provenance(provenance, cdbg)
    target_ids, array = _coerce_values(values, dtype=dtype, kind=kind)
    known = _known_targets(cdbg, target_type)
    missing = [target_id for target_id in target_ids.tolist() if target_id not in known]
    if missing:
        raise ContractError([f"missing annotation target: {target_id}" for target_id in missing])
    layer = AnnotationLayer(
        namespace=namespace,
        feature=feature,
        target_type=target_type,
        dtype=dtype,
        kind=kind,
        provenance=record,
        aggregation=None,
        target_ids=target_ids,
        values=array,
    )
    _store_layer(cdbg, layer, replace=replace)
    return cdbg


def get_node_annotations(
    cdbg: Cdbg,
    *,
    namespace: str,
    feature: str,
    target_type: str = "node",
) -> AnnotationLayer:
    """Return the columnar node layer for ``namespace`` and ``feature``.

    ``target_type`` is ``node`` (one row per unitig) or ``internal_node``
    (one row per CFA member). A missing layer raises ``ContractError``.
    """
    if target_type not in NODE_TARGET_TYPES:
        raise ContractError(["get_node_annotations target_type must be node or internal_node"])
    return _require_layer(cdbg, namespace, feature, target_type)


def get_edge_annotations(
    cdbg: Cdbg,
    *,
    namespace: str,
    feature: str,
    target_type: str = "edge",
) -> AnnotationLayer:
    """Return the columnar edge layer for ``namespace`` and ``feature``.

    ``target_type`` is ``edge`` (one CDBG link) or ``internal_edge`` (one
    CFA edge absorbed into a unitig). A missing layer raises ``ContractError``.
    """
    if target_type not in EDGE_TARGET_TYPES:
        raise ContractError(["get_edge_annotations target_type must be edge or internal_edge"])
    return _require_layer(cdbg, namespace, feature, target_type)


def transfer_annotations(
    cdbg: Cdbg,
    cfa: Any,
    *,
    namespace: str,
    provenance: Mapping[str, Any],
    node_columns: Sequence[str] | None = None,
    edge_columns: Sequence[str] | None = None,
    replace: bool = False,
) -> Cdbg:
    """Copy selected CFA columns onto the CDBG without aggregating them.

    Node columns become ``internal_node`` rows keyed by the CFA node id.
    An edge column becomes an ``internal_edge`` row when compaction absorbed
    that edge, and an ``edge`` row when the edge is still a CDBG link.
    The split uses ``unitig.internal_edge_ids`` and ``link.link_id``. Those
    ids are the CFA edge ids.
    Colour sets and orientations are not copied. A blank cell raises
    ``ContractError``. The function returns the same ``cdbg`` object and
    does not edit unitigs, links, or the mapping.
    """
    from metametro.formats.cdbg.validator import validate_cdbg
    from metametro.formats.cfa.validator import validate_cfa

    validate_cdbg(cdbg)
    _require_flag(replace, "replace")
    _require_name(namespace, "namespace")
    validate_cfa(cfa)
    record = _parse_provenance(provenance, cdbg)
    cfa_graph_id = cfa.metadata.get("graph_id")
    if not isinstance(cfa_graph_id, str) or cfa_graph_id == "":
        raise ContractError(["CFA metadata is missing graph_id"])
    mapped = {row.cfa_node_id for row in cdbg.mapping}
    if set(cfa.node_ids()) != mapped:
        raise ContractError(["CFA nodes do not match the CDBG mapping"])
    internal: dict[str, str] = {}
    for unitig in cdbg.unitigs:
        for edge_id in unitig.internal_edge_ids:
            if edge_id in internal:
                raise ContractError([f"duplicate edge id {edge_id}"])
            internal[edge_id] = unitig.unitig_id
    links = {link.link_id for link in cdbg.links}
    overlap = set(internal) & links
    if overlap:
        raise ContractError([f"CFA edge {edge_id} is both a link and an internal edge" for edge_id in sorted(overlap)])
    node_columns = _transfer_columns(cfa, "node", node_columns)
    edge_columns = _transfer_columns(cfa, "edge", edge_columns)
    if not node_columns and not edge_columns:
        raise ContractError(["no CFA columns selected for transfer"])
    planned: list[AnnotationLayer] = []
    for column in node_columns:
        planned.append(
            _column_layer(
                cdbg,
                rows=cfa.nodes,
                id_column="node_id",
                column=column,
                declared=_declared_type(cfa, "node", column),
                namespace=namespace,
                target_type="internal_node",
                provenance=record,
                cfa_graph_id=cfa_graph_id,
                table="nodes",
            )
        )
    edge_rows = {row["edge_id"]: row for row in cfa.edges}
    cfa_edges = set(edge_rows)
    if cfa_edges != set(internal) | links:
        raise ContractError(["CFA edges do not match CDBG links and internal edges"])
    for column in edge_columns:
        declared = _declared_type(cfa, "edge", column)
        for target_type, id_set in (("internal_edge", set(internal)), ("edge", links)):
            chosen = [edge_rows[edge_id] for edge_id in _ordered_edge_ids(edge_rows, id_set)]
            if not chosen:
                continue
            planned.append(
                _column_layer(
                    cdbg,
                    rows=chosen,
                    id_column="edge_id",
                    column=column,
                    declared=declared,
                    namespace=namespace,
                    target_type=target_type,
                    provenance=record,
                    cfa_graph_id=cfa_graph_id,
                    table="edges",
                )
            )
    _store_many(cdbg, planned, replace=replace)
    return cdbg


def aggregate_annotations(
    cdbg: Cdbg,
    *,
    namespace: str,
    feature: str,
    policy: str,
    provenance: Mapping[str, Any],
    source_target_type: str = "internal_node",
    weights: Mapping[str, float] | None = None,
    replace: bool = False,
) -> Cdbg:
    """Collapse per-member annotations onto one value per unitig.

    ``policy`` must be one of ``mean``, ``sum``, ``min``, ``max``, ``median``,
    ``weighted_mean``, ``union``, ``majority``, or ``keep_per_member``.
    Numeric policies are not applied to categories. A member with no value
    raises ``ContractError``. ``weighted_mean`` uses CFA member sequence
    lengths from the mapping unless ``weights`` is passed. Those weights are
    keyed by the source target id. ``keep_per_member`` does not write a
    unitig row; it records the policy on the existing layer. Other policies
    add a ``node`` layer and leave the source layer in place. The function
    returns the same ``cdbg`` object.
    """
    from metametro.formats.cdbg.validator import validate_cdbg

    validate_cdbg(cdbg)
    _require_flag(replace, "replace")
    if policy not in AGGREGATION_POLICIES:
        raise ContractError([f"unknown aggregation policy: {policy}"])
    if source_target_type not in {"internal_node", "internal_edge"}:
        raise ContractError(["aggregate_annotations source_target_type must be internal_node or internal_edge"])
    record = _parse_provenance(provenance, cdbg)
    if "aggregation" in record.parameters:
        raise ContractError(["provenance parameters already contain aggregation"])
    layer = _require_layer(cdbg, namespace, feature, source_target_type)
    if policy == "keep_per_member":
        if layer.aggregation is not None and not replace:
            raise ContractError(
                [
                    "annotation layer already records aggregation "
                    f"{layer.aggregation!r}; pass replace=True to overwrite"
                ]
            )
        _replace_layer(
            cdbg,
            _with_policy(layer, policy=policy, provenance=_with_aggregation_parameters(record, policy, layer, weights=None, used=None)),
        )
        return cdbg
    groups = _groups(cdbg, source_target_type)
    if not groups:
        raise ContractError(["no annotation targets to aggregate"])
    lookup = _value_index(layer)
    if weights is not None:
        unknown = [key for key in weights if key not in lookup]
        if unknown:
            raise ContractError([f"missing annotation target: {key}" for key in unknown])
    reduced_ids: list[str] = []
    reduced_values: list[Any] = []
    used_weights: dict[str, float] | None = {} if policy == "weighted_mean" else None
    out_dtype = layer.dtype
    out_kind = layer.kind
    for unitig_id, members in groups:
        missing = [member for member in members if member not in lookup]
        if missing:
            raise ContractError(
                [f"missing annotation value for {member}; refusing to impute" for member in missing]
            )
        positions = [lookup[member] for member in members]
        if policy == "weighted_mean":
            group_weights = _group_weights(cdbg, source_target_type, members, weights)
            if used_weights is not None:
                for member, weight in zip(members, group_weights):
                    used_weights[member] = float(weight)
        else:
            group_weights = None
        value, out_dtype, out_kind = _reduce(layer, positions, policy, group_weights, unitig_id)
        reduced_ids.append(unitig_id)
        reduced_values.append(value)
    output = AnnotationLayer(
        namespace=namespace,
        feature=feature,
        target_type="node",
        dtype=out_dtype,
        kind=out_kind,
        provenance=_with_aggregation_parameters(
            record,
            policy,
            layer,
            weights=weights,
            used=used_weights,
        ),
        aggregation=policy,
        target_ids=np.asarray(reduced_ids, dtype=object),
        values=_stack_reduced(reduced_values, out_dtype, out_kind),
    )
    _store_layer(cdbg, output, replace=replace)
    return cdbg


def annotation_feature_block(
    cdbg: Cdbg,
    specs: Sequence[tuple[str, str]] | None,
    keys: Sequence[str],
    target_type: str,
) -> tuple[np.ndarray, list[str], list[tuple[str, str]]]:
    """Build a float32 block aligned to ``keys`` from selected layers.

    ``target_type`` ``node`` reads unitig annotations. ``edge`` reads CDBG
    link annotations. An internal-edge layer is refused here because those
    edges are not CSR edges. Category layers are refused because ``X_node``
    and ``X_edge`` are float32. A key with no row raises ``ContractError``.

    The third value is one ``(namespace, source_annotation)`` pair per
    column. ``source_annotation`` is ``namespace:feature`` for the sidecar
    layer. A vector column ``namespace:feature:i`` keeps that same source.
    """
    parsed = _parse_specs(specs, "node_annotation" if target_type == "node" else "edge_annotation")
    width_names: list[str] = []
    sources: list[tuple[str, str]] = []
    blocks: list[np.ndarray] = []
    for namespace, feature in parsed:
        layer = _layer_for_features(cdbg, namespace, feature, target_type)
        if layer.dtype == "category":
            raise ContractError(
                [f"incompatible annotation dtype category for {namespace}:{feature}"]
            )
        lookup = _value_index(layer)
        missing = [key for key in keys if key not in lookup]
        if missing:
            raise ContractError([f"missing annotation target: {key}" for key in missing])
        positions = np.asarray([lookup[key] for key in keys], dtype=np.int64)
        block = np.asarray(layer.values[positions], dtype=np.float32)
        source = f"{namespace}:{feature}"
        if layer.kind == "scalar":
            block = block.reshape(len(keys), 1)
            width_names.append(source)
            sources.append((namespace, source))
        else:
            block = block.reshape(len(keys), layer.values.shape[1])
            width_names.extend(f"{source}:{index}" for index in range(block.shape[1]))
            sources.extend((namespace, source) for _ in range(block.shape[1]))
        blocks.append(block)
    if not blocks:
        return np.zeros((len(keys), 0), dtype=np.float32), [], []
    return np.hstack(blocks), width_names, sources


def annotation_errors(graph: Cdbg) -> list[str]:
    """Return sidecar invariant failures. An empty sidecar has none."""
    layers = getattr(graph, "annotations", None)
    if layers is None:
        return ["annotations must be a list"]
    if not isinstance(layers, list):
        return ["annotations must be a list"]
    errors: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for layer in layers:
        if not isinstance(layer, AnnotationLayer):
            errors.append("annotations must contain AnnotationLayer tables")
            continue
        key = (layer.namespace, layer.feature, layer.target_type)
        if key in seen:
            errors.append(
                f"annotation layer already exists: {layer.namespace}:{layer.feature}:{layer.target_type}"
            )
        seen.add(key)
        errors.extend(_layer_errors(graph, layer))
    return errors


def dump_annotations(graph: Cdbg, root: Path) -> None:
    """Write ``annotations/`` or remove a stale sidecar when there are no layers."""
    directory = root / "annotations"
    layers = list(graph.annotations or [])
    if not layers:
        if directory.exists():
            shutil.rmtree(directory)
        return
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    manifest: list[dict[str, Any]] = []
    stems: set[str] = set()
    for layer in layers:
        stem = _stem(layer.namespace, layer.feature, layer.target_type)
        if stem in stems:
            raise ContractError(
                [f"annotation file name {stem} is shared by two layers; namespace and feature must not contain '__'"]
            )
        stems.add(stem)
        entry: dict[str, Any] = {
            "namespace": layer.namespace,
            "feature": layer.feature,
            "target_type": layer.target_type,
            "dtype": layer.dtype,
            "kind": layer.kind,
            "aggregation": layer.aggregation,
            "provenance": _provenance_mapping(layer.provenance),
        }
        if layer.dtype == "category":
            write_tsv(
                directory / f"{stem}.tsv",
                ["target_id", "value"],
                [
                    {"target_id": target_id, "value": value}
                    for target_id, value in zip(layer.target_ids.tolist(), layer.values.tolist())
                ],
            )
            entry["table"] = f"{stem}.tsv"
        else:
            write_tsv(
                directory / f"{stem}.ids.tsv",
                ["target_id"],
                [{"target_id": target_id} for target_id in layer.target_ids.tolist()],
            )
            np.save(directory / f"{stem}.npy", np.asarray(layer.values))
            entry["ids"] = f"{stem}.ids.tsv"
            entry["values"] = f"{stem}.npy"
        manifest.append(entry)
    write_yaml(
        directory / "layers.yaml",
        {"annotation_sidecar_version": SIDECAR_VERSION, "layers": manifest},
    )


def load_annotations(root: Path) -> list[AnnotationLayer]:
    """Load the sidecar. A missing directory means the graph has no layers."""
    directory = root / "annotations"
    if not directory.exists():
        return []
    manifest_path = directory / "layers.yaml"
    if not manifest_path.is_file():
        raise ContractError(["missing required annotation file: layers.yaml"])
    manifest = read_yaml(manifest_path)
    version = str(manifest.get("annotation_sidecar_version", ""))
    if version != SIDECAR_VERSION:
        raise ContractError([f"incompatible annotation sidecar version: {version!r}"])
    raw_layers = manifest.get("layers")
    if not isinstance(raw_layers, list):
        raise ContractError(["annotation sidecar is missing layers"])
    expected = {"layers.yaml"}
    layers: list[AnnotationLayer] = []
    for entry in raw_layers:
        if not isinstance(entry, dict):
            raise ContractError(["annotation sidecar layer is not a mapping"])
        layer, files = _layer_from_entry(directory, entry)
        layers.append(layer)
        expected.update(files)
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    problems = []
    if missing:
        problems.append(f"missing required annotation file: {', '.join(missing)}")
    if extra:
        problems.append(f"unexpected annotation file: {', '.join(extra)}")
    if problems:
        raise ContractError(problems)
    return layers


def _layer_from_entry(directory: Path, entry: dict[str, Any]) -> tuple[AnnotationLayer, set[str]]:
    namespace = entry.get("namespace")
    feature = entry.get("feature")
    target_type = entry.get("target_type")
    dtype = entry.get("dtype")
    kind = entry.get("kind")
    if not all(isinstance(value, str) for value in (namespace, feature, target_type, dtype, kind)):
        raise ContractError(["annotation sidecar layer is missing a required field"])
    assert isinstance(namespace, str) and isinstance(feature, str) and isinstance(target_type, str)
    assert isinstance(dtype, str) and isinstance(kind, str)
    aggregation = entry.get("aggregation")
    if aggregation is not None and aggregation not in AGGREGATION_POLICIES:
        raise ContractError([f"unknown aggregation policy: {aggregation}"])
    provenance = _provenance_from_mapping(entry.get("provenance"))
    stem = _stem(namespace, feature, target_type)
    if dtype == "category":
        table = directory / f"{stem}.tsv"
        if not table.is_file():
            raise ContractError([f"missing required annotation file: {stem}.tsv"])
        header, rows = read_tsv(table)
        if header != ["target_id", "value"]:
            raise ContractError([f"malformed annotation table: {stem}.tsv"])
        target_ids = np.asarray([row["target_id"] for row in rows], dtype=object)
        values = np.asarray([row["value"] for row in rows], dtype=object)
        files = {f"{stem}.tsv"}
    else:
        ids_path = directory / f"{stem}.ids.tsv"
        values_path = directory / f"{stem}.npy"
        if not ids_path.is_file() or not values_path.is_file():
            raise ContractError([f"missing required annotation file: {stem}.ids.tsv or {stem}.npy"])
        header, rows = read_tsv(ids_path)
        if header != ["target_id"]:
            raise ContractError([f"malformed annotation table: {stem}.ids.tsv"])
        target_ids = np.asarray([row["target_id"] for row in rows], dtype=object)
        values = np.load(values_path, allow_pickle=False)
        files = {f"{stem}.ids.tsv", f"{stem}.npy"}
    return (
        AnnotationLayer(
            namespace=namespace,
            feature=feature,
            target_type=target_type,
            dtype=dtype,
            kind=kind,
            provenance=provenance,
            aggregation=aggregation,
            target_ids=target_ids,
            values=values,
        ),
        files,
    )


def _provenance_from_mapping(raw: Any) -> AnnotationProvenance:
    if not isinstance(raw, dict):
        raise ContractError(["missing provenance field: source"])
    missing = [name for name in _PROVENANCE_FIELDS if name not in raw]
    if missing:
        raise ContractError([f"missing provenance field: {name}" for name in missing])
    parameters = raw["parameters"]
    if not isinstance(parameters, dict):
        raise ContractError(["provenance parameters must be a mapping"])
    return AnnotationProvenance(
        source=str(raw["source"]),
        method=str(raw["method"]),
        version=str(raw["version"]),
        parameters=parameters,
        parent_graph_id=str(raw["parent_graph_id"]),
        parent_schema_version=str(raw["parent_schema_version"]),
        parent_contract_version=str(raw["parent_contract_version"]),
    )


def _layer_errors(graph: Cdbg, layer: AnnotationLayer) -> list[str]:
    errors: list[str] = []
    if not _NAME.fullmatch(layer.namespace):
        errors.append(f"annotation namespace {layer.namespace!r} is not a safe token")
    if not _NAME.fullmatch(layer.feature):
        errors.append(f"annotation feature {layer.feature!r} is not a safe token")
    if layer.target_type not in TARGET_TYPES:
        errors.append(f"unknown annotation target_type: {layer.target_type}")
    if layer.dtype not in DTYPES or layer.kind not in KINDS:
        errors.append(f"incompatible annotation dtype: {layer.dtype} kind {layer.kind}")
    if layer.kind == "vector" and layer.dtype == "category":
        errors.append("incompatible annotation dtype: category vectors are not supported")
    if layer.aggregation is not None and layer.aggregation not in AGGREGATION_POLICIES:
        errors.append(f"unknown aggregation policy: {layer.aggregation}")
    errors.extend(_provenance_errors(graph, layer.provenance))
    ids = layer.target_ids
    values = layer.values
    if not isinstance(ids, np.ndarray) or not isinstance(values, np.ndarray):
        errors.append("annotation values must be arrays")
        return errors
    if ids.ndim != 1:
        errors.append("annotation target ids must be a one-dimensional array")
        return errors
    if ids.shape[0] == 0:
        errors.append(f"annotation layer {layer.namespace}:{layer.feature} has no rows")
    if len(set(map(str, ids.tolist()))) != ids.shape[0]:
        errors.append(
            f"duplicate annotation target in {layer.namespace}:{layer.feature}:{layer.target_type}"
        )
    if layer.target_type in TARGET_TYPES:
        known = _known_targets(graph, layer.target_type)
        errors.extend(
            f"missing annotation target: {target_id}"
            for target_id in map(str, ids.tolist())
            if target_id not in known
        )
    if layer.dtype == "category":
        if values.ndim != 1 or values.shape[0] != ids.shape[0]:
            errors.append("incompatible annotation dtype: category values must align with target ids")
        elif any(not isinstance(value, str) or value == "" for value in values.tolist()):
            errors.append("incompatible annotation dtype: category values must be non-empty strings")
    elif layer.kind == "scalar":
        expected = np.float64 if layer.dtype == "float64" else np.int64
        if values.dtype != expected or values.ndim != 1 or values.shape[0] != ids.shape[0]:
            errors.append(f"incompatible annotation dtype: expected {layer.dtype} scalar column")
        elif layer.dtype == "float64" and not np.all(np.isfinite(values)):
            errors.append("missing annotation value: non-finite values are not imputed")
    else:
        expected = np.float64 if layer.dtype == "float64" else np.int64
        if (
            values.dtype != expected
            or values.ndim != 2
            or values.shape[0] != ids.shape[0]
            or values.shape[1] < 1
        ):
            errors.append(f"incompatible annotation dtype: expected {layer.dtype} vector column")
        elif layer.dtype == "float64" and not np.all(np.isfinite(values)):
            errors.append("missing annotation value: non-finite values are not imputed")
    return errors


def _provenance_errors(graph: Cdbg, provenance: AnnotationProvenance) -> list[str]:
    errors: list[str] = []
    if not isinstance(provenance, AnnotationProvenance):
        return ["missing provenance field: source"]
    for name in ("source", "method", "version", "parent_graph_id", "parent_schema_version", "parent_contract_version"):
        value = getattr(provenance, name)
        if not isinstance(value, str) or value == "":
            errors.append(f"missing provenance field: {name}")
    if not isinstance(provenance.parameters, dict):
        errors.append("provenance parameters must be a mapping")
    graph_id = graph.metadata.get("graph_id")
    if not isinstance(graph_id, str) or provenance.parent_graph_id != graph_id:
        errors.append("provenance parent_graph_id does not match the CDBG")
    schema = str(graph.metadata.get("schema_version", ""))
    if provenance.parent_schema_version != schema:
        errors.append("provenance parent_schema_version does not match the CDBG")
    contract = graph.metadata.get("contract_version")
    if not isinstance(contract, str) or provenance.parent_contract_version != contract:
        errors.append("provenance parent_contract_version does not match the CDBG")
    return errors


def _parse_provenance(raw: Mapping[str, Any], cdbg: Cdbg) -> AnnotationProvenance:
    if not isinstance(raw, Mapping):
        raise ContractError(["missing provenance field: source"])
    missing = [name for name in _PROVENANCE_FIELDS if name not in raw]
    if missing:
        raise ContractError([f"missing provenance field: {name}" for name in missing])
    unknown = sorted(set(raw) - set(_PROVENANCE_FIELDS))
    if unknown:
        raise ContractError([f"unknown provenance field: {name}" for name in unknown])
    parameters = raw["parameters"]
    if not isinstance(parameters, dict):
        raise ContractError(["provenance parameters must be a mapping"])
    record = AnnotationProvenance(
        source=_required_text(raw["source"], "source"),
        method=_required_text(raw["method"], "method"),
        version=_required_text(raw["version"], "version"),
        parameters=_plain_parameters(deepcopy(parameters)),
        parent_graph_id=_required_text(raw["parent_graph_id"], "parent_graph_id"),
        parent_schema_version=_required_text(raw["parent_schema_version"], "parent_schema_version"),
        parent_contract_version=_required_text(raw["parent_contract_version"], "parent_contract_version"),
    )
    errors = _provenance_errors(cdbg, record)
    if errors:
        raise ContractError(errors)
    return record


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ContractError([f"missing provenance field: {name}"])
    return value


def _plain_parameters(value: Any, path: str = "parameters") -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError([f"provenance {path} must be finite"])
        return value
    if isinstance(value, dict):
        plain: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError([f"provenance {path} keys must be strings"])
            plain[key] = _plain_parameters(item, f"{path}.{key}")
        return plain
    if isinstance(value, (list, tuple)):
        return [_plain_parameters(item, f"{path}[]") for item in value]
    raise ContractError([f"provenance {path} is not a plain value"])


def _coerce_values(values: Mapping[str, Any], *, dtype: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(values, Mapping) or isinstance(values, (str, bytes)):
        raise ContractError(["annotation values must be a mapping of target_id to value"])
    if len(values) == 0:
        raise ContractError(["annotation values are empty"])
    target_ids: list[str] = []
    parsed: list[Any] = []
    width: int | None = None
    for raw_id, raw_value in values.items():
        if not isinstance(raw_id, str) or raw_id == "" or any(char in raw_id for char in "\t\n\r"):
            raise ContractError([f"missing annotation target: {raw_id!r}"])
        target_ids.append(raw_id)
        if kind == "vector":
            row = _as_vector(raw_value, dtype, raw_id)
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ContractError(["incompatible annotation dtype: vector widths differ"])
            parsed.append(row)
        elif dtype == "float64":
            parsed.append(_as_float(raw_value, raw_id))
        elif dtype == "int64":
            parsed.append(_as_int(raw_value, raw_id))
        else:
            parsed.append(_as_category(raw_value, raw_id))
    ids = np.asarray(target_ids, dtype=object)
    if kind == "vector":
        array = np.asarray(parsed, dtype=np.float64 if dtype == "float64" else np.int64)
    elif dtype == "float64":
        array = np.asarray(parsed, dtype=np.float64)
    elif dtype == "int64":
        array = np.asarray(parsed, dtype=np.int64)
    else:
        array = np.asarray(parsed, dtype=object)
    return ids, array


def _as_float(value: Any, target_id: str) -> float:
    if isinstance(value, (bool, np.bool_)) or isinstance(value, (str, bytes, list, tuple)):
        raise ContractError([f"incompatible annotation dtype for {target_id}"])
    if isinstance(value, np.ndarray):
        raise ContractError([f"incompatible annotation dtype for {target_id}"])
    if isinstance(value, (int, float, np.floating, np.integer)):
        number = float(value)
        if not math.isfinite(number):
            raise ContractError([f"missing annotation value for {target_id}; refusing to impute"])
        return number
    raise ContractError([f"incompatible annotation dtype for {target_id}"])


def _as_int(value: Any, target_id: str) -> int:
    if isinstance(value, (bool, np.bool_)) or isinstance(value, (str, bytes, float, np.floating, list, tuple, np.ndarray)):
        raise ContractError([f"incompatible annotation dtype for {target_id}"])
    if isinstance(value, (int, np.integer)):
        number = int(value)
        if number < -2**63 or number > 2**63 - 1:
            raise ContractError([f"incompatible annotation dtype for {target_id}: value does not fit in int64"])
        return number
    raise ContractError([f"incompatible annotation dtype for {target_id}"])


def _as_category(value: Any, target_id: str) -> str:
    if not isinstance(value, str) or value == "" or any(char in value for char in "\t\n\r|"):
        raise ContractError([f"incompatible annotation dtype for {target_id}"])
    return value


def _as_vector(value: Any, dtype: str, target_id: str) -> list[float] | list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, np.ndarray)):
        raise ContractError([f"incompatible annotation dtype for {target_id}"])
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ContractError([f"incompatible annotation dtype for {target_id}"])
    items = list(value)
    if not items:
        raise ContractError([f"incompatible annotation dtype for {target_id}"])
    if dtype == "float64":
        return [_as_float(item, target_id) for item in items]
    if dtype == "int64":
        return [_as_int(item, target_id) for item in items]
    raise ContractError([f"incompatible annotation dtype: {dtype}"])


def _known_targets(cdbg: Cdbg, target_type: str) -> set[str]:
    if target_type == "node":
        return {unitig.unitig_id for unitig in cdbg.unitigs}
    if target_type == "edge":
        return {link.link_id for link in cdbg.links}
    if target_type == "internal_node":
        return {row.cfa_node_id for row in cdbg.mapping}
    if target_type == "internal_edge":
        return {edge_id for unitig in cdbg.unitigs for edge_id in unitig.internal_edge_ids}
    raise ContractError([f"unknown annotation target_type: {target_type}"])


def _require_layer(cdbg: Cdbg, namespace: str, feature: str, target_type: str) -> AnnotationLayer:
    found = [
        layer
        for layer in cdbg.annotations
        if layer.namespace == namespace and layer.feature == feature and layer.target_type == target_type
    ]
    if len(found) != 1:
        raise ContractError([f"missing annotation layer: {namespace}:{feature}:{target_type}"])
    return found[0]


def _store_layer(cdbg: Cdbg, layer: AnnotationLayer, *, replace: bool) -> None:
    _store_many(cdbg, [layer], replace=replace)


def _store_many(cdbg: Cdbg, layers: Sequence[AnnotationLayer], *, replace: bool) -> None:
    if not isinstance(cdbg.annotations, list):
        raise ContractError(["annotations must be a list"])
    positions = []
    for layer in layers:
        positions.append(_find_index(cdbg, layer.namespace, layer.feature, layer.target_type))
    collisions = [
        f"annotation layer already exists: {layer.namespace}:{layer.feature}:{layer.target_type}; pass replace=True to overwrite"
        for layer, index in zip(layers, positions)
        if index is not None
    ]
    if collisions and not replace:
        raise ContractError(collisions)
    previous = list(cdbg.annotations)
    for layer, index in zip(layers, positions):
        if index is None:
            cdbg.annotations.append(layer)
        else:
            cdbg.annotations[index] = layer
    errors = annotation_errors(cdbg)
    if errors:
        cdbg.annotations[:] = previous
        raise ContractError(errors)


def _find_index(cdbg: Cdbg, namespace: str, feature: str, target_type: str) -> int | None:
    matches = [
        index
        for index, layer in enumerate(cdbg.annotations)
        if layer.namespace == namespace and layer.feature == feature and layer.target_type == target_type
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ContractError([f"annotation layer already exists: {namespace}:{feature}:{target_type}"])
    return matches[0]


def _replace_layer(cdbg: Cdbg, layer: AnnotationLayer) -> None:
    index = _find_index(cdbg, layer.namespace, layer.feature, layer.target_type)
    if index is None:
        raise ContractError(
            [f"missing annotation layer: {layer.namespace}:{layer.feature}:{layer.target_type}"]
        )
    previous = list(cdbg.annotations)
    cdbg.annotations[index] = layer
    errors = annotation_errors(cdbg)
    if errors:
        cdbg.annotations[:] = previous
        raise ContractError(errors)


def _require_name(value: str, label: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ContractError([f"annotation {label} {value!r} is not a safe token"])


def _require_flag(value: bool, label: str) -> None:
    if not isinstance(value, bool):
        raise ContractError([f"{label} must be a bool"])


def _transfer_columns(cfa: Any, table: str, requested: Sequence[str] | None) -> list[str]:
    declared = cfa.metadata.get("features", {}).get(table, {})
    if not isinstance(declared, dict):
        raise ContractError([f"CFA metadata is missing features.{table}"])
    if requested is None:
        skip = {"color_set", "orientation"}
        return [
            name
            for name, kind in declared.items()
            if kind in {"float", "int", "str"} and name not in skip
        ]
    columns = list(requested)
    if len(columns) != len(set(columns)):
        raise ContractError([f"duplicate CFA {table} column in transfer"])
    for name in columns:
        if name not in declared:
            raise ContractError([f"undeclared CFA {table} column: {name}"])
    return columns


def _declared_type(cfa: Any, table: str, column: str) -> str:
    kind = cfa.metadata.get("features", {}).get(table, {}).get(column)
    if kind == "float":
        return "float64"
    if kind == "int":
        return "int64"
    if kind == "str":
        return "category"
    raise ContractError([f"incompatible annotation dtype for CFA column {column}"])


def _column_layer(
    cdbg: Cdbg,
    *,
    rows: Sequence[Mapping[str, str]],
    id_column: str,
    column: str,
    declared: str,
    namespace: str,
    target_type: str,
    provenance: AnnotationProvenance,
    cfa_graph_id: str,
    table: str,
) -> AnnotationLayer:
    raw: dict[str, Any] = {}
    for row in rows:
        target_id = row[id_column]
        cell = row.get(column, "")
        if cell == "":
            raise ContractError([f"missing annotation value for {target_id}; refusing to impute"])
        if declared == "float64":
            raw[target_id] = _parse_float_cell(cell, target_id)
        elif declared == "int64":
            raw[target_id] = _parse_int_cell(cell, target_id)
        else:
            raw[target_id] = _as_category(cell, target_id)
    target_ids, array = _coerce_values(raw, dtype=declared, kind="scalar")
    _require_name(column, "feature")
    parameters = deepcopy(provenance.parameters)
    if "transfer" in parameters:
        raise ContractError(["provenance parameters already contain transfer"])
    parameters["transfer"] = {"cfa_graph_id": cfa_graph_id, "column": column, "cfa_table": table}
    copied = AnnotationProvenance(
        source=provenance.source,
        method=provenance.method,
        version=provenance.version,
        parameters=parameters,
        parent_graph_id=provenance.parent_graph_id,
        parent_schema_version=provenance.parent_schema_version,
        parent_contract_version=provenance.parent_contract_version,
    )
    return AnnotationLayer(
        namespace=namespace,
        feature=column,
        target_type=target_type,
        dtype=declared,
        kind="scalar",
        provenance=copied,
        aggregation=None,
        target_ids=target_ids,
        values=array,
    )


def _parse_float_cell(cell: str, target_id: str) -> float:
    try:
        value = float(cell)
    except ValueError as exc:
        raise ContractError([f"incompatible annotation dtype for {target_id}"]) from exc
    if not math.isfinite(value):
        raise ContractError([f"missing annotation value for {target_id}; refusing to impute"])
    return value


def _parse_int_cell(cell: str, target_id: str) -> int:
    token = cell.strip()
    signed = token[:1] in "+-" and token[1:].isdigit()
    if signed or token.isdigit():
        return int(token)
    raise ContractError([f"incompatible annotation dtype for {target_id}"])


def _groups(cdbg: Cdbg, source_target_type: str) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    for unitig in cdbg.unitigs:
        if source_target_type == "internal_node":
            groups.append((unitig.unitig_id, list(unitig.members)))
            continue
        if unitig.internal_edge_ids:
            groups.append((unitig.unitig_id, list(unitig.internal_edge_ids)))
    return groups


def _value_index(layer: AnnotationLayer) -> dict[str, int]:
    return {str(target_id): index for index, target_id in enumerate(layer.target_ids.tolist())}


def _group_weights(
    cdbg: Cdbg,
    source_target_type: str,
    members: Sequence[str],
    weights: Mapping[str, float] | None,
) -> np.ndarray:
    if weights is None:
        if source_target_type != "internal_node":
            raise ContractError(["weighted_mean of internal_edge annotations requires weights"])
        by_node = {row.cfa_node_id: row.length for row in cdbg.mapping}
        chosen = []
        for member in members:
            if member not in by_node:
                raise ContractError([f"missing weight for {member}"])
            length = by_node[member]
            if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
                raise ContractError([f"missing weight for {member}"])
            chosen.append(float(length))
        return np.asarray(chosen, dtype=np.float64)
    chosen = []
    for member in members:
        if member not in weights:
            raise ContractError([f"missing weight for {member}"])
        raw = weights[member]
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, float, np.integer, np.floating)):
            raise ContractError([f"missing weight for {member}"])
        number = float(raw)
        if not math.isfinite(number) or number <= 0:
            raise ContractError([f"missing weight for {member}"])
        chosen.append(number)
    return np.asarray(chosen, dtype=np.float64)


def _reduce(
    layer: AnnotationLayer,
    positions: Sequence[int],
    policy: str,
    weights: np.ndarray | None,
    unitig_id: str,
) -> tuple[Any, str, str]:
    if policy in NUMERIC_REDUCTIONS and layer.dtype == "category":
        raise ContractError([f"incompatible annotation dtype category for policy {policy}"])
    if policy in {"union", "majority"} and layer.kind == "vector":
        raise ContractError([f"incompatible annotation dtype vector for policy {policy}"])
    if policy in {"union", "majority"} and layer.dtype == "float64":
        raise ContractError([f"incompatible annotation dtype float64 for policy {policy}"])
    index = np.asarray(list(positions), dtype=np.int64)
    if policy == "union":
        return _union(layer, index), "category", "scalar"
    if policy == "majority":
        return _majority(layer, index, unitig_id), layer.dtype, "scalar"
    rows = layer.values[index]
    if policy == "mean":
        return np.mean(rows.astype(np.float64), axis=0), "float64", layer.kind
    if policy == "median":
        return np.median(rows.astype(np.float64), axis=0), "float64", layer.kind
    if policy == "weighted_mean":
        if weights is None:
            raise ContractError(["weighted_mean requires weights"])
        return _weighted_mean(rows, weights), "float64", layer.kind
    if policy == "sum":
        return _sum_values(rows, layer.dtype), layer.dtype, layer.kind
    if policy == "min":
        return np.min(rows, axis=0), layer.dtype, layer.kind
    if policy == "max":
        return np.max(rows, axis=0), layer.dtype, layer.kind
    raise ContractError([f"unknown aggregation policy: {policy}"])


def _weighted_mean(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights, dtype=np.float64))
    if total <= 0:
        raise ContractError(["weighted_mean weights sum to 0"])
    if rows.ndim == 1:
        return np.asarray(np.dot(rows.astype(np.float64), weights) / total, dtype=np.float64)
    return np.asarray(np.dot(weights, rows.astype(np.float64)) / total, dtype=np.float64)


def _sum_values(rows: np.ndarray, dtype: str) -> Any:
    if dtype == "int64":
        if rows.ndim == 1:
            total = sum(int(value) for value in rows.tolist())
            _fit_int(total)
            return np.int64(total)
        columns = []
        for column in range(rows.shape[1]):
            total = sum(int(value) for value in rows[:, column].tolist())
            _fit_int(total)
            columns.append(total)
        return np.asarray(columns, dtype=np.int64)
    return np.sum(rows.astype(np.float64), axis=0)


def _fit_int(value: int) -> None:
    if value < -2**63 or value > 2**63 - 1:
        raise ContractError(["sum does not fit in int64"])


def _union(layer: AnnotationLayer, index: np.ndarray) -> str:
    if layer.dtype == "int64":
        tokens = [str(int(layer.values[position])) for position in index.tolist()]
    else:
        tokens = [str(layer.values[position]) for position in index.tolist()]
    if any("|" in token for token in tokens):
        raise ContractError(["union cannot join a category that already contains '|'"])
    return "|".join(sorted(set(tokens)))


def _majority(layer: AnnotationLayer, index: np.ndarray, unitig_id: str) -> Any:
    if layer.dtype == "int64":
        values = [int(layer.values[position]) for position in index.tolist()]
    else:
        values = [str(layer.values[position]) for position in index.tolist()]
    counts = Counter(values)
    best = max(counts.values())
    winners = sorted((value for value, count in counts.items() if count == best), key=str)
    if len(winners) != 1:
        raise ContractError([f"majority tie on unitig {unitig_id}"])
    if layer.dtype == "int64":
        return np.int64(winners[0])
    return str(winners[0])


def _stack_reduced(values: Sequence[Any], dtype: str, kind: str) -> np.ndarray:
    if dtype == "category":
        return np.asarray([str(value) for value in values], dtype=object)
    np_dtype = np.float64 if dtype == "float64" else np.int64
    if kind == "vector":
        rows = [np.asarray(value, dtype=np_dtype).reshape(-1) for value in values]
        return np.stack(rows)
    flat = [np.asarray(value, dtype=np_dtype).reshape(()).item() for value in values]
    return np.asarray(flat, dtype=np_dtype)


def _with_policy(layer: AnnotationLayer, *, policy: str, provenance: AnnotationProvenance) -> AnnotationLayer:
    return AnnotationLayer(
        namespace=layer.namespace,
        feature=layer.feature,
        target_type=layer.target_type,
        dtype=layer.dtype,
        kind=layer.kind,
        provenance=provenance,
        aggregation=policy,
        target_ids=np.array(layer.target_ids, copy=True),
        values=np.array(layer.values, copy=True),
    )


def _with_aggregation_parameters(
    provenance: AnnotationProvenance,
    policy: str,
    layer: AnnotationLayer,
    *,
    weights: Mapping[str, float] | None,
    used: dict[str, float] | None,
) -> AnnotationProvenance:
    parameters = deepcopy(provenance.parameters)
    payload: dict[str, Any] = {
        "policy": policy,
        "source_namespace": layer.namespace,
        "source_feature": layer.feature,
        "source_target_type": layer.target_type,
    }
    if policy == "weighted_mean":
        payload["weight_source"] = "caller" if weights is not None else "member_sequence_length"
        payload["weights"] = {key: used[key] for key in sorted(used or {})}
    parameters["aggregation"] = payload
    return AnnotationProvenance(
        source=provenance.source,
        method=provenance.method,
        version=provenance.version,
        parameters=parameters,
        parent_graph_id=provenance.parent_graph_id,
        parent_schema_version=provenance.parent_schema_version,
        parent_contract_version=provenance.parent_contract_version,
    )


def _layer_for_features(cdbg: Cdbg, namespace: str, feature: str, target_type: str) -> AnnotationLayer:
    matches = [layer for layer in cdbg.annotations if layer.namespace == namespace and layer.feature == feature]
    exact = [layer for layer in matches if layer.target_type == target_type]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ContractError([f"annotation layer already exists: {namespace}:{feature}:{target_type}"])
    found = sorted({layer.target_type for layer in matches})
    if target_type == "edge" and "internal_edge" in found:
        raise ContractError(
            [
                f"annotation {namespace}:{feature} is an internal unitig edge annotation; "
                "internal unitig edges are not CSR edges"
            ]
        )
    if found:
        raise ContractError(
            [f"annotation {namespace}:{feature} has target_type {found[0]}, not {target_type}"]
        )
    raise ContractError([f"missing annotation layer: {namespace}:{feature}:{target_type}"])


def _parse_specs(specs: Sequence[tuple[str, str]] | None, argument_name: str) -> list[tuple[str, str]]:
    if specs is None:
        return []
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise ContractError([f"{argument_name} entries must be (namespace, feature) pairs"])
    parsed: list[tuple[str, str]] = []
    for item in specs:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(part, str) and part != "" for part in item)
        ):
            raise ContractError([f"{argument_name} entries must be (namespace, feature) pairs"])
        parsed.append((item[0], item[1]))
    if len(parsed) != len(set(parsed)):
        raise ContractError([f"duplicate {argument_name} selection"])
    return parsed


def _stem(namespace: str, feature: str, target_type: str) -> str:
    _require_name(namespace, "namespace")
    _require_name(feature, "feature")
    if target_type not in TARGET_TYPES:
        raise ContractError([f"unknown annotation target_type: {target_type}"])
    return f"{namespace}__{feature}__{target_type}"


def _provenance_mapping(provenance: AnnotationProvenance) -> dict[str, Any]:
    return {
        "source": provenance.source,
        "method": provenance.method,
        "version": provenance.version,
        "parameters": deepcopy(provenance.parameters),
        "parent_graph_id": provenance.parent_graph_id,
        "parent_schema_version": provenance.parent_schema_version,
        "parent_contract_version": provenance.parent_contract_version,
    }


# CFA edge-table order helper used by transfer. Kept as a function so the
# call site stays a plain expression.
def _ordered_edge_ids(rows: Mapping[str, Mapping[str, str]], id_set: set[str]) -> list[str]:
    return [edge_id for edge_id in rows if edge_id in id_set]
