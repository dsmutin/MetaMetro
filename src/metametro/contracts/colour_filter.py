"""Filter colours on a CFA, a CDBG (ToCUMG), or a CGT.

The filter is not a new graph type and it is not one of the seven pipeline
stages. It returns a new graph of the same kind. Sequences, topology, feature
matrices, and labels stay as they were. Only colour ids, and the CGT columns
that belong to those ids, change.

A unitig colour on a CDBG remains the union of its CFA members after the
filter. On a CGT, ``node_colors``, ``edge_colors``, and the optional float32
colour weights lose a column together. Surviving columns stay in increasing
``color_id`` order.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import numpy as np

from metametro.errors import ContractError
from metametro.formats.cdbg.model import Cdbg
from metametro.formats.cdbg.validator import validate_cdbg
from metametro.formats.cfa.model import CfaGraph
from metametro.formats.cfa.validator import parse_color_set, validate_cfa
from metametro.formats.cgt.model import Cgt
from metametro.formats.cgt.validator import validate_cgt


def filter_colours(
    graph: CfaGraph | Cdbg | Cgt,
    *,
    color_ids: Sequence[int] | None = None,
    namespace: str | None = None,
    namespaces: Sequence[str] | None = None,
) -> CfaGraph | Cdbg | Cgt:
    """Return a copy of ``graph`` that keeps only the selected colours.

    Pass ``color_ids``, ``namespace``, ``namespaces``, or a combination.
    ``namespace`` and ``namespaces`` together are rejected. Several
    namespaces are a union. ``color_ids`` with a namespace is the
    intersection. A CGT has no namespace dictionary; pass ``color_ids``
    there. The input graph is not modified.
    """
    if namespace is not None and namespaces is not None:
        raise ContractError(["pass namespace or namespaces, not both"])
    if namespaces is not None:
        if isinstance(namespaces, (str, bytes)) or not isinstance(namespaces, Sequence) or len(namespaces) == 0:
            raise ContractError(["namespaces must be a non-empty sequence of strings"])
        keep: set[int] = set()
        for name in namespaces:
            keep |= _keep_ids(graph, color_ids=None, namespace=name)
        if color_ids is not None:
            requested = set(_id_list(color_ids))
            missing = [color_id for color_id in requested if color_id not in keep]
            if missing:
                raise ContractError([f"color_id {color_id} is not in the requested namespaces" for color_id in sorted(missing)])
            keep &= requested
    else:
        keep = _keep_ids(graph, color_ids=color_ids, namespace=namespace)
    note = namespace if namespaces is None else ",".join(str(name) for name in namespaces)
    if isinstance(graph, CfaGraph):
        return _filter_cfa(graph, keep, note)
    if isinstance(graph, Cdbg):
        return _filter_cdbg(graph, keep, note)
    if isinstance(graph, Cgt):
        return _filter_cgt(graph, keep, note)
    raise ContractError(["colour filter accepts a CFA, a CDBG, or a CGT"])


def _keep_ids(
    graph: CfaGraph | Cdbg | Cgt,
    *,
    color_ids: Sequence[int] | None,
    namespace: str | None,
) -> set[int]:
    if color_ids is None and namespace is None:
        raise ContractError(["colour filter requires color_ids or namespace"])
    if namespace is not None and (not isinstance(namespace, str) or namespace == ""):
        raise ContractError(["colour filter namespace must be a non-empty string"])
    requested = None if color_ids is None else _id_list(color_ids)
    if isinstance(graph, Cgt):
        if namespace is not None:
            raise ContractError(["a CGT has no colour namespace; filter by color_ids"])
        known = set(graph.color_ids)
        missing = [color_id for color_id in requested or [] if color_id not in known]
        if missing:
            raise ContractError([f"unknown color_id: {color_id}" for color_id in missing])
        return set(requested or [])
    rows = graph.colors
    if namespace is not None and rows is None:
        raise ContractError(["colour filter namespace requires a colour dictionary"])
    dictionary = _dictionary(rows)
    if namespace is not None:
        in_namespace = {color_id for color_id, row_namespace in dictionary.items() if row_namespace == namespace}
    else:
        in_namespace = set(dictionary)
    if requested is None:
        return in_namespace
    if rows is not None:
        missing = [color_id for color_id in requested if color_id not in dictionary]
        if missing:
            raise ContractError([f"unknown color_id: {color_id}" for color_id in missing])
        outside = [color_id for color_id in requested if color_id not in in_namespace]
        if outside:
            raise ContractError(
                [f"color_id {color_id} is not in namespace {namespace}" for color_id in outside]
            )
    return set(requested) & in_namespace if rows is not None else set(requested)


def _id_list(color_ids: Sequence[int]) -> list[int]:
    if isinstance(color_ids, (str, bytes)) or not isinstance(color_ids, Sequence):
        raise ContractError(["color_ids must be a sequence of integers"])
    found: list[int] = []
    for item in color_ids:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ContractError(["color_ids must be a sequence of integers"])
        found.append(int(item))
    return found


def _dictionary(rows: list[dict[str, str]] | None) -> dict[int, str]:
    if rows is None:
        return {}
    found: dict[int, str] = {}
    for row in rows:
        token = str(row.get("color_id", "")).strip()
        if not token.lstrip("-").isdigit():
            raise ContractError([f"malformed color_id: {token!r}"])
        found[int(token)] = str(row.get("namespace", ""))
    return found


def _select(color_ids: Sequence[int], keep: set[int]) -> list[int]:
    return sorted(color_id for color_id in color_ids if color_id in keep)


def _note(metadata: dict, keep: set[int], namespace: str | None) -> None:
    metadata["colour_filter"] = {"color_ids": sorted(keep), "namespace": namespace}


def _filter_cfa(graph: CfaGraph, keep: set[int], namespace: str | None) -> CfaGraph:
    copied = deepcopy(graph)
    for row in copied.nodes:
        _rewrite_cell(row, keep)
    for row in copied.edges:
        _rewrite_cell(row, keep)
    if copied.colors is not None:
        copied.colors = [row for row in copied.colors if int(str(row["color_id"]).strip()) in keep]
    _note(copied.metadata, keep, namespace)
    validate_cfa(copied)
    return copied


def _rewrite_cell(row: dict[str, str], keep: set[int]) -> None:
    for column in ("color_set", "colors"):
        if column not in row:
            continue
        row[column] = ",".join(str(color_id) for color_id in _select(parse_color_set(row[column]), keep))


def _filter_cdbg(graph: Cdbg, keep: set[int], namespace: str | None) -> Cdbg:
    copied = deepcopy(graph)
    for row in copied.mapping:
        row.color_ids = _select(row.color_ids, keep)
    by_node = {row.cfa_node_id: row.color_ids for row in copied.mapping}
    for unitig in copied.unitigs:
        members: set[int] = set()
        for node_id in unitig.members:
            members.update(by_node.get(node_id, []))
        unitig.color_ids = sorted(members)
        unitig.internal_edge_colors = [_select(group, keep) for group in unitig.internal_edge_colors]
    for link in copied.links:
        link.color_ids = _select(link.color_ids, keep)
    if copied.colors is not None:
        copied.colors = [row for row in copied.colors if int(str(row["color_id"]).strip()) in keep]
    _note(copied.metadata, keep, namespace)
    validate_cdbg(copied)
    return copied


def _filter_cgt(graph: Cgt, keep: set[int], namespace: str | None) -> Cgt:
    copied = deepcopy(graph)
    surviving = [color_id for color_id in copied.color_ids if color_id in keep]
    columns = [copied.color_ids.index(color_id) for color_id in surviving]
    copied.node_colors = _take_columns(copied.node_colors, columns)
    copied.edge_colors = _take_columns(copied.edge_colors, columns)
    if copied.node_color_weights is not None:
        copied.node_color_weights = _take_columns(copied.node_color_weights, columns)
    if copied.edge_color_weights is not None:
        copied.edge_color_weights = _take_columns(copied.edge_color_weights, columns)
    copied.color_ids = surviving
    _note(copied.metadata, keep, namespace)
    validate_cgt(copied)
    return copied


def _take_columns(array: np.ndarray, columns: list[int]) -> np.ndarray:
    if not columns:
        return np.zeros((array.shape[0], 0), dtype=array.dtype)
    return np.array(array[:, columns], copy=True)
