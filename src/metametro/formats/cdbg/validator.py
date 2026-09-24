"""CDBG invariants."""

from __future__ import annotations

from metametro.errors import ContractError
from metametro.formats.cdbg.annotations import annotation_errors
from metametro.formats.cdbg.model import SCHEMA_VERSION, Cdbg
from metametro.formats.cdbg.sequence import junction_overlaps, split_unitig
from metametro.formats.cfa.model import ALPHABET, ORIENTATIONS


def _color_dictionary(graph: Cdbg) -> set[int] | None:
    if graph.colors is None:
        return None
    found: set[int] = set()
    for row in graph.colors:
        token = str(row.get("color_id", "")).strip()
        if not token.lstrip("-").isdigit():
            raise ContractError([f"malformed color_id: {token!r}"])
        found.add(int(token))
    if len(found) != len(graph.colors):
        raise ContractError(["duplicate color_id"])
    return found


def validate_cdbg(graph: Cdbg) -> None:
    """Reject CDBG objects that break schema 1.0 invariants."""
    errors: list[str] = []
    version = str(graph.metadata.get("schema_version", ""))
    if version != SCHEMA_VERSION:
        errors.append(
            f"incompatible schema version: {version!r} (supported CDBG schema is {SCHEMA_VERSION})"
        )
    graph_type = str(graph.metadata.get("graph_type", ""))
    if not graph_type:
        errors.append("missing required field: graph_type")
    k_ok = isinstance(graph.k, int) and not isinstance(graph.k, bool) and graph.k > 0
    if graph_type == "de_bruijn" and not k_ok:
        errors.append("k must be an integer > 0")
    elif graph.k is not None and not k_ok:
        errors.append("k must be an integer > 0 when present")
    unitig_ids: set[str] = set()
    member_owner: dict[str, str] = {}
    for unitig in graph.unitigs:
        if unitig.unitig_id in unitig_ids:
            errors.append(f"duplicate unitig_id: {unitig.unitig_id}")
        unitig_ids.add(unitig.unitig_id)
        if unitig.sequence == "" or any(base not in ALPHABET for base in unitig.sequence):
            errors.append(f"malformed sequence for {unitig.unitig_id}")
        elif graph_type == "de_bruijn" and k_ok and len(unitig.sequence) < graph.k:
            errors.append(f"unitig {unitig.unitig_id} is shorter than k")
        junctions = max(0, len(unitig.members) - 1)
        if unitig.internal_overlaps and len(unitig.internal_overlaps) != junctions:
            errors.append(f"unitig {unitig.unitig_id} overlaps do not match its CFA members")
        if len(unitig.internal_edge_ids) != junctions:
            errors.append(
                f"unitig {unitig.unitig_id} internal edges do not match its CFA members"
            )
        if len(unitig.internal_edge_colors) != len(unitig.internal_edge_ids):
            errors.append(
                f"unitig {unitig.unitig_id} edge colours do not match its internal edges"
            )
        if graph.metadata.get("compaction") == "identity" and len(unitig.members) != 1:
            errors.append(f"identity unitig {unitig.unitig_id} has more than one CFA node")
        if not unitig.members:
            errors.append(f"unitig {unitig.unitig_id} has no CFA members")
        for node_id in unitig.members:
            if node_id in member_owner:
                errors.append(f"CFA node {node_id} is mapped to more than one unitig")
            member_owner[node_id] = unitig.unitig_id
    known_colors = _color_dictionary(graph)
    if known_colors is not None:
        for unitig in graph.unitigs:
            for color_id in unitig.color_ids:
                if color_id not in known_colors:
                    errors.append(f"undefined color {color_id} on {unitig.unitig_id}")
        for link in graph.links:
            for color_id in link.color_ids:
                if color_id not in known_colors:
                    errors.append(f"undefined color {color_id} on link {link.link_id}")
        for row in graph.mapping:
            for color_id in row.color_ids:
                if color_id not in known_colors:
                    errors.append(f"undefined color {color_id} on {row.cfa_node_id}")
    else:
        used = [color_id for unitig in graph.unitigs for color_id in unitig.color_ids]
        used.extend(color_id for link in graph.links for color_id in link.color_ids)
        used.extend(color_id for row in graph.mapping for color_id in row.color_ids)
        if used:
            errors.append("colour ids are set but the colour dictionary is missing")
    link_ids: set[str] = set()
    for link in graph.links:
        if link.link_id in link_ids:
            errors.append(f"duplicate link_id: {link.link_id}")
        link_ids.add(link.link_id)
        if link.source not in unitig_ids or link.target not in unitig_ids:
            errors.append(f"dangling link {link.link_id}: {link.source} -> {link.target}")
        if link.orientation is not None and link.orientation not in ORIENTATIONS:
            errors.append(f"link {link.link_id} has orientation {link.orientation!r}")
    seen_edges = set(link_ids)
    for unitig in graph.unitigs:
        for edge_id in unitig.internal_edge_ids:
            if edge_id == "":
                errors.append(f"unitig {unitig.unitig_id} has an empty internal edge id")
                continue
            if edge_id in seen_edges:
                errors.append(f"duplicate edge id {edge_id}")
            seen_edges.add(edge_id)
    by_node = {}
    for row in graph.mapping:
        if row.cfa_node_id in by_node:
            errors.append(f"duplicate mapping row for {row.cfa_node_id}")
            continue
        by_node[row.cfa_node_id] = row
    if set(by_node) != set(member_owner):
        errors.append("invalid mapping: mapping rows do not match unitig membership")
    for row in graph.mapping:
        if row.cfa_node_id not in by_node:
            continue
        if row.unitig_id != member_owner.get(row.cfa_node_id):
            errors.append(f"invalid mapping for {row.cfa_node_id}")
        if row.length <= 0:
            errors.append(f"invalid mapping length for {row.cfa_node_id}")
    for unitig in graph.unitigs:
        member_colors: set[int] = set()
        complete = True
        for node_id in unitig.members:
            row = by_node.get(node_id)
            if row is None:
                complete = False
                break
            member_colors.update(row.color_ids)
        if complete and set(unitig.color_ids) != member_colors:
            errors.append(f"unitig {unitig.unitig_id} colour is not the union of its CFA members")
        _check_path(graph, unitig, by_node, errors)
    errors.extend(annotation_errors(graph))
    if errors:
        raise ContractError(errors)


def _check_path(graph: Cdbg, unitig, by_node: dict, errors: list[str]) -> None:
    """Require mapping ordinals to follow ``unitig.members`` and tile the sequence."""
    rows = []
    aligned = True
    for ordinal, node_id in enumerate(unitig.members):
        row = by_node.get(node_id)
        if row is None or row.unitig_id != unitig.unitig_id:
            aligned = False
            continue
        if row.ordinal != ordinal:
            errors.append(
                f"mapping ordinal for {node_id} is not its position in {unitig.unitig_id}"
            )
            aligned = False
        if row.length <= 0:
            aligned = False
        rows.append(row)
    if not aligned or len(rows) != len(unitig.members):
        return
    if graph.metadata.get("compaction") == "identity":
        if rows[0].length != len(unitig.sequence):
            errors.append(f"identity unitig {unitig.unitig_id} has an inconsistent mapping")
        return
    if str(graph.metadata.get("graph_type", "")) == "de_bruijn" and not (
        isinstance(graph.k, int) and not isinstance(graph.k, bool) and graph.k > 0
    ):
        return
    k = graph.k if isinstance(graph.k, int) and not isinstance(graph.k, bool) and graph.k > 0 else None
    try:
        parts = split_unitig(
            unitig.sequence,
            [row.length for row in rows],
            junction_overlaps(unitig, k),
        )
    except ContractError as exc:
        errors.extend(exc.errors)
        return
    for row, part in zip(rows, parts):
        if len(part) != row.length:
            errors.append(f"restored length mismatch for {row.cfa_node_id}")
