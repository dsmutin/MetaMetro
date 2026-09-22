"""CDBG invariants."""

from __future__ import annotations

from metametro.errors import ContractError
from metametro.formats.cdbg.model import SCHEMA_VERSION, Cdbg
from metametro.formats.cfa.model import ALPHABET


def _color_dictionary(graph: Cdbg) -> set[int] | None:
    if graph.colors is None:
        return None
    found: set[int] = set()
    for row in graph.colors:
        token = str(row.get("color_id", "")).strip()
        if not token.lstrip("-").isdigit():
            raise ContractError([f"malformed color_id: {token!r}"])
        found.add(int(token))
    return found


def validate_cdbg(graph: Cdbg) -> None:
    """Reject CDBG objects that break schema 1.0 invariants."""
    errors: list[str] = []
    version = str(graph.metadata.get("schema_version", ""))
    if version != SCHEMA_VERSION:
        errors.append(
            f"incompatible schema version: {version!r} (supported CDBG schema is {SCHEMA_VERSION})"
        )
    if not isinstance(graph.k, int) or isinstance(graph.k, bool) or graph.k <= 0:
        errors.append("k must be an integer > 0")
    unitig_ids: set[str] = set()
    member_owner: dict[str, str] = {}
    for unitig in graph.unitigs:
        if unitig.unitig_id in unitig_ids:
            errors.append(f"duplicate unitig_id: {unitig.unitig_id}")
        unitig_ids.add(unitig.unitig_id)
        if unitig.sequence == "" or any(base not in ALPHABET for base in unitig.sequence):
            errors.append(f"malformed sequence for {unitig.unitig_id}")
        elif isinstance(graph.k, int) and graph.k > 0 and len(unitig.sequence) < graph.k:
            errors.append(f"unitig {unitig.unitig_id} is shorter than k")
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
    link_ids: set[str] = set()
    for link in graph.links:
        if link.link_id in link_ids:
            errors.append(f"duplicate link_id: {link.link_id}")
        link_ids.add(link.link_id)
        if link.source not in unitig_ids or link.target not in unitig_ids:
            errors.append(f"dangling link {link.link_id}: {link.source} -> {link.target}")
    mapped = {row.cfa_node_id: row for row in graph.mapping}
    if set(mapped) != set(member_owner):
        errors.append("invalid mapping: mapping rows do not match unitig membership")
    for row in graph.mapping:
        if row.unitig_id != member_owner.get(row.cfa_node_id):
            errors.append(f"invalid mapping for {row.cfa_node_id}")
        if row.length <= 0:
            errors.append(f"invalid mapping length for {row.cfa_node_id}")
    if errors:
        raise ContractError(errors)
