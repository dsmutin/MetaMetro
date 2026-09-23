"""Recover CFA ids from a CDBG and a CGT.

CFA node and edge ids are the semantic ids. A CDBG unitig id is only a
compacted path name. A CGT dense id is only a row number. This module reads
those links. It does not invent a genome, a sample, or any other biological
name for a unitig.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

from metametro.converters.cdbg_to_cgt import csr_link_order
from metametro.errors import ContractError
from metametro.formats.cdbg.model import Cdbg
from metametro.formats.cdbg.validator import validate_cdbg
from metametro.formats.cgt.model import Cgt
from metametro.formats.cgt.validator import validate_cgt


@dataclass(frozen=True)
class NodeLineage:
    """One CGT node row and the ids it already stores.

    ``source_id`` is the CDBG unitig id. ``cfa_node_ids`` is that unitig's
    CFA path, in member order.
    """

    dense_id: int
    source_id: str
    cfa_node_ids: tuple[str, ...]


def _dense_id(dense_id: int) -> int:
    if isinstance(dense_id, bool):
        raise ContractError(["dense_id must be an integer"])
    try:
        return operator.index(dense_id)
    except TypeError as exc:
        raise ContractError(["dense_id must be an integer"]) from exc


def node_lineage(cgt: Cgt, dense_id: int) -> NodeLineage:
    """Return the CDBG unitig id and CFA node ids for one CGT dense id."""
    validate_cgt(cgt)
    index = _dense_id(dense_id)
    if index < 0 or index >= cgt.num_nodes:
        raise ContractError([f"dense_id {index} is outside 0 .. {cgt.num_nodes - 1}"])
    row = cgt.mapping[index]
    return NodeLineage(
        dense_id=index,
        source_id=str(row["source_id"]),
        cfa_node_ids=tuple(row["cfa_node_ids"]),
    )


def assert_cgt_matches_cdbg(cgt: Cgt, cdbg: Cdbg) -> None:
    """Raise ``ContractError`` when a CGT row disagrees with its CDBG unitig.

    Every dense id must name a unitig, and that unitig's CFA member list must
    equal the CGT row's ``cfa_node_ids`` in the same order.
    """
    validate_cgt(cgt)
    validate_cdbg(cdbg)
    by_id = {unitig.unitig_id: unitig for unitig in cdbg.unitigs}
    if cgt.num_nodes != len(by_id):
        raise ContractError(["CGT node count does not match the CDBG"])
    for index in range(cgt.num_nodes):
        lineage = node_lineage(cgt, index)
        unitig = by_id.get(lineage.source_id)
        if unitig is None:
            raise ContractError([f"CGT source_id {lineage.source_id} is not a CDBG unitig"])
        if tuple(unitig.members) != lineage.cfa_node_ids:
            raise ContractError(
                [f"CGT CFA nodes for {lineage.source_id} do not match the CDBG path"]
            )


def link_cfa_edge_id(cdbg: Cdbg, link_id: str) -> str:
    """Return the CFA edge id stored on a CDBG link.

    Schema 1.0 keeps that id in ``link_id``. An unknown link is an error.
    """
    validate_cdbg(cdbg)
    found = [link.link_id for link in cdbg.links if link.link_id == link_id]
    if len(found) != 1:
        raise ContractError([f"unknown CDBG link: {link_id}"])
    return found[0]


def internal_cfa_edge_ids(cdbg: Cdbg, unitig_id: str) -> tuple[str, ...]:
    """Return CFA edge ids absorbed into one unitig, in path order."""
    validate_cdbg(cdbg)
    found = [unitig for unitig in cdbg.unitigs if unitig.unitig_id == unitig_id]
    if len(found) != 1:
        raise ContractError([f"unknown CDBG unitig: {unitig_id}"])
    return tuple(found[0].internal_edge_ids)


def resolve_sequence(cgt: Cgt, source_cdbg: Cdbg, dense_id: int) -> str:
    """Return the CDBG unitig sequence for one CGT dense id.

    CGT rows do not store DNA. The sequence is read from ``source_cdbg``
    after ``dense_id`` is mapped to a unitig id. A dense id outside
    ``0 .. N-1``, a unitig id that is not on the CDBG, an empty sequence, or
    any other CGT/CDBG membership mismatch raises ``ContractError``. No
    sequence is invented.
    """
    lineage = node_lineage(cgt, dense_id)
    cgt_graph = cgt.metadata.get("source", {}).get("graph_id") if isinstance(cgt.metadata.get("source"), dict) else None
    cdbg_graph = source_cdbg.metadata.get("graph_id")
    if (
        isinstance(cgt_graph, str)
        and cgt_graph != ""
        and isinstance(cdbg_graph, str)
        and cdbg_graph != ""
        and cgt_graph != cdbg_graph
    ):
        raise ContractError(
            [f"CGT source graph_id {cgt_graph} does not match CDBG graph_id {cdbg_graph}"]
        )
    assert_cgt_matches_cdbg(cgt, source_cdbg)
    found = [unitig for unitig in source_cdbg.unitigs if unitig.unitig_id == lineage.source_id]
    if len(found) != 1:
        raise ContractError([f"missing unitig {lineage.source_id}"])
    sequence = found[0].sequence
    if not isinstance(sequence, str) or sequence == "":
        raise ContractError([f"unitig {lineage.source_id} has no sequence"])
    return sequence


def cgt_edge_cfa_ids(cdbg: Cdbg) -> tuple[str, ...]:
    """Return the CFA edge id of each CSR edge in ``cdbg_to_cgt`` order.

    Each id is the CDBG link id. Edges absorbed into a unitig are not CSR
    edges; use ``internal_cfa_edge_ids`` for those.
    """
    validate_cdbg(cdbg)
    order = csr_link_order(cdbg)
    return tuple(cdbg.links[index].link_id for index in order)
