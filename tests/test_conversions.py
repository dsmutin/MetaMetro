"""Round-trip, identity, and determinism checks across the three formats."""

from __future__ import annotations

import pytest

from metametro.converters.cdbg_to_cfa import cdbg_to_cfa
from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
from metametro.fixtures import chain_cfa, mock_cfa, mock_cgt
from metametro.formats.cfa.validator import parse_color_set

pytestmark = pytest.mark.mandatory


def _edge_keys(graph) -> set[tuple[str, str, str]]:
    return {(row["edge_id"], row["source"], row["target"]) for row in graph.edges}


def _colors(row: dict[str, str]) -> set[int]:
    return set(parse_color_set(row.get("color_set", "")))


def test_bubble_round_trip_preserves_biology() -> None:
    """CFA to CDBG to CFA keeps sequences, edges, and colours on the bubble."""
    original = mock_cfa()
    restored = cdbg_to_cfa(cfa_to_cdbg(original))
    assert restored.sequences == original.sequences
    assert _edge_keys(restored) == _edge_keys(original)
    original_nodes = {row["node_id"]: _colors(row) for row in original.nodes}
    restored_nodes = {row["node_id"]: _colors(row) for row in restored.nodes}
    assert restored_nodes == original_nodes


def test_chain_round_trip_and_no_silent_loss() -> None:
    """Compaction changes node count and still restores every node and edge."""
    original = chain_cfa()
    cdbg = cfa_to_cdbg(original)
    assert len(cdbg.unitigs) == 3
    edge_ids = {row["edge_id"] for row in original.edges}
    kept = {link.link_id for link in cdbg.links}
    kept.update(edge_id for unitig in cdbg.unitigs for edge_id in unitig.internal_edge_ids)
    assert kept == edge_ids
    restored = cdbg_to_cfa(cdbg)
    assert restored.sequences == original.sequences
    assert _edge_keys(restored) == _edge_keys(original)


def test_identity_chain_through_cgt() -> None:
    """Every dense id points at one unitig and that unitig's CFA nodes."""
    cfa = mock_cfa()
    cdbg = cfa_to_cdbg(cfa)
    cgt = mock_cgt()
    member_of = {row.cfa_node_id: row.unitig_id for row in cdbg.mapping}
    seen = []
    for row in cgt.mapping:
        assert row["source_id"] in {unitig.unitig_id for unitig in cdbg.unitigs}
        for node_id in row["cfa_node_ids"]:
            assert member_of[node_id] == row["source_id"]
            seen.append(node_id)
    assert set(seen) == set(cfa.node_ids())


def test_colour_union_on_compacted_unitig() -> None:
    """Merged nodes keep every sample colour on the unitig."""
    cdbg = cfa_to_cdbg(chain_cfa())
    assert set(cdbg.unitigs[0].color_ids) == {0, 1}
    per_node = {row.cfa_node_id: set(row.color_ids) for row in cdbg.mapping}
    assert per_node["n000001"] == {0}
    assert per_node["n000002"] == {0, 1}


def test_conversion_is_deterministic() -> None:
    """The same CFA and parameters produce the same unitigs and dense ids."""
    left = cfa_to_cdbg(mock_cfa())
    right = cfa_to_cdbg(mock_cfa())
    assert [(u.unitig_id, u.sequence, u.members) for u in left.unitigs] == [
        (u.unitig_id, u.sequence, u.members) for u in right.unitigs
    ]
    assert [row["source_id"] for row in mock_cgt().mapping] == [
        row["source_id"] for row in mock_cgt().mapping
    ]
