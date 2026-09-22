"""Mandatory CDBG checks, including the compaction mock."""

from __future__ import annotations

import pytest

from metametro.errors import ContractError
from metametro.fixtures import chain_cdbg, chain_cfa, mock_cdbg
from metametro.formats.cdbg.io import dump_cdbg, load_cdbg
from metametro.formats.cdbg.validator import validate_cdbg

pytestmark = pytest.mark.mandatory


def test_bubble_cdbg_keeps_every_node(tmp_path) -> None:
    """Junctions are not merged, and every CFA node is mapped once."""
    graph = mock_cdbg()
    validate_cdbg(graph)
    assert len(graph.unitigs) == 4
    assert len(graph.links) == 5
    assert {row.cfa_node_id for row in graph.mapping} == {
        "n000001",
        "n000002",
        "n000003",
        "n000004",
    }
    dump_cdbg(graph, tmp_path / "cdbg")
    loaded = load_cdbg(tmp_path / "cdbg")
    assert [unitig.unitig_id for unitig in loaded.unitigs] == [unitig.unitig_id for unitig in graph.unitigs]


def test_chain_compacts_to_three_unitigs() -> None:
    """The documented chain becomes 3 unitigs and 4 links."""
    graph = chain_cdbg()
    members = [tuple(unitig.members) for unitig in graph.unitigs]
    assert members == [
        ("n000001", "n000002"),
        ("n000003", "n000004", "n000005"),
        ("n000006",),
    ]
    assert len(graph.links) == 4
    assert graph.unitigs[0].sequence == "CCGTAC"
    assert graph.unitigs[1].sequence == "ACGATGAC"
    assert set(graph.unitigs[0].color_ids) == {0, 1}


def test_short_unitig_rejected() -> None:
    """A unitig shorter than k is rejected."""
    graph = chain_cdbg()
    graph.unitigs[0].sequence = "AC"
    with pytest.raises(ContractError, match="shorter than k"):
        validate_cdbg(graph)


def test_dangling_link_rejected() -> None:
    """A link into a missing unitig is rejected."""
    graph = mock_cdbg()
    graph.links[0].target = "u999999"
    with pytest.raises(ContractError, match="dangling link"):
        validate_cdbg(graph)


def test_overlap_mismatch_is_not_silent() -> None:
    """A de Bruijn edge that does not overlap is refused."""
    from metametro.converters.cfa_to_cdbg import cfa_to_cdbg

    graph = chain_cfa()
    graph.sequences["n000002"] = "AAAA"
    with pytest.raises(ContractError, match="overlap mismatch"):
        cfa_to_cdbg(graph)
