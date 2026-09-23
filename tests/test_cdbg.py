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


def _plain_cfa(graph_type: str, sequences: dict[str, str], edges: list[dict[str, str]]):
    from metametro.formats.cfa.model import CfaGraph

    edge_features: dict[str, str] = {}
    header = ["edge_id", "source", "target"]
    if edges and "orientation" in edges[0]:
        edge_features["orientation"] = "orientation"
        header.append("orientation")
    if edges and "overlap" in edges[0]:
        edge_features["overlap"] = "int"
        header.append("overlap")
    return CfaGraph(
        metadata={
            "schema_version": "1.0",
            "graph_id": "plain",
            "graph_type": graph_type,
            "features": {"node": {}, "edge": edge_features},
        },
        sequences=sequences,
        nodes=[{"node_id": node_id} for node_id in sequences],
        edges=edges,
        node_header=["node_id"],
        edge_header=header,
    )


def test_non_de_bruijn_type_is_preserved_without_k() -> None:
    """An LCA graph with no overlap stays an LCA graph and does not gain k."""
    from metametro.converters.cdbg_to_cfa import cdbg_to_cfa
    from metametro.converters.cfa_to_cdbg import cfa_to_cdbg

    graph = _plain_cfa(
        "lca",
        {"a": "AAAA", "b": "CCCC"},
        [{"edge_id": "e1", "source": "a", "target": "b", "orientation": "++"}],
    )
    compacted = cfa_to_cdbg(graph)
    assert compacted.metadata["graph_type"] == "lca"
    assert compacted.metadata["compaction"] == "identity"
    assert compacted.k is None
    assert "k" not in compacted.metadata
    assert len(compacted.unitigs) == 2
    restored = cdbg_to_cfa(compacted)
    assert restored.metadata["graph_type"] == "lca"
    assert "k" not in restored.metadata
    assert restored.sequences == graph.sequences


def test_repeat_chain_compacts_with_edge_overlap() -> None:
    """A repeat graph merges on its own overlap and stays a repeat graph."""
    from metametro.converters.cdbg_to_cfa import cdbg_to_cfa
    from metametro.converters.cfa_to_cdbg import cfa_to_cdbg

    graph = _plain_cfa(
        "repeat",
        {"n1": "ACGTAA", "n2": "TAACCC", "n3": "CCCGGG"},
        [
            {"edge_id": "e1", "source": "n1", "target": "n2", "orientation": "++", "overlap": "3"},
            {"edge_id": "e2", "source": "n2", "target": "n3", "orientation": "++", "overlap": "3"},
        ],
    )
    compacted = cfa_to_cdbg(graph)
    assert compacted.metadata["graph_type"] == "repeat"
    assert compacted.metadata["compaction"] == "chain"
    assert compacted.k is None
    assert [tuple(unitig.members) for unitig in compacted.unitigs] == [("n1", "n2", "n3")]
    assert compacted.unitigs[0].sequence == "ACGTAACCCGGG"
    restored = cdbg_to_cfa(compacted)
    assert restored.metadata["graph_type"] == "repeat"
    assert restored.sequences == graph.sequences


def test_downstream_merge_keeps_internal_edges() -> None:
    """Merging the downstream pair first still records every internal edge."""
    from metametro.converters.cfa_to_cdbg import cfa_to_cdbg

    graph = _plain_cfa(
        "repeat",
        {"z": "ACGTAA", "a": "TAACCC", "m": "CCCGGG"},
        [
            {"edge_id": "ez", "source": "z", "target": "a", "orientation": "++", "overlap": "3"},
            {"edge_id": "ea", "source": "a", "target": "m", "orientation": "++", "overlap": "3"},
        ],
    )
    compacted = cfa_to_cdbg(graph)
    assert compacted.unitigs[0].members == ["z", "a", "m"]
    assert compacted.unitigs[0].internal_edge_ids == ["ez", "ea"]


def test_gfa_repeat_graph_compacts(tmp_path) -> None:
    """A Flye-style GFA loads as a repeat graph and compacts without k."""
    from metametro.contracts.assembly import gfa_to_cfa
    from metametro.converters.cfa_to_cdbg import cfa_to_cdbg

    gfa = tmp_path / "repeat.gfa"
    gfa.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                "S\tn1\tACGTAA",
                "S\tn2\tTAACCC",
                "S\tn3\tCCCGGG",
                "L\tn1\t+\tn2\t+\t3M",
                "L\tn2\t+\tn3\t+\t3M",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    graph = gfa_to_cfa(gfa, graph_id="flye")
    assert graph.metadata["graph_type"] == "repeat"
    assert "k" not in graph.metadata
    compacted = cfa_to_cdbg(graph)
    assert compacted.metadata["graph_type"] == "repeat"
    assert compacted.k is None
    assert compacted.unitigs[0].sequence == "ACGTAACCCGGG"


def test_read_colouring_follows_k_not_graph_type() -> None:
    """Read colouring accepts a non-de Bruijn graph that declares k."""
    from metametro.contracts.colouring import colour_by_reads

    graph = chain_cfa()
    graph.metadata["graph_type"] = "repeat"
    coloured = colour_by_reads(graph, [], ["s0"])
    assert coloured.metadata["graph_type"] == "repeat"
    graph.metadata.pop("k")
    with pytest.raises(ContractError, match="integer k"):
        colour_by_reads(graph, [], ["s0"])
