"""Mandatory checks for CFA, CDBG, and CGT identity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metametro.contracts.ds import run_ds
from metametro.converters.cdbg_to_cgt import cdbg_to_cgt
from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
from metametro.errors import ContractError
from metametro.fixtures import chain_cdbg, chain_cfa, mock_cdbg, mock_cgt
from metametro.formats.cdbg.io import load_cdbg
from metametro.formats.cdbg.validator import validate_cdbg
from metametro.formats.cfa.io import load_cfa
from metametro.formats.cgt.io import load_cgt
from metametro.formats.cgt.validator import validate_cgt
from metametro.identity import (
    assert_cgt_matches_cdbg,
    cgt_edge_cfa_ids,
    internal_cfa_edge_ids,
    link_cfa_edge_id,
    node_lineage,
)

pytestmark = pytest.mark.mandatory

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "minimal_metagenome"


def _path_edge_ids(cfa, members: list[str]) -> list[str]:
    found: list[str] = []
    for source, target in zip(members, members[1:]):
        matches = [
            row["edge_id"]
            for row in cfa.edges
            if row["source"] == source and row["target"] == target
        ]
        assert len(matches) == 1
        found.append(matches[0])
    return found


def test_dense_id_recovers_cdbg_unitig_and_cfa_nodes() -> None:
    """A CGT dense id names one unitig and that unitig's CFA path."""
    cfa = chain_cfa()
    cdbg = cfa_to_cdbg(cfa)
    cgt = cdbg_to_cgt(cdbg)
    assert_cgt_matches_cdbg(cgt, cdbg)
    seen: list[str] = []
    for dense_id in range(cgt.num_nodes):
        lineage = node_lineage(cgt, dense_id)
        assert lineage.dense_id == dense_id
        seen.extend(lineage.cfa_node_ids)
    assert seen == [node_id for unitig in cdbg.unitigs for node_id in unitig.members]
    assert set(seen) == set(cfa.node_ids())


def test_long_unitig_keeps_internal_cfa_edge_ids() -> None:
    """Each junction inside a unitig keeps the CFA edge that was merged."""
    cfa = chain_cfa()
    cdbg = cfa_to_cdbg(cfa)
    long = next(unitig for unitig in cdbg.unitigs if len(unitig.members) == 3)
    assert internal_cfa_edge_ids(cdbg, long.unitig_id) == tuple(_path_edge_ids(cfa, long.members))
    recovered = set(cgt_edge_cfa_ids(cdbg))
    for unitig in cdbg.unitigs:
        recovered.update(internal_cfa_edge_ids(cdbg, unitig.unitig_id))
    assert recovered == {row["edge_id"] for row in cfa.edges}
    for link in cdbg.links:
        assert link_cfa_edge_id(cdbg, link.link_id) == link.link_id


def _assert_permuted_link_payloads(order: list[int]) -> None:
    """Features, labels, and colours on slot j match the CFA edge of that slot."""
    cdbg = mock_cdbg()
    cdbg.links = [cdbg.links[index] for index in order]
    link_ids = [link.link_id for link in cdbg.links]
    by_id = {link.link_id: link for link in cdbg.links}
    features = np.arange(len(link_ids), dtype=np.float32).reshape(-1, 1)
    labels = np.arange(100, 100 + len(link_ids), dtype=np.int64)
    cgt = cdbg_to_cgt(
        cdbg,
        edge_features=features,
        edge_labels=labels,
        edge_feature_names=["slot"],
    )
    recovered = cgt_edge_cfa_ids(cdbg)
    assert list(recovered) != link_ids
    column = {color_id: index for index, color_id in enumerate(cgt.color_ids)}
    for slot, edge_id in enumerate(recovered):
        origin = link_ids.index(edge_id)
        assert cgt.edge_features[slot, 0] == np.float32(origin)
        assert int(cgt.edge_labels[slot]) == 100 + origin
        expected = np.zeros(len(cgt.color_ids), dtype=np.uint8)
        for color_id in by_id[edge_id].color_ids:
            expected[column[int(color_id)]] = 1
        assert np.array_equal(cgt.edge_colors[slot], expected)
        assert int(cgt.indices[slot]) == next(
            index
            for index, row in enumerate(cgt.mapping)
            if row["source_id"] == by_id[edge_id].target
        )


def test_csr_edge_ids_follow_permuted_links() -> None:
    """Reversed and shuffled links keep features, labels, and colours on the CFA edge."""
    count = len(mock_cdbg().links)
    _assert_permuted_link_payloads(list(reversed(range(count))))
    _assert_permuted_link_payloads([2, 0, 4, 1, 3])


def test_ds_result_uses_the_same_node_lineage() -> None:
    """A DS row carries the dense id, unitig id, and CFA nodes of that CGT row."""
    cgt = mock_cgt()
    cdbg = mock_cdbg()
    assert_cgt_matches_cdbg(cgt, cdbg)
    result = run_ds(cgt, epochs=1, seed=0)
    assert len(result["result"]) == cgt.num_nodes
    for row in result["result"]:
        lineage = node_lineage(cgt, row["dense_id"])
        assert row["source_id"] == lineage.source_id
        assert tuple(row["cfa_node_ids"]) == lineage.cfa_node_ids


def test_lineage_is_deterministic() -> None:
    """The same CFA produces the same unitig paths and CSR edge ids."""
    left = cfa_to_cdbg(chain_cfa())
    right = cfa_to_cdbg(chain_cfa())
    assert [internal_cfa_edge_ids(left, unitig.unitig_id) for unitig in left.unitigs] == [
        internal_cfa_edge_ids(right, unitig.unitig_id) for unitig in right.unitigs
    ]
    assert cgt_edge_cfa_ids(left) == cgt_edge_cfa_ids(right)
    assert [node_lineage(cdbg_to_cgt(left), index) for index in range(len(left.unitigs))] == [
        node_lineage(cdbg_to_cgt(right), index) for index in range(len(right.unitigs))
    ]


def test_on_disk_fixtures_match_the_compactor() -> None:
    """Checked-in CDBG and CGT files agree with compaction of their CFA."""
    pairs = [
        (FIXTURES / "chain" / "cfa", FIXTURES / "chain" / "cdbg", None),
        (FIXTURES / "bubble" / "cfa", FIXTURES / "bubble" / "cdbg", FIXTURES / "bubble" / "cgt"),
        (FIXTURES / "coloured_cfa", FIXTURES / "cdbg", FIXTURES / "cgt"),
    ]
    for cfa_path, cdbg_path, cgt_path in pairs:
        live = cfa_to_cdbg(load_cfa(cfa_path))
        disk = load_cdbg(cdbg_path)
        assert [
            (unitig.unitig_id, tuple(unitig.members), tuple(unitig.internal_edge_ids), tuple(unitig.internal_overlaps))
            for unitig in disk.unitigs
        ] == [
            (unitig.unitig_id, tuple(unitig.members), tuple(unitig.internal_edge_ids), tuple(unitig.internal_overlaps))
            for unitig in live.unitigs
        ]
        assert [(link.link_id, link.source, link.target, link.overlap) for link in disk.links] == [
            (link.link_id, link.source, link.target, link.overlap) for link in live.links
        ]
        if cgt_path is not None:
            assert_cgt_matches_cdbg(load_cgt(cgt_path), disk)


def test_dropped_internal_edges_rejected() -> None:
    """A long unitig that omits an internal CFA edge id does not validate."""
    graph = chain_cdbg()
    graph.unitigs[1].internal_edge_ids = graph.unitigs[1].internal_edge_ids[:1]
    graph.unitigs[1].internal_edge_colors = graph.unitigs[1].internal_edge_colors[:1]
    graph.unitigs[1].internal_overlaps = graph.unitigs[1].internal_overlaps[:1]
    with pytest.raises(ContractError, match="internal edges"):
        validate_cdbg(graph)


def test_swapped_mapping_ordinal_rejected() -> None:
    """A mapping ordinal that is not the member index is rejected."""
    graph = chain_cdbg()
    rows = [row for row in graph.mapping if row.unitig_id == "u000002"]
    rows[0].ordinal, rows[2].ordinal = rows[2].ordinal, rows[0].ordinal
    with pytest.raises(ContractError, match="mapping ordinal"):
        validate_cdbg(graph)


def test_duplicate_mapping_row_rejected() -> None:
    """Two mapping rows for one CFA node are rejected."""
    graph = chain_cdbg()
    graph.mapping.append(graph.mapping[0])
    with pytest.raises(ContractError, match="duplicate mapping row"):
        validate_cdbg(graph)


def test_unknown_link_rejected() -> None:
    """A link id that is not on the CDBG is not treated as a CFA edge."""
    with pytest.raises(ContractError, match="unknown CDBG link"):
        link_cfa_edge_id(chain_cdbg(), "e999999")


def test_empty_cfa_node_ids_rejected() -> None:
    """A CGT row with no CFA nodes is rejected."""
    graph = mock_cgt()
    graph.mapping[1]["cfa_node_ids"] = []
    with pytest.raises(ContractError, match="no CFA node ids"):
        validate_cgt(graph)


def test_colour_width_mismatch_rejected() -> None:
    """Colour-id columns must match the colour matrices."""
    graph = mock_cgt()
    graph.color_ids = [0]
    with pytest.raises(ContractError, match="color_ids"):
        validate_cgt(graph)


def test_dangling_cgt_source_rejected() -> None:
    """A dense id whose unitig is absent from the CDBG is rejected."""
    graph = mock_cgt()
    graph.mapping[0]["source_id"] = "u999999"
    with pytest.raises(ContractError, match="not a CDBG unitig"):
        assert_cgt_matches_cdbg(graph, mock_cdbg())


def test_cdbg_and_cgt_schema_mismatch_rejected() -> None:
    """Schema versions other than 1.0 are rejected."""
    cdbg = chain_cdbg()
    cdbg.metadata["schema_version"] = "0.1"
    with pytest.raises(ContractError, match="incompatible schema"):
        validate_cdbg(cdbg)
    cgt = mock_cgt()
    cgt.metadata["schema_version"] = "2.0"
    with pytest.raises(ContractError, match="incompatible schema"):
        validate_cgt(cgt)
