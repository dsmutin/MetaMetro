"""Mandatory checks for colour filtering on CFA, CDBG, and CGT."""

from __future__ import annotations

import numpy as np
import pytest

from metametro.contracts.colour_filter import filter_colours
from metametro.converters.cdbg_to_cgt import cdbg_to_cgt
from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
from metametro.errors import ContractError
from metametro.fixtures import chain_cdbg, mock_cfa, mock_cgt
from metametro.formats.cdbg.validator import validate_cdbg
from metametro.identity import cgt_edge_cfa_ids

pytestmark = pytest.mark.mandatory


def _signature_cfa(graph) -> tuple:
    return (
        tuple(graph.sequences.items()),
        tuple((row["edge_id"], row["source"], row["target"]) for row in graph.edges),
    )


def test_cfa_namespace_keeps_topology() -> None:
    graph = mock_cfa()
    before = _signature_cfa(graph)
    filtered = filter_colours(graph, namespace="sample", color_ids=[0])
    assert _signature_cfa(graph) == before
    assert filtered.nodes[1]["color_set"] in {"0", "0,1", "1"}
    multi = next(row for row in filtered.nodes if row["node_id"] == "n000002")
    assert multi["color_set"] == "0"
    assert all("1" not in row["color_set"].split(",") for row in filtered.nodes + filtered.edges)
    assert [row["color_id"] for row in filtered.colors] == ["0"]


def test_cdbg_filter_keeps_unitig_union_and_internal_edges() -> None:
    graph = chain_cdbg()
    sequences = [unitig.sequence for unitig in graph.unitigs]
    links = [(link.link_id, link.source, link.target) for link in graph.links]
    filtered = filter_colours(graph, color_ids=[0])
    assert [unitig.sequence for unitig in graph.unitigs] == sequences
    assert [(link.link_id, link.source, link.target) for link in filtered.links] == links
    validate_cdbg(filtered)
    by_node = {row.cfa_node_id: set(row.color_ids) for row in filtered.mapping}
    for unitig in filtered.unitigs:
        assert set(unitig.color_ids) == set().union(*(by_node[node_id] for node_id in unitig.members))
        assert all(color_id == 0 for color_id in unitig.color_ids)
        for group in unitig.internal_edge_colors:
            assert all(color_id == 0 for color_id in group)
    assert graph.metadata.get("colour_filter") is None


def test_cgt_filter_drops_columns_without_moving_edges() -> None:
    cdbg = chain_cdbg()
    original = cdbg_to_cgt(cdbg)
    width = original.node_colors.shape[1]
    original.node_color_weights = np.zeros(original.node_colors.shape, dtype=np.float32)
    original.edge_color_weights = np.zeros(original.edge_colors.shape, dtype=np.float32)
    original.node_color_weights[original.node_colors == 1] = 0.5
    original.edge_color_weights[original.edge_colors == 1] = 0.25
    features = np.array(original.node_features, copy=True)
    indices = np.array(original.indices, copy=True)
    edge_ids = cgt_edge_cfa_ids(cdbg)
    filtered = filter_colours(original, color_ids=[0])
    assert np.array_equal(original.indices, indices)
    assert np.array_equal(original.node_features, features)
    assert np.array_equal(filtered.indices, indices)
    assert np.array_equal(filtered.node_features, features)
    assert filtered.color_ids == [0]
    assert filtered.node_colors.shape[1] == 1
    assert filtered.edge_colors.shape == (filtered.num_edges, 1)
    assert filtered.node_color_weights.shape == filtered.node_colors.shape
    assert filtered.edge_color_weights.shape == filtered.edge_colors.shape
    assert cgt_edge_cfa_ids(cdbg) == edge_ids
    projected = cdbg_to_cgt(filter_colours(cdbg, color_ids=[0]))
    assert np.array_equal(projected.node_colors, filtered.node_colors)
    assert np.array_equal(projected.edge_colors, filtered.edge_colors)
    assert width >= 1


def test_cfa_namespaces_union_keeps_requested_layers() -> None:
    graph = mock_cfa()
    filtered = filter_colours(graph, namespaces=("sample",))
    assert {row["namespace"] for row in filtered.colors or []} == {"sample"}


def test_unknown_colour_and_cgt_namespace_are_rejected() -> None:
    with pytest.raises(ContractError, match="unknown color_id"):
        filter_colours(mock_cgt(), color_ids=[99])
    with pytest.raises(ContractError, match="no colour namespace"):
        filter_colours(mock_cgt(), namespace="sample")
    with pytest.raises(ContractError, match="requires color_ids or namespace"):
        filter_colours(mock_cfa())
