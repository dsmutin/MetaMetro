"""Mandatory checks for a coloured digraph CFA."""

from __future__ import annotations

import pytest

from metametro.digraph_cfa import coloured_digraph_cfa
from metametro.transit_cfa import overlap_mismatches

pytestmark = pytest.mark.mandatory


def test_attribute_colours_and_overlap() -> None:
    """Edge attributes become colours, and the endpoints share k-1 bases."""
    nodes = [
        {"node_id": "n1", "longitude": "0.1", "latitude": "1.2"},
        {"node_id": "n2", "longitude": "0.2", "latitude": "1.3"},
    ]
    edges = [{"source": "n1", "target": "n2", "type": "residential", "name": "Roxelweg"}]
    graph = coloured_digraph_cfa(
        nodes,
        edges,
        graph_id="toy_roxel",
        k=5,
        colour_columns=("type", "name"),
        keep_columns=("longitude", "latitude"),
    )
    assert overlap_mismatches(graph) == []
    assert {row["namespace"] for row in graph.colors or []} == {"type", "name"}
    assert graph.nodes[0]["longitude"] == "0.1"
    assert "type" in graph.edges[0]["color_set"] or graph.edges[0]["color_set"] != ""
