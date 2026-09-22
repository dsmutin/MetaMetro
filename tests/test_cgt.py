"""Mandatory CGT schema, topology, and alignment checks."""

from __future__ import annotations

import numpy as np
import pytest

from metametro.converters.cgt_to_pyg import edge_index_array
from metametro.errors import ContractError
from metametro.fixtures import mock_cfa, mock_cgt
from metametro.formats.cgt.io import dump_cgt, load_cgt
from metametro.formats.cgt.validator import validate_cgt

pytestmark = pytest.mark.mandatory


def test_bubble_tensor_layout(tmp_path) -> None:
    """The bubble is a 4x3 / 5x2 CSR tensor with dense ids 0..3."""
    graph = mock_cgt()
    validate_cgt(graph)
    assert graph.num_nodes == 4
    assert graph.num_edges == 5
    assert graph.node_features.shape == (4, 3)
    assert graph.edge_features.shape == (5, 2)
    assert graph.node_labels is not None and graph.node_labels.shape == (4,)
    assert graph.edge_labels is not None and graph.edge_labels.shape == (5,)
    assert list(graph.indptr) == [0, 2, 3, 4, 5]
    assert list(graph.indices) == [1, 2, 3, 3, 1]
    assert [row["dense_id"] for row in graph.mapping] == [0, 1, 2, 3]
    dump_cgt(graph, tmp_path / "cgt")
    loaded = load_cgt(tmp_path / "cgt")
    assert np.array_equal(loaded.indptr, graph.indptr)
    assert np.array_equal(loaded.node_features, graph.node_features)


def test_feature_alignment() -> None:
    """Row i of X_node is the CFA node stored at dense id i."""
    cfa = mock_cfa()
    graph = mock_cgt()
    coverage = {row["node_id"]: float(row["coverage"]) for row in cfa.nodes}
    for row, features in zip(graph.mapping, graph.node_features):
        assert features[0] == np.float32(coverage[row["cfa_node_ids"][0]])
    assert graph.node_features.dtype == np.float32
    assert graph.node_colors.dtype == np.uint8
    multi = graph.mapping[1]
    assert multi["cfa_node_ids"] == ["n000002"]
    assert list(graph.node_colors[1]) == [1, 1]


def test_edge_feature_follows_csr_slot() -> None:
    """indices[j] and edge_features[j] describe the same directed edge."""
    cfa = mock_cfa()
    graph = mock_cgt()
    coverage = {row["edge_id"]: float(row["coverage"]) for row in cfa.edges}
    # Bubble link order matches CSR order: e000001 .. e000005.
    expected = [coverage[f"e{index:06d}"] for index in range(1, 6)]
    assert list(graph.edge_features[:, 0]) == [np.float32(value) for value in expected]


def test_wrong_dtype_rejected() -> None:
    """A non-float32 feature matrix is rejected."""
    graph = mock_cgt()
    graph.node_features = graph.node_features.astype(np.float64)
    with pytest.raises(ContractError, match="node features"):
        validate_cgt(graph)


def test_invalid_topology_rejected() -> None:
    """An index past N is rejected."""
    graph = mock_cgt()
    graph.indices = graph.indices.copy()
    graph.indices[0] = 99
    with pytest.raises(ContractError, match="invalid topology"):
        validate_cgt(graph)


def test_pyg_edge_index_layout() -> None:
    """The adapter's edge_index agrees with CSR, without importing PyG."""
    graph = mock_cgt()
    edge_index = edge_index_array(graph)
    assert edge_index.shape == (2, 5)
    assert list(edge_index[0]) == [0, 0, 1, 2, 3]
    assert list(edge_index[1]) == list(graph.indices)
