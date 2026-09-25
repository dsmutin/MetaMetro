"""Mandatory checks for external CSR tensors and colour weights."""

from __future__ import annotations

import numpy as np
import pytest

from metametro.converters.csr_to_cgt import cgt_from_csr
from metametro.errors import ContractError
from metametro.fixtures import mock_cgt
from metametro.formats.cgt.io import dump_cgt, load_cgt
from metametro.formats.cgt.validator import validate_cgt

pytestmark = pytest.mark.mandatory


def test_csr_sorts_edges_and_keeps_colours_out_of_features() -> None:
    features = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    colors = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    weights = np.array([[0.2, 0.0], [0.0, 0.8], [0.4, 0.6]], dtype=np.float32)
    graph = cgt_from_csr(
        graph_id="bundle",
        indptr=np.array([0, 2, 3, 3], dtype=np.int64),
        indices=np.array([2, 0, 1], dtype=np.int64),
        node_features=features,
        source_ids=["a", "b", "c"],
        edge_features=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        node_colors=colors,
        node_color_weights=weights,
        source="test bundle",
    )
    assert graph.indices.tolist() == [0, 2, 1]
    assert graph.edge_features[:, 0].tolist() == [20.0, 10.0, 30.0]
    assert graph.edge_colors.shape == (3, 2)
    assert int(graph.edge_colors.sum()) == 0
    assert graph.edge_color_weights is not None
    assert float(graph.edge_color_weights.sum()) == 0.0
    assert graph.metadata["edge_feature_names"] == ["weight"]
    assert graph.node_features.tolist() == features.tolist()
    assert graph.metadata["node_feature_registry"][0]["feature_type"] == "feature"
    assert graph.mapping[0]["cfa_node_ids"] == ["a"]


def test_colour_weight_requires_mask_membership(tmp_path) -> None:
    graph = mock_cgt()
    width = graph.node_colors.shape[1]
    graph.node_color_weights = np.zeros(graph.node_colors.shape, dtype=np.float32)
    graph.edge_color_weights = np.zeros(graph.edge_colors.shape, dtype=np.float32)
    graph.node_color_weights[0, 0] = 1.5
    with pytest.raises(ContractError, match="lie in"):
        validate_cgt(graph)
    graph.node_color_weights[0, 0] = 0.0
    if width:
        zero_column = int(np.argmin(graph.node_colors[0]))
        if graph.node_colors[0, zero_column] == 0:
            graph.node_color_weights[0, zero_column] = 0.3
            with pytest.raises(ContractError, match="mask is 0"):
                validate_cgt(graph)
            graph.node_color_weights[0, zero_column] = 0.0
    validate_cgt(graph)
    dump_cgt(graph, tmp_path)
    loaded = load_cgt(tmp_path)
    assert loaded.node_color_weights is not None
    assert np.array_equal(loaded.node_color_weights, graph.node_color_weights)


def test_directory_without_colour_weights_still_loads() -> None:
    graph = mock_cgt()
    assert graph.node_color_weights is None
    validate_cgt(graph)


def test_duplicate_csr_target_is_rejected() -> None:
    with pytest.raises(ContractError, match="duplicate CSR edge"):
        cgt_from_csr(
            graph_id="dup",
            indptr=np.array([0, 2], dtype=np.int64),
            indices=np.array([0, 0], dtype=np.int64),
            node_features=np.zeros((1, 1), dtype=np.float32),
            source_ids=["n"],
            source="duplicate",
        )
