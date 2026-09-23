"""ToCUMG, geometric CGT, and a two-head GCN stay aligned with the CFA."""

from __future__ import annotations

import numpy as np
import pytest

from metametro.contracts.multitask import fit_multitask_gcn, infer_multitask_gcn
from metametro.digraph_cfa import coloured_digraph_cfa
from metametro.errors import ContractError
from metametro.states import build_geometric_states
from metametro.viz.states import plot_geometric_states

pytestmark = pytest.mark.mandatory


def _branched():
    nodes = [
        {"node_id": "a", "longitude": "0", "latitude": "0", "type": "bus"},
        {"node_id": "b", "longitude": "1", "latitude": "0", "type": "bus"},
        {"node_id": "c", "longitude": "1", "latitude": "1", "type": "tram"},
    ]
    edges = [
        {"source": "a", "target": "b", "type": "bus", "route": "1"},
        {"source": "a", "target": "c", "type": "tram", "route": "2"},
        {"source": "b", "target": "c", "type": "bus", "route": "1"},
    ]
    return coloured_digraph_cfa(
        nodes,
        edges,
        graph_id="branch",
        k=5,
        colour_columns=("type", "route"),
        keep_columns=("longitude", "latitude"),
    )


def test_unitig_coordinate_is_the_mean_of_its_stops() -> None:
    """The tensor longitude is the mean of the CFA nodes in that unitig."""
    states = build_geometric_states(_branched(), node_namespace="type", edge_namespaces=("type", "route"))
    assert states.cgt.num_nodes == 3
    assert states.cgt.num_edges == 3
    names = states.cgt.metadata["node_feature_names"]
    longitude = states.cgt.node_features[0, names.index("longitude")]
    by_id = {row["source_id"]: row for row in states.cgt.mapping}
    members = by_id["u000001"]["cfa_node_ids"]
    assert len(members) == 1
    assert states.node_class_names == ["bus", "tram"]
    assert "type=bus" in states.edge_label_names
    assert "route=1" in states.edge_label_names
    assert states.edge_targets.shape == (3, len(states.edge_label_names))
    assert set(np.unique(states.edge_targets)).issubset({0, 1})
    assert float(longitude) in {0.0, 1.0}


def test_two_types_on_one_node_are_one_class() -> None:
    """A stop that carries bus and tram is the single class bus+tram."""
    graph = _branched()
    tram = next(
        row["color_id"]
        for row in graph.colors
        if row["namespace"] == "type" and row["value"] == "tram"
    )
    for row in graph.nodes:
        if row["node_id"] == "a":
            row["color_set"] = f"{row['color_set']},{tram}"
    states = build_geometric_states(graph, node_namespace="type", edge_namespaces=("type", "route"))
    assert "bus+tram" in states.node_class_names
    assert "Unclassified" not in states.node_class_names or states.node_class_names[-1] == "Unclassified"


def test_missing_namespace_is_an_error() -> None:
    """A requested colour namespace that the CFA does not have stops the build."""
    with pytest.raises(ContractError, match="namespace missing"):
        build_geometric_states(_branched(), node_namespace="type", edge_namespaces=("missing",))


def test_gcn_inference_is_deterministic_and_leaves_the_tensor() -> None:
    """The same seed repeats, and inference does not write the CGT."""
    states = build_geometric_states(_branched(), node_namespace="type", edge_namespaces=("type", "route"))
    before = np.array(states.cgt.node_features, copy=True)
    model, metrics = fit_multitask_gcn(
        states.cgt,
        states.edge_targets,
        node_class_names=states.node_class_names,
        edge_label_names=states.edge_label_names,
        epochs=5,
        seed=0,
    )
    again, _ = fit_multitask_gcn(
        states.cgt,
        states.edge_targets,
        node_class_names=states.node_class_names,
        edge_label_names=states.edge_label_names,
        epochs=5,
        seed=0,
    )
    first = infer_multitask_gcn(model, states.cgt)
    second = infer_multitask_gcn(again, states.cgt)
    assert np.array_equal(first["node_class"], second["node_class"])
    assert np.allclose(first["edge_probability"], second["edge_probability"])
    assert first["node_probability"].shape == (3, len(states.node_class_names))
    assert first["edge_binary"].shape == states.edge_targets.shape
    assert set(np.unique(first["edge_binary"])).issubset({0, 1})
    assert np.allclose(first["node_probability"].sum(axis=1), 1.0)
    assert np.array_equal(before, states.cgt.node_features)
    assert "node_accuracy_heldout" in metrics
    assert "edge_micro_f1_heldout" in metrics


def test_three_state_pdf_has_three_pages(tmp_path) -> None:
    """CFA, ToCUMG, and CGT each occupy one PDF page."""
    states = build_geometric_states(_branched(), node_namespace="type", edge_namespaces=("type", "route"))
    output = tmp_path / "states.pdf"
    plot_geometric_states(_branched(), states, output, edge_namespace="type")
    data = output.read_bytes()
    assert data.startswith(b"%PDF")
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 3


def test_fr_layout_writes_the_same_three_pages(tmp_path) -> None:
    """Fruchterman–Reingold is a second placement of the same three pages."""
    states = build_geometric_states(_branched(), node_namespace="type", edge_namespaces=("type", "route"))
    output = tmp_path / "states_fr.pdf"
    plot_geometric_states(_branched(), states, output, edge_namespace="type", layout="fr")
    data = output.read_bytes()
    assert data.startswith(b"%PDF")
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 3
    with pytest.raises(ContractError, match="layout"):
        plot_geometric_states(_branched(), states, output, edge_namespace="type", layout="radial")
