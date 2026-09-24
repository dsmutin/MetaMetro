"""Mandatory checks for the CGT feature registry, predictions, and CSC index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import metametro
from metametro.contracts.ds import run_ds
from metametro.converters.cdbg_to_cgt import cdbg_to_cgt
from metametro.errors import ContractError
from metametro.fixtures import chain_cdbg, mock_cdbg, mock_cgt
from metametro.formats.cdbg.annotations import annotate_cdbg
from metametro.formats.cgt.io import load_cgt
from metametro.formats.cgt.predictions import edge_prediction, predictions_from_ds
from metametro.formats.cgt.topology import csc_from_cgt
from metametro.formats.cgt.validator import validate_cgt
from metametro.identity import cgt_edge_cfa_ids, node_lineage, resolve_sequence

pytestmark = pytest.mark.mandatory

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "minimal_metagenome"


def _provenance(cdbg) -> dict:
    return {
        "source": "unit-test",
        "method": "annotate_cdbg",
        "version": "test-1",
        "parameters": {"note": "registry"},
        "parent_graph_id": cdbg.metadata["graph_id"],
        "parent_schema_version": cdbg.metadata["schema_version"],
        "parent_contract_version": cdbg.metadata["contract_version"],
    }


def _snapshot(cgt) -> dict[str, np.ndarray]:
    arrays = {
        "indptr": cgt.indptr,
        "indices": cgt.indices,
        "node_features": cgt.node_features,
        "edge_features": cgt.edge_features,
        "node_colors": cgt.node_colors,
        "edge_colors": cgt.edge_colors,
    }
    if cgt.node_labels is not None:
        arrays["node_labels"] = cgt.node_labels
    if cgt.edge_labels is not None:
        arrays["edge_labels"] = cgt.edge_labels
    return {name: np.array(array, copy=True) for name, array in arrays.items()}


def test_feature_registry_names_namespace_source_and_dtype() -> None:
    """Named columns and sidecar columns are registered without a fake namespace."""
    cdbg = mock_cdbg()
    unitig_ids = [unitig.unitig_id for unitig in sorted(cdbg.unitigs, key=lambda item: item.unitig_id)]
    annotate_cdbg(
        cdbg,
        namespace="sample",
        feature="depth",
        target_type="node",
        values={unitig_id: float(index + 1) for index, unitig_id in enumerate(unitig_ids)},
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    annotate_cdbg(
        cdbg,
        namespace="composition",
        feature="kmer",
        target_type="node",
        values={unitig_id: [float(index), float(index + 1)] for index, unitig_id in enumerate(unitig_ids)},
        dtype="float64",
        kind="vector",
        provenance=_provenance(cdbg),
    )
    link_ids = [link.link_id for link in cdbg.links]
    annotate_cdbg(
        cdbg,
        namespace="sample",
        feature="support",
        target_type="edge",
        values={link_id: float(index + 3) for index, link_id in enumerate(link_ids)},
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    named = np.arange(len(unitig_ids) * 2, dtype=np.float32).reshape(len(unitig_ids), 2)
    named_edges = np.ones((len(link_ids), 1), dtype=np.float32)
    graph = cdbg_to_cgt(
        cdbg,
        node_features=named,
        edge_features=named_edges,
        node_feature_names=["coverage", "gc"],
        edge_feature_names=["weight"],
        node_annotation=[("sample", "depth"), ("composition", "kmer")],
        edge_annotation=[("sample", "support")],
    )
    assert graph.metadata["node_feature_names"] == [
        "coverage",
        "gc",
        "sample:depth",
        "composition:kmer:0",
        "composition:kmer:1",
    ]
    assert graph.metadata["edge_feature_names"] == ["weight", "sample:support"]
    node_registry = graph.metadata["node_feature_registry"]
    assert [row["name"] for row in node_registry] == graph.metadata["node_feature_names"]
    assert node_registry[0]["feature_type"] == "feature"
    assert node_registry[0]["namespace"] == ""
    assert node_registry[0]["source_annotation"] == ""
    assert node_registry[0]["positional"] is False
    assert node_registry[0]["dtype"] == "float32"
    assert node_registry[0]["normalization"] == "none"
    depth = node_registry[2]
    assert depth["name"] == "sample:depth"
    assert depth["namespace"] == "sample"
    assert depth["source_annotation"] == "sample:depth"
    assert depth["dtype"] == "float32"
    assert depth["positional"] is False
    kmer = node_registry[3]
    assert kmer["name"] == "composition:kmer:0"
    assert kmer["namespace"] == "composition"
    assert kmer["source_annotation"] == "composition:kmer"
    assert node_registry[4]["source_annotation"] == "composition:kmer"
    edge_registry = graph.metadata["edge_feature_registry"]
    assert edge_registry[0]["name"] == "weight"
    assert edge_registry[0]["namespace"] == ""
    assert edge_registry[0]["source_annotation"] == ""
    assert edge_registry[1]["namespace"] == "sample"
    assert edge_registry[1]["source_annotation"] == "sample:support"
    assert edge_registry[1]["dtype"] == "float32"
    assert np.array_equal(graph.node_features[:, :2], named)
    positional = cdbg_to_cgt(cdbg, node_features=named, edge_features=named_edges)
    assert positional.metadata["node_feature_names"] == ["f0", "f1"]
    assert all(row["positional"] is True for row in positional.metadata["node_feature_registry"])
    assert all(row["namespace"] == "" for row in positional.metadata["node_feature_registry"])
    assert all(row["source_annotation"] == "" for row in positional.metadata["node_feature_registry"])


def test_labels_are_not_feature_columns() -> None:
    """Training labels stay on y and are not concatenated into X."""
    cdbg = mock_cdbg()
    node_count = len(cdbg.unitigs)
    edge_count = len(cdbg.links)
    node_features = np.zeros((node_count, 2), dtype=np.float32)
    edge_features = np.zeros((edge_count, 1), dtype=np.float32)
    node_labels = np.arange(7, 7 + node_count, dtype=np.int64)
    edge_labels = np.arange(3, 3 + edge_count, dtype=np.int64)
    graph = cdbg_to_cgt(
        cdbg,
        node_features=node_features,
        edge_features=edge_features,
        node_labels=node_labels,
        edge_labels=edge_labels,
        node_feature_names=["coverage", "gc"],
        edge_feature_names=["weight"],
    )
    assert graph.node_features.shape == (node_count, 2)
    assert graph.edge_features.shape == (edge_count, 1)
    assert graph.node_labels is not None and graph.edge_labels is not None
    assert not np.array_equal(graph.node_features[:, 0], node_labels.astype(np.float32))
    assert not np.array_equal(graph.edge_features[:, 0], edge_labels.astype(np.float32))
    for record in graph.metadata["node_feature_registry"] + graph.metadata["edge_feature_registry"]:
        assert record["feature_type"] == "feature"


def test_predictions_join_lineage_and_do_not_mutate_cgt() -> None:
    """Node predictions join dense id, unitig id, and CFA ids. Arrays stay put."""
    cgt = mock_cgt()
    cdbg = mock_cdbg()
    before = _snapshot(cgt)
    identities = [id(cgt.node_features), id(cgt.edge_features), id(cgt.node_labels), id(cgt.indptr)]
    result = run_ds(cgt, epochs=1, seed=0)
    predictions = predictions_from_ds(result)
    assert len(predictions) == cgt.num_nodes
    assert result["metadata"]["num_predictions"] == cgt.num_nodes
    assert result["metadata"]["confidence"] == "softmax_probability"
    for prediction, row in zip(predictions, result["result"]):
        assert "predicted_label" in row and "probability" in row
        assert "csr_slot" not in row
        lineage = node_lineage(cgt, prediction.dense_id)
        assert prediction.target == "node"
        assert prediction.csr_slot is None
        assert prediction.dense_id == row["dense_id"] == lineage.dense_id
        assert prediction.source_id == row["source_id"] == lineage.source_id
        assert prediction.cfa_ids == tuple(row["cfa_node_ids"]) == lineage.cfa_node_ids
        assert prediction.predicted_class == row["predicted_label"] == row["predicted_class"]
        assert prediction.probability == row["probability"]
        assert prediction.confidence == row["confidence"] == prediction.probability
        assert prediction.model_id == row["model_id"] == "numpy_softmax_gcn"
        assert prediction.model_version == metametro.__version__
    edge = edge_prediction(
        cgt,
        0,
        predicted_class=1,
        probability=0.25,
        confidence=0.25,
        model_id="caller",
        model_version=metametro.__version__,
        cdbg=cdbg,
    )
    assert edge.target == "edge"
    assert edge.csr_slot == 0
    assert edge.cfa_ids == (cgt_edge_cfa_ids(cdbg)[0],)
    assert edge.source_id == edge.cfa_ids[0]
    assert edge.dense_id == 0
    assert edge.confidence == 0.25
    assert [id(cgt.node_features), id(cgt.edge_features), id(cgt.node_labels), id(cgt.indptr)] == identities
    after = _snapshot(cgt)
    for name, array in before.items():
        assert np.array_equal(after[name], array)
    with pytest.raises(ContractError, match="csr_slot"):
        edge_prediction(
            cgt,
            cgt.num_edges,
            predicted_class=0,
            probability=1.0,
            confidence=1.0,
            model_id="caller",
            model_version="test",
            cdbg=cdbg,
        )


def test_csc_matches_incoming_edges_without_reordering_features() -> None:
    """CSC lists predecessors. Edge feature rows stay in CSR slot order."""
    cgt = mock_cgt()
    features = cgt.edge_features
    before = np.array(features, copy=True)
    csc = csc_from_cgt(cgt)
    assert cgt.edge_features is features
    assert np.array_equal(cgt.edge_features, before)
    assert cgt.metadata["topology"] == "csr"
    expected_sources: list[list[int]] = [[] for _ in range(cgt.num_nodes)]
    expected_slots: list[list[int]] = [[] for _ in range(cgt.num_nodes)]
    for source in range(cgt.num_nodes):
        for slot in range(int(cgt.indptr[source]), int(cgt.indptr[source + 1])):
            target = int(cgt.indices[slot])
            expected_sources[target].append(source)
            expected_slots[target].append(slot)
    assert list(csc.indptr) == [0, 0, 2, 3, 5]
    for node in range(cgt.num_nodes):
        start = int(csc.indptr[node])
        stop = int(csc.indptr[node + 1])
        assert list(csc.indices[start:stop]) == expected_sources[node]
        assert list(csc.csr_slots[start:stop]) == expected_slots[node]
        for slot in csc.csr_slots[start:stop]:
            assert np.array_equal(cgt.edge_features[int(slot)], before[int(slot)])


def test_resolve_sequence_reads_the_unitig_and_rejects_a_bad_dense_id() -> None:
    """DNA stays on the CDBG. A bad dense id or the wrong graph is an error."""
    cgt = mock_cgt()
    cdbg = mock_cdbg()
    for dense_id in range(cgt.num_nodes):
        lineage = node_lineage(cgt, dense_id)
        unitig = next(item for item in cdbg.unitigs if item.unitig_id == lineage.source_id)
        assert resolve_sequence(cgt, cdbg, dense_id) == unitig.sequence
        assert unitig.sequence != ""
    with pytest.raises(ContractError, match="dense_id"):
        resolve_sequence(cgt, cdbg, cgt.num_nodes)
    with pytest.raises(ContractError, match="graph_id"):
        resolve_sequence(cgt, chain_cdbg(), 0)
    broken = mock_cgt()
    broken.mapping[0]["source_id"] = "missing"
    with pytest.raises(ContractError, match="not a CDBG unitig"):
        resolve_sequence(broken, mock_cdbg(), 0)


def test_old_cgt_metadata_without_registry_still_loads() -> None:
    """Schema 1.0 metadata that predates the registry still validates."""
    loaded = load_cgt(FIXTURES / "bubble" / "cgt")
    assert "node_feature_registry" not in loaded.metadata
    assert "edge_feature_registry" not in loaded.metadata
    validate_cgt(loaded)
    graph = mock_cgt()
    original = np.array(graph.node_features, copy=True)
    del graph.metadata["node_feature_registry"]
    del graph.metadata["edge_feature_registry"]
    validate_cgt(graph)
    assert np.array_equal(graph.node_features, original)


def test_missing_dict_label_and_colour_values() -> None:
    cdbg = chain_cdbg()
    with pytest.raises(ContractError, match="missing label"):
        cdbg_to_cgt(cdbg, node_labels={"missing": 1})
    graph = cdbg_to_cgt(cdbg, node_features=np.zeros((len(cdbg.unitigs), 1), dtype=np.float32))
    caller = np.zeros((len(cdbg.unitigs), 1), dtype=np.float32)
    copied = cdbg_to_cgt(cdbg, node_features=caller)
    caller[0, 0] = 9
    assert copied.node_features[0, 0] == 0
    graph.node_colors[0, 0] = 2
    with pytest.raises(ContractError, match="colours must be 0 or 1"):
        validate_cgt(graph)
    graph.node_colors[0, 0] = 1
    graph.metadata["node_feature_names"] = ("only", "extra")
    with pytest.raises(ContractError, match="node_feature_names"):
        validate_cgt(graph)
