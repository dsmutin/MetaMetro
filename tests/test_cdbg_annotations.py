"""Mandatory checks for the CDBG annotation sidecar."""

from __future__ import annotations

import numpy as np
import pytest

from metametro.converters.cdbg_to_cgt import cdbg_to_cgt
from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
from metametro.errors import ContractError
from metametro.fixtures import chain_cfa
from metametro.formats.cdbg.annotations import (
    aggregate_annotations,
    annotate_cdbg,
    get_edge_annotations,
    get_node_annotations,
    transfer_annotations,
)
from metametro.formats.cdbg.io import dump_cdbg, load_cdbg
from metametro.formats.cdbg.model import SCHEMA_VERSION
from metametro.formats.cfa.model import CfaGraph
from metametro.identity import cgt_edge_cfa_ids, internal_cfa_edge_ids, node_lineage

pytestmark = pytest.mark.mandatory


def _provenance(cdbg, **parameters: object) -> dict:
    payload = {"note": "test"}
    payload.update(parameters)
    return {
        "source": "unit-test",
        "method": "annotate_cdbg",
        "version": "test-1",
        "parameters": payload,
        "parent_graph_id": cdbg.metadata["graph_id"],
        "parent_schema_version": cdbg.metadata["schema_version"],
        "parent_contract_version": cdbg.metadata["contract_version"],
    }


def _length_weight_cdbg():
    """Two CFA nodes whose sequence lengths make weighted_mean unequal to mean."""
    sequences = {"n1": "ACGT", "n2": "GTACCC"}
    graph = CfaGraph(
        metadata={
            "schema_version": "1.0",
            "graph_id": "length_weights",
            "graph_type": "repeat",
            "features": {
                "node": {},
                "edge": {"orientation": "orientation", "overlap": "int"},
            },
        },
        sequences=sequences,
        nodes=[{"node_id": "n1"}, {"node_id": "n2"}],
        edges=[
            {
                "edge_id": "e1",
                "source": "n1",
                "target": "n2",
                "orientation": "++",
                "overlap": "2",
            }
        ],
        node_header=["node_id"],
        edge_header=["edge_id", "source", "target", "orientation", "overlap"],
    )
    compacted = cfa_to_cdbg(graph)
    assert [tuple(unitig.members) for unitig in compacted.unitigs] == [("n1", "n2")]
    assert {row.cfa_node_id: row.length for row in compacted.mapping} == {"n1": 4, "n2": 6}
    return compacted


def _coverage(cdbg, values: dict[str, float]):
    return annotate_cdbg(
        cdbg,
        namespace="sample",
        feature="coverage",
        target_type="internal_node",
        values=values,
        dtype="float64",
        provenance=_provenance(cdbg),
    )


def test_cfa_annotations_transfer_without_aggregation() -> None:
    """CFA node and edge columns land on the CDBG keyed by the original ids."""
    cfa = chain_cfa()
    expected_nodes = {}
    for index, row in enumerate(cfa.nodes, start=1):
        row["coverage"] = str(index * 10)
        expected_nodes[row["node_id"]] = float(index * 10)
    expected_edges = {}
    for row in cfa.edges:
        value = float(int(row["edge_id"].lstrip("e")))
        row["coverage"] = str(value)
        expected_edges[row["edge_id"]] = value
    cdbg = cfa_to_cdbg(cfa)
    unitigs = cdbg.unitigs
    transfer_annotations(
        cdbg,
        cfa,
        namespace="copied",
        provenance=_provenance(cdbg, method_note="transfer"),
        node_columns=["coverage", "label"],
        edge_columns=["coverage"],
    )
    assert cdbg.unitigs is unitigs
    assert cdbg.metadata["schema_version"] == SCHEMA_VERSION
    with pytest.raises(ContractError, match="missing annotation layer"):
        get_node_annotations(cdbg, namespace="copied", feature="coverage", target_type="node")
    nodes = get_node_annotations(cdbg, namespace="copied", feature="coverage", target_type="internal_node")
    assert dict(zip(nodes.target_ids.tolist(), nodes.values.tolist())) == expected_nodes
    labels = get_node_annotations(cdbg, namespace="copied", feature="label", target_type="internal_node")
    assert labels.dtype == "int64"
    assert labels.values.dtype == np.int64
    internal = get_edge_annotations(cdbg, namespace="copied", feature="coverage", target_type="internal_edge")
    links = get_edge_annotations(cdbg, namespace="copied", feature="coverage", target_type="edge")
    stored = dict(zip(internal.target_ids.tolist(), internal.values.tolist()))
    stored.update(zip(links.target_ids.tolist(), links.values.tolist()))
    assert stored == expected_edges


def test_node_annotation_survives_compaction() -> None:
    """A CFA node value is still addressable by that CFA id after compaction."""
    cfa = chain_cfa()
    cfa.nodes[0]["coverage"] = "10"
    cfa.nodes[1]["coverage"] = "20"
    cdbg = cfa_to_cdbg(cfa)
    transfer_annotations(
        cdbg,
        cfa,
        namespace="copied",
        provenance=_provenance(cdbg),
        node_columns=["coverage"],
        edge_columns=[],
    )
    long = next(unitig for unitig in cdbg.unitigs if unitig.members == ["n000001", "n000002"])
    layer = get_node_annotations(cdbg, namespace="copied", feature="coverage", target_type="internal_node")
    found = dict(zip(layer.target_ids.tolist(), layer.values.tolist()))
    assert found["n000001"] == 10.0
    assert found["n000002"] == 20.0
    assert long.unitig_id not in found


def test_internal_edge_annotation_survives_compaction() -> None:
    """An absorbed CFA edge keeps its own annotation and its CFA edge id."""
    cfa = chain_cfa()
    cdbg = cfa_to_cdbg(cfa)
    long = next(unitig for unitig in cdbg.unitigs if len(unitig.members) == 3)
    edge_ids = internal_cfa_edge_ids(cdbg, long.unitig_id)
    assert len(edge_ids) == 2
    annotate_cdbg(
        cdbg,
        namespace="reads",
        feature="support",
        target_type="internal_edge",
        values={edge_ids[0]: 4.0, edge_ids[1]: 8.0},
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    layer = get_edge_annotations(cdbg, namespace="reads", feature="support", target_type="internal_edge")
    assert dict(zip(layer.target_ids.tolist(), layer.values.tolist())) == {edge_ids[0]: 4.0, edge_ids[1]: 8.0}
    assert long.sequence == next(unitig.sequence for unitig in cdbg.unitigs if unitig.unitig_id == long.unitig_id)


def test_link_annotation() -> None:
    """A CDBG link annotation is keyed by the link id, which is the CFA edge id."""
    cfa = chain_cfa()
    cdbg = cfa_to_cdbg(cfa)
    link_id = cdbg.links[0].link_id
    annotate_cdbg(
        cdbg,
        namespace="reads",
        feature="support",
        target_type="edge",
        values={link.link_id: float(index) for index, link in enumerate(cdbg.links, start=1)},
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    layer = get_edge_annotations(cdbg, namespace="reads", feature="support")
    assert layer.target_type == "edge"
    assert link_id in set(layer.target_ids.tolist())
    assert len(layer.target_ids) == len(cdbg.links)


def test_multiple_namespaces() -> None:
    """The same targets can carry two namespaces without merging them."""
    cdbg = cfa_to_cdbg(chain_cfa())
    unitig_id = cdbg.unitigs[0].unitig_id
    annotate_cdbg(
        cdbg,
        namespace="sample",
        feature="coverage",
        target_type="node",
        values={unitig.unitig_id: 1.0 for unitig in cdbg.unitigs},
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    annotate_cdbg(
        cdbg,
        namespace="taxonomy",
        feature="coverage",
        target_type="node",
        values={unitig.unitig_id: 5.0 for unitig in cdbg.unitigs},
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    sample = get_node_annotations(cdbg, namespace="sample", feature="coverage")
    taxon = get_node_annotations(cdbg, namespace="taxonomy", feature="coverage")
    sample_values = dict(zip(sample.target_ids.tolist(), sample.values.tolist()))
    taxon_values = dict(zip(taxon.target_ids.tolist(), taxon.values.tolist()))
    assert sample_values[unitig_id] == 1.0
    assert taxon_values[unitig_id] == 5.0


def test_numeric_scalar_and_vector_features() -> None:
    """Scalar and vector numbers share the sidecar and keep their widths."""
    cdbg = cfa_to_cdbg(chain_cfa())
    annotate_cdbg(
        cdbg,
        namespace="composition",
        feature="gc",
        target_type="node",
        values={unitig.unitig_id: 0.5 for unitig in cdbg.unitigs},
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    annotate_cdbg(
        cdbg,
        namespace="composition",
        feature="kmer",
        target_type="node",
        values={unitig.unitig_id: [0.1, 0.2, 0.3, 0.4] for unitig in cdbg.unitigs},
        dtype="float64",
        kind="vector",
        provenance=_provenance(cdbg),
    )
    scalar = get_node_annotations(cdbg, namespace="composition", feature="gc")
    vector = get_node_annotations(cdbg, namespace="composition", feature="kmer")
    assert scalar.kind == "scalar"
    assert scalar.values.shape == (len(cdbg.unitigs),)
    assert vector.kind == "vector"
    assert vector.values.shape == (len(cdbg.unitigs), 4)
    assert vector.values.dtype == np.float64


def test_categorical_and_integer_annotations() -> None:
    """Categories stay strings and integer labels stay int64."""
    cdbg = cfa_to_cdbg(chain_cfa())
    annotate_cdbg(
        cdbg,
        namespace="genome",
        feature="call",
        target_type="internal_node",
        values={row.cfa_node_id: "genome_001" for row in cdbg.mapping},
        dtype="category",
        provenance=_provenance(cdbg),
    )
    annotate_cdbg(
        cdbg,
        namespace="genome",
        feature="label",
        target_type="internal_node",
        values={row.cfa_node_id: 2 for row in cdbg.mapping},
        dtype="int64",
        provenance=_provenance(cdbg),
    )
    categories = get_node_annotations(cdbg, namespace="genome", feature="call", target_type="internal_node")
    labels = get_node_annotations(cdbg, namespace="genome", feature="label", target_type="internal_node")
    assert set(categories.values.tolist()) == {"genome_001"}
    assert labels.values.dtype == np.int64
    assert set(int(value) for value in labels.values.tolist()) == {2}


def test_weighted_mean_uses_member_lengths_exactly() -> None:
    """n1 length 4 coverage 10 and n2 length 6 coverage 20 average to 16."""
    cdbg = _length_weight_cdbg()
    _coverage(cdbg, {"n1": 10.0, "n2": 20.0})
    with pytest.raises(ContractError, match="missing annotation layer"):
        get_node_annotations(cdbg, namespace="sample", feature="coverage", target_type="node")
    aggregate_annotations(
        cdbg,
        namespace="sample",
        feature="coverage",
        policy="weighted_mean",
        provenance=_provenance(cdbg, method="aggregate_annotations"),
    )
    layer = get_node_annotations(cdbg, namespace="sample", feature="coverage", target_type="node")
    assert layer.aggregation == "weighted_mean"
    assert layer.provenance.parameters["aggregation"]["weight_source"] == "member_sequence_length"
    assert layer.provenance.parameters["aggregation"]["weights"] == {"n1": 4.0, "n2": 6.0}
    assert float(layer.values[0]) == 16.0
    assert (10 * 4 + 20 * 6) / (4 + 6) == 16.0
    source = get_node_annotations(cdbg, namespace="sample", feature="coverage", target_type="internal_node")
    assert dict(zip(source.target_ids.tolist(), source.values.tolist())) == {"n1": 10.0, "n2": 20.0}

    other = _length_weight_cdbg()
    _coverage(other, {"n1": 10.0, "n2": 20.0})
    aggregate_annotations(
        other,
        namespace="sample",
        feature="coverage",
        policy="mean",
        provenance=_provenance(other),
    )
    mean_layer = get_node_annotations(other, namespace="sample", feature="coverage")
    assert float(mean_layer.values[0]) == 15.0

    weighted = _length_weight_cdbg()
    _coverage(weighted, {"n1": 10.0, "n2": 20.0})
    aggregate_annotations(
        weighted,
        namespace="sample",
        feature="coverage",
        policy="weighted_mean",
        provenance=_provenance(weighted),
        weights={"n1": 1.0, "n2": 3.0},
    )
    caller = get_node_annotations(weighted, namespace="sample", feature="coverage")
    assert float(caller.values[0]) == 17.5
    assert (10 * 1 + 20 * 3) / 4 == 17.5


def test_aggregation_does_not_impute_or_mix_categories() -> None:
    """Missing members, category means, and majority ties are errors."""
    partial = _length_weight_cdbg()
    _coverage(partial, {"n1": 10.0})
    with pytest.raises(ContractError, match="refusing to impute"):
        aggregate_annotations(
            partial,
            namespace="sample",
            feature="coverage",
            policy="weighted_mean",
            provenance=_provenance(partial),
        )
    cdbg = cfa_to_cdbg(chain_cfa())
    annotate_cdbg(
        cdbg,
        namespace="genome",
        feature="call",
        target_type="internal_node",
        values={row.cfa_node_id: "a" for row in cdbg.mapping},
        dtype="category",
        provenance=_provenance(cdbg),
    )
    with pytest.raises(ContractError, match="incompatible annotation dtype"):
        aggregate_annotations(
            cdbg,
            namespace="genome",
            feature="call",
            policy="mean",
            provenance=_provenance(cdbg),
        )
    tied = _length_weight_cdbg()
    annotate_cdbg(
        tied,
        namespace="genome",
        feature="call",
        target_type="internal_node",
        values={"n1": "a", "n2": "b"},
        dtype="category",
        provenance=_provenance(tied),
    )
    with pytest.raises(ContractError, match="majority tie"):
        aggregate_annotations(
            tied,
            namespace="genome",
            feature="call",
            policy="majority",
            provenance=_provenance(tied),
        )
    annotate_cdbg(
        cdbg,
        namespace="genome",
        feature="label",
        target_type="internal_node",
        values={
            "n000001": "a",
            "n000002": "a",
            "n000003": "a",
            "n000004": "a",
            "n000005": "b",
            "n000006": "b",
        },
        dtype="category",
        provenance=_provenance(cdbg),
        replace=False,
    )
    kept = get_node_annotations(cdbg, namespace="genome", feature="label", target_type="internal_node")
    before = list(kept.target_ids)
    aggregate_annotations(
        cdbg,
        namespace="genome",
        feature="label",
        policy="keep_per_member",
        provenance=_provenance(cdbg),
        source_target_type="internal_node",
    )
    still = get_node_annotations(cdbg, namespace="genome", feature="label", target_type="internal_node")
    assert still.aggregation == "keep_per_member"
    assert list(still.target_ids) == before
    assert len(still.values) == len(cdbg.mapping)
    aggregate_annotations(
        cdbg,
        namespace="genome",
        feature="label",
        policy="union",
        provenance=_provenance(cdbg),
    )
    union = get_node_annotations(cdbg, namespace="genome", feature="label", target_type="node")
    by_unitig = dict(zip(union.target_ids.tolist(), union.values.tolist()))
    long = next(unitig for unitig in cdbg.unitigs if len(unitig.members) == 3)
    assert by_unitig[long.unitig_id] == "a|b"


def test_provenance_is_required_and_recorded() -> None:
    """A layer without method, version, or parent ids is rejected."""
    cdbg = cfa_to_cdbg(chain_cfa())
    bad = _provenance(cdbg)
    del bad["method"]
    with pytest.raises(ContractError, match="missing provenance field: method"):
        annotate_cdbg(
            cdbg,
            namespace="sample",
            feature="coverage",
            target_type="node",
            values={cdbg.unitigs[0].unitig_id: 1.0},
            dtype="float64",
            provenance=bad,
        )
    annotate_cdbg(
        cdbg,
        namespace="sample",
        feature="coverage",
        target_type="node",
        values={unitig.unitig_id: 1.0 for unitig in cdbg.unitigs},
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    layer = get_node_annotations(cdbg, namespace="sample", feature="coverage")
    assert layer.provenance.source == "unit-test"
    assert layer.provenance.method == "annotate_cdbg"
    assert layer.provenance.version == "test-1"
    assert layer.provenance.parent_graph_id == cdbg.metadata["graph_id"]
    assert layer.provenance.parent_schema_version == "1.0"
    assert layer.provenance.parent_contract_version == "1.0"
    assert isinstance(layer.provenance.parameters, dict)


def test_annotation_sidecar_round_trip(tmp_path) -> None:
    """Layers survive dump/load, and a graph with no layers still has no sidecar."""
    plain = cfa_to_cdbg(chain_cfa())
    plain_dir = tmp_path / "plain"
    dump_cdbg(plain, plain_dir)
    assert not (plain_dir / "annotations").exists()
    reloaded_plain = load_cdbg(plain_dir)
    assert reloaded_plain.annotations == []
    assert [unitig.unitig_id for unitig in reloaded_plain.unitigs] == [unitig.unitig_id for unitig in plain.unitigs]

    cdbg = _length_weight_cdbg()
    _coverage(cdbg, {"n1": 10.0, "n2": 20.0})
    annotate_cdbg(
        cdbg,
        namespace="composition",
        feature="kmer",
        target_type="node",
        values={cdbg.unitigs[0].unitig_id: [1.0, 0.0, 0.0, 0.0]},
        dtype="float64",
        kind="vector",
        provenance=_provenance(cdbg),
    )
    annotate_cdbg(
        cdbg,
        namespace="genome",
        feature="call",
        target_type="internal_node",
        values={"n1": "genome_001", "n2": "genome_002"},
        dtype="category",
        provenance=_provenance(cdbg),
    )
    aggregate_annotations(
        cdbg,
        namespace="sample",
        feature="coverage",
        policy="weighted_mean",
        provenance=_provenance(cdbg),
    )
    directory = tmp_path / "cdbg"
    dump_cdbg(cdbg, directory)
    loaded = load_cdbg(directory)
    assert loaded.metadata["schema_version"] == "1.0"
    coverage = get_node_annotations(loaded, namespace="sample", feature="coverage", target_type="internal_node")
    assert dict(zip(coverage.target_ids.tolist(), coverage.values.tolist())) == {"n1": 10.0, "n2": 20.0}
    summary = get_node_annotations(loaded, namespace="sample", feature="coverage")
    assert summary.aggregation == "weighted_mean"
    assert float(summary.values[0]) == 16.0
    assert summary.provenance.parameters["aggregation"]["policy"] == "weighted_mean"
    vector = get_node_annotations(loaded, namespace="composition", feature="kmer")
    assert vector.values.shape == (1, 4)
    assert np.array_equal(vector.values, np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64))
    category = get_node_annotations(loaded, namespace="genome", feature="call", target_type="internal_node")
    assert dict(zip(category.target_ids.tolist(), category.values.tolist())) == {
        "n1": "genome_001",
        "n2": "genome_002",
    }
    dump_cdbg(cfa_to_cdbg(chain_cfa()), directory)
    assert not (directory / "annotations").exists()
    assert load_cdbg(directory).annotations == []


def test_cgt_alignment_keeps_cfa_nodes_joinable() -> None:
    """Selected rows follow dense ids and CSR order; CFA node rows stay joinable."""
    cfa = chain_cfa()
    expected_nodes = {}
    for index, row in enumerate(cfa.nodes, start=1):
        row["coverage"] = str(index * 10)
        expected_nodes[row["node_id"]] = float(index * 10)
    for row in cfa.edges:
        row["coverage"] = str(float(int(row["edge_id"].lstrip("e"))))
    cdbg = cfa_to_cdbg(cfa)
    transfer_annotations(
        cdbg,
        cfa,
        namespace="copied",
        provenance=_provenance(cdbg),
        node_columns=["coverage"],
        edge_columns=["coverage"],
    )
    annotate_cdbg(
        cdbg,
        namespace="unit",
        feature="marker",
        target_type="node",
        values={unitig.unitig_id: float(index) for index, unitig in enumerate(cdbg.unitigs, start=1)},
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    link_values = {link.link_id: float(int(link.link_id.lstrip("e"))) for link in cdbg.links}
    explicit = {unitig.unitig_id: [1.0] for unitig in cdbg.unitigs}
    cgt = cdbg_to_cgt(
        cdbg,
        node_features=explicit,
        node_annotation=[("unit", "marker")],
        edge_annotation=[("copied", "coverage")],
    )
    assert list(cgt.metadata["node_feature_names"]) == ["f0", "unit:marker"]
    assert list(cgt.metadata["edge_feature_names"]) == ["copied:coverage"]
    for row, features in zip(cgt.mapping, cgt.node_features):
        assert features[0] == np.float32(1.0)
        source = next(unitig for unitig in cdbg.unitigs if unitig.unitig_id == row["source_id"])
        expected_marker = float(cdbg.unitigs.index(source) + 1)
        assert features[1] == np.float32(expected_marker)
    recovered = cgt_edge_cfa_ids(cdbg)
    assert list(cgt.edge_features[:, 0]) == [np.float32(link_values[edge_id]) for edge_id in recovered]
    internal_values = {
        edge_id
        for unitig in cdbg.unitigs
        for edge_id in internal_cfa_edge_ids(cdbg, unitig.unitig_id)
    }
    assert internal_values.isdisjoint(recovered)
    assert cgt.num_edges == len(cdbg.links)
    for dense_id in range(cgt.num_nodes):
        lineage = node_lineage(cgt, dense_id)
        layer = get_node_annotations(cdbg, namespace="copied", feature="coverage", target_type="internal_node")
        found = dict(zip(layer.target_ids.tolist(), [float(value) for value in layer.values.tolist()]))
        for node_id in lineage.cfa_node_ids:
            assert found[node_id] == expected_nodes[node_id]
    annotate_cdbg(
        cdbg,
        namespace="copied",
        feature="junction",
        target_type="internal_edge",
        values={
            edge_id: 999.0
            for unitig in cdbg.unitigs
            for edge_id in internal_cfa_edge_ids(cdbg, unitig.unitig_id)
        },
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    linked = cdbg_to_cgt(cdbg, edge_annotation=[("copied", "coverage")])
    assert np.float32(999.0) not in set(linked.edge_features[:, 0].tolist())
    with pytest.raises(ContractError, match="internal unitig edges are not CSR edges"):
        cdbg_to_cgt(cdbg, edge_annotation=[("copied", "junction")])


def test_missing_annotation_target() -> None:
    """An id that is not on the graph is rejected."""
    cdbg = cfa_to_cdbg(chain_cfa())
    with pytest.raises(ContractError, match="missing annotation target: absent"):
        annotate_cdbg(
            cdbg,
            namespace="sample",
            feature="coverage",
            target_type="node",
            values={"absent": 1.0},
            dtype="float64",
            provenance=_provenance(cdbg),
        )
    assert cdbg.annotations == []


def test_incompatible_annotation_dtype() -> None:
    """A string scalar and a vector stored as a scalar are rejected."""
    cdbg = cfa_to_cdbg(chain_cfa())
    unitig_id = cdbg.unitigs[0].unitig_id
    with pytest.raises(ContractError, match="incompatible annotation dtype"):
        annotate_cdbg(
            cdbg,
            namespace="sample",
            feature="coverage",
            target_type="node",
            values={unitig_id: "high"},
            dtype="float64",
            provenance=_provenance(cdbg),
        )
    with pytest.raises(ContractError, match="incompatible annotation dtype"):
        annotate_cdbg(
            cdbg,
            namespace="sample",
            feature="coverage",
            target_type="node",
            values={unitig_id: [1.0, 2.0]},
            dtype="float64",
            kind="scalar",
            provenance=_provenance(cdbg),
        )
    assert cdbg.annotations == []


def test_annotation_overwrite_requires_replace() -> None:
    """A second write of the same layer fails unless replace is true."""
    cdbg = cfa_to_cdbg(chain_cfa())
    values = {unitig.unitig_id: 1.0 for unitig in cdbg.unitigs}
    annotate_cdbg(
        cdbg,
        namespace="sample",
        feature="coverage",
        target_type="node",
        values=values,
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    replaced = {unitig.unitig_id: 2.0 for unitig in cdbg.unitigs}
    with pytest.raises(ContractError, match="replace=True"):
        annotate_cdbg(
            cdbg,
            namespace="sample",
            feature="coverage",
            target_type="node",
            values=replaced,
            dtype="float64",
            provenance=_provenance(cdbg),
        )
    current = get_node_annotations(cdbg, namespace="sample", feature="coverage")
    assert set(current.values.tolist()) == {1.0}
    annotate_cdbg(
        cdbg,
        namespace="sample",
        feature="coverage",
        target_type="node",
        values=replaced,
        dtype="float64",
        provenance=_provenance(cdbg),
        replace=True,
    )
    updated = get_node_annotations(cdbg, namespace="sample", feature="coverage")
    assert set(updated.values.tolist()) == {2.0}


def test_int64_overflow_and_sidecar_name_collision() -> None:
    cdbg = cfa_to_cdbg(chain_cfa())
    unitig_id = cdbg.unitigs[0].unitig_id
    with pytest.raises(ContractError, match="does not fit in int64"):
        annotate_cdbg(
            cdbg,
            namespace="sample",
            feature="count",
            target_type="node",
            values={unitig_id: 2**63},
            dtype="int64",
            provenance=_provenance(cdbg),
        )
    annotate_cdbg(
        cdbg,
        namespace="a__b",
        feature="c",
        target_type="node",
        values={unitig_id: 1},
        dtype="int64",
        provenance=_provenance(cdbg),
    )
    annotate_cdbg(
        cdbg,
        namespace="a",
        feature="b__c",
        target_type="node",
        values={unitig_id: 2},
        dtype="int64",
        provenance=_provenance(cdbg),
    )
    with pytest.raises(ContractError, match="shared by two layers"):
        dump_cdbg(cdbg, "/tmp/metametro-sidecar-collision")
