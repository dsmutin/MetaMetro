#!/usr/bin/env python3
"""Annotate a CDBG sidecar and copy selected layers into a CGT.

The arithmetic in this script is a fixture check, not a biological result.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metametro.converters.cdbg_to_cgt import cdbg_to_cgt  # noqa: E402
from metametro.converters.cfa_to_cdbg import cfa_to_cdbg  # noqa: E402
from metametro.fixtures import chain_cfa  # noqa: E402
from metametro.formats.cdbg.annotations import (  # noqa: E402
    aggregate_annotations,
    annotate_cdbg,
    get_node_annotations,
    transfer_annotations,
)
from metametro.formats.cfa.model import CfaGraph  # noqa: E402
from metametro.identity import cgt_edge_cfa_ids, node_lineage  # noqa: E402


def _provenance(cdbg, method: str) -> dict:
    """Provenance recorded on every layer. The version is the package VERSION file."""
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    return {
        "source": "examples/cdbg_annotation",
        "method": method,
        "version": version,
        "parameters": {},
        "parent_graph_id": cdbg.metadata["graph_id"],
        "parent_schema_version": cdbg.metadata["schema_version"],
        "parent_contract_version": cdbg.metadata["contract_version"],
    }


def _length_weight_graph():
    """Repeat-graph chain whose member lengths are 4 and 6."""
    graph = CfaGraph(
        metadata={
            "schema_version": "1.0",
            "graph_id": "length_weights",
            "graph_type": "repeat",
            "features": {"node": {}, "edge": {"orientation": "orientation", "overlap": "int"}},
        },
        sequences={"n1": "ACGT", "n2": "GTACCC"},
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
    return cfa_to_cdbg(graph)


def demonstrate_weighted_mean() -> None:
    """Show that 16 appears only after an explicit weighted_mean."""
    cdbg = _length_weight_graph()
    annotate_cdbg(
        cdbg,
        namespace="sample",
        feature="coverage",
        target_type="internal_node",
        values={"n1": 10.0, "n2": 20.0},
        dtype="float64",
        provenance=_provenance(cdbg, "annotate_cdbg"),
    )
    aggregate_annotations(
        cdbg,
        namespace="sample",
        feature="coverage",
        policy="weighted_mean",
        provenance=_provenance(cdbg, "aggregate_annotations"),
    )
    summary = get_node_annotations(cdbg, namespace="sample", feature="coverage", target_type="node")
    members = get_node_annotations(cdbg, namespace="sample", feature="coverage", target_type="internal_node")
    assert float(summary.values[0]) == 16.0
    assert dict(zip(members.target_ids.tolist(), members.values.tolist())) == {"n1": 10.0, "n2": 20.0}
    print("weighted_mean coverage", float(summary.values[0]))
    print("per-member coverage kept", dict(zip(members.target_ids.tolist(), members.values.tolist())))


def demonstrate_cgt_selection() -> None:
    """Select a unitig marker and a link feature; keep CFA node coverage joinable."""
    cfa = chain_cfa()
    expected = {}
    for index, row in enumerate(cfa.nodes, start=1):
        row["coverage"] = str(index * 10)
        expected[row["node_id"]] = float(index * 10)
    for row in cfa.edges:
        row["coverage"] = str(float(int(row["edge_id"].lstrip("e"))))
    cdbg = cfa_to_cdbg(cfa)
    transfer_annotations(
        cdbg,
        cfa,
        namespace="copied",
        provenance=_provenance(cdbg, "transfer_annotations"),
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
        provenance=_provenance(cdbg, "annotate_cdbg"),
    )
    cgt = cdbg_to_cgt(
        cdbg,
        node_annotation=[("unit", "marker")],
        edge_annotation=[("copied", "coverage")],
    )
    assert list(cgt.metadata["node_feature_names"]) == ["unit:marker"]
    assert cgt.node_features.shape == (len(cdbg.unitigs), 1)
    assert cgt.edge_features.shape[0] == len(cdbg.links)
    for edge_id, value in zip(cgt_edge_cfa_ids(cdbg), cgt.edge_features[:, 0]):
        assert value == float(int(edge_id.lstrip("e")))
    for dense_id in range(cgt.num_nodes):
        lineage = node_lineage(cgt, dense_id)
        layer = get_node_annotations(cdbg, namespace="copied", feature="coverage", target_type="internal_node")
        found = dict(zip(layer.target_ids.tolist(), [float(item) for item in layer.values.tolist()]))
        for node_id in lineage.cfa_node_ids:
            assert found[node_id] == expected[node_id]
    print("CGT nodes", cgt.num_nodes, "CGT edges", cgt.num_edges)
    print("node features", list(cgt.metadata["node_feature_names"]))
    print("edge features", list(cgt.metadata["edge_feature_names"]))
    print("CFA node coverage remains on the sidecar, joined by node_lineage")


def main() -> int:
    """Run both checks."""
    demonstrate_weighted_mean()
    demonstrate_cgt_selection()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
