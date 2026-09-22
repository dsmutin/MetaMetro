"""Hand-built graphs that every contract test shares.

``mock_cfa`` is the branching bubble. ``chain_cfa`` is the compaction example
(6 CFA nodes, 3 unitigs, 4 links). ``synthetic_genomes`` is the Contract 1
mock with one shared interval and two private flanks.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from metametro.contracts.assembly import dbg_from_sequences, simulate_metagenome
from metametro.contracts.colouring import colour_by_reads
from metametro.converters.cdbg_to_cgt import cdbg_to_cgt
from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
from metametro.formats.cfa.io import dump_cfa
from metametro.formats.cfa.model import CfaGraph
from metametro.formats.cdbg.io import dump_cdbg
from metametro.formats.cdbg.model import Cdbg
from metametro.formats.cgt.io import dump_cgt
from metametro.formats.cgt.model import Cgt


def _gc(sequence: str) -> str:
    return f"{(sequence.count('G') + sequence.count('C')) / len(sequence):.2f}"


def _entropy(sequence: str) -> str:
    counts = [sequence.count(base) for base in "ACGTN"]
    total = sum(counts)
    value = 0.0
    for count in counts:
        if count:
            probability = count / total
            value -= probability * math.log2(probability)
    return f"{value:.2f}"


def _graph(
    *,
    graph_id: str,
    k: int,
    sequences: dict[str, str],
    nodes: list[dict[str, str]],
    edges: list[dict[str, str]],
    node_header: list[str],
    edge_header: list[str],
) -> CfaGraph:
    return CfaGraph(
        metadata={
            "schema_version": "1.0",
            "graph_id": graph_id,
            "graph_type": "de_bruijn",
            "k": k,
            "features": {
                "node": {
                    "coverage": "float",
                    "gc": "float",
                    "entropy": "float",
                    "color_set": "color_set",
                    "label": "int",
                },
                "edge": {
                    "orientation": "orientation",
                    "coverage": "float",
                    "support": "float",
                    "color_set": "color_set",
                    "label": "int",
                },
            },
            "source": {"samples": 2},
        },
        sequences=sequences,
        nodes=nodes,
        edges=edges,
        colors=[
            {"color_id": "0", "namespace": "sample", "value": "sample_1"},
            {"color_id": "1", "namespace": "sample", "value": "sample_2"},
        ],
        labels=[
            {"label_id": "0", "namespace": "genome", "value": "unclassified"},
            {"label_id": "1", "namespace": "genome", "value": "genome_001"},
            {"label_id": "2", "namespace": "genome", "value": "genome_002"},
        ],
        node_header=node_header,
        edge_header=edge_header,
    )


_NODE_HEADER = ["node_id", "coverage", "gc", "entropy", "color_set", "label"]
_EDGE_HEADER = ["edge_id", "source", "target", "orientation", "coverage", "support", "color_set", "label"]


def _node(node_id: str, sequence: str, coverage: str, color_set: str, label: str) -> dict[str, str]:
    return {
        "node_id": node_id,
        "coverage": coverage,
        "gc": _gc(sequence),
        "entropy": _entropy(sequence),
        "color_set": color_set,
        "label": label,
    }


def _edge(
    edge_id: str,
    source: str,
    target: str,
    coverage: str,
    support: str,
    color_set: str,
    label: str,
) -> dict[str, str]:
    return {
        "edge_id": edge_id,
        "source": source,
        "target": target,
        "orientation": "++",
        "coverage": coverage,
        "support": support,
        "color_set": color_set,
        "label": label,
    }


def mock_cfa() -> CfaGraph:
    """Four-node bubble: one branch, one join, one multi-colour node, one unclassified label."""
    sequences = {
        "n000001": "ACGT",
        "n000002": "GTAAC",
        "n000003": "GTTAC",
        "n000004": "ACGT",
    }
    nodes = [
        _node("n000001", sequences["n000001"], "12.4", "0", "1"),
        _node("n000002", sequences["n000002"], "18.1", "0,1", "1"),
        _node("n000003", sequences["n000003"], "7.2", "1", "2"),
        _node("n000004", sequences["n000004"], "9.0", "0", "0"),
    ]
    edges = [
        _edge("e000001", "n000001", "n000002", "12.1", "0.80", "0", "1"),
        _edge("e000002", "n000001", "n000003", "4.0", "0.20", "0,1", "0"),
        _edge("e000003", "n000002", "n000004", "17.8", "0.90", "0", "1"),
        _edge("e000004", "n000003", "n000004", "8.4", "0.40", "1", "2"),
        _edge("e000005", "n000004", "n000002", "3.2", "0.10", "0,1", "0"),
    ]
    return _graph(
        graph_id="mock_bubble",
        k=3,
        sequences=sequences,
        nodes=nodes,
        edges=edges,
        node_header=_NODE_HEADER,
        edge_header=_EDGE_HEADER,
    )


def chain_cfa() -> CfaGraph:
    """Six-node graph that compacts to three unitigs and four links."""
    sequences = {
        "n000001": "CCGT",
        "n000002": "GTAC",
        "n000003": "ACGA",
        "n000004": "GATG",
        "n000005": "TGAC",
        "n000006": "ACGGAC",
    }
    colors = ["0", "0,1", "1", "1", "0", "1"]
    labels = ["1", "1", "2", "2", "0", "2"]
    nodes = [
        _node(node_id, sequences[node_id], "1.0", color, label)
        for node_id, color, label in zip(sequences, colors, labels)
    ]
    pairs = [
        ("e000001", "n000001", "n000002", "0"),
        ("e000002", "n000002", "n000003", "0,1"),
        ("e000003", "n000002", "n000006", "1"),
        ("e000004", "n000003", "n000004", "1"),
        ("e000005", "n000004", "n000005", "1"),
        ("e000006", "n000005", "n000006", "0"),
        ("e000007", "n000006", "n000003", "1"),
    ]
    edges = [
        _edge(edge_id, source, target, "1.0", "1.0", color, "0")
        for edge_id, source, target, color in pairs
    ]
    return _graph(
        graph_id="mock_chain",
        k=3,
        sequences=sequences,
        nodes=nodes,
        edges=edges,
        node_header=_NODE_HEADER,
        edge_header=_EDGE_HEADER,
    )


def mock_cdbg() -> Cdbg:
    """CDBG of the bubble. Compaction does not merge junction nodes."""
    return cfa_to_cdbg(mock_cfa())


def chain_cdbg() -> Cdbg:
    """CDBG of the chain fixture."""
    return cfa_to_cdbg(chain_cfa())


def mock_cgt() -> Cgt:
    """CGT of the bubble, with features aligned through unitig ids."""
    cfa = mock_cfa()
    cdbg = cfa_to_cdbg(cfa)
    unitigs = sorted(cdbg.unitigs, key=lambda unitig: unitig.unitig_id)
    node_ids = [unitig.members[0] for unitig in unitigs]
    node_index = {row["node_id"]: row for row in cfa.nodes}
    node_features = np.asarray(
        [
            [float(node_index[node_id][column]) for column in ("coverage", "gc", "entropy")]
            for node_id in node_ids
        ],
        dtype=np.float32,
    )
    node_labels = np.asarray([int(node_index[node_id]["label"]) for node_id in node_ids], dtype=np.int64)
    edge_index = {row["edge_id"]: row for row in cfa.edges}
    link_ids = [link.link_id for link in cdbg.links]
    edge_features = np.asarray(
        [
            [float(edge_index[link_id][column]) for column in ("coverage", "support")]
            for link_id in link_ids
        ],
        dtype=np.float32,
    )
    edge_labels = np.asarray([int(edge_index[link_id]["label"]) for link_id in link_ids], dtype=np.int64)
    return cdbg_to_cgt(
        cdbg,
        node_features=node_features,
        edge_features=edge_features,
        node_labels=node_labels,
        edge_labels=edge_labels,
        node_feature_names=["coverage", "gc", "entropy"],
        edge_feature_names=["coverage", "support"],
    )


def synthetic_genomes() -> dict[str, str]:
    """Two genomes that share a 16-mer and differ in both flanks."""
    shared = "ACGTACGTACGTACGT"
    return {
        "genome_001": "AAAATTTT" + shared + "CCCCGGGG",
        "genome_002": "TTTTAAAA" + shared + "GGGGCCCC",
    }


def write_minimal_metagenome(root: str | Path) -> Path:
    """Write the on-disk contract fixture."""
    destination = Path(root)
    genomes = synthetic_genomes()
    genome_dir = destination / "genomes"
    genome_dir.mkdir(parents=True, exist_ok=True)
    for genome_id, sequence in genomes.items():
        (genome_dir / f"{genome_id}.fna").write_text(f">{genome_id}\n{sequence}\n", encoding="utf-8")
    simulate_metagenome(
        genomes,
        [
            ("sample_1", "genome_001", 4),
            ("sample_1", "genome_002", 1),
            ("sample_2", "genome_001", 2),
            ("sample_2", "genome_002", 4),
        ],
        destination / "reads",
        read_length=16,
        seed=1,
    )
    from metametro.contracts.assembly import read_fastq
    from metametro.tables import read_tsv

    graph = dbg_from_sequences(read_fastq(destination / "reads" / "reads.fastq"), k=8, graph_id="synthetic")
    dump_cfa(graph, destination / "graph")
    _, meta_rows = read_tsv(destination / "reads" / "metadata.tsv")
    sequences = dict(read_fastq(destination / "reads" / "reads.fastq"))
    coloured = colour_by_reads(
        graph,
        [(row["read_id"], row["sample_id"], sequences[row["read_id"]]) for row in meta_rows],
        ["sample_1", "sample_2"],
        min_vertex_depth=1,
        min_edge_kmer_density=2,
        operation="replace",
    )
    dump_cfa(coloured, destination / "coloured_cfa")
    cdbg = cfa_to_cdbg(coloured)
    dump_cdbg(cdbg, destination / "cdbg")
    cgt = cdbg_to_cgt(cdbg, node_feature_names=[], edge_feature_names=[])
    dump_cgt(cgt, destination / "cgt")
    dump_cfa(mock_cfa(), destination / "bubble" / "cfa")
    dump_cdbg(mock_cdbg(), destination / "bubble" / "cdbg")
    dump_cgt(mock_cgt(), destination / "bubble" / "cgt")
    dump_cfa(chain_cfa(), destination / "chain" / "cfa")
    dump_cdbg(chain_cdbg(), destination / "chain" / "cdbg")
    return destination
