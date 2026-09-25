"""Mandatory pipeline-contract checks on the shared mock."""

from __future__ import annotations

import numpy as np
import pytest

from metametro.contracts.assembly import dbg_from_sequences, read_fastq, simulate_metagenome
from metametro.contracts.colouring import colour_by_reads, colour_cfa, combine_colors
from metametro.contracts.ds import run_ds
from metametro.contracts.external import megahit_command, samovar_generate_command
from metametro.contracts.versions import CONTRACTS
from metametro.converters.cdbg_to_cgt import cdbg_to_cgt
from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
from metametro.errors import ContractError
from metametro.fixtures import mock_cfa, mock_cgt, synthetic_genomes
from metametro.formats.cfa.model import CfaGraph

pytestmark = pytest.mark.mandatory


def test_contract_versions_are_pinned() -> None:
    """Each pipeline contract has its own 1.0 pin."""
    assert set(CONTRACTS) == {
        "genome_to_metagenome",
        "metagenome_to_graph",
        "graph_to_cfa",
        "colour_cfa",
        "cfa_to_cdbg",
        "cdbg_to_cgt",
        "ds_on_cgt",
        "filter_colours",
    }
    assert set(CONTRACTS.values()) == {"1.0"}


def test_metagenome_keeps_genome_provenance(tmp_path) -> None:
    """Reads point at a source genome, including both abundance levels."""
    genomes = synthetic_genomes()
    paths = simulate_metagenome(
        genomes,
        [("sample_1", "genome_001", 4), ("sample_2", "genome_002", 2)],
        tmp_path,
        read_length=16,
        seed=1,
    )
    from metametro.tables import read_tsv

    _, rows = read_tsv(paths["metadata"])
    assert {row["genome_id"] for row in rows} == {"genome_001", "genome_002"}
    assert all(row["ambiguous"] == "no" for row in rows)
    assert len(read_fastq(paths["reads"])) == 6
    shared = "ACGTACGT"
    assert shared in genomes["genome_001"] and shared in genomes["genome_002"]


def test_dbg_from_reads_preserves_sequences(tmp_path) -> None:
    """Every k-mer of the reads is a node and successive k-mers are edges."""
    genomes = synthetic_genomes()
    simulate_metagenome(
        genomes,
        [("sample_1", "genome_001", 2)],
        tmp_path,
        read_length=16,
        seed=1,
    )
    graph = dbg_from_sequences(read_fastq(tmp_path / "reads.fastq"), k=8, graph_id="toy")
    assert graph.metadata["graph_type"] == "de_bruijn"
    assert all(len(sequence) == 8 for sequence in graph.sequences.values())
    assert graph.edges


def test_colour_threshold_and_topology() -> None:
    """Vertex depth >= 1 and edge k-mer density >= 2. Sequences stay put."""
    graph = CfaGraph(
        metadata={"schema_version": "1.0", "graph_id": "colour", "graph_type": "de_bruijn", "k": 4},
        sequences={"n000001": "ATGC", "n000002": "TGCA"},
        nodes=[{"node_id": "n000001"}, {"node_id": "n000002"}],
        edges=[{"edge_id": "e000001", "source": "n000001", "target": "n000002"}],
        node_header=["node_id"],
        edge_header=["edge_id", "source", "target"],
    )
    coloured = colour_by_reads(
        graph,
        [
            ("r1", "sample_1", "ATGCA"),
            ("r2", "sample_1", "ATGCA"),
            ("r3", "sample_2", "ATGC"),
        ],
        ["sample_1", "sample_2"],
    )
    assert coloured.sequences == graph.sequences
    assert [(row["source"], row["target"]) for row in coloured.edges] == [
        (row["source"], row["target"]) for row in graph.edges
    ]
    nodes = {row["node_id"]: row["color_set"] for row in coloured.nodes}
    edges = {row["edge_id"]: row["color_set"] for row in coloured.edges}
    assert nodes["n000001"] == "0,1"
    assert nodes["n000002"] == "0"
    assert edges["e000001"] == "0"


def test_colour_operations_are_explicit() -> None:
    """Replace, merge, intersect, and subtract do not invent a default mix."""
    assert combine_colors([0], [1], "merge") == [0, 1]
    assert combine_colors([0, 1], [1], "intersect") == [1]
    assert combine_colors([0, 1], [1], "subtract") == [0]
    graph = mock_cfa()
    replaced = colour_cfa(
        graph,
        {"n000001": [1], "n000002": [1], "n000003": [1], "n000004": [1]},
        operation="replace",
        target=("node",),
    )
    assert replaced.sequences == graph.sequences
    assert {row["node_id"]: row["color_set"] for row in replaced.nodes}["n000002"] == "1"
    with pytest.raises(ContractError, match="colour_operation"):
        combine_colors([0], [1], "overwrite")


def test_ds_is_deterministic_and_addressable() -> None:
    """Predictions match the input object count and do not mutate the CGT."""
    graph = mock_cgt()
    features = np.array(graph.node_features, copy=True)
    first = run_ds(graph, seed=0, epochs=5)
    second = run_ds(graph, seed=0, epochs=5)
    assert len(first["result"]) == graph.num_nodes
    assert [row["dense_id"] for row in first["result"]] == [0, 1, 2, 3]
    assert [row["predicted_label"] for row in first["result"]] == [
        row["predicted_label"] for row in second["result"]
    ]
    assert np.array_equal(graph.node_features, features)
    assert first["result"][0]["cfa_node_ids"] == ["n000001"]


def test_external_baselines_have_stable_commands(tmp_path) -> None:
    """Samovar ISS and MEGAHIT are the Contract 1 and Contract 2 baselines."""
    samovar = samovar_generate_command(tmp_path / "genomes", tmp_path / "out", total_reads=50)
    assert samovar[:2] == ["samovar", "generate"]
    assert "--genome_dir" in samovar
    megahit = megahit_command(tmp_path / "reads.fastq", tmp_path / "asm", k=21)
    assert megahit[0] == "megahit"
    assert "--k-min" in megahit and "21" in megahit
    with pytest.raises(ValueError):
        megahit_command(tmp_path / "reads.fastq", tmp_path / "asm", k=20)


def test_fastg_keeps_bubbles_and_oriented_edges(tmp_path) -> None:
    """MEGAHIT FASTG edges overlap by assembly k and keep a branch that reconverges."""
    from metametro.contracts.assembly import directed_bubble_sources, fastg_to_cfa, strand_junction_counts
    from metametro.converters.cfa_to_cdbg import cfa_to_cdbg

    text = """\
>NODE_1_length_6_cov_2.0000_ID_1:NODE_2_length_9_cov_1.0000_ID_3,NODE_3_length_9_cov_1.0000_ID_5;
ACGTAA
>NODE_1_length_6_cov_2.0000_ID_1';
TTACGT
>NODE_2_length_9_cov_1.0000_ID_3:NODE_4_length_7_cov_3.0000_ID_7;
TAATTTGAT
>NODE_2_length_9_cov_1.0000_ID_3';
ATCAAATTA
>NODE_3_length_9_cov_1.0000_ID_5:NODE_4_length_7_cov_3.0000_ID_7;
TAAACCGAT
>NODE_3_length_9_cov_1.0000_ID_5';
ATCGGTTTA
>NODE_4_length_7_cov_3.0000_ID_7;
GATACGT
>NODE_4_length_7_cov_3.0000_ID_7':NODE_2_length_9_cov_1.0000_ID_3',NODE_3_length_9_cov_1.0000_ID_5';
ACGTATC
"""
    path = tmp_path / "k3.fastg"
    path.write_text(text, encoding="utf-8")
    graph = fastg_to_cfa(path, k=3, graph_id="bubble")
    assert graph.metadata["k"] == 4
    assert graph.metadata["source"]["assembly_k"] == 3
    assert graph.sequences["n000001"] == "ACGTAA"
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 6
    assert "n000001" in directed_bubble_sources(graph.edges)
    assert strand_junction_counts(graph.edges)["branch_nodes"] >= 1
    forward = [
        (row["source"], row["target"])
        for row in graph.edges
        if row["orientation"] == "++"
    ]
    assert ("n000001", "n000003") in forward
    assert ("n000001", "n000005") in forward
    assert ("n000003", "n000007") in forward
    cdbg = cfa_to_cdbg(graph)
    assert len(cdbg.unitigs) == 4
    assert len(cdbg.links) == 6


def test_bubble_requires_reconvergence() -> None:
    """Two outs are a branch. They are a bubble only after the paths meet."""
    from metametro.contracts.assembly import directed_bubble_sources

    branch = [
        {"source": "a", "target": "b", "orientation": "++"},
        {"source": "a", "target": "c", "orientation": "++"},
    ]
    assert directed_bubble_sources(branch) == []
    bubble = branch + [
        {"source": "b", "target": "d", "orientation": "++"},
        {"source": "c", "target": "d", "orientation": "++"},
    ]
    assert directed_bubble_sources(bubble) == ["a"]
    parallel = [
        {"source": "a", "target": "b", "orientation": "++"},
        {"source": "a", "target": "b", "orientation": "+-"},
    ]
    assert directed_bubble_sources(parallel) == []


def test_reverse_complement_read_covers_node() -> None:
    """An unstranded read covers a node from either strand, and counts once."""
    graph = CfaGraph(
        metadata={"schema_version": "1.0", "graph_id": "rc", "graph_type": "de_bruijn", "k": 4},
        sequences={"n000001": "ATGC"},
        nodes=[{"node_id": "n000001"}],
        edges=[],
        node_header=["node_id"],
        edge_header=["edge_id", "source", "target"],
    )
    coloured = colour_by_reads(graph, [("r1", "sample_1", "GCAT")], ["sample_1"])
    assert coloured.nodes[0]["color_set"] == "0"
    both = colour_by_reads(
        graph,
        [("r1", "sample_1", "ATGC"), ("r1b", "sample_1", "GCAT")],
        ["sample_1"],
        min_vertex_depth=2,
    )
    assert both.nodes[0]["color_set"] == "0"


def test_oriented_junction_uses_reverse_complement() -> None:
    """A '-' endpoint is reverse-complemented before the (k+1)-mer is counted."""
    graph = CfaGraph(
        metadata={
            "schema_version": "1.0",
            "graph_id": "orient",
            "graph_type": "de_bruijn",
            "k": 4,
            "features": {"edge": {"orientation": "orientation"}},
        },
        sequences={"n000001": "ATGC", "n000002": "GCAT"},
        nodes=[{"node_id": "n000001"}, {"node_id": "n000002"}],
        edges=[
            {
                "edge_id": "e000001",
                "source": "n000001",
                "target": "n000002",
                "orientation": "+-",
            }
        ],
        node_header=["node_id"],
        edge_header=["edge_id", "source", "target", "orientation"],
    )
    # Forward target is GCAT; reverse complement is ATGC. Junction is ATGC + C.
    coloured = colour_by_reads(
        graph,
        [("r1", "sample_1", "ATGCC"), ("r2", "sample_1", "ATGCC")],
        ["sample_1"],
    )
    assert coloured.edges[0]["color_set"] == "0"


def test_contig_overlap_keeps_full_sequence() -> None:
    """An overlap edge is added only when the (k-1) bases match."""
    from metametro.contracts.assembly import contig_overlap_graph

    graph = contig_overlap_graph(
        [("a", "AACGTT"), ("b", "GTTAAA"), ("c", "CCCCCC")],
        k=3,
        graph_id="ov",
    )
    assert graph.sequences["n000001"] == "AACGTT"
    assert [(row["source"], row["target"]) for row in graph.edges] == [("n000002", "n000001")]


def test_end_to_end_mock_pipeline(tmp_path) -> None:
    """Genomes through a dummy DS stay aligned on one synthetic metagenome."""
    genomes = synthetic_genomes()
    simulate_metagenome(
        genomes,
        [("sample_1", "genome_001", 3), ("sample_2", "genome_002", 3)],
        tmp_path,
        read_length=16,
        seed=2,
    )
    from metametro.tables import read_tsv

    reads = read_fastq(tmp_path / "reads.fastq")
    _, meta = read_tsv(tmp_path / "metadata.tsv")
    sequence = {read_id: seq for read_id, seq in reads}
    graph = dbg_from_sequences(reads, k=8, graph_id="synthetic")
    coloured = colour_by_reads(
        graph,
        [(row["read_id"], row["sample_id"], sequence[row["read_id"]]) for row in meta],
        ["sample_1", "sample_2"],
    )
    assert coloured.sequences == graph.sequences
    cdbg = cfa_to_cdbg(coloured)
    assert {row.cfa_node_id for row in cdbg.mapping} == set(graph.node_ids())
    cgt = cdbg_to_cgt(cdbg)
    cgt.node_labels = np.zeros(cgt.num_nodes, dtype=np.int64)
    from metametro.formats.cgt.validator import validate_cgt

    validate_cgt(cgt)
    result = run_ds(cgt, epochs=2, seed=1)
    assert result["metadata"]["num_predictions"] == cgt.num_nodes
    assert {node for row in result["result"] for node in row["cfa_node_ids"]} == set(graph.node_ids())
