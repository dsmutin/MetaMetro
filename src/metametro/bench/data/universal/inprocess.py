"""In-process benchmark graphs. No download and no external assembler."""

from __future__ import annotations

from pathlib import Path

from metametro.bench.colourings import BuildContext
from metametro.bench.data.universal.bubbles import (
    error_bubble,
    nested_bubbles,
    shared_node,
    strain_bubble,
    synthetic_bubbles,
)
from metametro.contracts.assembly import dbg_from_sequences, read_fastq, simulate_metagenome
from metametro.fixtures import chain_cfa, mock_cfa, synthetic_genomes
from metametro.formats.cfa.model import CfaGraph
from metametro.tables import read_tsv, write_tsv


def _truth(outdir: Path, rows: list[dict[str, str]]) -> None:
    header = ["bubble_id", "type", "strain_ids", "source_id", "sink_id", "branch_ids", "expected_action"]
    destination = outdir / "ground_truth" / "bubbles.tsv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(destination, header, rows)


def _one(bubble_id: str, action: str, strains: str, source: str, sink: str, branches: str) -> dict[str, str]:
    return {
        "bubble_id": bubble_id,
        "type": "strain" if action == "retain" else "error",
        "strain_ids": strains,
        "source_id": source,
        "sink_id": sink,
        "branch_ids": branches,
        "expected_action": action,
    }


def prepare_bubble_strain_2(outdir: Path, _ctx: BuildContext) -> CfaGraph:
    """Two taxa and one retained bubble."""
    _truth(outdir, [_one("B1", "retain", "taxon_1,taxon_2", "S", "T", "A,B")])
    return strain_bubble()


def prepare_bubble_error_1(outdir: Path, _ctx: BuildContext) -> CfaGraph:
    """One taxon and one error bubble."""
    _truth(outdir, [_one("B1", "pop", "taxon_2", "S", "T", "A,E")])
    return error_bubble()


def prepare_bubble_nested_2(outdir: Path, _ctx: BuildContext) -> CfaGraph:
    """Two taxa and a nested bubble."""
    _truth(
        outdir,
        [
            _one("outer", "retain", "taxon_1,taxon_2", "S", "T", "A,B"),
            _one("inner", "retain", "taxon_2", "U", "V", "P,Q"),
        ],
    )
    return nested_bubbles()


def prepare_bubble_shared_2(outdir: Path, _ctx: BuildContext) -> CfaGraph:
    """Two taxa sharing one node and no bubble."""
    destination = outdir / "ground_truth" / "bubbles.tsv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(
        destination,
        ["bubble_id", "type", "strain_ids", "source_id", "sink_id", "branch_ids", "expected_action"],
        [],
    )
    return shared_node()


def prepare_bubble_strain_3_n50(outdir: Path, _ctx: BuildContext) -> CfaGraph:
    """Three strains and 50 disjoint bubbles (25 strain, 25 error), seed 42."""
    graph, truth = synthetic_bubbles()
    _truth(outdir, truth)
    return graph


def prepare_mock_bubble(_outdir: Path, _ctx: BuildContext) -> CfaGraph:
    """The shared four-node MetaMetro bubble fixture."""
    return mock_cfa()


def prepare_mock_chain(_outdir: Path, _ctx: BuildContext) -> CfaGraph:
    """The shared six-node compaction fixture."""
    return chain_cfa()


def prepare_synthetic_reads_2(outdir: Path, ctx: BuildContext) -> CfaGraph:
    """Two short genomes, in-process reads, and a de Bruijn graph at k=8."""
    genomes = synthetic_genomes()
    genome_dir = outdir / "genomes"
    genome_dir.mkdir(parents=True, exist_ok=True)
    for genome_id, sequence in genomes.items():
        (genome_dir / f"{genome_id}.fna").write_text(f">{genome_id}\n{sequence}\n", encoding="utf-8")
    paths = simulate_metagenome(
        genomes,
        [
            ("sample_1", "genome_001", 4),
            ("sample_1", "genome_002", 1),
            ("sample_2", "genome_001", 2),
            ("sample_2", "genome_002", 4),
        ],
        outdir / "reads",
        read_length=16,
        seed=1,
    )
    truth = paths["ground_truth"]
    destination = outdir / "ground_truth" / "read_to_genome.tsv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(truth.read_text(encoding="utf-8"), encoding="utf-8")
    fastq = read_fastq(paths["reads"])
    sequences = dict(fastq)
    _, meta = read_tsv(paths["metadata"])
    ctx.reads = [(row["read_id"], row["sample_id"], sequences[row["read_id"]]) for row in meta]
    ctx.samples = ["sample_1", "sample_2"]
    return dbg_from_sequences(fastq, k=8, graph_id="synthetic_reads_2")


def prepare_bubble_reads_2(outdir: Path, ctx: BuildContext) -> CfaGraph:
    """Two strains, one allele bubble, read length 12, k=5, seed 1.

    This is the BubbleBlower reads example. Sample colours come from the
    ``read_depth`` colouring. Genome ids stay in ``ground_truth/``.
    """
    left = "ACGTAGCTTG"
    allele_a = "CATGCA"
    allele_b = "GATCGA"
    right = "TGCCTAAGGC"
    genomes = {
        "strain_A": left + allele_a + right,
        "strain_B": left + allele_b + right,
    }
    genome_dir = outdir / "genomes"
    genome_dir.mkdir(parents=True, exist_ok=True)
    for name, sequence in genomes.items():
        (genome_dir / f"{name}.fna").write_text(f">{name}\n{sequence}\n", encoding="utf-8")
    paths = simulate_metagenome(
        genomes,
        [("sample_A", "strain_A", 20), ("sample_B", "strain_B", 20)],
        outdir / "reads",
        read_length=12,
        seed=1,
    )
    destination = outdir / "ground_truth" / "read_to_genome.tsv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(paths["ground_truth"].read_text(encoding="utf-8"), encoding="utf-8")
    fastq = read_fastq(paths["reads"])
    ctx.reads = []
    for read_id, sequence in fastq:
        sample = "sample_A" if read_id.startswith("sample_A") else "sample_B"
        ctx.reads.append((read_id, sample, sequence))
    ctx.samples = ["sample_A", "sample_B"]
    ctx.min_edge_kmer_density = 1
    return dbg_from_sequences(fastq, k=5, graph_id="bubble_reads_2")
