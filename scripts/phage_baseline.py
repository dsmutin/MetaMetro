"""Run the T-phage baseline: Samovar ISS, MEGAHIT intermediate graph, colouring, CGT, GCN.

The assembly graph is MEGAHIT's intermediate contig graph
(``intermediate_contigs/k*.contigs.fa`` via ``contig2fastg``), which keeps
bubbles and branch/join edges. Final contigs are linear and drop that structure.

A node label is the unique genome that contains a strict majority of the node's
assembly k-mers (both strands). Anything else is 0. The GCN does not modify the CGT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metametro.contracts.assembly import (
    directed_bubble_sources,
    fastg_to_cfa,
    load_genomes,
    read_fastq,
    strand_junction_counts,
)
from metametro.contracts.colouring import colour_by_reads
from metametro.contracts.ds import run_ds
from metametro.contracts.external import (
    contig2fastg_command,
    megahit_command,
    run_command,
    run_command_to_file,
    run_samovar_generate,
)
from metametro.converters.cdbg_to_cgt import cdbg_to_cgt
from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
from metametro.formats.cfa.io import dump_cfa
from metametro.formats.cdbg.io import dump_cdbg
from metametro.formats.cgt.io import dump_cgt
from metametro.tables import write_tsv

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def _assign_read(read_id: str, genome_ids: list[str]) -> tuple[str, str]:
    """Map a Samovar read id such as ``T1_0_0/1`` to one genome token."""
    token = read_id.split("_", 1)[0]
    chosen = [genome_id for genome_id in genome_ids if token == genome_id]
    if len(chosen) == 1:
        return chosen[0], "no"
    return "", "yes"


def _sample_id(fastq_name: str) -> str:
    """Sample token is the Samovar prefix of ``1_full_R1.fastq``."""
    if "_full_R" not in fastq_name:
        raise SystemExit(f"unexpected FASTQ name: {fastq_name}")
    return "sample_" + fastq_name.split("_full_R", 1)[0]


def _labels(sequences: dict[str, str], genomes: dict[str, str], k: int) -> tuple[list[str], dict[str, int]]:
    """Label a node by the genome that owns a strict majority of its k-mers."""
    names = sorted(genomes)
    label_of = {name: position + 1 for position, name in enumerate(names)}
    genome_kmers: dict[str, set[str]] = {}
    for name, sequence in genomes.items():
        strands = (sequence, sequence.translate(_COMPLEMENT)[::-1])
        kmers: set[str] = set()
        for strand in strands:
            if len(strand) >= k:
                kmers.update(strand[offset : offset + k] for offset in range(len(strand) - k + 1))
        genome_kmers[name] = kmers
    labels = {}
    for node_id, sequence in sequences.items():
        if len(sequence) < k:
            labels[node_id] = 0
            continue
        kmers = [sequence[offset : offset + k] for offset in range(len(sequence) - k + 1)]
        counts = [sum(kmer in genome_kmers[name] for kmer in kmers) for name in names]
        best = max(counts) if counts else 0
        winners = [position for position, count in enumerate(counts) if count == best and count > 0]
        if len(winners) == 1 and best * 2 > len(kmers):
            labels[node_id] = label_of[names[winners[0]]]
        else:
            labels[node_id] = 0
    return names, labels


def main() -> int:
    parser = argparse.ArgumentParser(description="T-phage CFA/CDBG/CGT baseline")
    parser.add_argument("--genomes", type=Path, default=Path("data/raw/genomes"))
    parser.add_argument("--work", type=Path, default=Path("data/work/phage_baseline"))
    parser.add_argument("--megahit", default="megahit")
    parser.add_argument("--samovar", default="samovar")
    parser.add_argument("--k", type=int, default=21)
    parser.add_argument("--total-reads", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--skip-simulate", action="store_true")
    parser.add_argument("--skip-assemble", action="store_true")
    args = parser.parse_args()

    genomes = load_genomes(args.genomes)
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    iss_dir = work / "iss"
    asm_dir = work / "megahit"
    if not args.skip_simulate:
        run_samovar_generate(
            args.genomes,
            iss_dir,
            n_samples=2,
            total_reads=args.total_reads,
            host_fraction=0,
            seed=1,
            samovar=args.samovar,
        )
    fastq = sorted((iss_dir / "initial").glob("*_full_R*.fastq"))
    if not fastq:
        raise SystemExit(f"no FASTQ under {iss_dir}")
    genome_ids = sorted(genomes)
    read_records: list[tuple[str, str, str]] = []
    for path in fastq:
        sample_id = _sample_id(path.name)
        for read_id, sequence in read_fastq(path):
            read_records.append((read_id, sample_id, sequence))
    reads_path = work / "reads.fastq"
    lines: list[str] = []
    meta = []
    for read_id, sample_id, sequence in read_records:
        genome_id, ambiguous = _assign_read(read_id, genome_ids)
        lines.extend([f"@{read_id}", sequence, "+", "I" * len(sequence)])
        meta.append(
            {
                "read_id": read_id,
                "sample_id": sample_id,
                "genome_id": genome_id,
                "ambiguous": ambiguous,
            }
        )
    reads_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_tsv(work / "metadata.tsv", ["read_id", "sample_id", "genome_id", "ambiguous"], meta)
    if not args.skip_assemble:
        if asm_dir.exists():
            raise SystemExit(f"refusing to reuse {asm_dir}; delete it or pass --skip-assemble")
        run_command(megahit_command(reads_path, asm_dir, k=args.k, megahit=args.megahit, min_count=2))
    intermediate = asm_dir / "intermediate_contigs" / f"k{args.k}.contigs.fa"
    if not intermediate.is_file():
        raise SystemExit(f"missing MEGAHIT intermediate graph: {intermediate}")
    toolkit = "megahit_toolkit"
    megahit_path = Path(args.megahit)
    if megahit_path.is_file():
        sibling = megahit_path.with_name("megahit_toolkit")
        if sibling.is_file():
            toolkit = str(sibling)
    fastg_path = work / f"k{args.k}.fastg"
    run_command_to_file(contig2fastg_command(intermediate, args.k, toolkit=toolkit), fastg_path)
    graph = fastg_to_cfa(fastg_path, k=args.k, graph_id="t_phages")
    dump_cfa(graph, work / "cfa")
    if len(meta) != len(read_records):
        raise SystemExit(f"metadata rows {len(meta)} do not match reads {len(read_records)}")
    samples = sorted({row["sample_id"] for row in meta})
    coloured = colour_by_reads(
        graph,
        [(row["read_id"], row["sample_id"], sequence) for row, (_, _, sequence) in zip(meta, read_records)],
        samples,
        min_vertex_depth=1,
        min_edge_kmer_density=2,
        operation="replace",
    )
    dump_cfa(coloured, work / "coloured_cfa")
    cdbg = cfa_to_cdbg(coloured)
    dump_cdbg(cdbg, work / "cdbg")
    names, node_label = _labels(graph.sequences, genomes, args.k)
    member_label = []
    for unitig in sorted(cdbg.unitigs, key=lambda item: item.unitig_id):
        values = {node_label[node_id] for node_id in unitig.members}
        member_label.append(next(iter(values)) if len(values) == 1 else 0)
    cgt = cdbg_to_cgt(
        cdbg,
        node_labels=np.asarray(member_label, dtype=np.int64),
        node_feature_names=[],
        edge_feature_names=[],
    )
    # Degree is a declared numeric feature so the convolution has an input.
    degree = np.diff(cgt.indptr).astype(np.float32).reshape(-1, 1)
    cgt.node_features = degree
    cgt.metadata["node_feature_names"] = ["out_degree"]
    dump_cgt(cgt, work / "cgt")
    numpy_result = run_ds(cgt, epochs=args.epochs, seed=1, backend="numpy")
    junctions = strand_junction_counts(graph.edges)
    bubbles = directed_bubble_sources(graph.edges)
    summary = {
        "genomes": names,
        "total_reads_parameter": args.total_reads,
        "reads": len(read_records),
        "ambiguous_reads": sum(row["ambiguous"] == "yes" for row in meta),
        "cfa_nodes": len(graph.nodes),
        "cfa_edges": len(graph.edges),
        "coloured_nodes": sum(row.get("color_set", "").strip() != "" for row in coloured.nodes),
        "coloured_edges": sum(row.get("color_set", "").strip() != "" for row in coloured.edges),
        "branch_nodes": junctions["branch_nodes"],
        "join_nodes": junctions["join_nodes"],
        "bubble_sources": len(bubbles),
        "nodes": cgt.num_nodes,
        "edges": cgt.num_edges,
        "unitigs": len(cdbg.unitigs),
        "graph_source": graph.metadata.get("source", {}).get("format", "megahit_fastg"),
        "assembly_k": args.k,
        "cfa_k": graph.metadata["k"],
        "label_counts": {
            str(label): int(sum(value == label for value in member_label))
            for label in sorted(set(member_label))
        },
        "numpy_predictions": numpy_result["metadata"]["num_predictions"],
    }
    try:
        pyg_result = run_ds(cgt, epochs=args.epochs, seed=1, backend="pyg")
        summary["pyg_predictions"] = pyg_result["metadata"]["num_predictions"]
        summary["pyg_backend"] = "pyg"
    except ImportError as exc:
        summary["pyg_backend"] = f"unavailable: {exc}"
    write_tsv(
        work / "ds_numpy.tsv",
        ["dense_id", "source_id", "cfa_node_ids", "predicted_label", "probability"],
        [
            {
                "dense_id": row["dense_id"],
                "source_id": row["source_id"],
                "cfa_node_ids": ",".join(row["cfa_node_ids"]),
                "predicted_label": row["predicted_label"],
                "probability": f"{row['probability']:.6f}",
            }
            for row in numpy_result["result"]
        ],
    )
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
