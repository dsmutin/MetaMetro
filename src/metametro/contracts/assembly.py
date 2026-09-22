"""Contract 1 and Contract 2: reads, then an assembly graph.

The in-process builders are the deterministic mock. Samovar ISS and MEGAHIT
are the external baselines; their command lines live in ``external.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

from metametro.errors import ContractError
from metametro.formats.cfa.model import CfaGraph
from metametro.tables import write_tsv

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
_FASTG_ID = re.compile(r"ID_(\d+)$")
_FASTG_COV = re.compile(r"_cov_([0-9.]+)")


def _fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current: str | None = None
    chunks: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "":
            continue
        if line.startswith(">"):
            if current is not None:
                records.append((current, "".join(chunks).upper()))
            token = line[1:].split()[0]
            current = token
            chunks = []
            continue
        chunks.append(line)
    if current is not None:
        records.append((current, "".join(chunks).upper()))
    return records


def load_genomes(genome_dir: str | Path) -> dict[str, str]:
    """Load nucleotide FASTA records. The record id is the genome id."""
    root = Path(genome_dir)
    genomes: dict[str, str] = {}
    files = sorted(path for path in root.iterdir() if path.suffix.lower() in {".fna", ".fa", ".fasta"})
    if not files:
        raise ContractError([f"no genome FASTA files in {root}"])
    for path in files:
        for genome_id, sequence in _fasta_records(path):
            if genome_id in genomes:
                raise ContractError([f"duplicate genome_id: {genome_id}"])
            if sequence == "" or any(base not in "ACGTN" for base in sequence):
                raise ContractError([f"malformed genome sequence: {genome_id}"])
            genomes[genome_id] = sequence
    return genomes


def simulate_metagenome(
    genomes: dict[str, str],
    abundances: list[tuple[str, str, int]],
    out_dir: str | Path,
    *,
    read_length: int = 16,
    seed: int = 1,
) -> dict[str, Path]:
    """Write FASTQ, metadata, and ground truth for a tiny metagenome.

    ``abundances`` is a list of ``(sample_id, genome_id, n_reads)``. Start
    positions are a deterministic function of ``seed`` and the read index.
    A read that cannot be tied to one genome is not emitted by this builder.
    """
    if read_length <= 0:
        raise ContractError(["read_length must be positive"])
    root = Path(out_dir)
    (root / "ground_truth").mkdir(parents=True, exist_ok=True)
    fastq_lines: list[str] = []
    rows: list[dict[str, str]] = []
    for sample_id, genome_id, n_reads in abundances:
        if genome_id not in genomes:
            raise ContractError([f"unknown genome_id: {genome_id}"])
        sequence = genomes[genome_id]
        if len(sequence) < read_length:
            raise ContractError([f"genome {genome_id} is shorter than read_length"])
        span = len(sequence) - read_length + 1
        for index in range(n_reads):
            start = (seed + index * 3 + sum(ord(char) for char in genome_id)) % span
            read = sequence[start : start + read_length]
            read_id = f"{sample_id}_{genome_id}_{index:04d}"
            fastq_lines.extend([f"@{read_id}", read, "+", "I" * read_length])
            rows.append(
                {
                    "read_id": read_id,
                    "sample_id": sample_id,
                    "genome_id": genome_id,
                    "ambiguous": "no",
                }
            )
    fastq_path = root / "reads.fastq"
    fastq_path.write_text("\n".join(fastq_lines) + "\n", encoding="utf-8")
    meta_path = root / "metadata.tsv"
    write_tsv(meta_path, ["read_id", "sample_id", "genome_id", "ambiguous"], rows)
    truth_path = root / "ground_truth" / "read_to_genome.tsv"
    write_tsv(truth_path, ["read_id", "sample_id", "genome_id", "ambiguous"], rows)
    return {"reads": fastq_path, "metadata": meta_path, "ground_truth": truth_path}


def read_fastq(path: str | Path) -> list[tuple[str, str]]:
    """Return ``(read_id, sequence)`` records from a FASTQ file."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 4 != 0:
        raise ContractError([f"malformed FASTQ: {path}"])
    records = []
    for offset in range(0, len(lines), 4):
        header = lines[offset]
        if not header.startswith("@") or lines[offset + 2] != "+":
            raise ContractError([f"malformed FASTQ record in {path}"])
        records.append((header[1:].split()[0], lines[offset + 1].upper()))
    return records


def dbg_from_sequences(
    sequences: list[tuple[str, str]],
    *,
    k: int,
    graph_id: str,
) -> CfaGraph:
    """Build a de Bruijn CFA from sequences. Node sequence is the k-mer."""
    if k <= 0:
        raise ContractError(["k must be > 0"])
    edges: set[tuple[str, str]] = set()
    seen: set[str] = set()
    for _, sequence in sequences:
        if any(base not in "ACGTN" for base in sequence):
            raise ContractError(["malformed sequence in graph construction"])
        if len(sequence) < k:
            continue
        kmers = [sequence[index : index + k] for index in range(len(sequence) - k + 1)]
        for kmer in kmers:
            seen.add(kmer)
        for left, right in zip(kmers, kmers[1:]):
            edges.add((left, right))
    ordered = sorted(seen)
    id_of = {kmer: f"n{index:06d}" for index, kmer in enumerate(ordered, start=1)}
    nodes = [{"node_id": id_of[kmer]} for kmer in ordered]
    edge_rows = []
    for index, (left, right) in enumerate(sorted(edges), start=1):
        edge_rows.append(
            {
                "edge_id": f"e{index:06d}",
                "source": id_of[left],
                "target": id_of[right],
            }
        )
    return CfaGraph(
        metadata={
            "schema_version": "1.0",
            "graph_id": graph_id,
            "graph_type": "de_bruijn",
            "k": k,
            "contract": "metagenome_to_graph",
            "contract_version": "1.0",
        },
        sequences={id_of[kmer]: kmer for kmer in ordered},
        nodes=nodes,
        edges=edge_rows,
        node_header=["node_id"],
        edge_header=["edge_id", "source", "target"],
    )


def contig_overlap_graph(
    records: list[tuple[str, str]],
    *,
    k: int,
    graph_id: str,
) -> CfaGraph:
    """One node per contig. An edge exists when the (k-1)-mer overlap is exact.

    This is the MEGAHIT baseline graph when expanding every contig into k-mers
    would be larger than the pure-Python compactor should materialize. Node
    sequence is the full contig, so sequence is not reduced to a k-mer.
    """
    if k <= 1:
        raise ContractError(["k must be > 1"])
    if not records:
        raise ContractError(["contig overlap graph requires at least one contig"])
    nodes = []
    sequences = {}
    for index, (name, sequence) in enumerate(records, start=1):
        if any(base not in "ACGTN" for base in sequence):
            raise ContractError([f"malformed contig sequence: {name}"])
        if len(sequence) < k:
            raise ContractError([f"contig {name} is shorter than k"])
        node_id = f"n{index:06d}"
        nodes.append({"node_id": node_id})
        sequences[node_id] = sequence
    overlap = k - 1
    edge_rows = []
    order = [row["node_id"] for row in nodes]
    edge_index = 1
    for source in order:
        for target in order:
            if source == target:
                continue
            if sequences[source][-overlap:] == sequences[target][:overlap]:
                edge_rows.append(
                    {
                        "edge_id": f"e{edge_index:06d}",
                        "source": source,
                        "target": target,
                    }
                )
                edge_index += 1
    return CfaGraph(
        metadata={
            "schema_version": "1.0",
            "graph_id": graph_id,
            "graph_type": "de_bruijn",
            "k": k,
            "contract": "metagenome_to_graph",
            "contract_version": "1.0",
            "source": {"format": "contig_overlap", "k": k},
        },
        sequences=sequences,
        nodes=nodes,
        edges=edge_rows,
        node_header=["node_id"],
        edge_header=["edge_id", "source", "target"],
    )


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def _fastg_endpoint(token: str) -> tuple[str, str, str]:
    """Return ``(node_id, strand, coverage)`` for one FASTG header token."""
    token = token.strip()
    reverse = token.endswith("'")
    if reverse:
        token = token[:-1]
    match = _FASTG_ID.search(token)
    if match is None:
        raise ContractError([f"FASTG record has no ID_ field: {token}"])
    coverage = _FASTG_COV.search(token)
    return f"n{int(match.group(1)):06d}", "-" if reverse else "+", coverage.group(1) if coverage else "0"


def _fastg_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "":
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks).upper()))
            header = line[1:].strip().rstrip(";")
            chunks = []
            continue
        chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks).upper()))
    if not records:
        raise ContractError([f"no FASTG records in {path}"])
    return records


def fastg_to_cfa(fastg_path: str | Path, *, k: int, graph_id: str) -> CfaGraph:
    """Load a MEGAHIT intermediate assembly graph from ``contig2fastg`` output.

    Each contig is stored once, as the forward sequence. A trailing ``'`` on a
    header or neighbor is the reverse strand and becomes the edge orientation.
    MEGAHIT writes edges that overlap by the assembly ``k`` (the shared k-mer).
    The CFA de Bruijn parameter is therefore ``k + 1``, so the contract overlap
    of ``k_cfa - 1`` is that same k-mer and compaction does not duplicate a base.
    """
    if k < 1:
        raise ContractError(["k must be > 0"])
    records = _fastg_records(Path(fastg_path))
    sequences: dict[str, str] = {}
    coverage: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []
    for header, sequence in records:
        if ":" in header:
            left, right = header.split(":", 1)
            neighbors = [piece for piece in right.split(",") if piece]
        else:
            left, neighbors = header, []
        source_id, source_strand, source_coverage = _fastg_endpoint(left)
        if any(base not in "ACGTN" for base in sequence):
            raise ContractError([f"malformed FASTG sequence: {source_id}"])
        forward = sequence if source_strand == "+" else _reverse_complement(sequence)
        if source_id in sequences and sequences[source_id] != forward:
            raise ContractError([f"FASTG reverse complement disagrees for {source_id}"])
        sequences[source_id] = forward
        if source_strand == "+":
            coverage[source_id] = source_coverage
        else:
            coverage.setdefault(source_id, source_coverage)
        for neighbor in neighbors:
            target_id, target_strand, _ = _fastg_endpoint(neighbor)
            edges.append((source_id, target_id, source_strand + target_strand))

    endpoints = {source for source, _, _ in edges} | {target for _, target, _ in edges}
    missing = sorted(endpoints - set(sequences))
    if missing:
        raise ContractError([f"FASTG edge endpoint has no sequence: {node_id}" for node_id in missing])
    cfa_k = k + 1
    short = [node_id for node_id, sequence in sequences.items() if len(sequence) < cfa_k]
    if short:
        raise ContractError(
            [f"node {node_id} is shorter than the FASTG overlap k+1={cfa_k}" for node_id in short]
        )
    overlap = k
    unique_edges = sorted(set(edges))

    def _strand(node_id: str, strand: str) -> str:
        sequence = sequences[node_id]
        return _reverse_complement(sequence) if strand == "-" else sequence

    for source_id, target_id, orientation in unique_edges:
        left = _strand(source_id, orientation[0])
        right = _strand(target_id, orientation[1])
        if left[-overlap:] != right[:overlap]:
            raise ContractError(
                [f"FASTG edge {source_id}->{target_id} {orientation} does not overlap by assembly k={k}"]
            )
    node_ids = sorted(sequences)
    return CfaGraph(
        metadata={
            "schema_version": "1.0",
            "graph_id": graph_id,
            "graph_type": "de_bruijn",
            "k": cfa_k,
            "contract": "metagenome_to_graph",
            "contract_version": "1.0",
            "features": {
                "node": {"coverage": "float"},
                "edge": {"orientation": "orientation"},
            },
            "source": {"format": "megahit_fastg", "assembly_k": k, "k": cfa_k},
        },
        sequences=sequences,
        nodes=[{"node_id": node_id, "coverage": coverage.get(node_id, "0")} for node_id in node_ids],
        edges=[
            {
                "edge_id": f"e{index:06d}",
                "source": source_id,
                "target": target_id,
                "orientation": orientation,
            }
            for index, (source_id, target_id, orientation) in enumerate(unique_edges, start=1)
        ],
        node_header=["node_id", "coverage"],
        edge_header=["edge_id", "source", "target", "orientation"],
    )


def _edge_orientation(row: dict[str, str]) -> str:
    orientation = row.get("orientation") or "++"
    if orientation not in {"++", "+-", "-+", "--"}:
        return "++"
    return orientation


def strand_junction_counts(edges: list[dict[str, str]]) -> dict[str, int]:
    """Count nodes with two or more edges on the same strand.

    FASTG stores both strands, so a linear contig has one forward link and one
    reverse link. Those are not a branch. A branch is two outs that leave on
    the same strand; a join is two ins that arrive on the same strand.
    """
    outgoing: dict[tuple[str, str], int] = {}
    incoming: dict[tuple[str, str], int] = {}
    for row in edges:
        orientation = _edge_orientation(row)
        outgoing[(row["source"], orientation[0])] = outgoing.get((row["source"], orientation[0]), 0) + 1
        incoming[(row["target"], orientation[1])] = incoming.get((row["target"], orientation[1]), 0) + 1
    return {
        "branch_nodes": len({node for (node, _), count in outgoing.items() if count >= 2}),
        "join_nodes": len({node for (node, _), count in incoming.items() if count >= 2}),
    }


def directed_bubble_sources(edges: list[dict[str, str]], *, max_depth: int = 4) -> list[str]:
    """Sources of two same-strand paths that meet again within ``max_depth`` edges.

    A step follows the arrival strand: an edge ``+-`` arrives on ``-``, so the
    next step must leave on ``-``. A branch that does not reconverge is not a bubble.
    """
    if max_depth < 1:
        raise ContractError(["bubble depth must be >= 1"])
    outgoing: dict[str, list[tuple[str, str, str]]] = {}
    for row in edges:
        orientation = _edge_orientation(row)
        outgoing.setdefault(row["source"], []).append((row["target"], orientation[0], orientation[1]))
    found: list[str] = []
    for source, departures in outgoing.items():
        by_strand: dict[str, list[tuple[str, str]]] = {}
        for target, source_strand, target_strand in departures:
            by_strand.setdefault(source_strand, []).append((target, target_strand))
        for starts in by_strand.values():
            unique = list(dict.fromkeys(starts))
            if len(unique) < 2:
                continue
            entered: list[set[str]] = []
            for start, arrival in unique:
                meeting: set[str] = set()
                seen: set[tuple[str, str]] = set()
                frontier = [(start, arrival)]
                for _ in range(max_depth):
                    nxt: list[tuple[str, str]] = []
                    for node, depart_strand in frontier:
                        key = (node, depart_strand)
                        if key in seen:
                            continue
                        seen.add(key)
                        for nxt_target, src_strand, nxt_arrival in outgoing.get(node, []):
                            if src_strand == depart_strand:
                                meeting.add(nxt_target)
                                nxt.append((nxt_target, nxt_arrival))
                    frontier = nxt
                entered.append(meeting)
            starts = {node for node, _ in unique}
            reconverge = False
            for left in range(len(entered)):
                for right in range(left + 1, len(entered)):
                    shared = entered[left] & entered[right]
                    crossed = (entered[left] & {unique[right][0]}) | (entered[right] & {unique[left][0]})
                    if shared - starts or crossed:
                        reconverge = True
            if reconverge:
                found.append(source)
                break
    return sorted(found)


def contigs_to_dbg(contig_fasta: str | Path, *, k: int, graph_id: str) -> CfaGraph:
    """Contract 2 baseline for MEGAHIT contigs: a de Bruijn graph at ``k``."""
    records = _fasta_records(Path(contig_fasta))
    if not records:
        raise ContractError([f"no contigs in {contig_fasta}"])
    graph = dbg_from_sequences(records, k=k, graph_id=graph_id)
    graph.metadata["source"] = {"format": "megahit_contigs", "k": k}
    return graph
