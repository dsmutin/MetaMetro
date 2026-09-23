"""Contract 4: colour an existing CFA without changing sequence or topology.

Vertices take a sample colour when that sample's read depth on the node is
at least ``min_vertex_depth`` (default 1). Edges take a sample colour when
the junction (k+1)-mer occurs at least ``min_edge_kmer_density`` times in
that sample (default 2). Reads are unstranded: the reverse complement of a
read covers a node, and the reverse complement of a junction counts toward
edge density. Existing colours are combined only through an explicit
``replace``, ``merge``, ``intersect``, or ``subtract`` operation.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from metametro.errors import ContractError
from metametro.formats.cfa.model import CfaGraph
from metametro.formats.cfa.validator import parse_color_set, validate_cfa

OPERATIONS = ("replace", "merge", "intersect", "subtract")


def combine_colors(existing: Iterable[int], incoming: Iterable[int], operation: str) -> list[int]:
    """Combine two colour sets. Unknown operations are rejected."""
    if operation not in OPERATIONS:
        raise ContractError([f"unknown colour_operation: {operation}"])
    left, right = set(existing), set(incoming)
    if operation == "replace":
        return sorted(right)
    if operation == "merge":
        return sorted(left | right)
    if operation == "intersect":
        return sorted(left & right)
    return sorted(left - right)


def _format_colors(color_ids: Sequence[int]) -> str:
    return ",".join(str(color_id) for color_id in color_ids)


def _existing(row: dict[str, str]) -> list[int]:
    raw = row.get("color_set", row.get("colors", ""))
    return parse_color_set(raw)


def colour_cfa(
    cfa: CfaGraph,
    node_colors: Mapping[str, Sequence[int]] | None = None,
    edge_colors: Mapping[str, Sequence[int]] | None = None,
    *,
    operation: str,
    colors: list[dict[str, str]] | None = None,
    target: Sequence[str] = ("node", "edge"),
) -> CfaGraph:
    """Return a new CFA with an updated colour layer.

    ``operation`` is required whenever the graph already has a colour set,
    and it is also required here so callers cannot rely on a silent default.
    Topology and sequences are copied unchanged.
    """
    validate_cfa(cfa)
    if operation not in OPERATIONS:
        raise ContractError([f"unknown colour_operation: {operation}"])
    unknown_targets = [name for name in target if name not in {"node", "edge"}]
    if unknown_targets:
        raise ContractError([f"unknown colour target: {', '.join(unknown_targets)}"])
    already = any(_existing(row) for row in cfa.nodes + cfa.edges)
    if already and operation is None:
        raise ContractError(["colour_operation is required when the CFA is already coloured"])
    nodes = [dict(row) for row in cfa.nodes]
    edges = [dict(row) for row in cfa.edges]
    if "node" in target and node_colors is not None:
        for row in nodes:
            incoming = node_colors.get(row["node_id"], _existing(row) if operation != "replace" else [])
            if row["node_id"] not in node_colors and operation == "replace":
                incoming = []
            row["color_set"] = _format_colors(combine_colors(_existing(row), incoming, operation))
            row.pop("colors", None)
    if "edge" in target and edge_colors is not None:
        for row in edges:
            if row["edge_id"] not in edge_colors and operation == "replace":
                incoming: Sequence[int] = []
            else:
                incoming = edge_colors.get(row["edge_id"], _existing(row))
            row["color_set"] = _format_colors(combine_colors(_existing(row), incoming, operation))
            row.pop("colors", None)
    metadata = dict(cfa.metadata)
    features = dict(metadata.get("features") or {})
    node_features = dict(features.get("node") or {})
    edge_features = dict(features.get("edge") or {})
    if "node" in target and node_colors is not None:
        node_features["color_set"] = "color_set"
    if "edge" in target and edge_colors is not None:
        edge_features["color_set"] = "color_set"
    features["node"] = node_features
    features["edge"] = edge_features
    metadata["features"] = features
    metadata["colour_operation"] = operation
    metadata["colour_target"] = list(target)
    node_header = list(cfa.node_header)
    edge_header = list(cfa.edge_header)
    if "color_set" in (nodes[0] if nodes else {}) and "color_set" not in node_header:
        node_header.append("color_set")
    if edges and "color_set" in edges[0] and "color_set" not in edge_header:
        edge_header.append("color_set")
    graph = CfaGraph(
        metadata=metadata,
        sequences=dict(cfa.sequences),
        nodes=nodes,
        edges=edges,
        colors=colors if colors is not None else (None if cfa.colors is None else [dict(row) for row in cfa.colors]),
        labels=None if cfa.labels is None else [dict(row) for row in cfa.labels],
        node_header=node_header,
        edge_header=edge_header,
    )
    validate_cfa(graph)
    return graph


_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def _oriented_sequence(sequence: str, strand: str) -> str:
    if strand == "-":
        return _reverse_complement(sequence)
    return sequence


def _junction(cfa: CfaGraph, row: dict[str, str], k: int) -> str:
    """Junction (k+1)-mer. A '-' endpoint uses the reverse complement."""
    orientation = row.get("orientation") or "++"
    if orientation not in {"++", "+-", "-+", "--"}:
        orientation = "++"
    source = _oriented_sequence(cfa.sequences[row["source"]], orientation[0])
    target = _oriented_sequence(cfa.sequences[row["target"]], orientation[1])
    if len(source) < k or len(target) < k:
        raise ContractError([f"edge {row['edge_id']} is shorter than k"])
    return source[-k:] + target[k - 1]


def _either_strand_count(counts: dict[str, int], motif: str) -> int:
    """Occurrences of ``motif`` plus its reverse complement, without double-counting a palindrome."""
    observed = counts.get(motif, 0)
    flipped = _reverse_complement(motif)
    if flipped != motif:
        observed += counts.get(flipped, 0)
    return observed


def _colour_kmers(cfa, by_sample, samples, sample_index, k, min_vertex_depth, min_edge_kmer_density):
    """Same depth and density rules, scanned once per read."""
    present: dict[str, dict[str, int]] = {sample: {} for sample in samples}
    density: dict[str, dict[str, int]] = {sample: {} for sample in samples}
    for sample, sample_reads in by_sample.items():
        seen_counts: dict[str, int] = {}
        junction_counts: dict[str, int] = {}
        for read in sample_reads:
            if len(read) >= k:
                seen: set[str] = set()
                for index in range(len(read) - k + 1):
                    kmer = read[index : index + k]
                    seen.add(kmer)
                    seen.add(_reverse_complement(kmer))
                for kmer in seen:
                    seen_counts[kmer] = seen_counts.get(kmer, 0) + 1
            if len(read) >= k + 1:
                for index in range(len(read) - k):
                    motif = read[index : index + k + 1]
                    junction_counts[motif] = junction_counts.get(motif, 0) + 1
        present[sample] = seen_counts
        density[sample] = junction_counts
    node_assignment = {}
    for row in cfa.nodes:
        sequence = cfa.sequences[row["node_id"]]
        node_assignment[row["node_id"]] = [
            sample_index[sample]
            for sample in samples
            if present[sample].get(sequence, 0) >= min_vertex_depth
        ]
    edge_assignment = {}
    for row in cfa.edges:
        junction = _junction(cfa, row, k)
        edge_assignment[row["edge_id"]] = [
            sample_index[sample]
            for sample in samples
            if _either_strand_count(density[sample], junction) >= min_edge_kmer_density
        ]
    return node_assignment, edge_assignment


def _read_covers(read: str, sequence: str) -> bool:
    """True when the read or its reverse complement covers the node sequence."""
    if read == "":
        return False
    flipped = _reverse_complement(read)
    if len(read) >= len(sequence):
        return sequence in read or sequence in flipped
    return read in sequence or flipped in sequence


def _colour_general(cfa, by_sample, samples, sample_index, k, min_vertex_depth, min_edge_kmer_density):
    node_assignment: dict[str, list[int]] = {}
    for row in cfa.nodes:
        node_id = row["node_id"]
        sequence = cfa.sequences[node_id]
        chosen = []
        for sample in samples:
            depth = sum(_read_covers(read, sequence) for read in by_sample[sample])
            if depth >= min_vertex_depth:
                chosen.append(sample_index[sample])
        node_assignment[node_id] = chosen
    edge_assignment: dict[str, list[int]] = {}
    for row in cfa.edges:
        junction = _junction(cfa, row, k)
        flipped = _reverse_complement(junction)
        motifs = (junction,) if flipped == junction else (junction, flipped)
        chosen = []
        for sample in samples:
            observed = sum(_count_motif(read, motif) for read in by_sample[sample] for motif in motifs)
            if observed >= min_edge_kmer_density:
                chosen.append(sample_index[sample])
        edge_assignment[row["edge_id"]] = chosen
    return node_assignment, edge_assignment


def _count_motif(text: str, motif: str) -> int:
    if motif == "":
        return 0
    count = 0
    start = 0
    while True:
        found = text.find(motif, start)
        if found < 0:
            return count
        count += 1
        start = found + 1


def colour_by_reads(
    cfa: CfaGraph,
    reads: Sequence[tuple[str, str, str]],
    samples: Sequence[str],
    *,
    min_vertex_depth: int = 1,
    min_edge_kmer_density: int = 2,
    operation: str = "replace",
) -> CfaGraph:
    """Colour nodes by read depth and edges by junction (k+1)-mer density.

    ``graph_type`` is not restricted. The graph must declare integer ``k``,
    because depth and junction density are counted in k-mers.

    ``reads`` entries are ``(read_id, sample_id, sequence)``. ``read_id`` is
    accepted so callers can keep provenance; it is not used as a colour.
    Samples absent from a node or edge contribute no colour. Depth is not
    imputed.
    """
    if min_vertex_depth < 1 or min_edge_kmer_density < 1:
        raise ContractError(["colour thresholds must be >= 1"])
    raw_k = cfa.metadata.get("k")
    if not isinstance(raw_k, int) or isinstance(raw_k, bool) or raw_k <= 0:
        raise ContractError(["read colouring requires integer k > 0"])
    k = raw_k
    sample_index = {sample: index for index, sample in enumerate(samples)}
    for _, sample_id, _ in reads:
        if sample_id not in sample_index:
            raise ContractError([f"read sample {sample_id} is not in the colour dictionary"])
    by_sample: dict[str, list[str]] = {sample: [] for sample in samples}
    for _, sample_id, sequence in reads:
        by_sample[sample_id].append(sequence.upper())
    kmer_nodes = all(len(cfa.sequences[row["node_id"]]) == k for row in cfa.nodes)
    if kmer_nodes:
        node_assignment, edge_assignment = _colour_kmers(
            cfa,
            by_sample,
            samples,
            sample_index,
            k,
            min_vertex_depth,
            min_edge_kmer_density,
        )
    else:
        node_assignment, edge_assignment = _colour_general(
            cfa,
            by_sample,
            samples,
            sample_index,
            k,
            min_vertex_depth,
            min_edge_kmer_density,
        )
    dictionary = [
        {"color_id": str(index), "namespace": "sample", "value": sample}
        for index, sample in enumerate(samples)
    ]
    return colour_cfa(
        cfa,
        node_assignment,
        edge_assignment,
        operation=operation,
        colors=dictionary,
        target=("node", "edge"),
    )
