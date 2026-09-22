"""Read and write a CFA directory."""

from __future__ import annotations

from pathlib import Path

from metametro.errors import ContractError
from metametro.formats.cfa.model import CfaGraph
from metametro.formats.cfa.validator import validate_cfa
from metametro.tables import read_tsv, read_yaml, write_tsv, write_yaml


def _read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal current, chunks
        if current is None:
            return
        sequences[current] = "".join(chunks).upper()
        current = None
        chunks = []

    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if line == "":
            continue
        if line.startswith(">"):
            flush()
            token = line[1:].split()[0] if len(line) > 1 else ""
            if token == "":
                raise ContractError([f"{path}:{line_number}: FASTA record has no node_id"])
            if token in sequences or current == token:
                raise ContractError([f"duplicate node_id in nodes.fna: {token}"])
            current = token
            continue
        if current is None:
            raise ContractError([f"{path}:{line_number}: sequence line before a FASTA header"])
        chunks.append(line)
    flush()
    return sequences


def _write_fasta(path: Path, sequences: dict[str, str], order: list[str]) -> None:
    lines: list[str] = []
    for node_id in order:
        lines.append(f">{node_id}")
        lines.append(sequences[node_id])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_cfa(path: str | Path, *, validate: bool = True) -> CfaGraph:
    """Load a CFA directory.

    When ``validate`` is true, ``validate_cfa`` runs before the object is returned.
    """
    root = Path(path)
    required = ["metadata.yaml", "nodes.fna", "nodes.tsv", "edges.tsv"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ContractError([f"missing required CFA file: {name}" for name in missing])
    metadata = read_yaml(root / "metadata.yaml")
    sequences = _read_fasta(root / "nodes.fna")
    node_header, nodes = read_tsv(root / "nodes.tsv")
    edge_header, edges = read_tsv(root / "edges.tsv")
    colors = None
    labels = None
    if (root / "colors.tsv").is_file():
        _, colors = read_tsv(root / "colors.tsv")
    if (root / "labels.tsv").is_file():
        _, labels = read_tsv(root / "labels.tsv")
    graph = CfaGraph(
        metadata=metadata,
        sequences=sequences,
        nodes=nodes,
        edges=edges,
        colors=colors,
        labels=labels,
        node_header=node_header,
        edge_header=edge_header,
    )
    if validate:
        validate_cfa(graph)
    return graph


def dump_cfa(graph: CfaGraph, path: str | Path) -> None:
    """Write a CFA directory. The graph is validated first."""
    validate_cfa(graph)
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    write_yaml(root / "metadata.yaml", graph.metadata)
    _write_fasta(root / "nodes.fna", graph.sequences, graph.node_ids())
    write_tsv(root / "nodes.tsv", graph.node_header, graph.nodes)
    write_tsv(root / "edges.tsv", graph.edge_header, graph.edges)
    if graph.colors is not None:
        header = list(graph.colors[0].keys()) if graph.colors else ["color_id", "namespace", "value"]
        write_tsv(root / "colors.tsv", header, graph.colors)
    if graph.labels is not None:
        header = list(graph.labels[0].keys()) if graph.labels else ["label_id", "namespace", "value"]
        write_tsv(root / "labels.tsv", header, graph.labels)
