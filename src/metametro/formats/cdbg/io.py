"""Read and write a CDBG directory."""

from __future__ import annotations

from pathlib import Path

from metametro.errors import ContractError
from metametro.formats.cdbg.model import SCHEMA_VERSION, Cdbg, Link, NodeMap, Unitig
from metametro.formats.cdbg.validator import validate_cdbg
from metametro.formats.cfa.validator import parse_color_set
from metametro.tables import read_tsv, read_yaml, write_tsv, write_yaml


def _join_ids(values: list[int] | list[str]) -> str:
    return ",".join(str(value) for value in values)


def _decode_color_piece(piece: str) -> list[int]:
    if piece in {"", "-"}:
        return []
    return parse_color_set(piece)


def _encode_edge_colors(groups: list[list[int]]) -> str:
    return "|".join(_join_ids(colors) if colors else "-" for colors in groups)


def dump_cdbg(graph: Cdbg, path: str | Path) -> None:
    """Write unitigs, links, the color dictionary, and the CFA mapping."""
    validate_cdbg(graph)
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    write_yaml(root / "metadata.yaml", graph.metadata)
    fasta = []
    unitig_rows = []
    for unitig in graph.unitigs:
        fasta.append(f">{unitig.unitig_id}")
        fasta.append(unitig.sequence)
        unitig_rows.append(
            {
                "unitig_id": unitig.unitig_id,
                "color_set": _join_ids(unitig.color_ids),
                "cfa_nodes": ",".join(unitig.members),
                "internal_edge_ids": ",".join(unitig.internal_edge_ids),
                "internal_edge_colors": _encode_edge_colors(unitig.internal_edge_colors),
            }
        )
    (root / "unitigs.fna").write_text("\n".join(fasta) + ("\n" if fasta else ""), encoding="utf-8")
    write_tsv(
        root / "unitigs.tsv",
        ["unitig_id", "color_set", "cfa_nodes", "internal_edge_ids", "internal_edge_colors"],
        unitig_rows,
    )
    write_tsv(
        root / "links.tsv",
        ["link_id", "source", "target", "orientation", "color_set"],
        [
            {
                "link_id": link.link_id,
                "source": link.source,
                "target": link.target,
                "orientation": "" if link.orientation is None else link.orientation,
                "color_set": _join_ids(link.color_ids),
            }
            for link in graph.links
        ],
    )
    write_tsv(
        root / "mapping.tsv",
        ["cfa_node_id", "unitig_id", "ordinal", "length", "color_set"],
        [
            {
                "cfa_node_id": row.cfa_node_id,
                "unitig_id": row.unitig_id,
                "ordinal": row.ordinal,
                "length": row.length,
                "color_set": _join_ids(row.color_ids),
            }
            for row in graph.mapping
        ],
    )
    if graph.colors is not None:
        header = list(graph.colors[0].keys()) if graph.colors else ["color_id", "namespace", "value"]
        write_tsv(root / "colors.tsv", header, graph.colors)
    if graph.labels is not None:
        header = list(graph.labels[0].keys()) if graph.labels else ["label_id", "namespace", "value"]
        write_tsv(root / "labels.tsv", header, graph.labels)


def _read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "":
            continue
        if line.startswith(">"):
            if current is not None:
                sequences[current] = "".join(chunks).upper()
            current = line[1:].split()[0]
            chunks = []
            continue
        chunks.append(line)
    if current is not None:
        sequences[current] = "".join(chunks).upper()
    return sequences


def load_cdbg(path: str | Path, *, validate: bool = True) -> Cdbg:
    """Load a CDBG directory."""
    root = Path(path)
    required = ["metadata.yaml", "unitigs.fna", "unitigs.tsv", "links.tsv", "mapping.tsv"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ContractError([f"missing required CDBG file: {name}" for name in missing])
    metadata = read_yaml(root / "metadata.yaml")
    if str(metadata.get("schema_version", "")) != SCHEMA_VERSION:
        raise ContractError(
            [f"incompatible schema version: {metadata.get('schema_version')!r}"]
        )
    sequences = _read_fasta(root / "unitigs.fna")
    _, unitig_rows = read_tsv(root / "unitigs.tsv")
    unitigs: list[Unitig] = []
    for row in unitig_rows:
        unitig_id = row["unitig_id"]
        if unitig_id not in sequences:
            raise ContractError([f"unitig {unitig_id} has no sequence"])
        members = [token for token in row["cfa_nodes"].split(",") if token]
        internal = [token for token in row.get("internal_edge_ids", "").split(",") if token]
        raw_colors = row.get("internal_edge_colors", "")
        color_groups = [] if raw_colors == "" else [_decode_color_piece(piece) for piece in raw_colors.split("|")]
        unitigs.append(
            Unitig(
                unitig_id=unitig_id,
                sequence=sequences[unitig_id],
                members=members,
                color_ids=parse_color_set(row.get("color_set", "")),
                internal_edge_ids=internal,
                internal_edge_colors=color_groups,
            )
        )
    _, link_rows = read_tsv(root / "links.tsv")
    links = [
        Link(
            link_id=row["link_id"],
            source=row["source"],
            target=row["target"],
            orientation=row["orientation"] or None,
            color_ids=parse_color_set(row.get("color_set", "")),
        )
        for row in link_rows
    ]
    _, map_rows = read_tsv(root / "mapping.tsv")
    mapping = [
        NodeMap(
            cfa_node_id=row["cfa_node_id"],
            unitig_id=row["unitig_id"],
            ordinal=int(row["ordinal"]),
            length=int(row["length"]),
            color_ids=parse_color_set(row.get("color_set", "")),
        )
        for row in map_rows
    ]
    colors = labels = None
    if (root / "colors.tsv").is_file():
        _, colors = read_tsv(root / "colors.tsv")
    if (root / "labels.tsv").is_file():
        _, labels = read_tsv(root / "labels.tsv")
    k = metadata.get("k")
    graph = Cdbg(
        metadata=metadata,
        k=int(k) if isinstance(k, int) else -1,
        unitigs=unitigs,
        links=links,
        mapping=mapping,
        colors=colors,
        labels=labels,
    )
    if validate:
        validate_cdbg(graph)
    return graph
