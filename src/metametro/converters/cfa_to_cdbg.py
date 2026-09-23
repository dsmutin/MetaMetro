"""CFA to CDBG compaction.

CDBG is the on-disk form of a ToCUMG (totally coloured universal
metagenomic graph). Compaction keeps ``graph_type``. It does not relabel a
repeat graph, an LCA graph, or any other graph as a de Bruijn graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from metametro.errors import ContractError
from metametro.formats.cdbg.model import SCHEMA_VERSION, Cdbg, Link, NodeMap, Unitig
from metametro.formats.cfa.model import CfaGraph
from metametro.formats.cfa.validator import parse_color_set, validate_cfa


@dataclass
class _Super:
    representative: str
    members: list[str]
    sequence: str
    color_ids: set[int]
    internal_edge_ids: list[str] = field(default_factory=list)
    internal_edge_colors: list[list[int]] = field(default_factory=list)
    internal_overlaps: list[int] = field(default_factory=list)
    outs: list[dict] = field(default_factory=list)


def _row_colors(row: dict[str, str]) -> list[int]:
    raw = row.get("color_set", row.get("colors", ""))
    return sorted(set(parse_color_set(raw)))


def _metadata_overlap(metadata: dict) -> int | None:
    if "overlap" not in metadata or metadata.get("overlap") is None:
        return None
    raw = metadata["overlap"]
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ContractError(["overlap must be an integer >= 0"])
    return raw


def _edge_overlap(row: dict[str, str], *, graph_type: str, k: int | None, default: int | None) -> int | None:
    raw = row.get("overlap")
    if raw not in (None, ""):
        token = str(raw).strip()
        if not token.isdigit():
            raise ContractError([f"overlap on edge {row.get('edge_id', '')} must be an integer >= 0"])
        return int(token)
    if graph_type == "de_bruijn":
        if k is None or k < 1:
            raise ContractError(["de Bruijn compaction requires integer k > 0"])
        return k - 1
    return default


def _sequences_overlap(left: str, right: str, overlap: int) -> bool:
    if overlap < 0:
        return False
    if overlap == 0:
        return True
    if len(left) < overlap or len(right) < overlap:
        return False
    return left[-overlap:] == right[:overlap]


def cfa_to_cdbg(cfa: CfaGraph) -> Cdbg:
    """Compact a CFA graph into a ToCUMG and keep a CFA node mapping.

    Any ``graph_type`` is preserved. A forward edge (``++`` or orientation
    omitted) is merged when the source has out-degree 1, the target has
    in-degree 1, and the two nodes are distinct. The overlap is ``k - 1`` for
    ``de_bruijn``, the edge ``overlap`` column when that column is present,
    or metadata ``overlap`` otherwise. A defined overlap that does not match
    the sequences raises ``ContractError``. A graph with no overlap contract
    keeps one unitig per node (``compaction: identity``) and the same type.
    ``k`` is stored only when the CFA declared it.

    Unitig colours are the union of member node colours. Per-node colours stay
    on the mapping.
    """
    validate_cfa(cfa)
    graph_type = str(cfa.metadata.get("graph_type"))
    raw_k = cfa.metadata.get("k")
    if isinstance(raw_k, bool) or (raw_k is not None and not isinstance(raw_k, int)):
        raise ContractError(["k must be an integer > 0 when present"])
    k: int | None = raw_k
    if graph_type == "de_bruijn" and (k is None or k <= 0):
        raise ContractError(["de Bruijn compaction requires integer k > 0"])
    if k is not None and k <= 0:
        raise ContractError(["k must be an integer > 0 when present"])
    default_overlap = _metadata_overlap(cfa.metadata)
    sequences = cfa.sequences
    if graph_type == "de_bruijn" and k is not None:
        short = [node_id for node_id, sequence in sequences.items() if len(sequence) < k]
        if short:
            raise ContractError([f"node {node_id} is shorter than k" for node_id in short])

    node_colors = {row["node_id"]: _row_colors(row) for row in cfa.nodes}
    node_lengths = {node_id: len(sequence) for node_id, sequence in sequences.items()}
    supers = {
        node_id: _Super(
            representative=node_id,
            members=[node_id],
            sequence=sequences[node_id],
            color_ids=set(node_colors.get(node_id, [])),
        )
        for node_id in cfa.node_ids()
    }
    saw_overlap = False
    overlaps_used: set[int] = set()
    for row in sorted(cfa.edges, key=lambda item: item["edge_id"]):
        orientation = row.get("orientation") or None
        if orientation == "":
            orientation = None
        overlap = _edge_overlap(row, graph_type=graph_type, k=k, default=default_overlap)
        if overlap is not None:
            saw_overlap = True
            overlaps_used.add(overlap)
            if orientation in (None, "++") and not _sequences_overlap(
                sequences[row["source"]], sequences[row["target"]], overlap
            ):
                raise ContractError(
                    [f"overlap mismatch on edge {row['edge_id']}; refusing to drop the edge"]
                )
        supers[row["source"]].outs.append(
            {
                "edge_id": row["edge_id"],
                "target": row["target"],
                "orientation": orientation,
                "color_ids": _row_colors(row),
                "overlap": overlap,
            }
        )

    if saw_overlap:
        _compact(supers)

    ordered = sorted(supers.values(), key=lambda item: tuple(item.members))
    id_of = {}
    unitigs: list[Unitig] = []
    for index, super_node in enumerate(ordered, start=1):
        unitig_id = f"u{index:06d}"
        id_of[super_node.representative] = unitig_id
        unitigs.append(
            Unitig(
                unitig_id=unitig_id,
                sequence=super_node.sequence,
                members=list(super_node.members),
                color_ids=sorted(super_node.color_ids),
                internal_edge_ids=list(super_node.internal_edge_ids),
                internal_edge_colors=[list(colors) for colors in super_node.internal_edge_colors],
                internal_overlaps=list(super_node.internal_overlaps),
            )
        )
    links = []
    for super_node in ordered:
        source = id_of[super_node.representative]
        for edge in super_node.outs:
            links.append(
                Link(
                    link_id=edge["edge_id"],
                    source=source,
                    target=id_of[edge["target"]],
                    orientation=edge["orientation"],
                    color_ids=list(edge["color_ids"]),
                    overlap=edge["overlap"],
                )
            )
    links.sort(key=lambda link: (link.source, link.target, link.link_id))
    mapping: list[NodeMap] = []
    for unitig in unitigs:
        for ordinal, node_id in enumerate(unitig.members):
            mapping.append(
                NodeMap(
                    cfa_node_id=node_id,
                    unitig_id=unitig.unitig_id,
                    ordinal=ordinal,
                    length=node_lengths[node_id],
                    color_ids=list(node_colors.get(node_id, [])),
                )
            )
    if graph_type == "de_bruijn":
        compaction = "de_bruijn_chain"
    elif saw_overlap:
        compaction = "chain"
    else:
        compaction = "identity"
    metadata: dict = {
        "schema_version": SCHEMA_VERSION,
        "graph_id": cfa.metadata.get("graph_id"),
        "graph_type": graph_type,
        "contract": "cfa_to_cdbg",
        "contract_version": "1.0",
        "compaction": compaction,
        "source": {"format": "cfa", "graph_id": cfa.metadata.get("graph_id")},
    }
    if k is not None:
        metadata["k"] = k
    if len(overlaps_used) == 1:
        metadata["overlap"] = next(iter(overlaps_used))
    return Cdbg(
        metadata=metadata,
        k=k,
        unitigs=unitigs,
        links=links,
        mapping=mapping,
        colors=None if cfa.colors is None else [dict(row) for row in cfa.colors],
        labels=None if cfa.labels is None else [dict(row) for row in cfa.labels],
    )


def _compact(supers: dict[str, _Super]) -> None:
    while True:
        indeg = {key: 0 for key in supers}
        for super_node in supers.values():
            for edge in super_node.outs:
                indeg[edge["target"]] += 1
        candidates: list[tuple[str, str, str]] = []
        for representative in sorted(supers):
            super_node = supers[representative]
            if len(super_node.outs) != 1:
                continue
            edge = super_node.outs[0]
            target = edge["target"]
            if target == representative or indeg[target] != 1:
                continue
            if edge["orientation"] not in (None, "++"):
                continue
            if edge["overlap"] is None:
                continue
            candidates.append((representative, edge["edge_id"], target))
        if not candidates:
            return
        representative, edge_id, target = candidates[0]
        _merge(supers, representative, target, edge_id)


def _merge(supers: dict[str, _Super], representative: str, target: str, edge_id: str) -> None:
    left = supers[representative]
    right = supers[target]
    consumed = left.outs[0]
    overlap = consumed["overlap"]
    if overlap is None or not _sequences_overlap(left.sequence, right.sequence, overlap):
        raise ContractError([f"overlap mismatch on edge {edge_id}; refusing to drop the edge"])
    left.sequence = left.sequence + right.sequence[overlap:]
    left.members.extend(right.members)
    left.color_ids |= right.color_ids
    left.internal_edge_ids.append(edge_id)
    left.internal_edge_ids.extend(right.internal_edge_ids)
    left.internal_edge_colors.append(list(consumed["color_ids"]))
    left.internal_edge_colors.extend(list(colors) for colors in right.internal_edge_colors)
    left.internal_overlaps.append(overlap)
    left.internal_overlaps.extend(right.internal_overlaps)
    left.outs = list(right.outs)
    for super_node in supers.values():
        for edge in super_node.outs:
            if edge["target"] == target:
                edge["target"] = representative
    del supers[target]
