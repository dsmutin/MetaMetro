"""CFA to CDBG compaction."""

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
    outs: list[dict] = field(default_factory=list)


def _row_colors(row: dict[str, str]) -> list[int]:
    raw = row.get("color_set", row.get("colors", ""))
    return sorted(set(parse_color_set(raw)))


def _overlaps(left: str, right: str, k: int) -> bool:
    overlap = k - 1
    if overlap == 0:
        return True
    if len(left) < overlap or len(right) < overlap:
        return False
    return left[-overlap:] == right[:overlap]


def cfa_to_cdbg(cfa: CfaGraph) -> Cdbg:
    """Compact a CFA graph into a CDBG and keep a CFA node mapping.

    De Bruijn graphs merge an edge ``u -> v`` when ``u`` has out-degree 1,
    ``v`` has in-degree 1, the edge is forward (``++`` or orientation omitted),
    and the sequences overlap by ``k - 1``. A mismatch raises ``ContractError``
    instead of dropping the edge. Other graph types become identity unitigs.
    Unitig colours are the union of member node colours. Per-node colours stay
    on the mapping.
    """
    validate_cfa(cfa)
    graph_type = str(cfa.metadata.get("graph_type"))
    de_bruijn = graph_type == "de_bruijn"
    k = int(cfa.metadata["k"]) if de_bruijn else int(cfa.metadata.get("k") or 1)
    sequences = cfa.sequences
    if de_bruijn:
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
    for row in sorted(cfa.edges, key=lambda item: item["edge_id"]):
        orientation = row.get("orientation") or None
        if orientation == "":
            orientation = None
        colors = _row_colors(row)
        if de_bruijn and orientation in (None, "++"):
            if not _overlaps(sequences[row["source"]], sequences[row["target"]], k):
                raise ContractError(
                    [f"overlap mismatch on edge {row['edge_id']}; refusing to drop the edge"]
                )
        supers[row["source"]].outs.append(
            {
                "edge_id": row["edge_id"],
                "target": row["target"],
                "orientation": orientation,
                "color_ids": colors,
            }
        )

    if de_bruijn:
        _compact(supers, k)

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
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "graph_id": cfa.metadata.get("graph_id"),
        "graph_type": graph_type,
        "k": k,
        "contract": "cfa_to_cdbg",
        "contract_version": "1.0",
        "compaction": "de_bruijn_chain" if de_bruijn else "identity",
        "source": {"format": "cfa", "graph_id": cfa.metadata.get("graph_id")},
    }
    return Cdbg(
        metadata=metadata,
        k=k,
        unitigs=unitigs,
        links=links,
        mapping=mapping,
        colors=None if cfa.colors is None else [dict(row) for row in cfa.colors],
        labels=None if cfa.labels is None else [dict(row) for row in cfa.labels],
    )


def _compact(supers: dict[str, _Super], k: int) -> None:
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
            candidates.append((representative, edge["edge_id"], target))
        if not candidates:
            return
        representative, edge_id, target = candidates[0]
        _merge(supers, representative, target, edge_id, k)


def _merge(supers: dict[str, _Super], representative: str, target: str, edge_id: str, k: int) -> None:
    left = supers[representative]
    right = supers[target]
    overlap = k - 1
    if left.sequence[-overlap:] != right.sequence[:overlap]:
        raise ContractError([f"overlap mismatch on edge {edge_id}; refusing to drop the edge"])
    left.sequence = left.sequence + right.sequence[overlap:]
    left.members.extend(right.members)
    left.color_ids |= right.color_ids
    consumed = left.outs[0]
    left.internal_edge_ids.append(edge_id)
    left.internal_edge_colors.append(list(consumed["color_ids"]))
    left.outs = list(right.outs)
    for super_node in supers.values():
        for edge in super_node.outs:
            if edge["target"] == target:
                edge["target"] = representative
    del supers[target]
