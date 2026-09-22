"""Expand a CDBG back to CFA using the stored mapping.

The restored CFA keeps node and edge identifiers, sequences, connectivity,
and colours. Numeric feature columns are not part of the CDBG contract and
are not reconstructed; join them from the original CFA through the mapping.
"""

from __future__ import annotations

from metametro.errors import ContractError
from metametro.formats.cdbg.model import Cdbg
from metametro.formats.cdbg.validator import validate_cdbg
from metametro.formats.cfa.model import SCHEMA_VERSION as CFA_SCHEMA
from metametro.formats.cfa.model import CfaGraph


def _split_unitig(sequence: str, lengths: list[int], k: int) -> list[str]:
    overlap = k - 1
    if not lengths:
        return []
    pieces = [sequence[: lengths[0]]]
    cursor = lengths[0]
    for length in lengths[1:]:
        extra = length - overlap
        if extra < 0 or cursor + extra > len(sequence):
            raise ContractError(["cannot split unitig sequence with the stored node lengths"])
        pieces.append(pieces[-1][-overlap:] + sequence[cursor : cursor + extra])
        cursor += extra
    if cursor != len(sequence):
        raise ContractError(["unitig sequence length does not match mapped node lengths"])
    return pieces


def cdbg_to_cfa(cdbg: Cdbg) -> CfaGraph:
    """Rebuild CFA sequences and topology from a CDBG."""
    validate_cdbg(cdbg)
    by_unitig: dict[str, list] = {}
    for row in cdbg.mapping:
        by_unitig.setdefault(row.unitig_id, []).append(row)
    sequences: dict[str, str] = {}
    nodes: list[dict[str, str]] = []
    for unitig in cdbg.unitigs:
        rows = sorted(by_unitig[unitig.unitig_id], key=lambda item: item.ordinal)
        lengths = [row.length for row in rows]
        if cdbg.metadata.get("compaction") == "identity":
            if len(rows) != 1 or rows[0].length != len(unitig.sequence):
                raise ContractError([f"identity unitig {unitig.unitig_id} has an inconsistent mapping"])
            parts = [unitig.sequence]
        else:
            parts = _split_unitig(unitig.sequence, lengths, cdbg.k)
        for row, sequence in zip(rows, parts):
            if len(sequence) != row.length:
                raise ContractError([f"restored length mismatch for {row.cfa_node_id}"])
            sequences[row.cfa_node_id] = sequence
            color_set = ",".join(str(color_id) for color_id in row.color_ids)
            nodes.append({"node_id": row.cfa_node_id, "color_set": color_set})
    nodes.sort(key=lambda row: row["node_id"])
    edges: list[dict[str, str]] = []
    for unitig in cdbg.unitigs:
        rows = sorted(by_unitig[unitig.unitig_id], key=lambda item: item.ordinal)
        for index, edge_id in enumerate(unitig.internal_edge_ids):
            colors = unitig.internal_edge_colors[index] if index < len(unitig.internal_edge_colors) else []
            edges.append(
                {
                    "edge_id": edge_id,
                    "source": rows[index].cfa_node_id,
                    "target": rows[index + 1].cfa_node_id,
                    "orientation": "++",
                    "color_set": ",".join(str(color_id) for color_id in colors),
                }
            )
    for link in cdbg.links:
        edges.append(
            {
                "edge_id": link.link_id,
                "source": _endpoint_node(by_unitig[link.source], start=False),
                "target": _endpoint_node(by_unitig[link.target], start=True),
                "orientation": "++" if link.orientation in (None, "++") else link.orientation,
                "color_set": ",".join(str(color_id) for color_id in link.color_ids),
            }
        )
    edges.sort(key=lambda row: row["edge_id"])
    has_colors = cdbg.colors is not None
    metadata = {
        "schema_version": CFA_SCHEMA,
        "graph_id": cdbg.metadata.get("graph_id"),
        "graph_type": cdbg.metadata.get("graph_type") or "de_bruijn",
        "k": cdbg.k,
        "features": {
            "node": {"color_set": "color_set"},
            "edge": {"orientation": "orientation", "color_set": "color_set"},
        },
    }
    if not has_colors:
        metadata["features"] = {
            "node": {},
            "edge": {"orientation": "orientation"},
        }
        for row in nodes:
            row.pop("color_set", None)
        for row in edges:
            row.pop("color_set", None)
    node_header = ["node_id"] + (["color_set"] if has_colors else [])
    edge_header = ["edge_id", "source", "target", "orientation"] + (["color_set"] if has_colors else [])
    return CfaGraph(
        metadata=metadata,
        sequences=sequences,
        nodes=nodes,
        edges=edges,
        colors=None if cdbg.colors is None else [dict(row) for row in cdbg.colors],
        labels=None if cdbg.labels is None else [dict(row) for row in cdbg.labels],
        node_header=node_header,
        edge_header=edge_header,
    )


def _endpoint_node(rows: list, *, start: bool) -> str:
    ordered = sorted(rows, key=lambda item: item.ordinal)
    return ordered[0].cfa_node_id if start else ordered[-1].cfa_node_id
