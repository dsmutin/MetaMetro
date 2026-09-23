"""Expand a CDBG back to CFA using the stored mapping.

The restored CFA keeps node and edge identifiers, sequences, connectivity,
and colours. Numeric feature columns are not part of the CDBG contract and
are not reconstructed; join them from the original CFA through the mapping.
"""

from __future__ import annotations

from metametro.errors import ContractError
from metametro.formats.cdbg.model import Cdbg
from metametro.formats.cdbg.validator import validate_cdbg
from metametro.formats.cdbg.sequence import junction_overlaps, split_unitig
from metametro.formats.cfa.model import SCHEMA_VERSION as CFA_SCHEMA
from metametro.formats.cfa.model import CfaGraph


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
            parts = split_unitig(unitig.sequence, lengths, junction_overlaps(unitig, cdbg.k))
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
        if len(unitig.internal_edge_colors) != len(unitig.internal_edge_ids):
            raise ContractError(
                [f"unitig {unitig.unitig_id} edge colours do not match its internal edges"]
            )
        for index, edge_id in enumerate(unitig.internal_edge_ids):
            colors = unitig.internal_edge_colors[index]
            edge = {
                "edge_id": edge_id,
                "source": rows[index].cfa_node_id,
                "target": rows[index + 1].cfa_node_id,
                "orientation": "++",
                "color_set": ",".join(str(color_id) for color_id in colors),
            }
            if index < len(unitig.internal_overlaps):
                edge["overlap"] = str(unitig.internal_overlaps[index])
            edges.append(edge)
    for link in cdbg.links:
        edge = {
            "edge_id": link.link_id,
            "source": _endpoint_node(by_unitig[link.source], start=False),
            "target": _endpoint_node(by_unitig[link.target], start=True),
            "orientation": "++" if link.orientation in (None, "++") else link.orientation,
            "color_set": ",".join(str(color_id) for color_id in link.color_ids),
        }
        if link.overlap is not None:
            edge["overlap"] = str(link.overlap)
        edges.append(edge)
    edges.sort(key=lambda row: row["edge_id"])
    if edges and not all("overlap" in row for row in edges):
        for row in edges:
            row.pop("overlap", None)
    has_colors = cdbg.colors is not None
    has_overlap = bool(edges) and all("overlap" in row for row in edges)
    graph_type = str(cdbg.metadata.get("graph_type") or "")
    if not graph_type:
        raise ContractError(["CDBG metadata is missing graph_type"])
    edge_features: dict[str, str] = {"orientation": "orientation"}
    if has_colors:
        edge_features["color_set"] = "color_set"
    if has_overlap:
        edge_features["overlap"] = "int"
    metadata: dict = {
        "schema_version": CFA_SCHEMA,
        "graph_id": cdbg.metadata.get("graph_id"),
        "graph_type": graph_type,
        "features": {
            "node": {"color_set": "color_set"} if has_colors else {},
            "edge": edge_features,
        },
    }
    if cdbg.k is not None:
        metadata["k"] = cdbg.k
    if isinstance(cdbg.metadata.get("overlap"), int) and not isinstance(cdbg.metadata.get("overlap"), bool):
        metadata["overlap"] = cdbg.metadata["overlap"]
    if not has_colors:
        for row in nodes:
            row.pop("color_set", None)
        for row in edges:
            row.pop("color_set", None)
    node_header = ["node_id"] + (["color_set"] if has_colors else [])
    edge_header = ["edge_id", "source", "target", "orientation"]
    if has_colors:
        edge_header.append("color_set")
    if has_overlap:
        edge_header.append("overlap")
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
