"""Repeat-graph CFA records shared by the in-process benchmarks.

No overlap column is written, so compaction keeps one unitig per node.
"""

from __future__ import annotations

from metametro.formats.cfa.model import CfaGraph


def _color_set(colors: list[int]) -> str:
    return ",".join(str(color) for color in colors)


def records_to_cfa(
    *,
    graph_id: str,
    nodes: list[dict],
    links: list[dict],
    colors: list[dict[str, str]],
) -> CfaGraph:
    """Build a repeat-graph CFA from node and link records.

    ``nodes`` use ``id``, ``sequence``, ``colors``, and ``coverage``.
    ``links`` use ``id``, ``source``, ``target``, ``colors``, and ``coverage``.
    """
    if not nodes:
        from metametro.errors import ContractError

        raise ContractError(["a benchmark CFA needs at least one node"])
    sequences = {str(node["id"]): str(node["sequence"]) for node in nodes}
    node_rows = [
        {
            "node_id": str(node["id"]),
            "coverage": str(float(node["coverage"])),
            "color_set": _color_set([int(color) for color in node["colors"]]),
        }
        for node in nodes
    ]
    edge_rows = [
        {
            "edge_id": str(link["id"]),
            "source": str(link["source"]),
            "target": str(link["target"]),
            "orientation": str(link.get("orientation", "++")),
            "coverage": str(float(link["coverage"])),
            "color_set": _color_set([int(color) for color in link["colors"]]),
        }
        for link in links
    ]
    return CfaGraph(
        metadata={
            "schema_version": "1.0",
            "graph_id": graph_id,
            "graph_type": "repeat",
            "contract": "metagenome_to_graph",
            "contract_version": "1.0",
            "features": {
                "node": {"coverage": "float", "color_set": "color_set"},
                "edge": {
                    "orientation": "orientation",
                    "coverage": "float",
                    "color_set": "color_set",
                },
            },
        },
        sequences=sequences,
        nodes=node_rows,
        edges=edge_rows,
        colors=colors,
        node_header=["node_id", "coverage", "color_set"],
        edge_header=["edge_id", "source", "target", "orientation", "coverage", "color_set"],
    )
