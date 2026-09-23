"""Build a de Bruijn CFA from a directed graph and its attributes.

Node sequences use the same repeat-junction tape as the transit CFA, so every
forward edge overlaps by ``k - 1``. Each named attribute becomes a colour
namespace. An element carries the colour of every non-empty attribute value
it has.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from metametro.contracts.colouring import colour_cfa
from metametro.errors import ContractError
from metametro.formats.cfa.model import SCHEMA_VERSION, CfaGraph
from metametro.transit_cfa import repeat_junction_sequences


def coloured_digraph_cfa(
    nodes: Sequence[Mapping[str, str]],
    edges: Sequence[Mapping[str, str]],
    *,
    graph_id: str,
    k: int = 21,
    colour_columns: Sequence[str],
    keep_columns: Sequence[str] = (),
) -> CfaGraph:
    """Return a coloured CFA for this directed graph.

    ``nodes`` require ``node_id``. ``edges`` require ``source`` and ``target``.
    ``colour_columns`` are the attribute names copied into ``colors.tsv``.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ContractError(["digraph CFA requires integer k >= 2"])
    if not nodes:
        raise ContractError(["digraph CFA needs at least one node"])
    if not colour_columns:
        raise ContractError(["digraph CFA needs at least one colour column"])
    node_ids = [row["node_id"] for row in nodes]
    if any(node_id == "" for node_id in node_ids) or len(node_ids) != len(set(node_ids)):
        raise ContractError(["node ids must be present and unique"])
    known = set(node_ids)
    directed: list[tuple[str, str]] = []
    edge_rows: list[dict[str, str]] = []
    for index, row in enumerate(edges, start=1):
        source = row.get("source", "")
        target = row.get("target", "")
        if source not in known or target not in known:
            raise ContractError([f"edge {index} has an endpoint outside the node table"])
        directed.append((source, target))
        edge_rows.append(
            {
                "edge_id": f"e{index:06d}",
                "source": source,
                "target": target,
                "orientation": "++",
            }
        )
    sequences = repeat_junction_sequences(node_ids, directed, k)
    colors, node_colors, edge_colors = _colours(nodes, edge_rows, edges, colour_columns)
    node_features = {column: "str" for column in colour_columns if any(row.get(column, "") != "" for row in nodes)}
    node_features.update(_kept(nodes, keep_columns))
    edge_features = {"orientation": "orientation"}
    edge_features.update(
        {column: "str" for column in colour_columns if any(row.get(column, "") != "" for row in edges)}
    )
    node_table = []
    for row in nodes:
        stored = {"node_id": row["node_id"]}
        for column in node_features:
            stored[column] = _cell(row.get(column, ""))
        node_table.append(stored)
    for stored, raw in zip(edge_rows, edges):
        for column in edge_features:
            if column == "orientation":
                continue
            stored[column] = _cell(raw.get(column, ""))
    bare = CfaGraph(
        metadata={
            "schema_version": SCHEMA_VERSION,
            "graph_id": graph_id,
            "graph_type": "de_bruijn",
            "k": k,
            "contract": "graph_to_cfa",
            "contract_version": "1.0",
            "sequence_rule": "repeat_junction_stream",
            "features": {"node": node_features, "edge": edge_features},
        },
        sequences=sequences,
        nodes=node_table,
        edges=edge_rows,
        node_header=["node_id", *node_features.keys()],
        edge_header=["edge_id", "source", "target", *edge_features.keys()],
    )
    return colour_cfa(
        bare,
        node_colors,
        edge_colors,
        operation="replace",
        colors=colors,
        target=("node", "edge"),
    )


def _colours(nodes, edge_rows, raw_edges, columns):
    values: dict[str, list[str]] = {column: [] for column in columns}
    seen: dict[str, set[str]] = {column: set() for column in columns}
    for column in columns:
        for row in list(nodes) + list(raw_edges):
            value = _cell(row.get(column, ""))
            if value and value not in seen[column]:
                seen[column].add(value)
                values[column].append(value)
    colors: list[dict[str, str]] = []
    index_of: dict[tuple[str, str], int] = {}
    next_id = 0
    for column in columns:
        for value in values[column]:
            index_of[(column, value)] = next_id
            colors.append({"color_id": str(next_id), "namespace": column, "value": value})
            next_id += 1
    if not colors:
        raise ContractError(["colour columns have no values"])

    def paint(row: Mapping[str, str]) -> list[int]:
        ids = []
        for column in columns:
            value = _cell(row.get(column, ""))
            if value:
                ids.append(index_of[(column, value)])
        return sorted(ids)

    node_colors = {row["node_id"]: paint(row) for row in nodes}
    edge_colors = {stored["edge_id"]: paint(raw) for stored, raw in zip(edge_rows, raw_edges)}
    return colors, node_colors, edge_colors


def _kept(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> dict[str, str]:
    features: dict[str, str] = {}
    for column in columns:
        values = [row.get(column, "") for row in rows if row.get(column, "") != ""]
        if not values:
            continue
        numeric = True
        for value in values:
            try:
                float(value)
            except ValueError:
                numeric = False
                break
        features[column] = "float" if numeric else "str"
    return features


def _cell(text: str) -> str:
    return " ".join(str(text).replace("\t", " ").split())
