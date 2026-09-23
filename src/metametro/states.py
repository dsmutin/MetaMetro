"""CFA to a ToCUMG and a geometric coloured graph tensor.

The tensor stores mean longitude and latitude of each unitig, so the same
drawing can be read back from the arrays. Node classes and edge multi-labels
are derived from colour namespaces. They are targets, not input features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metametro.converters.cdbg_to_cgt import cdbg_to_cgt
from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
from metametro.errors import ContractError
from metametro.formats.cdbg.model import Cdbg
from metametro.formats.cfa.model import CfaGraph
from metametro.formats.cgt.model import Cgt


@dataclass
class GeometricStates:
    """One CFA compacted to a ToCUMG and a coordinate-carrying CGT.

    ``node_labels`` has shape ``(N,)`` and indexes ``node_class_names``.
    ``edge_targets`` has shape ``(E, L)`` in CSR order. Each column is an
    independent 0/1 label named in ``edge_label_names``.
    """

    cdbg: Cdbg
    cgt: Cgt
    node_class_names: list[str]
    edge_label_names: list[str]
    edge_targets: np.ndarray


def build_geometric_states(
    cfa: CfaGraph,
    *,
    node_namespace: str,
    edge_namespaces: tuple[str, ...],
) -> GeometricStates:
    """Compact ``cfa`` and attach geometry plus colour targets.

    A unitig coordinate is the mean longitude and latitude of its CFA nodes.
    A numeric ``coverage`` column is averaged the same way and copied onto the
    link that kept that CFA edge. The node class is the sorted combination of
    ``node_namespace`` colours on the unitig. A unitig with none of those
    colours takes the majority value of that namespace on its incident links.
    A remaining unitig is ``Unclassified``. Each edge column is one colour in
    ``edge_namespaces``.
    """
    if not edge_namespaces:
        raise ContractError(["edge label namespaces are required"])
    longitudes, latitudes = _coordinates(cfa)
    coverage = _optional_float(cfa.nodes, "coverage", "node_id", "node")
    edge_coverage = _optional_float(cfa.edges, "coverage", "edge_id", "edge")
    cdbg = cfa_to_cdbg(cfa)
    if not cdbg.colors:
        raise ContractError(["CFA has no colours to turn into labels"])
    palette = {int(row["color_id"]): row for row in cdbg.colors}
    _require_namespaces(palette, (node_namespace, *edge_namespaces))

    node_features: dict[str, np.ndarray] = {}
    for unitig in cdbg.unitigs:
        values = [longitudes[member] for member in unitig.members]
        row = [float(np.mean(values)), float(np.mean([latitudes[member] for member in unitig.members]))]
        if coverage is not None:
            row.append(float(np.mean([coverage[member] for member in unitig.members])))
        node_features[unitig.unitig_id] = np.asarray(row, dtype=np.float32)
    node_names = ["longitude", "latitude"] + (["coverage"] if coverage is not None else [])
    edge_features = None
    edge_names: list[str] = []
    if edge_coverage is not None and cdbg.links:
        missing = [link.link_id for link in cdbg.links if link.link_id not in edge_coverage]
        if missing:
            raise ContractError([f"link {missing[0]} has no coverage on the CFA edge"])
        edge_features = {
            link.link_id: np.asarray([edge_coverage[link.link_id]], dtype=np.float32) for link in cdbg.links
        }
        edge_names = ["coverage"]
    cgt = cdbg_to_cgt(
        cdbg,
        node_features=node_features,
        edge_features=edge_features,
        node_feature_names=node_names,
        edge_feature_names=edge_names,
    )
    names, labels, edge_names_out, targets = _targets(
        cgt,
        palette,
        node_namespace=node_namespace,
        edge_namespaces=edge_namespaces,
    )
    cgt.node_labels = labels
    cgt.metadata["node_class_names"] = list(names)
    cgt.metadata["edge_label_names"] = list(edge_names_out)
    cgt.metadata["geometry"] = "mean longitude and latitude of the CFA nodes in each unitig"
    return GeometricStates(
        cdbg=cdbg,
        cgt=cgt,
        node_class_names=names,
        edge_label_names=edge_names_out,
        edge_targets=targets,
    )


def _coordinates(cfa: CfaGraph) -> tuple[dict[str, float], dict[str, float]]:
    if "longitude" not in cfa.node_header or "latitude" not in cfa.node_header:
        raise ContractError(["geometric tensor requires longitude and latitude columns"])
    longitudes: dict[str, float] = {}
    latitudes: dict[str, float] = {}
    for row in cfa.nodes:
        try:
            longitude = float(row["longitude"])
            latitude = float(row["latitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError([f"node {row.get('node_id', '')} has a non-numeric coordinate"]) from exc
        if not np.isfinite(longitude) or not np.isfinite(latitude):
            raise ContractError([f"node {row['node_id']} has a non-finite coordinate"])
        longitudes[row["node_id"]] = longitude
        latitudes[row["node_id"]] = latitude
    return longitudes, latitudes


def _optional_float(
    rows: list[dict[str, str]],
    column: str,
    id_column: str,
    kind: str,
) -> dict[str, float] | None:
    if not rows or column not in rows[0]:
        return None
    values: dict[str, float] = {}
    for row in rows:
        try:
            number = float(row[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError([f"{kind} {row.get(id_column, '')} has a non-numeric {column}"]) from exc
        if not np.isfinite(number):
            raise ContractError([f"{kind} {row[id_column]} has a non-finite {column}"])
        values[row[id_column]] = number
    return values


def _require_namespaces(palette: dict[int, dict[str, str]], namespaces: tuple[str, ...]) -> None:
    present = {row["namespace"] for row in palette.values()}
    missing = [name for name in namespaces if name not in present]
    if missing:
        raise ContractError([f"colour namespace {name} is not in the CFA" for name in missing])


def _targets(
    cgt: Cgt,
    palette: dict[int, dict[str, str]],
    *,
    node_namespace: str,
    edge_namespaces: tuple[str, ...],
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    column = {color_id: index for index, color_id in enumerate(cgt.color_ids)}
    node_columns = [color_id for color_id in cgt.color_ids if palette[color_id]["namespace"] == node_namespace]
    edge_columns = [
        color_id
        for namespace in edge_namespaces
        for color_id in cgt.color_ids
        if palette[color_id]["namespace"] == namespace
    ]
    edge_label_names = [f"{palette[color_id]['namespace']}={palette[color_id]['value']}" for color_id in edge_columns]
    edge_index = np.asarray([column[color_id] for color_id in edge_columns], dtype=np.int64)
    if edge_index.size:
        targets = cgt.edge_colors[:, edge_index].astype(np.uint8, copy=True)
    else:
        targets = np.zeros((cgt.num_edges, 0), dtype=np.uint8)

    sources = np.repeat(np.arange(cgt.num_nodes, dtype=np.int64), np.diff(cgt.indptr))
    raw_names: list[str] = []
    for node in range(cgt.num_nodes):
        owned = [
            palette[color_id]["value"]
            for color_id in node_columns
            if cgt.node_colors[node, column[color_id]]
        ]
        if owned:
            raw_names.append("+".join(sorted(owned)))
            continue
        raw_names.append(_majority_incident(node, node_columns, column, palette, cgt, sources))
    ordered = sorted({name for name in raw_names if name != "Unclassified"})
    if any(name == "Unclassified" for name in raw_names):
        ordered.append("Unclassified")
    index = {name: position for position, name in enumerate(ordered)}
    labels = np.asarray([index[name] for name in raw_names], dtype=np.int64)
    return ordered, labels, edge_label_names, targets


def _majority_incident(
    node: int,
    colour_ids: list[int],
    column: dict[int, int],
    palette: dict[int, dict[str, str]],
    cgt: Cgt,
    sources: np.ndarray,
) -> str:
    slots = np.flatnonzero((sources == node) | (cgt.indices == node))
    counts: dict[str, int] = {}
    for slot in slots:
        for color_id in colour_ids:
            if cgt.edge_colors[int(slot), column[color_id]]:
                value = palette[color_id]["value"]
                counts[value] = counts.get(value, 0) + 1
    if not counts:
        return "Unclassified"
    best = max(counts.values())
    return sorted(name for name, count in counts.items() if count == best)[0]
