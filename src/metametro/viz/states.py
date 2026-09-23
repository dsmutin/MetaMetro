"""Geographic drawings of a CFA, its ToCUMG, and its geometric tensor.

The three pages share one numeric scale when coverage is present. A graph
without coverage colours edges by a colour namespace and nodes by degree.
"""

from __future__ import annotations

import math
from pathlib import Path

from metametro.errors import ContractError
from metametro.formats.cfa.model import CfaGraph
from metametro.states import GeometricStates

_SET1 = (
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00",
    "#FFFF33",
    "#A65628",
    "#F781BF",
    "#999999",
)
_UNCLASSIFIED = "#333333"
_OTHER = "#CCCCCC"
_NA = "#CCCCCC"


def plot_geometric_states(
    cfa: CfaGraph,
    states: GeometricStates,
    path: str | Path,
    *,
    edge_namespace: str | None = None,
) -> None:
    """Write a three-page PDF: CFA, ToCUMG, then the geometric tensor.

    Node and edge coverage use one YlGnBu scale across the pages. Without
    coverage, edges use Set1 on ``edge_namespace`` and nodes use degree.
    Panel names are the only titles. Legends sit outside the axes.
    """
    output = Path(path)
    if output.suffix.lower() != ".pdf":
        raise ContractError(["state figure must be a .pdf file"])
    pages = _pages(cfa, states, edge_namespace)
    latitudes = [row[1] for page in pages for row in page["positions"]]
    aspect = 1.0 / math.cos(math.radians(sum(latitudes) / len(latitudes)))
    plt = _pyplot()
    from matplotlib.backends.backend_pdf import PdfPages

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for page in pages:
            _draw_page(plt, pdf, page, aspect)


def _pages(cfa: CfaGraph, states: GeometricStates, edge_namespace: str | None) -> list[dict]:
    cfa_xy = _cfa_positions(cfa)
    order = [row["node_id"] for row in cfa.nodes]
    index = {node_id: position for position, node_id in enumerate(order)}
    cfa_edges = [(index[row["source"]], index[row["target"]]) for row in cfa.edges]
    unitig_index = {unitig.unitig_id: position for position, unitig in enumerate(states.cdbg.unitigs)}
    tensor_xy = _feature_positions(states.cgt)
    link_edges = [(unitig_index[link.source], unitig_index[link.target]) for link in states.cdbg.links]
    sources = _sources(states.cgt)
    tensor_edges = list(zip(sources.tolist(), states.cgt.indices.tolist()))
    cfa_positions = _as_array(order, cfa_xy)
    unitig_positions = _unitig_array(states, cfa_xy)
    numeric = "coverage" in (states.cgt.metadata.get("node_feature_names") or [])
    if numeric:
        by_node = {row["node_id"]: float(row["coverage"]) for row in cfa.nodes}
        by_edge = {row["edge_id"]: float(row["coverage"]) for row in cfa.edges}
        pages = [
            _page("CFA", cfa_positions, cfa_edges, _column(cfa.nodes, "coverage"), _column(cfa.edges, "coverage"), "float", "Coverage", "Coverage", []),
            _page("ToCUMG", unitig_positions, link_edges, _unitig_coverage(states, by_node), _link_coverage(states, by_edge), "float", "Coverage", "Coverage", []),
            _page(
                "CGT",
                tensor_xy,
                tensor_edges,
                _named_feature(states.cgt, "coverage", node=True),
                _named_feature(states.cgt, "coverage", node=False),
                "float",
                "Coverage",
                "Coverage",
                [],
            ),
        ]
        low, high = _shared_limits(pages)
        for page in pages:
            page["limits"] = (low, high)
        return pages
    if edge_namespace is None:
        raise ContractError(["a graph without coverage needs edge_namespace for the drawing"])
    cfa_edge_values, _ = _discrete_edges(cfa, states, edge_namespace, level="cfa")
    link_values, _ = _discrete_edges(cfa, states, edge_namespace, level="tocumg")
    tensor_edge_values, _ = _discrete_edges(cfa, states, edge_namespace, level="cgt")
    categories = _merge_categories(cfa_edge_values, link_values, tensor_edge_values)
    return [
        _page("CFA", cfa_positions, cfa_edges, _degree(len(order), cfa_edges), cfa_edge_values, "discrete", "Degree (count)", edge_namespace, categories),
        _page("ToCUMG", unitig_positions, link_edges, _degree(len(states.cdbg.unitigs), link_edges), link_values, "discrete", "Degree (count)", edge_namespace, categories),
        _page("CGT", tensor_xy, tensor_edges, _degree(states.cgt.num_nodes, tensor_edges), tensor_edge_values, "discrete", "Degree (count)", edge_namespace, categories),
    ]


def _page(name, positions, edges, node_values, edge_values, kind, node_legend, edge_legend, categories) -> dict:
    return {
        "name": name,
        "positions": positions,
        "edges": edges,
        "node_values": node_values,
        "edge_values": edge_values,
        "kind": kind,
        "node_legend": node_legend,
        "edge_legend": edge_legend,
        "categories": categories,
    }


def _cfa_positions(cfa: CfaGraph) -> dict[str, tuple[float, float]]:
    if "longitude" not in cfa.node_header or "latitude" not in cfa.node_header:
        raise ContractError(["state figure needs longitude and latitude"])
    return {row["node_id"]: (float(row["longitude"]), float(row["latitude"])) for row in cfa.nodes}


def _as_array(order: list[str], positions: dict[str, tuple[float, float]]):
    import numpy as np

    return np.asarray([positions[node_id] for node_id in order], dtype=float)


def _unitig_array(states: GeometricStates, positions: dict[str, tuple[float, float]]):
    import numpy as np

    rows = []
    for unitig in states.cdbg.unitigs:
        stacked = np.asarray([positions[member] for member in unitig.members], dtype=float)
        rows.append(stacked.mean(axis=0))
    return np.asarray(rows, dtype=float)


def _feature_positions(cgt):
    import numpy as np

    names = list(cgt.metadata.get("node_feature_names") or [])
    if "longitude" not in names or "latitude" not in names:
        raise ContractError(["geometric tensor is missing longitude or latitude"])
    return np.column_stack(
        [cgt.node_features[:, names.index("longitude")], cgt.node_features[:, names.index("latitude")]]
    ).astype(float)


def _column(rows: list[dict[str, str]], name: str):
    import numpy as np

    return np.asarray([float(row[name]) for row in rows], dtype=float)


def _unitig_coverage(states: GeometricStates, by_node: dict[str, float]):
    import numpy as np

    return np.asarray(
        [float(np.mean([by_node[member] for member in unitig.members])) for unitig in states.cdbg.unitigs],
        dtype=float,
    )


def _shared_limits(pages: list[dict]) -> tuple[float, float]:
    values = [
        float(value)
        for page in pages
        for series in (page["node_values"], page["edge_values"])
        for value in series
    ]
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if low == high:
        high = low + 1.0
    return low, high


def _merge_categories(*groups: list[str]) -> list[str]:
    labels = [label for group in groups for label in group]
    categories = sorted({label for label in labels if label not in {"Unclassified", "Other"}})
    if "Other" in labels:
        categories.append("Other")
    if "Unclassified" in labels:
        categories.append("Unclassified")
    return categories


def _link_coverage(states: GeometricStates, by_edge: dict[str, float]):
    import numpy as np

    return np.asarray([by_edge[link.link_id] for link in states.cdbg.links], dtype=float)


def _named_feature(cgt, name: str, *, node: bool):
    names = list(cgt.metadata["node_feature_names" if node else "edge_feature_names"])
    matrix = cgt.node_features if node else cgt.edge_features
    return matrix[:, names.index(name)].astype(float)


def _degree(count: int, edges: list[tuple[int, int]]):
    import numpy as np

    degree = np.zeros(count, dtype=float)
    for source, target in edges:
        degree[source] += 1
        degree[target] += 1
    return degree


def _sources(cgt):
    import numpy as np

    return np.repeat(np.arange(cgt.num_nodes), np.diff(cgt.indptr))


def _discrete_edges(cfa, states, namespace: str, *, level: str):
    palette = {int(row["color_id"]): row for row in states.cdbg.colors or []}
    wanted = {color_id for color_id, row in palette.items() if row["namespace"] == namespace}
    if not wanted:
        raise ContractError([f"colour namespace {namespace} is not in the CFA"])

    def label(color_ids: list[int]) -> str:
        values = sorted(palette[color_id]["value"] for color_id in color_ids if color_id in wanted)
        if not values:
            return "Unclassified"
        if len(values) == 1:
            return values[0]
        return "Other"

    if level == "cfa":
        from metametro.formats.cfa.validator import parse_color_set

        labels = [label(parse_color_set(row.get("color_set", ""))) for row in cfa.edges]
    elif level == "tocumg":
        labels = [label(link.color_ids) for link in states.cdbg.links]
    else:
        column = {color_id: index for index, color_id in enumerate(states.cgt.color_ids)}
        labels = []
        for slot in range(states.cgt.num_edges):
            present = [color_id for color_id in wanted if states.cgt.edge_colors[slot, column[color_id]]]
            labels.append(label(present))
    categories = sorted({item for item in labels if item not in {"Unclassified", "Other"}})
    if "Other" in labels:
        categories.append("Other")
    if "Unclassified" in labels:
        categories.append("Unclassified")
    return labels, categories


def _draw_page(plt, pdf, page: dict, aspect: float) -> None:
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    figure, axis = plt.subplots(figsize=(7.4, 5.6))
    figure.subplots_adjust(left=0.12, right=0.78, bottom=0.12, top=0.92)
    positions = page["positions"]
    edges = page["edges"]
    segments = [[positions[source], positions[target]] for source, target in edges]
    if page["kind"] == "float":
        node_scale = _float_scale(page["node_values"], plt, page.get("limits"))
        edge_scale = node_scale
        if segments:
            axis.add_collection(
                LineCollection(segments, colors=[_paint(value, edge_scale) for value in page["edge_values"]], linewidths=0.4)
            )
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            s=_marker_size(len(positions)),
            c=[_paint(value, node_scale) for value in page["node_values"]],
            linewidths=0,
            zorder=3,
        )
        _colourbar(figure, node_scale, page["node_legend"], [0.82, 0.18, 0.02, 0.64], plt)
    else:
        colours = _discrete_map(page["categories"])
        edge_colours = [colours[value] for value in page["edge_values"]]
        if segments:
            axis.add_collection(LineCollection(segments, colors=edge_colours, linewidths=0.6))
        node_scale = _float_scale(page["node_values"], plt)
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            s=_marker_size(len(positions)),
            c=[_paint(value, node_scale) for value in page["node_values"]],
            linewidths=0,
            zorder=3,
        )
        _colourbar(figure, node_scale, page["node_legend"], [0.80, 0.18, 0.02, 0.64], plt)
        handles = [
            Line2D([0], [0], color=colours[name], linewidth=1.5, label=name) for name in page["categories"]
        ]
        figure.legend(handles=handles, title=page["edge_legend"], loc="center left", bbox_to_anchor=(0.84, 0.5), frameon=False, fontsize=7, title_fontsize=8)
    _frame(axis, positions, aspect, page["name"])
    pdf.savefig(figure)
    plt.close(figure)


def _float_scale(values, plt, limits: tuple[float, float] | None = None):
    from matplotlib.colors import LogNorm, Normalize

    if limits is None:
        low = float(min(values)) if len(values) else 0.0
        high = float(max(values)) if len(values) else 1.0
        if low == high:
            high = low + 1.0
    else:
        low, high = limits
    if low > 0 and high > low:
        norm = LogNorm(vmin=low, vmax=high)
    else:
        norm = Normalize(vmin=low, vmax=high)
    return {"norm": norm, "cmap": plt.get_cmap("YlGnBu")}


def _paint(value, scale) -> tuple:
    return scale["cmap"](scale["norm"](value))


def _discrete_map(categories: list[str]) -> dict[str, str]:
    colours: dict[str, str] = {}
    cursor = 0
    for name in categories:
        if name == "Unclassified":
            colours[name] = _UNCLASSIFIED
        elif name == "Other":
            colours[name] = _OTHER
        else:
            colours[name] = _SET1[cursor % len(_SET1)]
            cursor += 1
    return colours


def _colourbar(figure, scale, legend: str, box: list[float], plt) -> None:
    from matplotlib.cm import ScalarMappable

    axis = figure.add_axes(box)
    bar = figure.colorbar(ScalarMappable(norm=scale["norm"], cmap=scale["cmap"]), cax=axis)
    bar.set_label(legend, fontsize=8)
    bar.ax.tick_params(labelsize=8)


def _frame(axis, positions, aspect: float, name: str) -> None:
    xs = positions[:, 0]
    ys = positions[:, 1]
    span_x = float(xs.max() - xs.min()) if len(xs) else 1.0
    span_y = float(ys.max() - ys.min()) if len(ys) else 1.0
    pad_x = span_x * 0.03 if span_x else 0.5
    pad_y = span_y * 0.03 if span_y else 0.5
    axis.set_xlim(float(xs.min()) - pad_x, float(xs.max()) + pad_x)
    axis.set_ylim(float(ys.min()) - pad_y, float(ys.max()) + pad_y)
    axis.set_xlabel("Longitude (°E)", fontsize=9)
    axis.set_ylabel("Latitude (°N)", fontsize=9)
    axis.set_title(name, fontsize=10, pad=4)
    axis.tick_params(labelsize=8, length=3, width=0.6)
    axis.set_aspect(aspect, adjustable="box")
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)


def _marker_size(count: int) -> float:
    if count < 2:
        return 18.0
    return max(0.4, 90.0 / math.sqrt(count))


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt
