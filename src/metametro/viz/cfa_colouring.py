"""Faceted CFA colouring figures.

Every node and every edge is drawn in every facet. An element that does not
carry the facet colour stays in the panel with the NA colour. It is not
dropped.
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from metametro.errors import ContractError
from metametro.formats.cfa.model import CfaGraph
from metametro.formats.cfa.validator import parse_color_set, validate_cfa

NA_COLOUR = "#CCCCCC"
UNCLASSIFIED_COLOUR = "#333333"
OTHER_COLOUR = "#CCCCCC"
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
_ROUTE_NUMBER = re.compile(r"^(\d+)")


@dataclass(frozen=True)
class ColourLabel:
    """How one layer (nodes or edges) reads colours.tsv.

    ``namespace`` selects dictionary rows. ``matches`` keeps the values that
    belong to the current facet level. ``reduce`` turns that list into one
    visual value. ``None`` is NA. ``palette`` is a matplotlib colormap name
    for numeric values, or ``Set1`` for text values. ``limits`` fixes the
    numeric scale; otherwise the scale spans the finite values.
    """

    namespace: str
    legend: str
    matches: Callable[[str, str], bool]
    reduce: Callable[[Sequence[str]], float | str | None]
    palette: str = "YlGnBu"
    limits: tuple[float, float] | None = None


@dataclass(frozen=True)
class ColourFacet:
    """One panel per level of a colour namespace.

    ``level_labels`` are the panel captions. They are not a figure title.
    """

    namespace: str
    levels: tuple[str, ...] | None = None
    level_labels: Mapping[str, str] | None = None

    def caption(self, level: str) -> str:
        """Return the panel caption for one facet level."""
        if self.level_labels and level in self.level_labels:
            return self.level_labels[level]
        return level


@dataclass(frozen=True)
class ColourFrames:
    """Resolved values. Each facet maps every node id and every edge id."""

    levels: tuple[str, ...]
    nodes: dict[str, dict[str, float | str | None]]
    edges: dict[str, dict[str, float | str | None]]


def route_number(value: str) -> int | None:
    """Return the leading integer in a route colour such as ``bus:145Б``.

    The token after the first colon is the short name. A value with no
    digits has no route number.
    """
    parts = value.split(":")
    token = parts[1] if len(parts) >= 2 else parts[0]
    match = _ROUTE_NUMBER.match(token)
    if match is None:
        return None
    return int(match.group(1))


def even_route_percent(values: Sequence[str]) -> float | None:
    """Percent of route colours whose number is even.

    An empty list is NA. A route with no number counts in the denominator
    and not in the numerator. This is not a passenger share.
    """
    if not values:
        return None
    even = sum(1 for value in values if (number := route_number(value)) is not None and number % 2 == 0)
    return 100.0 * even / len(values)


def count_colours(values: Sequence[str]) -> float:
    """Return how many colour values were matched. Zero is a real zero."""
    return float(len(values))


def cfa_colour_frames(
    graph: CfaGraph,
    *,
    node_label: ColourLabel,
    edge_label: ColourLabel,
    facet: ColourFacet,
) -> ColourFrames:
    """Resolve node and edge values for every facet level.

    A node or edge that lacks the facet colour is stored as ``None`` and is
    still present in the mapping.
    """
    validate_cfa(graph)
    if not graph.colors:
        raise ContractError(["CFA colouring figure needs colors.tsv"])
    by_id, by_value = _colour_index(graph.colors)
    levels = _facet_levels(by_value, facet)
    nodes = {level: {} for level in levels}
    edges = {level: {} for level in levels}
    for level in levels:
        facet_id = by_value[(facet.namespace, level)]
        for row in graph.nodes:
            nodes[level][row["node_id"]] = _visual_value(row, facet_id, level, node_label, by_id)
        for row in graph.edges:
            edges[level][row["edge_id"]] = _visual_value(row, facet_id, level, edge_label, by_id)
    _reject_mixed_kinds(nodes, edges)
    return ColourFrames(levels=levels, nodes=nodes, edges=edges)


def circle_positions(node_ids: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Place nodes on the unit circle in the given order."""
    ids = list(node_ids)
    if not ids:
        raise ContractError(["CFA colouring figure needs at least one node"])
    if len(ids) == 1:
        return {ids[0]: (0.0, 0.0)}
    positions: dict[str, tuple[float, float]] = {}
    for index, node_id in enumerate(ids):
        angle = 2.0 * math.pi * index / len(ids)
        positions[node_id] = (math.cos(angle), math.sin(angle))
    return positions


def plot_cfa_colouring(
    graph: CfaGraph,
    *,
    node_label: ColourLabel,
    edge_label: ColourLabel,
    facet: ColourFacet,
    path: str | Path,
    x_label: str,
    y_label: str,
    positions: Mapping[str, tuple[float, float]] | None = None,
    aspect: float = 1.0,
) -> ColourFrames:
    """Write a PDF or SVG of the faceted colouring and return the frames.

    Positions are required for every node when ``positions`` is set. Omitted
    positions use :func:`circle_positions`. The figure has no title. Facet
    names are panel labels. Legends sit outside the panels. NA is gray.
    """
    frames = cfa_colour_frames(graph, node_label=node_label, edge_label=edge_label, facet=facet)
    placed = dict(positions) if positions is not None else circle_positions(graph.node_ids())
    missing = [node_id for node_id in graph.node_ids() if node_id not in placed]
    if missing:
        raise ContractError(
            [f"{len(missing)} nodes have no position; first missing id is {missing[0]}"]
        )
    output = Path(path)
    if output.suffix.lower() not in {".pdf", ".svg"}:
        raise ContractError(["CFA colouring figure must be a .pdf or .svg file"])
    plt = _pyplot()
    _write_figure(
        graph,
        frames,
        placed,
        node_label=node_label,
        edge_label=edge_label,
        facet=facet,
        path=output,
        x_label=x_label,
        y_label=y_label,
        aspect=aspect,
        plt=plt,
    )
    return frames


def _colour_index(
    rows: Sequence[Mapping[str, str]],
) -> tuple[dict[int, tuple[str, str]], dict[tuple[str, str], int]]:
    by_id: dict[int, tuple[str, str]] = {}
    by_value: dict[tuple[str, str], int] = {}
    for row in rows:
        color_id = int(row["color_id"])
        namespace = row.get("namespace", "")
        value = row.get("value", row.get("name", ""))
        if namespace == "" or value == "":
            raise ContractError([f"color {color_id} needs namespace and value"])
        by_id[color_id] = (namespace, value)
        by_value[(namespace, value)] = color_id
    return by_id, by_value


def _facet_levels(
    by_value: Mapping[tuple[str, str], int],
    facet: ColourFacet,
) -> tuple[str, ...]:
    present = sorted(value for namespace, value in by_value if namespace == facet.namespace)
    if not present:
        raise ContractError([f"colour namespace {facet.namespace!r} is not in colors.tsv"])
    if facet.levels is None:
        return tuple(present)
    missing = [level for level in facet.levels if (facet.namespace, level) not in by_value]
    if missing:
        raise ContractError(
            [f"facet level {missing[0]!r} is not in namespace {facet.namespace!r}"]
        )
    return tuple(facet.levels)


def _visual_value(
    row: Mapping[str, str],
    facet_id: int,
    level: str,
    label: ColourLabel,
    by_id: Mapping[int, tuple[str, str]],
) -> float | str | None:
    owned = set(parse_color_set(row.get("color_set", "")))
    if facet_id not in owned:
        return None
    matched: set[str] = set()
    for color_id in owned:
        pair = by_id.get(color_id)
        if pair is None:
            continue
        namespace, value = pair
        if namespace == label.namespace and label.matches(level, value):
            matched.add(value)
    return _coerce(label.reduce(sorted(matched)))


def _coerce(value: float | str | None) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ContractError([f"colour reduce must return a number, a string, or None, found {value!r}"])
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _reject_mixed_kinds(
    nodes: Mapping[str, Mapping[str, float | str | None]],
    edges: Mapping[str, Mapping[str, float | str | None]],
) -> None:
    for layer, frames in (("node", nodes), ("edge", edges)):
        kinds = {
            "str" if isinstance(value, str) else "float"
            for level_map in frames.values()
            for value in level_map.values()
            if value is not None
        }
        if len(kinds) > 1:
            raise ContractError([f"{layer} colour mixes numeric and text values"])


def _pyplot():
    try:
        import matplotlib
    except ImportError as exc:
        raise ContractError(["matplotlib is required to plot a CFA colouring"]) from exc
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _write_figure(
    graph: CfaGraph,
    frames: ColourFrames,
    positions: Mapping[str, tuple[float, float]],
    *,
    node_label: ColourLabel,
    edge_label: ColourLabel,
    facet: ColourFacet,
    path: Path,
    x_label: str,
    y_label: str,
    aspect: float,
    plt,
) -> None:
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    node_scale = _scale(frames.nodes, node_label, plt)
    edge_scale = _scale(frames.edges, edge_label, plt)
    level_count = len(frames.levels)
    figure, axes = plt.subplots(
        1,
        level_count,
        figsize=(3.15 * level_count + 2.2, 3.8),
        squeeze=False,
    )
    figure.subplots_adjust(left=0.07, right=0.76, bottom=0.16, top=0.90, wspace=0.22)
    panels = list(axes[0])
    span = _span(positions)
    for axis, level in zip(panels, frames.levels):
        _draw_edges(axis, graph, frames.edges[level], positions, edge_scale, span, LineCollection)
        _draw_nodes(axis, graph, frames.nodes[level], positions, node_scale)
        axis.set_title(facet.caption(level), fontsize=10, pad=4)
        _style_axis(axis, x_label, y_label, aspect, positions)
    _colourbar(figure, panels, node_scale, node_label.legend, [0.80, 0.28, 0.018, 0.52], plt)
    _colourbar(figure, panels, edge_scale, edge_label.legend, [0.90, 0.28, 0.018, 0.52], plt)
    na_legend = figure.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=NA_COLOUR, markersize=5, label="NA node"),
            Line2D([0], [0], color=NA_COLOUR, linewidth=1.5, label="NA edge"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.40, 0.01),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    figure.add_artist(na_legend)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def _draw_edges(axis, graph, values, positions, scale, span, line_collection) -> None:
    plain: list = []
    marked: list = []
    colours: list = []
    for row in graph.edges:
        segment = _segment(positions[row["source"]], positions[row["target"]], span)
        value = values[row["edge_id"]]
        if value is None or scale is None:
            plain.append(segment)
            continue
        marked.append(segment)
        colours.append(_paint(value, scale))
    if plain:
        axis.add_collection(line_collection(plain, colors=NA_COLOUR, linewidths=0.3, zorder=1))
    if marked:
        axis.add_collection(line_collection(marked, colors=colours, linewidths=0.45, zorder=2))


def _draw_nodes(axis, graph, values, positions, scale) -> None:
    plain_x: list[float] = []
    plain_y: list[float] = []
    marked_x: list[float] = []
    marked_y: list[float] = []
    colours: list = []
    for row in graph.nodes:
        x_coord, y_coord = positions[row["node_id"]]
        value = values[row["node_id"]]
        if value is None or scale is None:
            plain_x.append(x_coord)
            plain_y.append(y_coord)
            continue
        marked_x.append(x_coord)
        marked_y.append(y_coord)
        colours.append(_paint(value, scale))
    if plain_x:
        axis.scatter(plain_x, plain_y, s=6, c=NA_COLOUR, linewidths=0, zorder=3)
    if marked_x:
        axis.scatter(marked_x, marked_y, s=7, c=colours, linewidths=0, zorder=4)


def _segment(
    source: tuple[float, float],
    target: tuple[float, float],
    span: float,
) -> list[tuple[float, float]]:
    if source != target:
        return [source, target]
    radius = 0.02 * span if span > 0 else 0.05
    loop = []
    for step in range(9):
        angle = 2.0 * math.pi * step / 8
        loop.append((source[0] + radius * math.cos(angle), source[1] + radius * math.sin(angle)))
    return loop


def _span(positions: Mapping[str, tuple[float, float]]) -> float:
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)


def _paint(value: float | str | None, scale) -> str | tuple:
    if value is None or scale is None:
        return NA_COLOUR
    if scale["kind"] == "float":
        return scale["cmap"](scale["norm"](value))
    return scale["colours"][value]


def _scale(frames: Mapping[str, Mapping[str, float | str | None]], label: ColourLabel, plt):
    values = [value for level_map in frames.values() for value in level_map.values() if value is not None]
    if not values:
        return None
    if isinstance(values[0], str):
        return {"kind": "str", "colours": _discrete_colours(values, label.palette)}
    return {"kind": "float", "cmap": plt.get_cmap(label.palette), "norm": _norm(values, label.limits, plt)}


def _norm(values: Sequence[float], limits: tuple[float, float] | None, plt):
    from matplotlib.colors import Normalize

    if limits is not None:
        low, high = limits
    else:
        low, high = min(values), max(values)
    if low == high:
        high = low + 1.0
    if low > high:
        raise ContractError([f"colour limits {low} > {high}"])
    return Normalize(vmin=low, vmax=high)


def _discrete_colours(values: Sequence[str], palette: str) -> dict[str, str]:
    if palette != "Set1":
        raise ContractError([f"text colours use palette Set1, found {palette!r}"])
    ordered = sorted({value for value in values if value not in {"Other", "Unclassified"}})
    if "Other" in values:
        ordered.append("Other")
    colours: dict[str, str] = {}
    cursor = 0
    for value in ordered:
        if value == "Unclassified":
            colours[value] = UNCLASSIFIED_COLOUR
            continue
        if value == "Other":
            colours[value] = OTHER_COLOUR
            continue
        colours[value] = _SET1[cursor % len(_SET1)]
        cursor += 1
    if "Unclassified" in values:
        colours["Unclassified"] = UNCLASSIFIED_COLOUR
    return colours


def _colourbar(figure, panels, scale, legend: str, box: list[float], plt) -> None:
    if scale is None or scale["kind"] != "float":
        if scale is not None and scale["kind"] == "str":
            _discrete_legend(figure, panels, scale, legend)
        return
    from matplotlib.cm import ScalarMappable

    mappable = ScalarMappable(norm=scale["norm"], cmap=scale["cmap"])
    axis = figure.add_axes(box)
    bar = figure.colorbar(mappable, cax=axis)
    bar.set_label(legend, fontsize=8)
    bar.ax.tick_params(labelsize=8)


def _discrete_legend(figure, panels, scale, legend: str) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colour, markersize=5, label=name)
        for name, colour in scale["colours"].items()
    ]
    legend_artist = figure.legend(
        handles=handles,
        title=legend,
        loc="center left",
        bbox_to_anchor=(0.78, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )
    figure.add_artist(legend_artist)


def _style_axis(axis, x_label: str, y_label: str, aspect: float, positions) -> None:
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    pad_x = span_x * 0.03 if span_x else 0.5
    pad_y = span_y * 0.03 if span_y else 0.5
    axis.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    axis.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    axis.set_xlabel(x_label, fontsize=9)
    axis.set_ylabel(y_label, fontsize=9)
    axis.tick_params(labelsize=8, length=3, width=0.6)
    axis.set_aspect(aspect, adjustable="box")
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
