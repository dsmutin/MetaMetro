"""Facet the Saint Petersburg CFA by bus, tram, and trolleybus.

Node colour is simulated passengers summed over routes. Edge colour is the
mean of the two stops. Panels are stacked vertically. Stops and hops that
do not carry the facet mode stay drawn in NA grey.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

from metametro.errors import ContractError
from metametro.formats.cfa.io import load_cfa

from metametro.viz.cfa_colouring import (
    PINK_YELLOW_GREEN,
    ColourFacet,
    ColourLabel,
    nearest_neighbor_ratio,
    plot_cfa_colouring,
    spring_positions,
)


def _graph_positions(graph) -> dict[str, tuple[float, float]]:
    """Read longitude and latitude stored on the CFA nodes."""
    if "longitude" not in graph.node_header or "latitude" not in graph.node_header:
        raise ContractError(["CFA nodes need longitude and latitude columns"])
    return {
        row["node_id"]: (float(row["longitude"]), float(row["latitude"]))
        for row in graph.nodes
    }


def _choose_layout(graph):
    """Pick the Fruchterman–Reingold spread closest to ideal spacing.

    The score is ``nearest_neighbor_ratio``: median nearest-neighbor distance
    divided by ``spread * sqrt(1 / n)``. A ratio near 1 is the equilibrium
    spacing. Axes then fit the drawing, so a larger spread only rescales it.
    """
    count = len(graph.nodes)
    best_key = (-1.0, -1.0)
    best_spread = 1.0
    best_positions = None
    best_ratio = 0.0
    for spread in (1.0, 2.0, 4.0):
        positions = spring_positions(graph, seed=0, iterations=40, spread=spread)
        ratio = nearest_neighbor_ratio(positions, spread=spread)
        distance = ratio * spread * math.sqrt(1.0 / count)
        print(f"spread {spread:g} median_nn {distance:.4f} ratio {ratio:.3f}")
        key = (ratio, distance)
        if key > best_key:
            best_key = key
            best_spread = spread
            best_positions = positions
            best_ratio = ratio
    print(f"using spread {best_spread:g}")
    return best_positions, best_ratio


def _colouring(graph, **kwargs):
    """Draw one page. Both scales use the pink–yellow–light-green gradient."""
    plot_cfa_colouring(
        graph,
        node_label=ColourLabel(
            namespace="coverage",
            legend="Simulated passengers (sum)",
            column="coverage",
            palette=PINK_YELLOW_GREEN,
        ),
        edge_label=ColourLabel(
            namespace="coverage",
            legend="Simulated passengers (mean)",
            column="coverage",
            palette=PINK_YELLOW_GREEN,
        ),
        facet=ColourFacet(
            namespace="transport_type",
            levels=("bus", "tram", "trolley"),
            level_labels={"bus": "Bus", "tram": "Tram", "trolley": "Trolleybus"},
        ),
        **kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    """Write the faceted colouring PDF.

    Page 1 uses stop longitude and latitude. Page 2 uses the Fruchterman–Reingold
    spread whose median nearest-neighbor distance is closest to the ideal
    spacing. Panels are stacked vertically. Both pages use the same gradient.
    """
    parser = argparse.ArgumentParser(description="Plot the Saint Petersburg CFA colouring.")
    parser.add_argument("--cfa", type=Path, default=Path("data/work/spb_ground_transit/cfa"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/work/spb_ground_transit/cfa_colouring.pdf"),
    )
    args = parser.parse_args(argv)
    try:
        graph = load_cfa(args.cfa)
        positions = _graph_positions(graph)
        latitudes = [point[1] for point in positions.values()]
        aspect = 1.0 / math.cos(math.radians(sum(latitudes) / len(latitudes)))
        layout, ratio = _choose_layout(graph)
        print(f"final median_nn ratio {ratio:.3f}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(args.out) as pdf:
            _colouring(
                graph,
                positions=positions,
                path=args.out,
                x_label="Longitude (°E)",
                y_label="Latitude (°N)",
                aspect=aspect,
                facet_along="y",
                pdf_pages=pdf,
            )
            _colouring(
                graph,
                positions=layout,
                path=args.out,
                x_label="Layout x",
                y_label="Layout y",
                facet_along="y",
                pdf_pages=pdf,
            )
    except ContractError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
