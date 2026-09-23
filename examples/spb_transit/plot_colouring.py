"""Facet the Saint Petersburg CFA by bus, tram, and trolleybus.

Node colour is the percent of that mode's routes whose number is even.
Edge colour is how many of that mode's routes use the hop. Stops and hops
that do not carry the facet mode stay on the map in NA grey.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import zipfile
from pathlib import Path

from metametro.errors import ContractError
from metametro.formats.cfa.io import load_cfa
from metametro.viz.cfa_colouring import (
    ColourFacet,
    ColourLabel,
    count_colours,
    even_route_percent,
    plot_cfa_colouring,
)


def _mode_route(level: str, value: str) -> bool:
    """Keep route colours that belong to this facet mode."""
    return value.startswith(f"{level}:")


def load_positions(gtfs_zip: Path, node_ids: list[str]) -> dict[str, tuple[float, float]]:
    """Read stop longitude and latitude for CFA ids ``s`` + stop id."""
    if not gtfs_zip.is_file() or gtfs_zip.stat().st_size == 0:
        raise ContractError([f"GTFS zip is missing or empty: {gtfs_zip}"])
    wanted = {node_id[1:]: node_id for node_id in node_ids}
    found: dict[str, tuple[float, float]] = {}
    with zipfile.ZipFile(gtfs_zip) as archive:
        if "stops.txt" not in archive.namelist():
            raise ContractError(["GTFS zip is missing stops.txt"])
        reader = csv.DictReader(io.TextIOWrapper(archive.open("stops.txt"), encoding="utf-8", newline=""))
        for row in reader:
            stop_id = (row.get("stop_id") or "").strip()
            if stop_id not in wanted:
                continue
            lat = (row.get("stop_lat") or "").strip()
            lon = (row.get("stop_lon") or "").strip()
            try:
                found[wanted[stop_id]] = (float(lon), float(lat))
            except ValueError as exc:
                raise ContractError([f"stop {stop_id} has a non-numeric coordinate"]) from exc
    missing = [node_id for node_id in node_ids if node_id not in found]
    if missing:
        raise ContractError(
            [f"{len(missing)} CFA nodes have no GTFS coordinate; first missing id is {missing[0]}"]
        )
    return found


def main(argv: list[str] | None = None) -> int:
    """Write the faceted colouring PDF."""
    parser = argparse.ArgumentParser(description="Plot the Saint Petersburg CFA colouring.")
    parser.add_argument("--cfa", type=Path, default=Path("data/work/spb_ground_transit/cfa"))
    parser.add_argument("--gtfs", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/work/spb_ground_transit/cfa_colouring.pdf"),
    )
    args = parser.parse_args(argv)
    try:
        graph = load_cfa(args.cfa)
        positions = load_positions(args.gtfs, graph.node_ids())
        latitudes = [point[1] for point in positions.values()]
        aspect = 1.0 / math.cos(math.radians(sum(latitudes) / len(latitudes)))
        plot_cfa_colouring(
            graph,
            node_label=ColourLabel(
                namespace="route",
                legend="Even-numbered routes (%)",
                matches=_mode_route,
                reduce=even_route_percent,
                palette="YlGnBu",
                limits=(0.0, 100.0),
            ),
            edge_label=ColourLabel(
                namespace="route",
                legend="Routes (count)",
                matches=_mode_route,
                reduce=count_colours,
                palette="YlGnBu",
            ),
            facet=ColourFacet(
                namespace="transport_type",
                levels=("bus", "tram", "trolley"),
                level_labels={"bus": "Bus", "tram": "Tram", "trolley": "Trolleybus"},
            ),
            positions=positions,
            path=args.out,
            x_label="Longitude (°E)",
            y_label="Latitude (°N)",
            aspect=aspect,
        )
    except (ContractError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
