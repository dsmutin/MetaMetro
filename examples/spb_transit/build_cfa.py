"""Build a CFA of Saint Petersburg ground transit from an ORGP GTFS zip.

The zip is the public feed at
https://transport.orgp.spb.ru/Portal/transport/internalapi/gtfs/feed.zip
Passenger counts are not in that feed. Coverage is a seeded normal draw
per route, summed on the stops that route serves. A node is one named stop:
feed ids with the same name at most 60 m apart are merged. That 60 m cut is
where the same-name cross-mode nearest-distance histogram stops falling.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
from metametro.errors import ContractError
from metametro.formats.cfa.io import dump_cfa
from metametro.transit_cfa import TRANSPORT_TYPES, TransitRoute, overlap_mismatches, transit_cfa
from metametro.transit_stops import StopPlace, cluster_stops

SAME_STOP_METRES = 60.0

_REQUIRED = ("routes.txt", "stops.txt", "trips.txt", "stop_times.txt")


def _rows(archive: zipfile.ZipFile, name: str) -> csv.DictReader:
    if name not in archive.namelist():
        raise ContractError([f"GTFS zip is missing {name}"])
    handle = io.TextIOWrapper(archive.open(name), encoding="utf-8", newline="")
    return csv.DictReader(handle)


def load_ground_graph(
    gtfs_zip: Path,
) -> tuple[dict[str, str], dict[str, tuple[float, float]], list[TransitRoute], list[tuple[str, list[str]]]]:
    """Read stops, ground routes, and ordered stop lists from a GTFS zip.

    Routes whose ``transport_type`` is not bus, trolley, or tram are skipped.
    A trip that names a route id absent from ``routes.txt`` is an error.
    """
    if not gtfs_zip.is_file() or gtfs_zip.stat().st_size == 0:
        raise ContractError([f"GTFS zip is missing or empty: {gtfs_zip}"])
    with zipfile.ZipFile(gtfs_zip) as archive:
        missing = [name for name in _REQUIRED if name not in archive.namelist()]
        if missing:
            raise ContractError([f"GTFS zip is missing {name}" for name in missing])
        stops: dict[str, str] = {}
        coordinates: dict[str, tuple[float, float]] = {}
        for row in _rows(archive, "stops.txt"):
            stop_id = (row.get("stop_id") or "").strip()
            if stop_id == "":
                raise ContractError(["stops.txt row has an empty stop_id"])
            stops[stop_id] = (row.get("stop_name") or "").strip()
            try:
                coordinates[stop_id] = (float(row["stop_lon"]), float(row["stop_lat"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractError([f"stop {stop_id} has a non-numeric coordinate"]) from exc
        routes: list[TransitRoute] = []
        known_ids: set[str] = set()
        other_ids: set[str] = set()
        skipped_modes: dict[str, int] = defaultdict(int)
        for row in _rows(archive, "routes.txt"):
            route_id = (row.get("route_id") or "").strip()
            short_name = (row.get("route_short_name") or "").strip()
            kind = (row.get("transport_type") or "").strip()
            if route_id == "" or short_name == "" or kind == "":
                raise ContractError(
                    [f"routes.txt row {route_id or '?'} is missing id, name, or transport_type"]
                )
            if route_id in known_ids:
                raise ContractError([f"duplicate route_id in routes.txt: {route_id}"])
            known_ids.add(route_id)
            if kind not in TRANSPORT_TYPES:
                skipped_modes[kind] += 1
                other_ids.add(route_id)
                continue
            routes.append(TransitRoute(route_id, short_name, kind))
        trip_route: dict[str, str] = {}
        for row in _rows(archive, "trips.txt"):
            trip_id = (row.get("trip_id") or "").strip()
            route_id = (row.get("route_id") or "").strip()
            if trip_id == "" or route_id == "":
                raise ContractError(["trips.txt row is missing trip_id or route_id"])
            if route_id not in known_ids:
                raise ContractError([f"trip {trip_id} references unknown route_id: {route_id}"])
            if route_id in other_ids:
                continue
            trip_route[trip_id] = route_id
        visits: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for row in _rows(archive, "stop_times.txt"):
            trip_id = (row.get("trip_id") or "").strip()
            if trip_id not in trip_route:
                continue
            stop_id = (row.get("stop_id") or "").strip()
            sequence = (row.get("stop_sequence") or "").strip()
            if stop_id == "" or not sequence.lstrip("-").isdigit():
                raise ContractError([f"stop_times.txt trip {trip_id} has a bad stop row"])
            visits[trip_id].append((int(sequence), stop_id))
    trips: list[tuple[str, list[str]]] = []
    trips_without_stops = 0
    for trip_id, route_id in sorted(trip_route.items()):
        ordered = [stop_id for _sequence, stop_id in sorted(visits.get(trip_id, []))]
        if not ordered:
            trips_without_stops += 1
            continue
        trips.append((route_id, ordered))
    if not trips:
        raise ContractError(["no ground trip has stop_times"])
    if trips_without_stops:
        print(
            f"skipped {trips_without_stops} ground trips that have no stop_times",
            file=sys.stderr,
        )
    if skipped_modes:
        print(
            "skipped non-ground routes: "
            + ", ".join(f"{kind}={count}" for kind, count in sorted(skipped_modes.items())),
            file=sys.stderr,
        )
    return stops, coordinates, routes, trips


def physical_stops(
    stops: dict[str, str],
    coordinates: dict[str, tuple[float, float]],
    trips: list[tuple[str, list[str]]],
) -> tuple[dict[str, str], dict[str, tuple[float, float]], list[tuple[str, list[str]]], int]:
    """Collapse feed ids that are the same named stop within 60 m."""
    used = [stop_id for _route_id, stop_ids in trips for stop_id in stop_ids]
    missing = [stop_id for stop_id in used if stop_id not in coordinates]
    if missing:
        raise ContractError([f"stop {missing[0]} has no coordinate"])
    places = [
        StopPlace(
            stop_id,
            stops[stop_id],
            coordinates[stop_id][1],
            coordinates[stop_id][0],
        )
        for stop_id in sorted(set(used))
    ]
    clustered = cluster_stops(places, metres=SAME_STOP_METRES)
    names = {item.stop_id: item.name for item in clustered}
    centroids = {item.stop_id: (item.longitude, item.latitude) for item in clustered}
    member_of = {member: item.stop_id for item in clustered for member in item.members}
    rewritten: list[tuple[str, list[str]]] = []
    for route_id, stop_ids in trips:
        sequence: list[str] = []
        for stop_id in stop_ids:
            cluster_id = member_of[stop_id]
            if sequence and sequence[-1] == cluster_id:
                continue
            sequence.append(cluster_id)
        if sequence:
            rewritten.append((route_id, sequence))
    return names, centroids, rewritten, len(places)


def main(argv: list[str] | None = None) -> int:
    """Write the CFA directory and compact it once to check overlaps."""
    parser = argparse.ArgumentParser(description="Build a Saint Petersburg ground-transit CFA from GTFS.")
    parser.add_argument("--gtfs", type=Path, required=True, help="Path to the ORGP GTFS zip")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/work/spb_ground_transit/cfa"),
        help="Output CFA directory",
    )
    parser.add_argument("--k", type=int, default=21, help="De Bruijn k. Sequences have length 2*(k-1).")
    args = parser.parse_args(argv)
    try:
        feed_stops, coordinates, routes, trips = load_ground_graph(args.gtfs)
        stops, centroids, trips, feed_count = physical_stops(feed_stops, coordinates, trips)
        graph = transit_cfa(
            stops,
            routes,
            trips,
            graph_id="spb_ground_transit",
            k=args.k,
            passenger_seed=0,
            coordinates=centroids,
        )
    except (ContractError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    graph.metadata["source"] = {
        "feed": "https://transport.orgp.spb.ru/Portal/transport/internalapi/gtfs/feed.zip",
        "modes": list(TRANSPORT_TYPES),
        "note": (
            "A node is one named stop. Feed ids with the same name at most 60 m apart are one node. "
            "Coverage is simulated: each route draws Normal(mean=stop count, sd=sqrt(stop count)) "
            "with seed 0, the draw is added to every stop of that route, and an edge is the mean of its stops. "
            "The GTFS feed has no passenger counts. "
            "Ground trips with no stop_times are omitted and counted on stderr."
        ),
        "feed_stops": feed_count,
        "physical_stops": len(stops),
        "same_stop_metres": SAME_STOP_METRES,
    }
    mismatches = overlap_mismatches(graph)
    if mismatches:
        print(f"overlap mismatch on {len(mismatches)} edges", file=sys.stderr)
        return 1
    compacted = cfa_to_cdbg(graph)
    dump_cfa(graph, args.out)
    print(f"nodes {len(graph.nodes)}")
    print(f"edges {len(graph.edges)}")
    print(f"colors {len(graph.colors or [])}")
    print(f"k {graph.metadata['k']}")
    print(f"sequence_length {2 * (int(graph.metadata['k']) - 1)}")
    print(f"unitigs {len(compacted.unitigs)}")
    print(f"links {len(compacted.links)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
