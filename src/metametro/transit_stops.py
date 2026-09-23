"""Group GTFS stop ids that are the same named stop.

A GTFS feed gives each mode, and often each side of a street, its own stop
id. Those ids are not separate stops when they share a name and stand within
``metres`` of each other. The Saint Petersburg feed's nearest cross-mode
distances for one name fall until 60 m and then stop falling, so the example
passes 60.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from metametro.errors import ContractError

_EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class StopPlace:
    """One feed stop id before physical stops are merged."""

    stop_id: str
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class PhysicalStop:
    """One node: the feed ids that name the same place."""

    stop_id: str
    name: str
    latitude: float
    longitude: float
    members: tuple[str, ...]


def cluster_stops(places: list[StopPlace], *, metres: float) -> list[PhysicalStop]:
    """Merge places with the same name whose distance is at most ``metres``.

    The link is transitive. The kept id is the smallest member id. The
    coordinate is the mean of the members. Different names are never merged,
    however close they are.
    """
    if not math.isfinite(metres) or metres < 0:
        raise ContractError(["stop clustering distance must be a finite number >= 0"])
    if not places:
        raise ContractError(["stop clustering needs at least one stop"])
    seen: set[str] = set()
    for place in places:
        if place.stop_id == "" or place.stop_id in seen:
            raise ContractError([f"duplicate or empty stop_id: {place.stop_id or '?'}"])
        seen.add(place.stop_id)
        if not math.isfinite(place.latitude) or not math.isfinite(place.longitude):
            raise ContractError([f"stop {place.stop_id} has a non-finite coordinate"])
    parent = {place.stop_id: place.stop_id for place in places}

    def find(stop_id: str) -> str:
        root = stop_id
        while parent[root] != root:
            root = parent[root]
        while parent[stop_id] != root:
            parent[stop_id], stop_id = root, parent[stop_id]
        return root

    by_name: dict[str, list[StopPlace]] = defaultdict(list)
    for place in places:
        by_name[_name_key(place.name)].append(place)
    for group in by_name.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                if _haversine_m(left, right) <= metres:
                    left_root, right_root = find(left.stop_id), find(right.stop_id)
                    if left_root == right_root:
                        continue
                    if left_root > right_root:
                        left_root, right_root = right_root, left_root
                    parent[right_root] = left_root
    grouped: dict[str, list[StopPlace]] = defaultdict(list)
    for place in places:
        grouped[find(place.stop_id)].append(place)
    stops: list[PhysicalStop] = []
    for members in grouped.values():
        ordered = sorted(members, key=lambda item: item.stop_id)
        stops.append(
            PhysicalStop(
                stop_id=ordered[0].stop_id,
                name=ordered[0].name.strip() or "unnamed",
                latitude=sum(item.latitude for item in ordered) / len(ordered),
                longitude=sum(item.longitude for item in ordered) / len(ordered),
                members=tuple(item.stop_id for item in ordered),
            )
        )
    stops.sort(key=lambda item: item.stop_id)
    return stops


def _name_key(name: str) -> str:
    return " ".join(name.casefold().split())


def _haversine_m(left: StopPlace, right: StopPlace) -> float:
    lat1 = math.radians(left.latitude)
    lat2 = math.radians(right.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(right.longitude - left.longitude)
    height = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(height)))
