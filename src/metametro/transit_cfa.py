"""Build a de Bruijn CFA from a ground-transit stop graph.

Nodes are stops. A directed edge is one ordered pair of stops that follow
each other on at least one trip. Sequences come from a single base-4 tape so
that every edge overlaps by ``k - 1``, the same repeat-junction rule compaction
checks. Edge coverage is the mean route count of bus, trolley, and tram
at the two stops. A mode with no route counts as zero.
Colours are every transport type and every route, applied with ``replace``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from metametro.contracts.colouring import colour_cfa
from metametro.errors import ContractError
from metametro.formats.cfa.model import SCHEMA_VERSION, CfaGraph

TRANSPORT_TYPES = ("bus", "trolley", "tram")
_ALPHABET = "ACGT"


@dataclass(frozen=True)
class TransitRoute:
    """One ground route that can colour stops and hops."""

    route_id: str
    short_name: str
    transport_type: str


def transit_cfa(
    stops: Mapping[str, str],
    routes: Sequence[TransitRoute],
    trips: Sequence[tuple[str, Sequence[str]]],
    *,
    graph_id: str = "ground_transit",
    k: int = 21,
) -> CfaGraph:
    """Return a coloured de Bruijn CFA for this stop graph.

    ``stops`` maps a stop id to its name. ``trips`` are ``(route_id, stop ids
    in visit order)``. Only stops that appear in a trip become nodes.

    Each node sequence has length ``2 * (k - 1)``. Junctions that an edge
    forces to be equal share one block of a base-4 counter tape, so the
    source suffix equals the target prefix. The sequence is longer than ``k``
    because a transit graph does not satisfy the extra one-base shift that
    length-``k`` de Bruijn nodes would require between a node's own prefix
    and suffix.

    A GTFS stop belongs to one mode, so a count of modes is 1 everywhere.
    Node coverage is the mean route count over bus, trolley, and tram
    (modes with no route contribute 0). Edge coverage is the mean of the
    two endpoint coverages. That is not a passenger count. The feed used
    by the example has none.

    The colour dictionary has one ``transport_type`` row per mode and one
    ``route`` row per route. A stop or hop carries every colour that applies.
    """
    _require_k(k)
    route_by_id = _routes_by_id(routes)
    stop_routes, edge_routes = _collect(stops, route_by_id, trips)
    if not stop_routes:
        raise ContractError(["transit CFA needs at least one stop visit"])

    node_ids = [f"s{stop_id}" for stop_id in sorted(stop_routes)]
    id_of = {stop_id: f"s{stop_id}" for stop_id in stop_routes}
    directed = sorted(
        (id_of[source], id_of[target]) for source, target in edge_routes
    )
    sequences = _assign_sequences(node_ids, directed, k)
    coverage_of = {
        stop_id: _type_coverage(route_ids, route_by_id) for stop_id, route_ids in stop_routes.items()
    }

    nodes: list[dict[str, str]] = []
    for stop_id in sorted(stop_routes):
        nodes.append(
            {
                "node_id": id_of[stop_id],
                "stop_name": _cell(stops[stop_id]),
                "coverage": _float_cell(coverage_of[stop_id]),
            }
        )
    edges: list[dict[str, str]] = []
    edge_key_to_id: dict[tuple[str, str], str] = {}
    for index, (source_id, target_id) in enumerate(directed, start=1):
        edge_id = f"e{index:06d}"
        source_stop = source_id[1:]
        target_stop = target_id[1:]
        edge_key_to_id[(source_stop, target_stop)] = edge_id
        mean_coverage = (coverage_of[source_stop] + coverage_of[target_stop]) / 2
        edges.append(
            {
                "edge_id": edge_id,
                "source": source_id,
                "target": target_id,
                "orientation": "++",
                "coverage": _float_cell(mean_coverage),
            }
        )

    colors, node_colors, edge_colors = _colour_maps(
        route_by_id, stop_routes, edge_routes, id_of, edge_key_to_id
    )
    bare = CfaGraph(
        metadata={
            "schema_version": SCHEMA_VERSION,
            "graph_id": graph_id,
            "graph_type": "de_bruijn",
            "k": k,
            "contract": "graph_to_cfa",
            "contract_version": "1.0",
            "sequence_rule": "repeat_junction_stream",
            "coverage_rule": "mean_route_count_over_bus_trolley_tram",
            "features": {
                "node": {"stop_name": "str", "coverage": "float"},
                "edge": {"orientation": "orientation", "coverage": "float"},
            },
        },
        sequences=sequences,
        nodes=nodes,
        edges=edges,
        node_header=["node_id", "stop_name", "coverage"],
        edge_header=["edge_id", "source", "target", "orientation", "coverage"],
    )
    return colour_cfa(
        bare,
        node_colors,
        edge_colors,
        operation="replace",
        colors=colors,
        target=("node", "edge"),
    )


def overlap_mismatches(graph: CfaGraph) -> list[str]:
    """Return edge ids whose forward sequences do not share ``k - 1`` bases."""
    k = graph.metadata.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ContractError(["overlap check requires integer k >= 2"])
    overlap = k - 1
    mismatches: list[str] = []
    for row in graph.edges:
        orientation = row.get("orientation") or "++"
        if orientation != "++":
            continue
        left = graph.sequences[row["source"]]
        right = graph.sequences[row["target"]]
        if len(left) < overlap or len(right) < overlap or left[-overlap:] != right[:overlap]:
            mismatches.append(row["edge_id"])
    return mismatches


def _require_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ContractError(["transit CFA requires integer k >= 2"])


def _routes_by_id(routes: Sequence[TransitRoute]) -> dict[str, TransitRoute]:
    found: dict[str, TransitRoute] = {}
    for route in routes:
        if route.route_id == "" or route.short_name.strip() == "":
            raise ContractError(["route_id and short_name are required"])
        if route.transport_type not in TRANSPORT_TYPES:
            raise ContractError(
                [
                    f"unsupported transport_type {route.transport_type!r}; "
                    f"expected one of {', '.join(TRANSPORT_TYPES)}"
                ]
            )
        if route.route_id in found:
            raise ContractError([f"duplicate route_id: {route.route_id}"])
        found[route.route_id] = route
    if not found:
        raise ContractError(["transit CFA needs at least one route"])
    return found


def _collect(
    stops: Mapping[str, str],
    route_by_id: Mapping[str, TransitRoute],
    trips: Sequence[tuple[str, Sequence[str]]],
) -> tuple[dict[str, set[str]], dict[tuple[str, str], set[str]]]:
    stop_routes: dict[str, set[str]] = defaultdict(set)
    edge_routes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for route_id, stop_ids in trips:
        if route_id not in route_by_id:
            raise ContractError([f"trip references unknown route_id: {route_id}"])
        if not stop_ids:
            raise ContractError([f"route {route_id} has a trip with no stops"])
        missing = [stop_id for stop_id in stop_ids if stop_id not in stops]
        if missing:
            raise ContractError([f"unknown stop_id: {missing[0]}"])
        previous: str | None = None
        for stop_id in stop_ids:
            stop_routes[stop_id].add(route_id)
            if previous is not None:
                edge_routes[(previous, stop_id)].add(route_id)
            previous = stop_id
    return stop_routes, edge_routes


def _type_coverage(route_ids: set[str], route_by_id: Mapping[str, TransitRoute]) -> float:
    counts = Counter(route_by_id[route_id].transport_type for route_id in route_ids)
    return sum(counts.get(kind, 0) for kind in TRANSPORT_TYPES) / len(TRANSPORT_TYPES)


def _assign_sequences(
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str]],
    k: int,
) -> dict[str, str]:
    overlap = k - 1
    parent: dict[str, str] = {}

    def add(key: str) -> None:
        parent.setdefault(key, key)

    def find(key: str) -> str:
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:
            parent[key], key = root, parent[key]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for node_id in node_ids:
        add(f"p:{node_id}")
        add(f"s:{node_id}")
    for source, target in edges:
        union(f"s:{source}", f"p:{target}")

    roots = sorted({find(key) for key in parent})
    limit = 4**overlap
    if len(roots) > limit:
        raise ContractError(
            [
                f"junction stream has {len(roots)} classes but only {limit} "
                f"distinct strings of length {overlap}"
            ]
        )
    block = {root: _junction_block(index, overlap) for index, root in enumerate(roots)}
    return {
        node_id: block[find(f"p:{node_id}")] + block[find(f"s:{node_id}")]
        for node_id in node_ids
    }


def _junction_block(index: int, length: int) -> str:
    """Return one block of the base-4 counter tape.

    Block ``index`` is the ``length``-digit base-4 writing of ``index``.
    Concatenated in index order, the blocks are one stream. An edge does not
    consume a new block: it reuses the source suffix as the target prefix.
    """
    digits: list[str] = []
    value = index
    for _ in range(length):
        digits.append(_ALPHABET[value % 4])
        value //= 4
    return "".join(reversed(digits))


def _colour_maps(
    route_by_id: Mapping[str, TransitRoute],
    stop_routes: Mapping[str, set[str]],
    edge_routes: Mapping[tuple[str, str], set[str]],
    id_of: Mapping[str, str],
    edge_key_to_id: Mapping[tuple[str, str], str],
) -> tuple[list[dict[str, str]], dict[str, list[int]], dict[str, list[int]]]:
    colors: list[dict[str, str]] = []
    type_id = {kind: index for index, kind in enumerate(TRANSPORT_TYPES)}
    for kind, color_id in type_id.items():
        colors.append(
            {"color_id": str(color_id), "namespace": "transport_type", "value": kind}
        )
    bases = Counter(
        f"{route.transport_type}:{route.short_name}" for route in route_by_id.values()
    )
    route_color: dict[str, int] = {}
    next_id = len(TRANSPORT_TYPES)
    for route in sorted(
        route_by_id.values(),
        key=lambda item: (item.transport_type, item.short_name, item.route_id),
    ):
        base = f"{route.transport_type}:{route.short_name}"
        value = base if bases[base] == 1 else f"{base}:{route.route_id}"
        route_color[route.route_id] = next_id
        colors.append(
            {"color_id": str(next_id), "namespace": "route", "value": value}
        )
        next_id += 1

    def paint(route_ids: set[str]) -> list[int]:
        ids = {type_id[route_by_id[route_id].transport_type] for route_id in route_ids}
        ids.update(route_color[route_id] for route_id in route_ids)
        return sorted(ids)

    node_colors = {
        id_of[stop_id]: paint(route_ids) for stop_id, route_ids in stop_routes.items()
    }
    edge_colors = {
        edge_key_to_id[(source, target)]: paint(route_ids)
        for (source, target), route_ids in edge_routes.items()
    }
    return colors, node_colors, edge_colors


def _cell(text: str) -> str:
    return " ".join(text.replace("\t", " ").split()) or "unnamed"


def _float_cell(value: float) -> str:
    return format(value, ".10f")
