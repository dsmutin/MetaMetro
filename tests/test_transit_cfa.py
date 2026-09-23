"""Mandatory checks for the ground-transit CFA builder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
from metametro.errors import ContractError
from metametro.formats.cfa.validator import validate_cfa
from metametro.transit_cfa import TransitRoute, overlap_mismatches, transit_cfa
from metametro.transit_stops import StopPlace, cluster_stops

pytestmark = pytest.mark.mandatory


def _sample():
    """Three stops, three modes, and a branch plus a join."""
    stops = {"a": "Alpha", "b": "Beta", "c": "Gamma", "d": "Delta"}
    routes = [
        TransitRoute("bus1", "10", "bus"),
        TransitRoute("tram3", "3", "tram"),
        TransitRoute("trol5", "5", "trolley"),
    ]
    trips = [
        ("bus1", ["a", "b", "c"]),
        ("tram3", ["a", "b", "d"]),
        ("trol5", ["b", "c"]),
    ]
    return stops, routes, trips


def test_neighbors_share_repeat_overlap_and_compaction_accepts_it() -> None:
    """Every hop overlaps by k-1, and a branch reuses one junction."""
    stops, routes, trips = _sample()
    graph = transit_cfa(stops, routes, trips, graph_id="toy_transit", k=5)
    validate_cfa(graph)
    assert overlap_mismatches(graph) == []
    overlap = 4
    sequences = graph.sequences
    assert sequences["sa"][-overlap:] == sequences["sb"][:overlap]
    assert sequences["sb"][-overlap:] == sequences["sc"][:overlap]
    assert sequences["sb"][-overlap:] == sequences["sd"][:overlap]
    assert len(sequences["sa"]) == 8
    compacted = cfa_to_cdbg(graph)
    assert {item.cfa_node_id for item in compacted.mapping} == set(graph.node_ids())


def test_coverage_is_simulated_passenger_flow() -> None:
    """Node coverage sums route draws. Edge coverage averages the endpoints.

    Each route draws Normal(mean=stop count, sd=sqrt(stop count)) with seed 0.
    Draws are in route-id order. A negative draw would be zero.
    """
    stops, routes, trips = _sample()
    graph = transit_cfa(stops, routes, trips, k=5, passenger_seed=0)
    rng = np.random.default_rng(0)
    bus = max(0.0, float(rng.normal(3.0, math.sqrt(3.0))))
    tram = max(0.0, float(rng.normal(3.0, math.sqrt(3.0))))
    trolley = max(0.0, float(rng.normal(2.0, math.sqrt(2.0))))
    expected = {
        "sa": bus + tram,
        "sb": bus + tram + trolley,
        "sc": bus + trolley,
        "sd": tram,
    }
    node_coverage = {row["node_id"]: float(row["coverage"]) for row in graph.nodes}
    for node_id, value in expected.items():
        assert node_coverage[node_id] == pytest.approx(value)
    edge_coverage = {(row["source"], row["target"]): float(row["coverage"]) for row in graph.edges}
    assert edge_coverage[("sa", "sb")] == pytest.approx((expected["sa"] + expected["sb"]) / 2)
    assert edge_coverage[("sb", "sc")] == pytest.approx((expected["sb"] + expected["sc"]) / 2)
    assert edge_coverage[("sb", "sd")] == pytest.approx((expected["sb"] + expected["sd"]) / 2)
    assert graph.metadata["coverage_rule"] == "simulated_passenger_normal"
    assert graph.metadata["passenger_seed"] == 0


def test_same_name_within_the_cutoff_is_one_stop() -> None:
    """A node is a named place. Close ids merge. A far copy of the name does not."""
    places = [
        StopPlace("1", "Alpha", 60.0, 30.0),
        StopPlace("2", "Alpha", 60.0002, 30.0),
        StopPlace("3", "Alpha", 60.02, 30.0),
        StopPlace("4", "Beta", 60.0001, 30.0),
    ]
    clustered = cluster_stops(places, metres=60.0)
    by_id = {item.stop_id: item for item in clustered}
    assert set(by_id) == {"1", "3", "4"}
    assert by_id["1"].members == ("1", "2")
    assert by_id["3"].members == ("3",)
    assert by_id["4"].members == ("4",)


def test_all_mode_and_route_colours_are_applied() -> None:
    """Each stop and hop carries every mode and route that uses it."""
    stops, routes, trips = _sample()
    graph = transit_cfa(stops, routes, trips, k=5)
    assert graph.metadata["colour_operation"] == "replace"
    by_value = {row["value"]: int(row["color_id"]) for row in graph.colors or []}
    assert {row["namespace"] for row in graph.colors or []} == {"transport_type", "route"}
    assert set(by_value) >= {"bus", "trolley", "tram", "bus:10", "tram:3", "trolley:5"}
    nodes = {row["node_id"]: set(row["color_set"].split(",")) for row in graph.nodes}
    beta = nodes["sb"]
    assert str(by_value["bus"]) in beta
    assert str(by_value["trolley"]) in beta
    assert str(by_value["tram"]) in beta
    assert str(by_value["bus:10"]) in beta
    assert str(by_value["tram:3"]) in beta
    assert str(by_value["trolley:5"]) in beta
    edges = {(row["source"], row["target"]): set(row["color_set"].split(",")) for row in graph.edges}
    assert str(by_value["tram:3"]) in edges[("sb", "sd")]
    assert str(by_value["trolley:5"]) not in edges[("sb", "sd")]


def test_same_inputs_produce_the_same_sequences() -> None:
    """The junction tape is a counter, so the CFA is deterministic."""
    stops, routes, trips = _sample()
    first = transit_cfa(stops, routes, trips, k=5)
    second = transit_cfa(stops, routes, trips, k=5)
    assert first.sequences == second.sequences
    assert [row["edge_id"] for row in first.edges] == [row["edge_id"] for row in second.edges]


def test_missing_stop_and_unknown_mode_fail() -> None:
    """A trip cannot name a stop or a mode that was not supplied."""
    stops, routes, _trips = _sample()
    with pytest.raises(ContractError, match="unknown stop_id"):
        transit_cfa(stops, routes, [("bus1", ["a", "missing"])], k=5)
    with pytest.raises(ContractError, match="transport_type"):
        transit_cfa(
            stops,
            [TransitRoute("ship1", "1", "ship")],
            [("ship1", ["a"])],
            k=5,
        )
