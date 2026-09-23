"""Mandatory checks for faceted CFA colouring figures."""

from __future__ import annotations

import pytest

from metametro.errors import ContractError
from metametro.transit_cfa import TransitRoute, transit_cfa
from metametro.viz.cfa_colouring import (
    NA_COLOUR,
    OTHER_COLOUR,
    ColourFacet,
    ColourLabel,
    cfa_colour_frames,
    count_colours,
    even_route_percent,
    plot_cfa_colouring,
)

pytestmark = pytest.mark.mandatory


def _graph():
    """Bus stops a-b and a tram continuation b-c."""
    stops = {"a": "Alpha", "b": "Beta", "c": "Gamma"}
    routes = [
        TransitRoute("bus2", "2", "bus"),
        TransitRoute("bus3", "3", "bus"),
        TransitRoute("tram1", "1", "tram"),
    ]
    trips = [
        ("bus2", ["a", "b"]),
        ("bus3", ["a", "b"]),
        ("tram1", ["b", "c"]),
    ]
    return transit_cfa(stops, routes, trips, graph_id="viz", k=5)


def _route_label(legend: str, reduce, limits=None) -> ColourLabel:
    return ColourLabel(
        namespace="route",
        legend=legend,
        matches=lambda level, value: value.startswith(f"{level}:"),
        reduce=reduce,
        palette="YlGnBu",
        limits=limits,
    )


def _frames():
    return cfa_colour_frames(
        _graph(),
        node_label=_route_label("Even-numbered routes (%)", even_route_percent, (0.0, 100.0)),
        edge_label=_route_label("Routes (count)", count_colours),
        facet=ColourFacet(namespace="transport_type", levels=("bus", "tram")),
    )


def test_absent_facet_members_stay_as_na() -> None:
    """A stop with no bus colour remains in the bus panel as NA."""
    frames = _frames()
    assert set(frames.nodes["bus"]) == {"sa", "sb", "sc"}
    assert set(frames.edges["bus"]) == {"e000001", "e000002"}
    assert frames.nodes["bus"]["sc"] is None
    assert frames.nodes["tram"]["sa"] is None
    assert frames.edges["tram"]["e000001"] is None
    assert frames.edges["bus"]["e000002"] is None
    assert frames.nodes["bus"]["sa"] == pytest.approx(50.0)
    assert frames.nodes["tram"]["sc"] == pytest.approx(0.0)
    assert frames.edges["bus"]["e000001"] == pytest.approx(2.0)
    assert frames.edges["tram"]["e000002"] == pytest.approx(1.0)


def test_even_route_percent_uses_the_leading_number() -> None:
    """A letter suffix does not change the route number. No digits is not even."""
    assert even_route_percent(["bus:2", "bus:3"]) == pytest.approx(50.0)
    assert even_route_percent(["bus:145Б", "bus:10"]) == pytest.approx(50.0)
    assert even_route_percent(["tram:А"]) == pytest.approx(0.0)
    assert even_route_percent([]) is None


def test_unknown_facet_level_is_rejected() -> None:
    """A requested facet level must exist in the colour dictionary."""
    with pytest.raises(ContractError, match="facet level"):
        cfa_colour_frames(
            _graph(),
            node_label=_route_label("Even-numbered routes (%)", even_route_percent),
            edge_label=_route_label("Routes (count)", count_colours),
            facet=ColourFacet(namespace="transport_type", levels=("ship",)),
        )


def test_plot_keeps_na_nodes_and_writes_pdf(tmp_path) -> None:
    """The PDF is a vector file and the frames still list every stop."""
    output = tmp_path / "colouring.pdf"
    frames = plot_cfa_colouring(
        _graph(),
        node_label=_route_label("Even-numbered routes (%)", even_route_percent, (0.0, 100.0)),
        edge_label=_route_label("Routes (count)", count_colours),
        facet=ColourFacet(
            namespace="transport_type",
            levels=("bus", "tram"),
            level_labels={"bus": "Bus", "tram": "Tram"},
        ),
        path=output,
        x_label="Layout x",
        y_label="Layout y",
        positions={"sa": (0.0, 0.0), "sb": (1.0, 0.0), "sc": (1.0, 1.0)},
    )
    assert output.is_file() and output.stat().st_size > 0
    assert output.read_bytes().startswith(b"%PDF")
    assert frames.nodes["bus"]["sc"] is None


def test_other_category_is_gray_and_last() -> None:
    """Discrete Set1 puts Other last and paints it gray80."""
    from metametro.viz.cfa_colouring import _discrete_colours

    colours = _discrete_colours(["bus", "Other", "tram"], "Set1")
    assert list(colours) == ["bus", "tram", "Other"]
    assert colours["Other"] == OTHER_COLOUR
    assert colours["Other"] == NA_COLOUR
