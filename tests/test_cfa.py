"""Mandatory CFA schema and structural checks."""

from __future__ import annotations

import pytest

from metametro.errors import ContractError
from metametro.fixtures import mock_cfa
from metametro.formats.cfa.io import dump_cfa, load_cfa
from metametro.formats.cfa.validator import validate_cfa

pytestmark = pytest.mark.mandatory


def test_valid_minimal_bubble(tmp_path) -> None:
    """The branching mock loads, validates, and round-trips through disk."""
    graph = mock_cfa()
    validate_cfa(graph)
    dump_cfa(graph, tmp_path / "cfa")
    loaded = load_cfa(tmp_path / "cfa")
    assert loaded.node_ids() == graph.node_ids()
    assert loaded.sequences == graph.sequences
    assert len(loaded.edges) == 5


def test_missing_required_field() -> None:
    """A CFA without graph_id is rejected."""
    graph = mock_cfa()
    del graph.metadata["graph_id"]
    with pytest.raises(ContractError, match="graph_id"):
        validate_cfa(graph)


def test_incompatible_schema_version() -> None:
    """Draft schema 0.1 is not the pinned contract."""
    graph = mock_cfa()
    graph.metadata["schema_version"] = "0.1"
    with pytest.raises(ContractError, match="incompatible schema"):
        validate_cfa(graph)


def test_wrong_dtype() -> None:
    """A non-numeric coverage value is rejected."""
    graph = mock_cfa()
    graph.nodes[0]["coverage"] = "high"
    with pytest.raises(ContractError, match="wrong dtype"):
        validate_cfa(graph)


def test_duplicate_node_id() -> None:
    """Duplicate node identifiers are rejected."""
    graph = mock_cfa()
    graph.nodes.append(dict(graph.nodes[0]))
    with pytest.raises(ContractError, match="duplicate node_id"):
        validate_cfa(graph)


def test_dangling_edge() -> None:
    """An edge into a missing node is rejected."""
    graph = mock_cfa()
    graph.edges[0]["target"] = "n999999"
    with pytest.raises(ContractError, match="dangling edge"):
        validate_cfa(graph)


def test_undefined_color_and_label() -> None:
    """Colour and label ids must exist in their dictionaries."""
    graph = mock_cfa()
    graph.nodes[0]["color_set"] = "9"
    with pytest.raises(ContractError, match="undefined color"):
        validate_cfa(graph)
    graph = mock_cfa()
    graph.nodes[0]["label"] = "9"
    with pytest.raises(ContractError, match="undefined label"):
        validate_cfa(graph)


def test_malformed_sequence() -> None:
    """Sequences outside ACGTN are rejected."""
    graph = mock_cfa()
    graph.sequences["n000001"] = "ACGX"
    with pytest.raises(ContractError, match="malformed sequence"):
        validate_cfa(graph)


def test_sequence_length_column_rejected() -> None:
    """Length is not stored as an authoritative column."""
    graph = mock_cfa()
    graph.nodes[0]["sequence_length"] = "4"
    graph.metadata["features"]["node"]["sequence_length"] = "int"
    with pytest.raises(ContractError, match="sequence_length"):
        validate_cfa(graph)


def test_missing_file(tmp_path) -> None:
    """A directory without edges.tsv is rejected."""
    graph = mock_cfa()
    dump_cfa(graph, tmp_path / "cfa")
    (tmp_path / "cfa" / "edges.tsv").unlink()
    with pytest.raises(ContractError, match="edges.tsv"):
        load_cfa(tmp_path / "cfa")
