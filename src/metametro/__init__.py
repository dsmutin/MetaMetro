"""metametro: Metagenomic assembly totally coloured graph representations"""

from __future__ import annotations

from pathlib import Path


def _version() -> str:
    """Return the package version from the repository VERSION file."""
    path = Path(__file__).resolve().parents[2] / "VERSION"
    return path.read_text(encoding="utf-8").strip()


__version__ = _version()

from metametro.contracts import CONTRACTS, run_ds
from metametro.converters import cdbg_to_cfa, cdbg_to_cgt, cfa_to_cdbg, edge_index_array, to_pyg
from metametro.fixtures import chain_cfa, chain_cdbg, mock_cdbg, mock_cfa, mock_cgt
from metametro.identity import (
    assert_cgt_matches_cdbg,
    cgt_edge_cfa_ids,
    internal_cfa_edge_ids,
    link_cfa_edge_id,
    node_lineage,
)
from metametro.formats.cdbg import dump_cdbg, load_cdbg, validate_cdbg
from metametro.formats.cfa import dump_cfa, load_cfa, validate_cfa
from metametro.formats.cgt import dump_cgt, load_cgt, validate_cgt

__all__ = [
    "CONTRACTS",
    "assert_cgt_matches_cdbg",
    "cdbg_to_cfa",
    "cgt_edge_cfa_ids",
    "cdbg_to_cgt",
    "cfa_to_cdbg",
    "chain_cdbg",
    "chain_cfa",
    "dump_cdbg",
    "dump_cfa",
    "dump_cgt",
    "edge_index_array",
    "internal_cfa_edge_ids",
    "link_cfa_edge_id",
    "load_cdbg",
    "load_cfa",
    "load_cgt",
    "mock_cdbg",
    "mock_cfa",
    "mock_cgt",
    "node_lineage",
    "run_ds",
    "to_pyg",
    "validate_cdbg",
    "validate_cfa",
    "validate_cgt",
]
