"""metametro: Metagenomic assembly totally coloured graph representations"""

from __future__ import annotations

from pathlib import Path


def _version() -> str:
    """Return the package version from the repository VERSION file."""
    path = Path(__file__).resolve().parents[2] / "VERSION"
    return path.read_text(encoding="utf-8").strip()


__version__ = _version()

from metametro.contracts import CONTRACTS, run_ds
from metametro.edits import (
    EditProposal,
    GraphEdit,
    apply_edit_proposal,
    validate_edit_application,
    validate_edit_proposal,
)
from metametro.converters import cdbg_to_cfa, cdbg_to_cgt, cfa_to_cdbg, cgt_from_csr, edge_index_array, to_pyg
from metametro.fixtures import chain_cfa, chain_cdbg, mock_cdbg, mock_cfa, mock_cgt
from metametro.identity import (
    assert_cgt_matches_cdbg,
    cgt_edge_cfa_ids,
    internal_cfa_edge_ids,
    link_cfa_edge_id,
    node_lineage,
    resolve_sequence,
)
from metametro.formats.cdbg import (
    aggregate_annotations,
    annotate_cdbg,
    dump_cdbg,
    get_edge_annotations,
    get_node_annotations,
    load_cdbg,
    transfer_annotations,
    validate_cdbg,
)
from metametro.formats.cfa import dump_cfa, load_cfa, validate_cfa
from metametro.formats.cgt import (
    GraphPrediction,
    csc_from_cgt,
    dump_cgt,
    edge_prediction,
    load_cgt,
    predictions_from_ds,
    validate_cgt,
)

__all__ = [
    "CONTRACTS",
    "EditProposal",
    "GraphEdit",
    "apply_edit_proposal",
    "aggregate_annotations",
    "annotate_cdbg",
    "assert_cgt_matches_cdbg",
    "cdbg_to_cfa",
    "GraphPrediction",
    "cgt_edge_cfa_ids",
    "csc_from_cgt",
    "cdbg_to_cgt",
    "cgt_from_csr",
    "cfa_to_cdbg",
    "chain_cdbg",
    "chain_cfa",
    "dump_cdbg",
    "dump_cfa",
    "dump_cgt",
    "edge_index_array",
    "edge_prediction",
    "get_edge_annotations",
    "get_node_annotations",
    "internal_cfa_edge_ids",
    "link_cfa_edge_id",
    "load_cdbg",
    "load_cfa",
    "load_cgt",
    "mock_cdbg",
    "mock_cfa",
    "mock_cgt",
    "node_lineage",
    "predictions_from_ds",
    "resolve_sequence",
    "run_ds",
    "to_pyg",
    "transfer_annotations",
    "validate_cdbg",
    "validate_edit_application",
    "validate_edit_proposal",
    "validate_cfa",
    "validate_cgt",
]
