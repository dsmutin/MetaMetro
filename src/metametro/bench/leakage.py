"""Refuse evaluation targets that were written into a graph a model can read.

Ground truth belongs under ``ground_truth/``. It must not be a feature column
or a colour namespace on CFA, CDBG, or CGT.
"""

from __future__ import annotations

from metametro.errors import ContractError
from metametro.formats.cfa.model import CfaGraph
from metametro.formats.cgt.model import Cgt

_FORBIDDEN = (
    "truth_taxon",
    "truth_taxon_id",
    "expected_action",
    "genome_abundance",
    "target_metric",
    "gold_standard",
    "gold_taxon",
)
_FORBIDDEN_NAMESPACES = frozenset({"truth", "gold", "target", "expected_action"})


def _names(graph: CfaGraph) -> list[str]:
    found: list[str] = []
    features = graph.metadata.get("features") or {}
    if isinstance(features, dict):
        for kind in ("node", "edge"):
            declared = features.get(kind) or {}
            if isinstance(declared, dict):
                found.extend(str(name) for name in declared)
    found.extend(graph.node_header)
    found.extend(graph.edge_header)
    return found


def assert_no_target_leak(graph: CfaGraph, *, where: str = "CFA") -> None:
    """Raise ``ContractError`` when a target column or namespace is on the graph."""
    bad = []
    for name in _names(graph):
        folded = name.lower()
        if any(token in folded for token in _FORBIDDEN):
            bad.append(f"{where} column {name}")
    for row in graph.colors or []:
        namespace = str(row.get("namespace", "")).lower()
        if namespace in _FORBIDDEN_NAMESPACES:
            bad.append(f"{where} colour namespace {namespace}")
    if bad:
        raise ContractError(["target leaked into the graph: " + "; ".join(bad)])


def assert_cgt_no_target_leak(graph: Cgt) -> None:
    """Raise ``ContractError`` when a CGT feature name is an evaluation target."""
    names: list[str] = []
    for key in ("node_feature_names", "edge_feature_names"):
        raw = graph.metadata.get(key) or []
        if isinstance(raw, list):
            names.extend(str(item) for item in raw)
    bad = [name for name in names if any(token in name.lower() for token in _FORBIDDEN)]
    if bad:
        raise ContractError(["target leaked into CGT features: " + ", ".join(bad)])
