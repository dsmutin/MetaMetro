"""Model predictions stored outside a CGT.

A prediction is not a feature column, a training label, or a colour. These
objects do not write into CGT arrays. ``run_ds`` still returns its existing
row keys. ``predictions_from_ds`` wraps that mapping.

The NumPy and PyG trainers emit one softmax probability per class. They do
not emit a second uncertainty estimate. ``run_ds`` therefore sets
``confidence`` equal to the probability of ``predicted_class``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from metametro.errors import ContractError
from metametro.formats.cdbg.model import Cdbg
from metametro.formats.cgt.model import Cgt
from metametro.formats.cgt.validator import validate_cgt
from metametro.identity import assert_cgt_matches_cdbg, cgt_edge_cfa_ids


@dataclass(frozen=True)
class GraphPrediction:
    """One predicted class, joinable without reading a feature matrix.

    ``dense_id``, ``source_id``, and ``cfa_ids`` are the join keys. For
    ``target="node"``, ``source_id`` is the CDBG unitig id and ``cfa_ids``
    are the member CFA node ids. For ``target="edge"``, ``dense_id`` is the
    CSR source node, ``source_id`` is the CDBG link id, ``cfa_ids`` is that
    link's CFA edge id, and ``csr_slot`` is the CSR position of the link.

    ``predicted_class`` is the model class. ``probability`` is that class's
    probability. ``confidence`` is a separate field. For the softmax models
    in ``run_ds`` the two numbers are equal, because those models have no
    other uncertainty output.
    """

    dense_id: int
    source_id: str
    cfa_ids: tuple[str, ...]
    predicted_class: int
    probability: float
    confidence: float
    model_id: str
    model_version: str
    target: str
    csr_slot: int | None = None


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or isinstance(value, np.bool_):
        raise ContractError([f"{label} must be an integer"])
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ContractError([f"{label} must be an integer"])


def _probability(value: object, label: str) -> float:
    if isinstance(value, bool) or isinstance(value, np.bool_):
        raise ContractError([f"{label} must be a float"])
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not np.isfinite(number) or number < 0.0 or number > 1.0:
            raise ContractError([f"{label} must be a finite probability in [0, 1]"])
        return number
    raise ContractError([f"{label} must be a float"])


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ContractError([f"{label} must be a non-empty string"])
    return value


def predictions_from_ds(result: Mapping[str, Any]) -> tuple[GraphPrediction, ...]:
    """Wrap node rows from ``run_ds`` as predictions.

    Existing result keys are read and left in place. ``predicted_class`` is
    the integer ``predicted_label``. ``confidence`` is the row's
    ``confidence``. ``run_ds`` sets that field to the softmax probability of
    the predicted class. This function does not train a model, does not
    invent edge predictions, and does not accept a CGT, so it cannot write
    CGT arrays.
    """
    if not isinstance(result, Mapping):
        raise ContractError(["DS result must be a mapping"])
    rows = result.get("result")
    metadata = result.get("metadata")
    if not isinstance(rows, list):
        raise ContractError(["DS result is missing result rows"])
    if not isinstance(metadata, Mapping):
        raise ContractError(["DS result is missing metadata"])
    model_id = _text(metadata.get("model_id"), "model_id")
    model_version = _text(metadata.get("model_version"), "model_version")
    predictions: list[GraphPrediction] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ContractError([f"DS result row {index} must be a mapping"])
        if "csr_slot" in row:
            raise ContractError(["DS result contains an edge prediction"])
        predicted_label = _integer(row.get("predicted_label"), "predicted_label")
        if "predicted_class" in row:
            predicted_class = _integer(row.get("predicted_class"), "predicted_class")
            if predicted_class != predicted_label:
                raise ContractError(["predicted_class does not match predicted_label"])
        else:
            predicted_class = predicted_label
        if "confidence" not in row:
            raise ContractError(["DS result row is missing confidence"])
        cfa_node_ids = row.get("cfa_node_ids")
        if (
            not isinstance(cfa_node_ids, list)
            or not cfa_node_ids
            or any(not isinstance(node_id, str) or node_id == "" for node_id in cfa_node_ids)
        ):
            raise ContractError([f"DS result row {index} has no CFA node ids"])
        predictions.append(
            GraphPrediction(
                dense_id=_integer(row.get("dense_id"), "dense_id"),
                source_id=_text(row.get("source_id"), "source_id"),
                cfa_ids=tuple(cfa_node_ids),
                predicted_class=predicted_class,
                probability=_probability(row.get("probability"), "probability"),
                confidence=_probability(row.get("confidence"), "confidence"),
                model_id=_text(row.get("model_id", model_id), "model_id"),
                model_version=_text(row.get("model_version", model_version), "model_version"),
                target="node",
                csr_slot=None,
            )
        )
    return tuple(predictions)


def edge_prediction(
    cgt: Cgt,
    csr_slot: int,
    *,
    predicted_class: int,
    probability: float,
    confidence: float,
    model_id: str,
    model_version: str,
    cdbg: Cdbg,
) -> GraphPrediction:
    """Build one edge prediction for an existing CSR slot.

    ``run_ds`` does not call this and does not invent edge predictions. The
    caller supplies the class, probability, and confidence. ``cdbg`` supplies
    the CFA edge id through ``cgt_edge_cfa_ids``. ``dense_id`` is the source
    node of the CSR slot. ``source_id`` is the CDBG link id, which schema 1.0
    stores as that CFA edge id. The CGT arrays are not written.
    """
    validate_cgt(cgt)
    assert_cgt_matches_cdbg(cgt, cdbg)
    slot = _integer(csr_slot, "csr_slot")
    if slot < 0 or slot >= cgt.num_edges:
        raise ContractError([f"csr_slot {slot} is outside 0 .. {cgt.num_edges - 1}"])
    edge_ids = cgt_edge_cfa_ids(cdbg)
    if len(edge_ids) != cgt.num_edges:
        raise ContractError(["CGT edge count does not match the CDBG links"])
    source = int(np.searchsorted(cgt.indptr, slot, side="right") - 1)
    if source < 0 or source >= cgt.num_nodes:
        raise ContractError([f"csr_slot {slot} has no CSR source node"])
    if not (int(cgt.indptr[source]) <= slot < int(cgt.indptr[source + 1])):
        raise ContractError([f"csr_slot {slot} is outside its CSR source range"])
    cfa_edge_id = edge_ids[slot]
    return GraphPrediction(
        dense_id=source,
        source_id=cfa_edge_id,
        cfa_ids=(cfa_edge_id,),
        predicted_class=_integer(predicted_class, "predicted_class"),
        probability=_probability(probability, "probability"),
        confidence=_probability(confidence, "confidence"),
        model_id=_text(model_id, "model_id"),
        model_version=_text(model_version, "model_version"),
        target="edge",
        csr_slot=slot,
    )
