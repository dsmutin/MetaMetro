"""NumPy archive plus metadata for a CGT directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from metametro.errors import ContractError
from metametro.formats.cgt.model import Cgt
from metametro.formats.cgt.validator import validate_cgt
from metametro.tables import read_tsv, read_yaml, write_tsv, write_yaml


def dump_cgt(graph: Cgt, path: str | Path) -> None:
    """Write a CGT directory. Arrays are NumPy ``.npy`` files."""
    validate_cgt(graph)
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    write_yaml(root / "metadata.yaml", graph.metadata)
    np.save(root / "indptr.npy", graph.indptr)
    np.save(root / "indices.npy", graph.indices)
    np.save(root / "node_features.npy", graph.node_features)
    np.save(root / "edge_features.npy", graph.edge_features)
    np.save(root / "node_colors.npy", graph.node_colors)
    np.save(root / "edge_colors.npy", graph.edge_colors)
    _save_or_remove(root / "node_color_weights.npy", graph.node_color_weights)
    _save_or_remove(root / "edge_color_weights.npy", graph.edge_color_weights)
    if graph.node_labels is not None:
        np.save(root / "node_labels.npy", graph.node_labels)
    if graph.edge_labels is not None:
        np.save(root / "edge_labels.npy", graph.edge_labels)
    write_tsv(
        root / "mapping.tsv",
        ["dense_id", "source_id", "cfa_node_ids"],
        [
            {
                "dense_id": row["dense_id"],
                "source_id": row["source_id"],
                "cfa_node_ids": ",".join(row["cfa_node_ids"]),
            }
            for row in graph.mapping
        ],
    )
    write_tsv(
        root / "color_ids.tsv",
        ["column", "color_id"],
        [{"column": index, "color_id": color_id} for index, color_id in enumerate(graph.color_ids)],
    )


def _optional(root: Path, name: str) -> np.ndarray | None:
    path = root / name
    if not path.is_file():
        return None
    return np.load(path)


def _save_or_remove(path: Path, array: np.ndarray | None) -> None:
    """Write an optional array, or delete a stale file from an earlier dump."""
    if array is None:
        if path.is_file():
            path.unlink()
        return
    np.save(path, array)


def load_cgt(path: str | Path, *, validate: bool = True) -> Cgt:
    """Load a CGT directory."""
    root = Path(path)
    required = [
        "metadata.yaml",
        "indptr.npy",
        "indices.npy",
        "node_features.npy",
        "edge_features.npy",
        "node_colors.npy",
        "edge_colors.npy",
        "mapping.tsv",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ContractError([f"missing required CGT file: {name}" for name in missing])
    metadata: dict[str, Any] = read_yaml(root / "metadata.yaml")
    header, rows = read_tsv(root / "mapping.tsv")
    required_columns = {"dense_id", "source_id", "cfa_node_ids"}
    missing_columns = sorted(required_columns - set(header))
    if missing_columns:
        raise ContractError(
            [f"missing required CGT mapping column: {column}" for column in missing_columns]
        )
    mapping = [
        {
            "dense_id": int(row["dense_id"]),
            "source_id": row["source_id"],
            "cfa_node_ids": [token for token in row["cfa_node_ids"].split(",") if token],
        }
        for row in rows
    ]
    color_ids: list[int] = []
    if (root / "color_ids.tsv").is_file():
        _, color_rows = read_tsv(root / "color_ids.tsv")
        color_ids = [int(row["color_id"]) for row in color_rows]
    graph = Cgt(
        metadata=metadata,
        indptr=np.load(root / "indptr.npy"),
        indices=np.load(root / "indices.npy"),
        node_features=np.load(root / "node_features.npy"),
        edge_features=np.load(root / "edge_features.npy"),
        node_colors=np.load(root / "node_colors.npy"),
        edge_colors=np.load(root / "edge_colors.npy"),
        mapping=mapping,
        node_labels=_optional(root, "node_labels.npy"),
        edge_labels=_optional(root, "edge_labels.npy"),
        color_ids=color_ids,
        node_color_weights=_optional(root, "node_color_weights.npy"),
        edge_color_weights=_optional(root, "edge_color_weights.npy"),
    )
    if validate:
        validate_cgt(graph)
    return graph
