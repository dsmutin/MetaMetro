"""Stable identity of a built graph or of a pinned contract.

The digest covers topology, sequence, and colour sets. It does not cover
unitig ids, dense ids, or filesystem paths.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from metametro.formats.cfa.model import CfaGraph


def _colour(row: dict[str, str]) -> str:
    return row.get("color_set", row.get("colors", ""))


def graph_identity(graph: CfaGraph, *, colourings: tuple[str, ...]) -> str:
    """SHA-256 of the coloured CFA, independent of directory layout."""
    lines = [
        f"graph_id\t{graph.metadata.get('graph_id', '')}",
        f"graph_type\t{graph.metadata.get('graph_type', '')}",
        f"k\t{graph.metadata.get('k', '')}",
        "colourings\t" + ",".join(colourings),
    ]
    for row in graph.nodes:
        node_id = row["node_id"]
        lines.append(f"node\t{node_id}\t{graph.sequences[node_id]}\t{_colour(row)}")
    for row in graph.edges:
        lines.append(
            f"edge\t{row['edge_id']}\t{row['source']}\t{row['target']}\t"
            f"{row.get('orientation', '++')}\t{_colour(row)}"
        )
    for row in graph.colors or []:
        lines.append(f"palette\t{row.get('color_id', '')}\t{row.get('namespace', '')}\t{row.get('value', '')}")
    payload = "\n".join(lines) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contract_identity(path: Path) -> str:
    """SHA-256 of ``contract/`` and ``ground_truth/`` only.

    ``manifest.yaml`` and ``identity.sha256`` are not part of the digest, so
    recording the digest does not change it.
    """
    files: list[Path] = []
    for folder in ("contract", "ground_truth"):
        root = path / folder
        if root.is_dir():
            files.extend(item for item in root.rglob("*") if item.is_file())
    files.sort()
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_identity(path: Path, digest: str) -> None:
    """Write ``identity.sha256`` as ``<digest>  identity``."""
    path.write_text(f"{digest}  identity\n", encoding="utf-8")
