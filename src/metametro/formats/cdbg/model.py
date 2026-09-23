"""In-memory ToCUMG (totally coloured universal metagenomic graph).

The on-disk name remains CDBG. ``graph_type`` is whatever the source graph
declared (de Bruijn, repeat, LCA, or another type). ``k`` is present only
when that graph declared it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass
class Unitig:
    """One compacted path of CFA nodes."""

    unitig_id: str
    sequence: str
    members: list[str]
    color_ids: list[int]
    internal_edge_ids: list[str] = field(default_factory=list)
    internal_edge_colors: list[list[int]] = field(default_factory=list)
    internal_overlaps: list[int] = field(default_factory=list)


@dataclass
class Link:
    """An adjacency that compaction did not absorb."""

    link_id: str
    source: str
    target: str
    orientation: str | None
    color_ids: list[int] = field(default_factory=list)
    overlap: int | None = None


@dataclass
class NodeMap:
    """Where one CFA node sits inside a unitig."""

    cfa_node_id: str
    unitig_id: str
    ordinal: int
    length: int
    color_ids: list[int]


@dataclass
class Cdbg:
    """ToCUMG plus the mandatory CFA mapping.

    ``k`` is ``None`` when the source graph did not declare a de Bruijn ``k``.
    ``annotations`` is an optional columnar sidecar. It is not a feature
    matrix and it is not stored on each unitig. An empty list means the
    graph has no annotation layers.
    """

    metadata: dict[str, Any]
    k: int | None
    unitigs: list[Unitig]
    links: list[Link]
    mapping: list[NodeMap]
    colors: list[dict[str, str]] | None = None
    labels: list[dict[str, str]] | None = None
    annotations: list = field(default_factory=list)
