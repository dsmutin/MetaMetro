"""In-memory compacted colored de Bruijn graph."""

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


@dataclass
class Link:
    """An adjacency that compaction did not absorb."""

    link_id: str
    source: str
    target: str
    orientation: str | None
    color_ids: list[int] = field(default_factory=list)


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
    """Compact colored graph plus the mandatory CFA mapping."""

    metadata: dict[str, Any]
    k: int
    unitigs: list[Unitig]
    links: list[Link]
    mapping: list[NodeMap]
    colors: list[dict[str, str]] | None = None
    labels: list[dict[str, str]] | None = None
