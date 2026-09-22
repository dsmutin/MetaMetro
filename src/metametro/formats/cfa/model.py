"""In-memory CFA graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "1.0"
ALPHABET = frozenset("ACGTN")
ORIENTATIONS = frozenset({"++", "+-", "-+", "--"})
FEATURE_TYPES = frozenset({"float", "int", "str", "color_set", "orientation"})
REJECTED_COLUMNS = frozenset({"length", "sequence_length"})


@dataclass
class CfaGraph:
    """Canonical semantic graph.

    Rows keep string cells so dump/load can round-trip the TSV text.
    Sequences are the only authoritative sequence store.
    """

    metadata: dict[str, Any]
    sequences: dict[str, str]
    nodes: list[dict[str, str]]
    edges: list[dict[str, str]]
    colors: list[dict[str, str]] | None = None
    labels: list[dict[str, str]] | None = None
    node_header: list[str] = field(default_factory=lambda: ["node_id"])
    edge_header: list[str] = field(default_factory=lambda: ["edge_id", "source", "target"])

    def node_ids(self) -> list[str]:
        """Return node identifiers in table order."""
        return [row["node_id"] for row in self.nodes]
