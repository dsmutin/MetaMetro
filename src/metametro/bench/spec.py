"""Benchmark names, assemblers, and the properties that enter the output path."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchSpec:
    """One benchmark the ``benchbuild`` command can materialise.

    ``properties`` is the last path segment under
    ``data/bench/{name}/{assembler}/{properties}``. It records the knobs that
    change the graph (read budget, k, seed) and is not a machine path.
    """

    name: str
    assembler: str
    properties: str
    summary: str
    scoring: tuple[str, ...]
    kind: str
    legacy_names: tuple[str, ...] = ()
    score_rank: str = ""
    total_reads: int = 0
    pair_count: int = 0
    domain_count: int = 0
    k: int = 0
    parent: str = ""

    def path_parts(self) -> tuple[str, str, str]:
        """Return ``(name, assembler, properties)`` for the default output tree."""
        return (self.name, self.assembler, self.properties)
