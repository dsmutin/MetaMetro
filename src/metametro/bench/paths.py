"""Output locations for a benchmark build. Results are gitignored."""

from __future__ import annotations

from pathlib import Path

from metametro.bench.spec import BenchSpec


def repo_root() -> Path:
    """Return the MetaMetro checkout that contains this package."""
    return Path(__file__).resolve().parents[3]


def default_outdir(spec: BenchSpec, *, root: Path | None = None) -> Path:
    """Return ``data/bench/{name}/{assembler}/{properties}`` under ``root``.

    ``root`` defaults to the repository. Callers that pass ``--outdir`` replace
    this path entirely; they do not append another segment.
    """
    base = repo_root() if root is None else Path(root)
    name, assembler, properties = spec.path_parts()
    return base / "data" / "bench" / name / assembler / properties
