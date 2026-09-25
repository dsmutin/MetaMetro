"""Build one benchmark into CFA, CDBG, and CGT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from metametro.bench.colourings import BuildContext
from metametro.bench.materialize import Materialized, materialize
from metametro.bench.paths import default_outdir
from metametro.bench.registry import PREPARE, resolve
from metametro.bench.spec import BenchSpec
from metametro.errors import ContractError
from metametro.tables import write_yaml


@dataclass(frozen=True)
class BuildResult:
    """One finished ``benchbuild`` run."""

    spec: BenchSpec
    outdir: Path
    identity: str
    colourings: tuple[str, ...]
    status: str


def build(
    name: str,
    *,
    outdir: Path | None = None,
    colourings: tuple[str, ...] | None = None,
    root: Path | None = None,
) -> BuildResult:
    """Build ``name`` and write the three graph structures.

    ``colourings`` restricts the registry. ``None`` applies every colouring
    that can run on this build, including colourings registered later.
    """
    spec = resolve(name)
    if spec.kind != "inprocess":
        raise ContractError([f"{spec.name} is not an in-process benchmark"])
    destination = Path(outdir) if outdir is not None else default_outdir(spec, root=root)
    if destination.exists() and any(destination.iterdir()):
        raise ContractError([f"output directory is not empty: {destination}"])
    ctx = BuildContext(selected=colourings)
    graph = PREPARE[spec.name](destination, ctx)
    made: Materialized = materialize(graph, destination, ctx)
    manifest = {
        "bench": spec.name,
        "assembler": spec.assembler,
        "properties": spec.properties,
        "status": "built",
        "colourings": list(made.colourings),
        "structures": ["cfa", "cdbg", "cgt"],
        "identity": made.identity,
        "scoring": list(spec.scoring),
    }
    if name != spec.name:
        manifest["requested_name"] = name
    write_yaml(destination / "manifest.yaml", manifest)
    return BuildResult(
        spec=spec,
        outdir=destination,
        identity=made.identity,
        colourings=made.colourings,
        status="built",
    )
