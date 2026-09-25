"""Build one benchmark into a contract and, when tools allow, three graphs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from metametro.bench.colourings import BuildContext
from metametro.bench.data.universal.catalog import (
    require_programs,
    write_community_contract,
    write_external_contract,
)
from metametro.bench.data.universal.community import highest_megahit_contigs, load_pairs
from metametro.bench.identity import contract_identity, write_identity
from metametro.bench.materialize import Materialized, materialize
from metametro.bench.paths import default_outdir, repo_root
from metametro.bench.registry import PREPARE, resolve
from metametro.bench.spec import BenchSpec
from metametro.contracts.assembly import fastg_to_cfa
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
    execute: bool = False,
    gtfs: Path | None = None,
) -> BuildResult:
    """Build ``name``.

    In-process benchmarks always write CFA, CDBG, and CGT. Community and
    external benchmarks always write their contract. They download and assemble
    only when ``execute`` is true, and they stop if a required program is
    missing. ``colourings`` restricts the registry. ``None`` applies every
    colouring that can run, including colourings registered later.
    """
    spec = resolve(name)
    destination = Path(outdir) if outdir is not None else default_outdir(spec, root=root)
    if destination.exists() and any(destination.iterdir()):
        raise ContractError([f"output directory is not empty: {destination}"])
    if spec.kind == "inprocess":
        return _inprocess(spec, destination, colourings, requested=name)
    if spec.kind == "community":
        return _community(spec, destination, colourings, execute=execute, requested=name)
    if spec.kind == "external":
        return _external(
            spec, destination, colourings, execute=execute, gtfs=gtfs, requested=name
        )
    raise ContractError([f"unknown benchmark kind {spec.kind}"])


def _finish_manifest(
    spec: BenchSpec,
    destination: Path,
    *,
    status: str,
    identity: str,
    colourings: tuple[str, ...],
    requested: str,
) -> None:
    manifest = {
        "bench": spec.name,
        "assembler": spec.assembler,
        "properties": spec.properties,
        "status": status,
        "colourings": list(colourings),
        "identity": identity,
        "scoring": list(spec.scoring),
    }
    if requested != spec.name:
        manifest["requested_name"] = requested
    write_yaml(destination / "manifest.yaml", manifest)


def _inprocess(
    spec: BenchSpec,
    destination: Path,
    colourings: tuple[str, ...] | None,
    *,
    requested: str,
) -> BuildResult:
    ctx = BuildContext(selected=colourings)
    graph = PREPARE[spec.name](destination, ctx)
    made: Materialized = materialize(graph, destination, ctx)
    _finish_manifest(
        spec,
        destination,
        status="built",
        identity=made.identity,
        colourings=made.colourings,
        requested=requested,
    )
    return BuildResult(
        spec=spec,
        outdir=destination,
        identity=made.identity,
        colourings=made.colourings,
        status="built",
    )


def _community(
    spec: BenchSpec,
    destination: Path,
    colourings: tuple[str, ...] | None,
    *,
    execute: bool,
    requested: str,
) -> BuildResult:
    write_community_contract(spec, destination)
    if not execute:
        digest = contract_identity(destination)
        write_identity(destination / "identity.sha256", digest)
        _finish_manifest(
            spec,
            destination,
            status="contract",
            identity=digest,
            colourings=(),
            requested=requested,
        )
        return BuildResult(
            spec=spec, outdir=destination, identity=digest, colourings=(), status="contract"
        )
    require_programs(spec, ("datasets", "samovar", "megahit", "megahit_toolkit"))
    graph = _assemble_community(spec, destination)
    ctx = BuildContext(selected=colourings)
    made = materialize(graph, destination, ctx)
    _finish_manifest(
        spec,
        destination,
        status="built",
        identity=made.identity,
        colourings=made.colourings,
        requested=requested,
    )
    return BuildResult(
        spec=spec,
        outdir=destination,
        identity=made.identity,
        colourings=made.colourings,
        status="built",
    )


def _assemble_community(spec: BenchSpec, destination: Path):
    from metametro.bench.data.universal import ncbi_pipeline as pipe

    root = repo_root()
    pipe.configure(
        example=destination / "contract",
        work=destination / "work",
        root=root,
        graph_id=spec.name,
        score_rank=spec.score_rank,
        total_reads=spec.total_reads,
        report_path=root / "data" / "raw" / f"{spec.name}_assembly_data_report.jsonl",
        zip_name=f"ncbi_{spec.name}.zip",
    )
    if spec.parent:
        pipe.stage_download = lambda: _require_parent_fasta(spec, root)
    for stage in ("download", "layout", "simulate", "assemble"):
        getattr(pipe, f"stage_{stage}")()
    fastg = destination / "work" / "megahit" / "assembly.fastg"
    if not fastg.is_file() or fastg.stat().st_size == 0:
        raise ContractError([f"{spec.name} assembled no MEGAHIT FASTG"])
    k_dir = destination / "work" / "megahit" / "intermediate_contigs"
    names = [path.name for path in k_dir.iterdir()] if k_dir.is_dir() else []
    chosen = highest_megahit_contigs(names)
    if chosen is None:
        raise ContractError([f"{spec.name} has no MEGAHIT k-mer contig graph"])
    k_value = int(Path(chosen).name.split(".", 1)[0][1:])
    return fastg_to_cfa(fastg, k=k_value, graph_id=spec.name)


def _require_parent_fasta(spec: BenchSpec, root: Path) -> None:
    rows = load_pairs(root / "src" / "metametro" / "bench" / "data" / "pins" / spec.name / "accessions.tsv")
    missing = []
    for row in rows:
        path = root / "data" / "raw" / "fasta" / f"{row['accession']}.fna"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(row["accession"])
    if missing:
        shown = ", ".join(missing[:8])
        if len(missing) > 8:
            shown += ", ..."
        raise ContractError([f"{spec.name} reuses FASTA from {spec.parent}; missing {shown}"])


def _external(
    spec: BenchSpec,
    destination: Path,
    colourings: tuple[str, ...] | None,
    *,
    execute: bool,
    gtfs: Path | None,
    requested: str,
) -> BuildResult:
    write_external_contract(spec, destination)
    if not execute:
        digest = contract_identity(destination)
        write_identity(destination / "identity.sha256", digest)
        _finish_manifest(
            spec,
            destination,
            status="contract",
            identity=digest,
            colourings=(),
            requested=requested,
        )
        return BuildResult(
            spec=spec, outdir=destination, identity=digest, colourings=(), status="contract"
        )
    if spec.name.startswith("phage_species_5"):
        require_programs(spec, ("samovar", "megahit", "megahit_toolkit"))
        genomes = repo_root() / "data" / "raw" / "genomes"
        missing = [name for name in ("T1", "T3", "T4", "T5", "T7") if not (genomes / f"{name}.fna").is_file()]
        if missing:
            raise ContractError([f"missing genome FASTA under data/raw/genomes: {', '.join(missing)}"])
        script = repo_root() / "scripts" / "phage_baseline.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--genomes",
                str(genomes),
                "--work",
                str(destination / "work"),
                "--k",
                str(spec.k),
                "--total-reads",
                str(spec.total_reads),
                "--samovar",
                shutil.which("samovar") or "samovar",
                "--megahit",
                shutil.which("megahit") or "megahit",
            ],
            check=False,
        )
        if completed.returncode != 0:
            raise ContractError([f"phage baseline exited {completed.returncode}"])
        from metametro.formats.cfa.io import load_cfa

        coloured = destination / "work" / "coloured_cfa"
        if not coloured.is_dir():
            raise ContractError(["phage baseline did not write coloured_cfa"])
        graph = load_cfa(coloured)
        ctx = BuildContext(selected=colourings)
        made = materialize(graph, destination, ctx)
        _finish_manifest(
            spec,
            destination,
            status="built",
            identity=made.identity,
            colourings=made.colourings,
            requested=requested,
        )
        return BuildResult(
            spec=spec,
            outdir=destination,
            identity=made.identity,
            colourings=made.colourings,
            status="built",
        )
    if spec.name == "spb_ground_transit" and (gtfs is None or not Path(gtfs).is_file()):
        raise ContractError(["spb_ground_transit needs an existing GTFS zip passed as --gtfs"])
    if spec.name == "roxel":
        require_programs(spec, ("Rscript",))
    raise ContractError([f"{spec.name} contract is written; the external graph build did not run"])
