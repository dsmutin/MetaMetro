"""Build one benchmark into a contract and, when tools allow, three graphs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from metametro.bench.colourings import BuildContext
from metametro.bench.data.universal.catalog import (
    classifier_pin,
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


def build_all(
    *,
    execute: bool = False,
    gtfs: Path | None = None,
    root: Path | None = None,
) -> list[BuildResult]:
    """Build every registered benchmark.

    In-process graphs are always materialised. Community and external benches
    write their contract. They download and assemble only when ``execute`` is
    true. A directory that already has ``manifest.yaml`` and ``identity.sha256``
    is left as it is.
    """
    from metametro.bench.registry import list_specs

    results: list[BuildResult] = []
    for spec in list_specs():
        destination = default_outdir(spec, root=root)
        if (destination / "manifest.yaml").is_file() and (destination / "identity.sha256").is_file():
            results.append(
                BuildResult(
                    spec=spec,
                    outdir=destination,
                    identity=(destination / "identity.sha256").read_text(encoding="utf-8").split()[0],
                    colourings=(),
                    status="present",
                )
            )
            continue
        results.append(
            build(
                spec.name,
                execute=execute if spec.kind != "inprocess" else False,
                gtfs=gtfs,
                root=root,
            )
        )
    return results


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
    ctx = _community_context(spec, destination, colourings)
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
    _install_classifier_library(spec, destination, root)
    _try_classifier_indexes(destination)
    return fastg_to_cfa(fastg, k=k_value, graph_id=spec.name)


def _install_classifier_library(spec: BenchSpec, destination: Path, root: Path) -> None:
    """Copy the full non-synonymous ``db`` FASTA set used to build classifiers."""
    rows = load_pairs(classifier_pin(spec) / "accessions.tsv")
    library = destination / "work" / "classifier_db"
    library.mkdir(parents=True, exist_ok=True)
    missing = []
    for row in rows:
        if row["role"] != "db":
            continue
        source = root / "data" / "raw" / "fasta" / f"{row['accession']}.fna"
        if not source.is_file() or source.stat().st_size == 0:
            missing.append(row["accession"])
            continue
        target = library / source.name
        if not target.is_file():
            shutil.copyfile(source, target)
    if missing:
        shown = ", ".join(missing[:8])
        if len(missing) > 8:
            shown += ", ..."
        raise ContractError([f"{spec.name} classifier library missing FASTA: {shown}"])


def _try_classifier_indexes(destination: Path) -> None:
    """Build Kraken2 and Kaiju indexes when those tools can run. Skip if they cannot."""
    from metametro.bench.data.universal import ncbi_pipeline as pipe

    kraken = shutil.which("kraken2")
    kaiju = shutil.which("kaiju")
    samovar = shutil.which("samovar")
    if kraken is None and kaiju is None:
        return
    if samovar is None:
        return
    try:
        pipe.stage_databases()
    except SystemExit:
        return


def _community_context(spec: BenchSpec, destination: Path, colourings: tuple[str, ...] | None) -> BuildContext:
    """Point colourings at classifier databases built from the full ``db`` set."""
    work = destination / "work"
    kraken_db = work / "kraken_db"
    kaiju_db = work / "kaiju_db"
    kraken_out = work / "classifier" / "contigs.kraken"
    kaiju_out = work / "classifier" / "contigs.kaiju"
    return BuildContext(
        selected=colourings,
        work_dir=work,
        kraken_db=kraken_db if (kraken_db / "hash.k2d").is_file() else None,
        kaiju_db=kaiju_db if kaiju_db.is_dir() else None,
        kraken_output=kraken_out if kraken_out.is_file() else None,
        kaiju_output=kaiju_out if kaiju_out.is_file() else None,
    )


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
    if spec.name == "roxel":
        require_programs(spec, ("Rscript",))
        from metametro.bench.data.universal.roxel import build_roxel_cfa

        graph = build_roxel_cfa(k=spec.k or 21)
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
    if spec.name == "spb_ground_transit":
        if gtfs is None or not Path(gtfs).is_file():
            raise ContractError(["spb_ground_transit needs an existing GTFS zip passed as --gtfs"])
        script = repo_root() / "examples" / "spb_transit" / "build_cfa.py"
        staged = destination / "cfa_source"
        completed = subprocess.run(
            [sys.executable, str(script), "--gtfs", str(gtfs), "--out", str(staged), "--k", "21"],
            check=False,
        )
        if completed.returncode != 0:
            raise ContractError([f"GTFS CFA build exited {completed.returncode}"])
        from metametro.formats.cfa.io import load_cfa

        graph = load_cfa(staged)
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
    raise ContractError([f"no executor for {spec.name}"])
