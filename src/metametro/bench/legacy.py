"""Run a pinned community with the same stage flags the old example scripts used."""

from __future__ import annotations

from pathlib import Path

from metametro.bench.data.universal.catalog import pin_dir, write_community_contract
from metametro.bench.paths import default_outdir, repo_root
from metametro.bench.registry import resolve


def main_for(name: str, argv: list[str] | None = None) -> int:
    """Configure the NCBI pipeline for ``name`` and run the requested stage.

    Outputs go to the MetaMetro bench directory. A missing program stops the
    run inside the pipeline. This does not invent reads or a graph.
    """
    from metametro.bench.build import _require_parent_fasta
    from metametro.bench.data.universal import ncbi_pipeline as pipe

    spec = resolve(name)
    if spec.kind != "community":
        raise SystemExit(f"{name} is not a community benchmark")
    destination = default_outdir(spec)
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "contract" / "accessions.tsv").is_file():
        write_community_contract(spec, destination)
    root = repo_root()
    pipe.configure(
        example=pin_dir(spec.name),
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
    pipe.main(argv)
    return 0
