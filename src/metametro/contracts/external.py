"""External baselines: Samovar ISS (Contract 1) and MEGAHIT (Contract 2)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def samovar_generate_command(
    genome_dir: str | Path,
    output_dir: str | Path,
    *,
    n_samples: int = 2,
    total_reads: int = 200,
    host_fraction: float = 0,
    seed: int = 1,
    samovar: str = "samovar",
) -> list[str]:
    """Argv for ``samovar generate``, which drives InSilicoSeq."""
    return [
        samovar,
        "generate",
        "--genome_dir",
        str(genome_dir),
        "--output_dir",
        str(output_dir),
        "--n_samples",
        str(n_samples),
        "--total_reads",
        str(total_reads),
        "--host_fraction",
        str(host_fraction),
        "--seed",
        str(seed),
    ]


def megahit_command(
    reads: str | Path,
    output_dir: str | Path,
    *,
    k: int = 21,
    threads: int = 4,
    megahit: str = "megahit",
    min_count: int = 2,
) -> list[str]:
    """Argv for a single-k MEGAHIT assembly. ``k`` must be odd and at least 15."""
    if k < 15 or k % 2 == 0:
        raise ValueError("MEGAHIT k must be an odd integer >= 15")
    return [
        megahit,
        "-r",
        str(reads),
        "-o",
        str(output_dir),
        "--k-min",
        str(k),
        "--k-max",
        str(k),
        "--k-step",
        "2",
        "--min-count",
        str(min_count),
        "--keep-tmp-files",
        "-t",
        str(threads),
    ]


def contig2fastg_command(
    contigs: str | Path,
    k: int,
    *,
    toolkit: str = "megahit_toolkit",
) -> list[str]:
    """Argv for ``megahit_toolkit contig2fastg``. FASTG is written to stdout."""
    if k < 15 or k % 2 == 0:
        raise ValueError("MEGAHIT k must be an odd integer >= 15")
    return [toolkit, "contig2fastg", str(k), str(contigs)]


def run_samovar_generate(
    genome_dir: str | Path,
    output_dir: str | Path,
    *,
    n_samples: int = 2,
    total_reads: int = 200,
    host_fraction: float = 0,
    seed: int = 1,
    samovar: str = "samovar",
) -> None:
    """Write the Samovar ISS config and execute the generated Snakemake script.

    ``samovar generate`` only writes ``.generate/generate.sh``. The reads
    appear after that script runs.
    """
    run_command(
        samovar_generate_command(
            genome_dir,
            output_dir,
            n_samples=n_samples,
            total_reads=total_reads,
            host_fraction=host_fraction,
            seed=seed,
            samovar=samovar,
        )
    )
    script = Path(output_dir) / ".generate" / "generate.sh"
    if not script.is_file():
        raise FileNotFoundError(script)
    run_command(["bash", str(script), "--directory", str(output_dir)])


def run_command(argv: list[str]) -> None:
    """Run a baseline command. Fail if the executable is missing or it exits non-zero."""
    if shutil.which(argv[0]) is None and not Path(argv[0]).is_file():
        raise FileNotFoundError(argv[0])
    subprocess.run(argv, check=True)


def run_command_to_file(argv: list[str], destination: str | Path) -> None:
    """Run a baseline command and write its stdout to ``destination``."""
    if shutil.which(argv[0]) is None and not Path(argv[0]).is_file():
        raise FileNotFoundError(argv[0])
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            subprocess.run(argv, check=True, stdout=handle)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(path)
