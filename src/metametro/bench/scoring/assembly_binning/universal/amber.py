"""Genome-binning scores from CAMI AMBER.

``amber.py`` comes from https://github.com/CAMI-challenge/AMBER, tag
``v2.0.17-beta`` (commit ``089ea20e83811e090ab69814e033adc9fc963892``).
This module does not reimplement purity or completeness. It writes the CAMI
files and reads ``f1_score_seq`` from AMBER's ``results.tsv``.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from metametro.errors import ContractError

AMBER_REPO = "https://github.com/CAMI-challenge/AMBER"
AMBER_TAG = "v2.0.17-beta"
AMBER_COMMIT = "089ea20e83811e090ab69814e033adc9fc963892"
_HEADER = "@Version:0.9.0\n@SampleID:SAMPLEID\n"


def amber_executable() -> Path:
    """Return ``amber.py`` from ``AMBER_SRC`` or from ``PATH``.

    ``AMBER_SRC`` is a checkout of ``AMBER_REPO`` at ``AMBER_TAG``. A missing
    script is an error. A substitute scorer is not used.
    """
    source = os.environ.get("AMBER_SRC")
    if source:
        path = Path(source) / "amber.py"
        if path.is_file():
            return path
        raise ContractError(
            [
                f"AMBER_SRC={source} has no amber.py. "
                f"Clone {AMBER_REPO} at {AMBER_TAG} ({AMBER_COMMIT})."
            ]
        )
    found = shutil.which("amber.py")
    if found:
        return Path(found)
    raise ContractError(
        [
            "amber.py is not on PATH and AMBER_SRC is unset. "
            f"Clone {AMBER_REPO} at {AMBER_TAG} ({AMBER_COMMIT}) and set AMBER_SRC."
        ]
    )


def amber_command(gold: Path, bins: Path, output: Path, *, label: str = "metametro") -> list[str]:
    """Argv for one AMBER genome-binning run."""
    return [
        str(amber_executable()),
        "-g",
        str(gold),
        "-l",
        label,
        "-o",
        str(output),
        "--silent",
        str(bins),
    ]


def write_binning_tsv(path: Path, sequence_ids: list[str], bin_ids: np.ndarray) -> None:
    """Write a CAMI binning file: one predicted bin id per sequence."""
    if len(sequence_ids) != len(bin_ids):
        raise ContractError([f"{len(sequence_ids)} sequence ids and {len(bin_ids)} bin ids"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_HEADER, "@@SEQUENCEID\tBINID\n"]
    for sequence_id, bin_id in zip(sequence_ids, bin_ids):
        lines.append(f"{sequence_id}\t{int(bin_id)}\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_gold_tsv(path: Path, gold: dict[str, tuple[str, int]], sequence_ids: list[str]) -> None:
    """Write the CAMI gold file for ``sequence_ids``, in that order."""
    missing = [sequence_id for sequence_id in sequence_ids if sequence_id not in gold]
    if missing:
        sample = ", ".join(missing[:5])
        raise ContractError([f"{len(missing)} sequences are absent from the gold standard, including {sample}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_HEADER, "@@SEQUENCEID\tBINID\tLENGTH\n"]
    for sequence_id in sequence_ids:
        genome, length = gold[sequence_id]
        if length <= 0:
            raise ContractError([f"{sequence_id} has length {length}"])
        lines.append(f"{sequence_id}\t{genome}\t{length}\n")
    path.write_text("".join(lines), encoding="utf-8")


def score_bins(
    sequence_ids: list[str],
    bin_ids: np.ndarray,
    gold: dict[str, tuple[str, int]],
    work: Path,
    *,
    label: str = "metametro",
) -> dict[str, float]:
    """Run ``amber.py`` and return sequence-level precision, recall, and F1."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    bins = work / "bins.tsv"
    gold_path = work / "gold_standard_genome.tsv"
    write_binning_tsv(bins, sequence_ids, bin_ids)
    write_gold_tsv(gold_path, gold, sequence_ids)
    out = work / "amber"
    out.mkdir(parents=True, exist_ok=True)
    argv = amber_command(gold_path, bins, out, label=label)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    done = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    results = out / "results.tsv"
    if not results.is_file():
        detail = (done.stderr or done.stdout or "").strip()
        raise ContractError([f"amber.py wrote no results.tsv (exit {done.returncode}). {detail}"])
    return read_amber_results(results)


def read_amber_results(path: Path) -> dict[str, float]:
    """Read ``f1_score_seq`` and the matching precision and recall."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    tools = [row for row in rows if "gold" not in str(row.get("Tool", "")).lower()]
    if len(tools) != 1:
        raise ContractError([f"{path} has {len(tools)} tool rows"])
    row = tools[0]
    required = ("f1_score_seq", "precision_avg_seq", "recall_avg_seq")
    missing = [name for name in required if name not in row]
    if missing:
        raise ContractError([f"{path} is missing {', '.join(missing)}"])
    return {
        "amber_f1": float(row["f1_score_seq"]),
        "amber_ap": float(row["precision_avg_seq"]),
        "amber_ar": float(row["recall_avg_seq"]),
    }
