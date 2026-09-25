"""Mandatory checks for the AMBER binning scorer. The binary is not required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metametro.bench.scoring.assembly_binning.universal.amber import (
    AMBER_COMMIT,
    AMBER_REPO,
    AMBER_TAG,
    amber_executable,
    read_amber_results,
    write_binning_tsv,
    write_gold_tsv,
)
from metametro.errors import ContractError

pytestmark = pytest.mark.mandatory


def test_amber_pin_is_the_cami_repository() -> None:
    assert AMBER_REPO == "https://github.com/CAMI-challenge/AMBER"
    assert AMBER_TAG == "v2.0.17-beta"
    assert AMBER_COMMIT == "089ea20e83811e090ab69814e033adc9fc963892"


def test_cami_files_keep_sequence_order(tmp_path: Path) -> None:
    bins = tmp_path / "bins.tsv"
    gold = tmp_path / "gold.tsv"
    write_binning_tsv(bins, ["c1", "c2"], np.array([0, 1]))
    write_gold_tsv(gold, {"c1": ("g1", 10), "c2": ("g2", 20)}, ["c1", "c2"])
    assert "c1\t0" in bins.read_text(encoding="utf-8")
    assert gold.read_text(encoding="utf-8").strip().splitlines()[-1] == "c2\tg2\t20"


def test_missing_amber_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AMBER_SRC", raising=False)
    monkeypatch.setattr(
        "metametro.bench.scoring.assembly_binning.universal.amber.shutil.which",
        lambda _name: None,
    )
    with pytest.raises(ContractError, match="amber.py"):
        amber_executable()


def test_read_results_uses_the_tool_row(tmp_path: Path) -> None:
    path = tmp_path / "results.tsv"
    path.write_text(
        "Tool\tf1_score_seq\tprecision_avg_seq\trecall_avg_seq\n"
        "gold standard\t1\t1\t1\n"
        "metametro\t0.5\t0.4\t0.6\n",
        encoding="utf-8",
    )
    scores = read_amber_results(path)
    assert scores == {"amber_f1": 0.5, "amber_ap": 0.4, "amber_ar": 0.6}
