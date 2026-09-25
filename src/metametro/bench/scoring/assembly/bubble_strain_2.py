"""Assembly score for the two-taxon bubble.

The debubbler action lives in ``ground_truth/bubbles.tsv``. This module does
not read that action out of the graph.
"""

from __future__ import annotations

from pathlib import Path

from metametro.tables import read_tsv


def expected_action(outdir: Path) -> str:
    """Return the single ground-truth action for ``bubble_strain_2``."""
    _header, rows = read_tsv(outdir / "ground_truth" / "bubbles.tsv")
    if len(rows) != 1:
        raise ValueError(f"bubble_strain_2 ground truth has {len(rows)} rows")
    return rows[0]["expected_action"]
