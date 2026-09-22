#!/usr/bin/env python3
"""Train the NumPy GCN on the bubble fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metametro.contracts.ds import run_ds  # noqa: E402
from metametro.fixtures import mock_cgt  # noqa: E402


def run() -> int:
    """Run Contract 7 on the shared four-node graph."""
    result = run_ds(mock_cgt(), epochs=5, seed=0)
    if result["metadata"]["num_predictions"] != 4:
        print("unexpected prediction count", file=sys.stderr)
        return 1
    payload = {
        "status": "ok",
        "ok": True,
        "contract": result["metadata"]["contract"],
        "num_predictions": result["metadata"]["num_predictions"],
        "dense_ids": [row["dense_id"] for row in result["result"]],
        "cfa_node_ids": [row["cfa_node_ids"] for row in result["result"]],
    }
    out = Path(__file__).resolve().parent / "data" / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
