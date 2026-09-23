#!/usr/bin/env python3
"""Show the CGT feature registry, a node prediction, and the derived CSC index.

The script checks structural joins. It does not report accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metametro.contracts.ds import run_ds  # noqa: E402
from metametro.fixtures import mock_cdbg, mock_cgt  # noqa: E402
from metametro.formats.cgt.predictions import predictions_from_ds  # noqa: E402
from metametro.formats.cgt.topology import csc_from_cgt  # noqa: E402
from metametro.identity import resolve_sequence  # noqa: E402


def run() -> int:
    """Check registry, prediction join keys, sequence lookup, and CSC."""
    cgt = mock_cgt()
    cdbg = mock_cdbg()
    registry = cgt.metadata["node_feature_registry"]
    if any(row["feature_type"] != "feature" for row in registry):
        print("feature registry contains a non-feature column", file=sys.stderr)
        return 1
    if cgt.node_labels is None or cgt.node_features.shape[1] != len(registry):
        print("labels and features are not separate", file=sys.stderr)
        return 1
    result = run_ds(cgt, epochs=1, seed=0)
    predictions = predictions_from_ds(result)
    if len(predictions) != cgt.num_nodes:
        print("prediction count does not match nodes", file=sys.stderr)
        return 1
    if any(item.confidence != item.probability for item in predictions):
        print("softmax confidence was not the class probability", file=sys.stderr)
        return 1
    sequence = resolve_sequence(cgt, cdbg, 0)
    unitig_id = cgt.mapping[0]["source_id"]
    expected = next(unitig.sequence for unitig in cdbg.unitigs if unitig.unitig_id == unitig_id)
    if sequence != expected:
        print("resolved sequence does not match the unitig", file=sys.stderr)
        return 1
    features = cgt.edge_features
    csc = csc_from_cgt(cgt)
    if cgt.edge_features is not features or int(csc.indptr[-1]) != cgt.num_edges:
        print("CSC reordered edge features or dropped an edge", file=sys.stderr)
        return 1
    print(
        {
            "status": "ok",
            "nodes": cgt.num_nodes,
            "feature_columns": [row["name"] for row in registry],
            "predictions": len(predictions),
            "sequence_length": len(sequence),
            "csc_edges": int(csc.indptr[-1]),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
