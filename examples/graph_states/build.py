"""Write ToCUMG, a geometric CGT, three state drawings, and GCN inference.

Node features used by the GCN are longitude and latitude only. Coverage, when
the CFA has it, is stored on the tensor for the drawing and is not a model
input. Edge targets are independent 0/1 columns for every colour in the
requested namespaces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from metametro.contracts.multitask import fit_multitask_gcn, infer_multitask_gcn
from metametro.errors import ContractError
from metametro.formats.cdbg.io import dump_cdbg
from metametro.formats.cfa.io import load_cfa
from metametro.formats.cgt.io import dump_cgt
from metametro.states import build_geometric_states
from metametro.viz.states import plot_geometric_states


def export_graph(
    cfa_path: Path,
    out: Path,
    *,
    node_namespace: str,
    edge_namespaces: tuple[str, ...],
    draw_edge_namespace: str | None,
    epochs: int,
    seed: int,
) -> dict:
    """Build the three states, draw them, fit the GCN, and write inference."""
    cfa = load_cfa(cfa_path)
    states = build_geometric_states(cfa, node_namespace=node_namespace, edge_namespaces=edge_namespaces)
    out.mkdir(parents=True, exist_ok=True)
    dump_cdbg(states.cdbg, out / "cdbg")
    dump_cgt(states.cgt, out / "cgt")
    np.save(out / "cgt" / "edge_targets.npy", states.edge_targets)
    plot_geometric_states(
        cfa,
        states,
        out / "states.pdf",
        edge_namespace=draw_edge_namespace,
    )
    model, metrics = fit_multitask_gcn(
        states.cgt,
        states.edge_targets,
        node_class_names=states.node_class_names,
        edge_label_names=states.edge_label_names,
        epochs=epochs,
        seed=seed,
    )
    inference = infer_multitask_gcn(model, states.cgt)
    gcn = out / "gcn"
    gcn.mkdir(parents=True, exist_ok=True)
    _write_nodes(gcn / "node_predictions.tsv", states, inference)
    np.savez(
        gcn / "edge_predictions.npz",
        probability=inference["edge_probability"],
        binary=inference["edge_binary"],
        label_names=np.asarray(states.edge_label_names),
    )
    _write_positive_edges(gcn / "edge_predictions.tsv", states, inference)
    serializable = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in metrics.items()
    }
    (gcn / "metrics.json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    return {
        "unitigs": len(states.cdbg.unitigs),
        "links": len(states.cdbg.links),
        "tensor_nodes": states.cgt.num_nodes,
        "tensor_edges": states.cgt.num_edges,
        "node_classes": states.node_class_names,
        "edge_labels": len(states.edge_label_names),
        "metrics": serializable,
    }


def _write_nodes(path: Path, states, inference) -> None:
    lines = ["dense_id\tsource_id\ttrue_class\tpredicted_class\tprobability"]
    labels = states.cgt.node_labels
    for row, predicted, name, probability in zip(
        states.cgt.mapping,
        inference["node_class"],
        inference["node_class_name"],
        inference["node_probability"],
    ):
        true = states.node_class_names[int(labels[int(row["dense_id"])])]
        lines.append(
            f"{row['dense_id']}\t{row['source_id']}\t{true}\t{name}\t{float(probability[int(predicted)]):.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_positive_edges(path: Path, states, inference) -> None:
    sources = np.repeat(np.arange(states.cgt.num_nodes), np.diff(states.cgt.indptr))
    lines = ["edge_slot\tsource\ttarget\tlabel\ttarget_bit\tpredicted_bit\tprobability"]
    binary = inference["edge_binary"]
    probability = inference["edge_probability"]
    targets = states.edge_targets
    names = states.edge_label_names
    for slot in range(states.cgt.num_edges):
        columns = np.flatnonzero((targets[slot] == 1) | (binary[slot] == 1))
        for column in columns:
            lines.append(
                f"{slot}\t{int(sources[slot])}\t{int(states.cgt.indices[slot])}\t{names[int(column)]}\t"
                f"{int(targets[slot, column])}\t{int(binary[slot, column])}\t{float(probability[slot, column]):.6f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Export one geometric pipeline from a CFA directory."""
    parser = argparse.ArgumentParser(description="Export ToCUMG, a geometric CGT, drawings, and GCN inference.")
    parser.add_argument("--cfa", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--node-namespace", required=True)
    parser.add_argument("--edge-namespaces", required=True, help="Comma-separated colour namespaces.")
    parser.add_argument("--draw-edge-namespace", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        summary = export_graph(
            args.cfa,
            args.out,
            node_namespace=args.node_namespace,
            edge_namespaces=tuple(part for part in args.edge_namespaces.split(",") if part),
            draw_edge_namespace=args.draw_edge_namespace,
            epochs=args.epochs,
            seed=args.seed,
        )
    except (ContractError, OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"unitigs {summary['unitigs']}")
    print(f"links {summary['links']}")
    print(f"tensor {summary['tensor_nodes']} nodes {summary['tensor_edges']} edges")
    print(f"node classes {summary['node_classes']}")
    print(f"edge labels {summary['edge_labels']}")
    for key, value in summary["metrics"].items():
        print(f"{key} {value}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
