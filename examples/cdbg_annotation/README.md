# CDBG annotation sidecar

This example stores measurements on a compacted graph and then copies two of them into a coloured graph tensor.

CFA is the semantic source of truth. CDBG (a ToCUMG, totally coloured universal metagenomic graph) is the compacted graph plus an optional columnar annotation sidecar. CGT is the dense float32 matrix used by a model.

Those are not the same object:

- A sidecar layer has one row per annotated target. Targets may be a whole unitig (`node`), one original CFA node (`internal_node`), one CFA edge absorbed into a unitig (`internal_edge`), or one external CDBG link (`edge`). A unitig id is not a biological name.
- Missing targets stay missing. The layer does not invent zeros, and it does not average member values unless `aggregate_annotations` is called with a policy.
- A CGT node matrix has one row for every dense id, in unitig-id order. A CGT edge matrix has one row for every CSR link. Internal unitig edges are not CSR edges, so they are not rows of `X_edge`.
- `cdbg_to_cgt(..., node_annotation=..., edge_annotation=...)` only appends numeric unitig-level and link-level layers. Per-CFA-node rows stay on the CDBG and are joined afterwards with `node_lineage`.

The script checks a known weighted mean. CFA nodes `n1` (length 4, coverage 10) and `n2` (length 6, coverage 20) compact into one unitig. The length-weighted mean is `(10 * 4 + 20 * 6) / (4 + 6) = 16`. The unweighted mean is 15. Compaction does not compute either number.

Run it from the repository root:

```bash
PYTHONPATH=src python examples/cdbg_annotation/run.py
```
