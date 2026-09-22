# Graph formats

Pinned schema for every format is **1.0**. Draft snippets that used `schema_version: "0.1"` are not accepted. A consumer must reject any other schema version.

Pipeline position:

```text
Initial data → CFA → CDBG → Coloured Graph Tensor → analysis
```

| Format | Role | Human-readable | ML-ready | Mutable |
| --- | --- | --- | --- | --- |
| CFA | Canonical exchange format | Yes | No | Yes |
| CDBG | Compact colored de Bruijn graph | Partially | Indirectly | Limited |
| CGT | Computation / ML representation | No | Yes | Yes |

Physical files may gain optional columns and optional metadata. The changes in [contracts.md](contracts.md) that alter identifier, sequence, edge, colour, feature-order, dtype, topology, or mandatory-field semantics are breaking and require `schema_version: "2.0"` plus a migration.

## CFA

CFA is the semantic source of truth. It does not depend on Bifrost, PyTorch, or any graph engine, and it does not store an ML memory layout.

Directory:

```text
cfa/
├── metadata.yaml    # required
├── nodes.fna        # required
├── nodes.tsv        # required
├── edges.tsv        # required
├── colors.tsv       # optional
└── labels.tsv       # optional
```

`metadata.yaml` requires `schema_version`, `graph_id`, and `graph_type`. A `de_bruijn` graph also requires integer `k > 0`.

```yaml
schema_version: "1.0"
graph_id: "mock_bubble"
graph_type: "de_bruijn"
k: 3
features:
  node:
    coverage: float
    gc: float
    color_set: color_set
    label: int
  edge:
    orientation: orientation
    coverage: float
    color_set: color_set
```

Declared feature types are `float`, `int`, `str`, `color_set`, and `orientation` (`++`, `+-`, `-+`, `--`). Extra columns without a declared type are invalid. `length` and `sequence_length` are rejected: the sequence length is `len(sequence)`.

`nodes.fna` holds one unique `node_id` per record. The alphabet is `ACGTN` after ASCII uppercasing. `nodes.tsv` starts with `node_id`. `edges.tsv` starts with `edge_id`, `source`, `target`. Sources and targets must name existing nodes.

`colors.tsv` uses integer `color_id` plus `namespace` and `value` (`name` is accepted as an alias of `value`). A `color_set` cell is a comma-separated list of those ids, for example `0,1`. `labels.tsv` is `label_id`, `namespace`, `value`. Taxonomy, functional annotation, and sample colour stay in different namespaces. Sample colour is not a taxonomic label.

Invariants checked by `validate_cfa`: unique node and edge ids, no dangling edges, every colour and label id defined, sequences in the alphabet, schema version `1.0`.

## CDBG

CDBG stores compacted topology, unitig sequences, sample colours, and the CFA mapping. It is not an annotation database and it does not store dense feature matrices. Unitig ids are not permanent biological ids.

```text
cdbg/
├── metadata.yaml
├── unitigs.fna
├── unitigs.tsv
├── links.tsv
├── mapping.tsv
├── colors.tsv       # optional, copied from CFA
└── labels.tsv       # optional dictionary only
```

Every unitig has `unitig_id`, `sequence` (length at least `k`), and `color_set`. `mapping.tsv` is mandatory:

```text
cfa_node_id    unitig_id    ordinal    length    color_set
```

`length` here is transfer metadata for splitting a unitig back into CFA nodes. It is not a CFA sequence column. Each CFA node belongs to exactly one unitig.

A Bifrost `graph.gfa` / `graph.color.bfg` / `graph.bfi` bundle is an allowed backend for a later implementation. This repository's minimal backend is the pure-Python compactor in `converters/cfa_to_cdbg.py`. It does not call Bifrost.

### Compaction rule (schema 1.0)

For `graph_type: de_bruijn`, repeatedly merge the lexicographically first edge `u → v` such that `u` has out-degree 1, `v` has in-degree 1, `u` is not `v`, and the orientation is `++` or omitted. The surviving sequence is `seq(u) + seq(v)[k-1:]`. The `(k-1)` overlap must match; a mismatch raises and the edge is kept. Non-forward orientations are not fused. Graphs that are not de Bruijn graphs use identity unitigs (one CFA node each).

Unitig colour is the union of member node colours. Per-node colours remain on the mapping, so annotation transfer does not depend on that union. Every original edge is either an internal edge of one unitig or a link. Nothing is dropped.

The chain fixture compacts as:

```text
u000001 → n000001, n000002
u000002 → n000003, n000004, n000005
u000003 → n000006
```

with 7 CFA edges and 4 links. The bubble fixture does not merge, because every junction has degree other than 1.

`CFA → CDBG → CFA` restores sequences, edge endpoints, edge ids, and colours. Numeric feature columns are not part of CDBG and are not restored; join them from the original CFA through the mapping.

## Coloured Graph Tensor

CGT is the runtime ML object. It has no DNA sequence requirement, no dense `N×N` adjacency, and no per-node Python objects. Topology is CSR (`indptr`, `indices`). Optional CSC is not part of schema 1.0.

Nodes are renumbered `0 .. N-1` in unitig-id order. `mapping.tsv` stores `dense_id`, `source_id` (CDBG unitig id), and `cfa_node_ids`.

```text
X_node  float32  (N, F_v)     F_v = 0 is allowed
X_edge  float32  (E, F_e)     F_e = 0 is allowed
y_node  int64    (N,)         optional
y_edge  int64    (E,)         optional
C_node  uint8    (N, S)
C_edge  uint8    (E, S)
```

`indices[j]`, `edge_features[j]`, `edge_labels[j]`, and `edge_colors[j]` are the same directed adjacency entry. Node row `i` matches dense id `i`. Feature names and dtypes are in `metadata.yaml`. Colour columns follow sorted `color_id`. For this minimal schema the colour matrix is dense; a bitset or a sparse matrix is a non-breaking physical choice as long as the node-to-colour-set semantics stay the same.

On-disk arrays are NumPy `.npy` files. That choice is not part of the logical contract.

## PyG adapter

`cgt_to_pyg.to_pyg` builds `torch_geometric.data.Data(x, edge_index, edge_attr, y)` from the CGT. `edge_index` has shape `(2, E)` and the same column order as CSR. The CGT remains the canonical object. `edge_index_array` exposes that layout without importing PyTorch Geometric.

## Shared fixtures

`mock_cfa()` is the bubble: 4 nodes, 5 edges, 2 samples, labels `genome_001`, `genome_002`, and `unclassified`, one multi-colour node (`n000002`), and one multi-colour edge. `chain_cfa()` is the compaction mock. `synthetic_genomes()` is the two-genome metagenome mock with a shared 16-mer and private flanks.
