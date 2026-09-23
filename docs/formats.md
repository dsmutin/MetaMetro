# Graph formats

Pinned schema for every format is **1.0**. Draft snippets that used `schema_version: "0.1"` are not accepted. A consumer must reject any other schema version.

Pipeline position:

```text
Initial data → CFA → CDBG → Coloured Graph Tensor → analysis
```

| Format | Role | Human-readable | ML-ready | Mutable |
| --- | --- | --- | --- | --- |
| CFA | Canonical exchange format | Yes | No | Yes |
| CDBG | ToCUMG: totally coloured universal metagenomic graph | Partially | Indirectly | Limited |
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

CDBG is the stored form of a ToCUMG (totally coloured universal metagenomic graph). It stores compacted topology, unitig sequences, sample colours, the CFA mapping, and an optional columnar annotation sidecar. The graph before and after compaction has the same `graph_type`. That type may be `de_bruijn`, `repeat`, `lca`, or any other declared type. A de Bruijn `k` is stored only when the source graph declared it. The sidecar is not a dense feature matrix and it is not the CGT. CFA remains the semantic source of truth. A unitig id is not a biological name.

```text
cdbg/
├── metadata.yaml
├── unitigs.fna
├── unitigs.tsv
├── links.tsv
├── mapping.tsv
├── colors.tsv       # optional, copied from CFA
├── labels.tsv       # optional dictionary only
└── annotations/     # optional; omitted when the graph has no layers
    ├── layers.yaml
    ├── <namespace>__<feature>__<target_type>.tsv       # categories
    ├── <namespace>__<feature>__<target_type>.ids.tsv   # numeric target ids
    └── <namespace>__<feature>__<target_type>.npy       # numeric values
```

Every unitig has `unitig_id`, `sequence`, and `color_set`. For `graph_type: de_bruijn` the sequence is at least `k`. `unitigs.tsv` may carry `internal_overlaps`, and `links.tsv` may carry `overlap`; both are the integer overlap used at that junction. `mapping.tsv` is mandatory:

```text
cfa_node_id    unitig_id    ordinal    length    color_set
```

`length` here is transfer metadata for splitting a unitig back into CFA nodes. It is not a CFA sequence column. Each CFA node belongs to exactly one unitig. `validate_cdbg` requires mapping ordinal `i` to be member `i` of that unitig, and exactly one internal CFA edge id per junction in that order. A unitig that drops those ids does not load.

A Bifrost `graph.gfa` / `graph.color.bfg` / `graph.bfi` bundle is an allowed backend for a later implementation. This repository's minimal backend is the pure-Python compactor in `converters/cfa_to_cdbg.py`. It does not call Bifrost.

### Compaction rule (schema 1.0)

Repeatedly merge the lexicographically first edge `u → v` such that `u` has out-degree 1, `v` has in-degree 1, `u` is not `v`, and the orientation is `++` or omitted. The surviving sequence is `seq(u) + seq(v)[overlap:]`. For `graph_type: de_bruijn` the overlap is `k - 1`. For any other type it is the edge `overlap` column when that column is set, otherwise metadata `overlap`. The overlapped bases must match; a mismatch raises and the edge is kept. Non-forward orientations are not fused. A graph with no overlap contract keeps identity unitigs (one CFA node each) and keeps its `graph_type`. `gfa_to_cfa` loads a Flye-style repeat graph (`S` segments and `L` links) with `graph_type: repeat` and the CIGAR overlap, and does not invent `k`.

Unitig colour is the union of member node colours. Per-node colours remain on the mapping, so annotation transfer does not depend on that union. Every original edge is either an internal edge of one unitig or a link. Nothing is dropped.

The chain fixture compacts as:

```text
u000001 → n000001, n000002
u000002 → n000003, n000004, n000005
u000003 → n000006
```

with 7 CFA edges and 4 links. The bubble fixture does not merge, because every junction has degree other than 1.

`CFA → CDBG → CFA` restores sequences, edge endpoints, edge ids, and colours. Compaction does not copy numeric CFA columns and does not aggregate them. `cdbg_to_cfa` does not write the annotation sidecar back into CFA columns.

### Annotation sidecar

The sidecar is optional. A schema-1.0 directory with no `annotations/` directory loads as a CDBG with an empty layer list. Adding the sidecar does not change schema 1.0: unitig ids, links, sequences, colours, and the mapping mean what they meant before.

Each layer is one table, not a Python object per graph node and not an `N × F` matrix on the core graph. A record has:

```text
target_type = node | edge | internal_node | internal_edge
target_id
namespace
feature
value
dtype
source
```

`node` annotates a whole unitig. `internal_node` annotates one CFA member of that unitig. `internal_edge` annotates one CFA edge absorbed into the unitig. `edge` annotates one external CDBG link. Those four are not interchangeable, and a unitig id is not a stand-in for the CFA node id.

`layers.yaml` records, for every layer, `source`, `method`, `version`, `parameters`, the parent CDBG `graph_id`, and the parent schema and contract versions. A missing field is an error. Numeric values live in a NumPy array aligned to an id table. Categories live in a TSV. Scalar and vector values share this layout; a vector is a 2-d array `(n, width)`, still one array for the layer.

`transfer_annotations` copies selected CFA columns after compaction. Node columns stay keyed by the CFA node id (`internal_node`). An edge column is `internal_edge` when that edge was absorbed, and `edge` when it is still a link. The copy does not average, sum, or otherwise mix member values.

`aggregate_annotations` writes one unitig-level (`node`) value only when the caller names a policy: `mean`, `sum`, `min`, `max`, `median`, `weighted_mean`, `union`, `majority`, or `keep_per_member`. `weighted_mean` uses member sequence lengths from the mapping unless the caller passes weights. `median` of an even count is the mean of the two central values. `union` writes a category whose tokens are sorted and joined with `|`. `majority` raises `ContractError` on a tie. A missing member value raises `ContractError`. It is not filled in. `keep_per_member` records the policy and leaves one row per member. The source layer stays in place when a policy writes a new unitig layer.

The same feature may exist in more than one namespace. Writing the same namespace, feature, and target type twice raises `ContractError` unless the caller passes `replace=True`.

This sidecar is not a CGT feature matrix. A CGT matrix has one float32 row for every dense id and one float32 row for every CSR edge, with no gaps. The sidecar may omit targets, may keep one row per original CFA node, and may store categories. Internal unitig edges are sidecar rows and are not CSR edges, so they are not copied into `X_edge`. `cdbg_to_cgt(..., node_annotation=[(namespace, feature)], edge_annotation=[...])` appends only unitig-level and link-level numeric layers, in dense-id order and CSR order. Per-CFA-node rows stay on the CDBG and are joined through `node_lineage` after tensorization. The existing `node_features`, `edge_features`, and label arguments still build those arrays on their own.

## Coloured Graph Tensor

CGT is the runtime ML object. It has no DNA sequence requirement, no dense `N×N` adjacency, and no per-node Python objects. It is a computational representation, not the biological source of truth and not a fourth semantic graph format. Topology stored on disk is CSR (`indptr`, `indices`). `metadata.topology` stays `csr`.

Nodes are renumbered `0 .. N-1` in unitig-id order. `mapping.tsv` stores `dense_id`, `source_id` (CDBG unitig id), and `cfa_node_ids`. `metametro.identity` reads that row, and it reads each CSR edge slot back to the CDBG link id. The link id is the CFA edge id. Internal unitig edges are not CSR edges. A unitig id is not a biological name. DNA stays on the CDBG unitig. `resolve_sequence(cgt, source_cdbg, dense_id)` reads it through that mapping. A missing unitig, a dense id outside range, or a CGT/CDBG mismatch raises `ContractError`.

```text
X_node  float32  (N, F_v)     features. F_v = 0 is allowed
X_edge  float32  (E, F_e)     features. F_e = 0 is allowed
y_node  int64    (N,)         training labels, optional
y_edge  int64    (E,)         training labels, optional
C_node  uint8    (N, S)       colours
C_edge  uint8    (E, S)       colours
```

Features, training labels, colours, predictions, and confidence are different objects:

| Object | Role |
| --- | --- |
| `X_node` / `X_edge` | Numeric features used by a model. Described by the feature registry. |
| `y_node` / `y_edge` | Integer training labels. They are not columns of `X` and are not concatenated into `X`. |
| `C_node` / `C_edge` | Sample colours. A colour is not a feature and not a label. |
| prediction | A separate object joined by `dense_id`, `source_id`, and CFA ids. It is not written into the CGT arrays. |
| confidence | A field on that prediction. The NumPy and PyG softmax models set it equal to the predicted class probability. That is not a second uncertainty estimator and not an accuracy. |

`indices[j]`, `edge_features[j]`, `edge_labels[j]`, and `edge_colors[j]` are the same directed adjacency entry. Node row `i` matches dense id `i`. Feature names and dtypes are in `metadata.yaml`. Colour columns follow sorted `color_id`. For this minimal schema the colour matrix is dense; a bitset or a sparse matrix is a non-breaking physical choice as long as the node-to-colour-set semantics stay the same.

`node_feature_names` and `edge_feature_names` remain the column order. `node_feature_registry` and `edge_feature_registry` are optional extra metadata for the same columns. A schema-1.0 directory written without those keys still loads. Loading does not invent a registry and does not change array values. Each registry entry records:

```text
name                  # the column name; f0, f1 only when the caller passed an unnamed ndarray
feature_type          # feature
namespace             # sidecar namespace, or empty when the column is not from an annotation layer
source_annotation     # empty, or namespace:feature for the sidecar layer (vector columns share that source)
normalization         # recorded, not applied; none means the values were copied unchanged
dtype                 # float32
positional            # true only for an unnamed ndarray; namespace is then empty
```

An unnamed ndarray is marked positional. Those names are not a biological namespace. A caller-supplied name list as wide as the explicit array names that block, and sidecar names are appended after it. A list as wide as the full matrix, including sidecar columns, is stored as given. Either way the explicit block has an empty namespace. A sidecar column uses `namespace:feature` or `namespace:feature:i` unless the full name list replaced that string, and the registry still records the layer as `source_annotation`.

CSR stays canonical because every feature, label, and colour row is defined on a source-major slot. `csc_from_cgt` derives an incoming-neighbor index (`indptr`, `indices`, and the CSR slot of each incoming edge) from those arrays. It does not write the CGT, does not reorder `edge_features`, and is not required to load a directory. CSC is a view for predecessor walks. It is not a stored topology and not a dense adjacency.

On-disk arrays are NumPy `.npy` files. That choice is not part of the logical contract. Predictions are not files in the CGT directory.

## PyG adapter

`cgt_to_pyg.to_pyg` builds `torch_geometric.data.Data(x, edge_index, edge_attr, y)` from the CGT. `edge_index` has shape `(2, E)` and the same column order as CSR. The CGT remains the canonical object. `edge_index_array` exposes that layout without importing PyTorch Geometric.

## Shared fixtures

`mock_cfa()` is the bubble: 4 nodes, 5 edges, 2 samples, labels `genome_001`, `genome_002`, and `unclassified`, one multi-colour node (`n000002`), and one multi-colour edge. `chain_cfa()` is the compaction mock. `synthetic_genomes()` is the two-genome metagenome mock with a shared 16-mer and private flanks.
