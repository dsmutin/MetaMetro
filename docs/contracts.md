# Pipeline contracts

Each stage is a contract. The implementation underneath it is not. Contract versions are independent of the package `VERSION` file and are pinned in `metametro.contracts.CONTRACTS`.

| Contract | Version | Responsibility |
| --- | --- | --- |
| 1 genome → metagenome | 1.0 | Biological simulation |
| 2 metagenome → graph | 1.0 | Graph construction |
| 3 graph → CFA | 1.0 | Canonical representation |
| 4 CFA colouring | 1.0 | Sample annotation on CFA |
| 5 CFA → CDBG | 1.0 | Compaction |
| 6 CDBG → CGT | 1.0 | Tensorization |
| 7 DS on CGT | 1.0 | Graph analysis |

Colouring is a CFA operation. Other measurements may be copied onto the CDBG annotation sidecar after compaction; that copy is not part of Contract 5 and does not change topology. CDBG owns compacted topology. CGT owns the dense numeric arrays used at runtime. The sidecar is not a fourth graph format and it is not a CGT matrix. Downstream analysis still sees the CGT API.

Identity that must survive:

```text
genome → read → CFA node → CDBG unitig → CGT dense id → DS result
CFA edge → CDBG link or internal edge id → CGT CSR slot (links only)
```

`metametro.identity` reads those ids from the objects above. It does not add a fourth graph format. A CDBG link id is the CFA edge id. A CGT CSR slot is that link id after the same sort used for edge features. Edges absorbed into a unitig stay on the unitig and are not CSR edges.

## Contract 1 — genome → metagenome

Input: a directory of nucleotide FASTA files. The FASTA record id is `genome_id` and must be unique. Taxonomy and strain metadata are optional.

Output:

```text
reads.fastq
metadata.tsv          # read_id, sample_id, genome_id, ambiguous
ground_truth/read_to_genome.tsv
```

`ambiguous=yes` is required when a read cannot be tied to one genome. The in-process mock never emits ambiguous reads: each read is sliced from one genome. The external baseline is `samovar generate` (InSilicoSeq). See `samovar_generate_command`.

The mock uses two genomes that share `ACGTACGTACGTACGT` and differ in both flanks, with more than one read count per sample.

Breaking: a read format that drops `genome_id`, or ground truth that cannot be recovered. Adding metadata columns is non-breaking.

## Contract 2 — metagenome → graph

Input: FASTQ, or an existing assembly the backend can read. Output is a directed graph whose vertices have `node_id` and `sequence`, and whose edges have `edge_id`, `source`, and `target`. De Bruijn graphs publish `k`. Coverage is optional; when present its unit must be stated.

The in-process builder `dbg_from_sequences` emits one node per observed k-mer and one edge per successive pair inside a sequence. Node ids are assigned in sorted k-mer order, so the same sequences produce the same ids.

The external baseline runs MEGAHIT. `contigs_to_dbg` turns `final.contigs.fa` into a k-mer de Bruijn graph. `contig_overlap_graph` keeps each contig as one node and adds an edge only when the sequences overlap by exactly `k - 1`. The T-phage script does not use those two builders for the baseline graph. It reads `intermediate_contigs/k{k}.contigs.fa` through `megahit_toolkit contig2fastg`. That FASTG still has bubbles and branch/join edges; `final.contigs.fa` is the linear result after those are resolved. `fastg_to_cfa` stores each contig once. A trailing `'` is the reverse strand and is stored as edge orientation `++`, `+-`, `-+`, or `--`. MEGAHIT edges overlap by the assembly `k`, so the CFA de Bruijn parameter is `k + 1` and compaction checks that same overlap. `megahit_command` passes `--keep-tmp-files`.

Breaking: losing sequences, connectivity, or unique ids. Adding coverage or swapping the assembler while preserving those semantics is non-breaking.

## Contract 3 — graph → CFA

Input: any graph that already satisfies Contract 2. Output: the minimal CFA directory (`metadata.yaml`, `nodes.fna`, `nodes.tsv`, `edges.tsv`). CFA ids are the semantic ids. A backend id that differs from the CFA id must be stored beside it; the minimal de Bruijn builder uses the CFA id as the only id, so no extra map is written.

Breaking: changing what `node_id` means, dropping sequence, or making the topology unrecoverable. New optional columns are non-breaking.

## Contract 4 — colouring

Colouring is an operation on a CFA, not a new graph type. It does not change node ids, edge ids, sequences, or endpoints.

A colour is a set of integer ids. Nodes and edges are coloured separately. The call states the operation: `replace`, `merge`, `intersect`, or `subtract`. There is no implicit overwrite.

Minimal read-colouring rule used here:

- Vertex: sample `S` colours node `V` when the number of reads from `S` that cover `V` is at least `min_vertex_depth` (default 1). Covering means the node sequence occurs in the read, or, when the read is shorter than the node, the read occurs in the node. The reverse complement of the read counts as the same read, once.
- Edge: sample `S` colours edge `E` when the junction `(k+1)`-mer `source[-k:] + target[k-1]`, or that junction's reverse complement, occurs at least `min_edge_kmer_density` times in reads from `S` (default 2). Occurrences are counted with overlap. A palindromic junction is not counted twice. When the edge has an orientation, a `-` endpoint is reverse-complemented before that slice. A missing orientation is `++`. A sample with zero observations stays uncoloured; depth is not imputed.

The colour dictionary namespace for this rule is `sample`. Taxonomic labels stay in `labels.tsv` under namespace `genome` and are not stored as sample colours.

The bubble mock has a node coloured `{0}`, a node coloured `{0,1}`, and a node coloured `{1}`, plus an edge coloured `{0,1}`.

Breaking: changing colour-set semantics, or a colouring pass that edits topology. New namespaces are non-breaking.

## Contract 5 — CFA → CDBG

Input: CFA sequences, topology, and colours when they exist. An uncoloured CFA may produce an uncoloured CDBG. `graph_type` may be `de_bruijn` or any other declared type (`repeat`, `lca`, and so on).

Output: a ToCUMG (totally coloured universal metagenomic graph), stored as CDBG: unitigs, links, colours, and `cfa_node_id → unitig_id` with path order. Compaction does not change `graph_type` and does not add `k` unless the CFA already had it. The compaction rule and the chain mock (6 nodes, 7 edges, 3 unitigs, 4 links) are specified in [formats.md](formats.md).

Preserved: nucleotide content (under the overlap rule), connectivity (internal edge or link), colours (per node on the mapping; union on the unitig), provenance through CFA ids. Changed: number of nodes, unitig ids, and the topology encoding.

Compaction does not copy CFA feature columns and does not aggregate them. After compaction, `transfer_annotations` may store selected columns on the optional sidecar: a CFA node value stays an `internal_node` row keyed by that CFA id, and a CFA edge value stays an `internal_edge` row or an `edge` row (a CDBG link). `aggregate_annotations` writes a unitig-level row only for an explicit policy (`mean`, `sum`, `min`, `max`, `median`, `weighted_mean`, `union`, `majority`, `keep_per_member`). `weighted_mean` uses member sequence lengths unless the caller passes weights. A missing value raises `ContractError` and is not imputed. Every layer records source, method, version, parameters, parent graph id, and parent schema and contract versions. The sidecar is optional files, so schema 1.0 and this contract stay 1.0.

Breaking: dropping the mapping, sequences, or colours, or changing the compaction rule without a schema bump. Another backend is non-breaking. Optional annotation files are non-breaking.

## Contract 6 — CDBG → CGT

Input: a CDBG plus optional feature and label arrays. Arrays follow unitig-id order for nodes and `cdbg.links` order for edges; conversion then permutes edges into CSR order. A dict keyed by unitig id or link id is also accepted. `F_v = 0` and `F_e = 0` are valid. `node_annotation` and `edge_annotation` are optional `(namespace, feature)` pairs. Node pairs must already be unitig-level sidecar rows. Edge pairs must already be link-level sidecar rows. Those columns are appended after the explicit feature columns, in dense-id order and in CSR order. An internal-edge layer is not written into `X_edge`. A per-CFA-node layer is not silently aggregated into `X_node`; join it with `node_lineage`. A missing selected target raises `ContractError`.

Output: the CGT in [formats.md](formats.md). Construction is one pass over unitigs, links, and feature rows. It does not look up a graph object per feature assignment. The CGT is still not a semantic graph format.

Labels are a separate integer layer. They are not concatenated into `X_node` or `X_edge`. Colours stay on `C_node` / `C_edge`. The feature registry records each `X` column's name, type `feature`, namespace, source annotation, normalization, and dtype. `node_feature_names` and `edge_feature_names` remain. A missing registry on an older directory is valid and does not change the arrays. Normalization is stored as `none` unless a caller recorded another string; the converter does not rescale values. An unnamed ndarray is marked positional with an empty namespace. A sidecar column keeps its `namespace:feature` name and records that layer as the source annotation.

CSR remains the canonical topology. `csc_from_cgt` may derive incoming adjacency for predecessor walks. That index is not stored, is not required on load, and does not reorder `edge_features`. `resolve_sequence` reads a unitig sequence from the source CDBG. The CGT row does not gain a DNA column. `cgt_from_csr` is a separate entry for an external CSR graph. It is not Contract 6: it does not read a CDBG, and it does not place colours in `X`. Optional colour weights are float32 scores on the existing colour columns. Schema 1.0 directories without those files still load.

Breaking: changing dense-id order, CSR meaning, feature order, or dtypes. A different physical container (Arrow, memmap, torch) is non-breaking if the logical arrays match. Optional registry metadata and a derived CSC index are non-breaking.

## Contract 7 — analysis on CGT

Input: a CGT, plus training settings. This implementation trains a minimal graph convolution whose target is the integer genome label on each node (`0` = unclassified or shared).

The NumPy model aggregates each node with itself and its outgoing neighbors, then applies a linear classifier with softmax cross-entropy. The seed fixes the initial weights. `backend="pyg"` trains a two-layer `GCNConv` network when PyTorch Geometric is installed. The PyG object is an adapter; training still reads and returns CGT ids.

Output:

```text
result[]: dense_id, source_id, cfa_node_ids, predicted_label, probability,
          predicted_class, confidence, model_id, model_version
mapping
metadata: contract, backend, model_id, model_version, confidence, seed, epochs
```

`predicted_label` and `probability` stay. `predicted_class` is the same integer as `predicted_label`. `confidence` is that class probability. The current models return only a softmax distribution, so confidence is not a separate uncertainty estimate and not an accuracy. `predictions_from_ds` builds one node `GraphPrediction` per result row. Those objects are joinable by `dense_id`, `source_id`, and CFA node ids. They are not written into the CGT. `run_ds` does not emit edge predictions. `edge_prediction` can represent one edge when the caller supplies the class, the probability, the confidence, and a CDBG. It stores the CSR slot and the CFA edge id and does not modify the CGT.

The function copies arrays before training and raises if the input CGT changed. A model that appends embeddings must return a new CGT or a result table. It must not write into the caller's arrays.

The mock check is: prediction count equals node count, and each prediction still carries the original CFA ids. Accuracy on the four-node bubble is not a performance claim.

Breaking: results that cannot be joined back to `dense_id` or CFA id, or removing `predicted_label` or `probability`. Adding prediction fields is non-breaking. A different model is non-breaking.

## Edit proposals

An edit proposal is not one of the seven contracts and not a graph format. `validate_edit_proposal` checks a proposal and returns nothing. `apply_edit_proposal` returns a new CDBG when the proposal is valid. The input is not modified. A check that would make apply fail is also a validation error. New unitig ids record their parent unitig ids and are not biological names. CFA node ids on the surviving members stay the original ids. Schema 1.0 and the contract versions stay 1.0 because `edit_provenance` is optional metadata. A directory that lacks it still loads.

## Breaking-change rule

Non-breaking additions: optional metadata, optional features, annotation namespaces, and new backends that honour the same contract version.

Breaking changes bump that contract's schema to `2.0` and ship a conversion. Package `VERSION` still moves on its own (feature → minor, fix → patch, release → major) and is not a substitute for `schema_version`.
