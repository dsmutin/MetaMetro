# Representation audit

Audit of the schema-1.0 path CFA → CDBG (ToCUMG) → CGT. CFA is the semantic source of truth. CDBG is the compacted graph. CGT is the ML/runtime tensor. Package version was `0.7.1` before this change and is `0.8.0` after the identity helpers. No schema or contract version changed.

## 1. CFA loading, dumping, and validation

- **Current invariant.** A CFA directory has `metadata.yaml`, `nodes.fna`, `nodes.tsv`, and `edges.tsv`. Schema version is `1.0`. Node and edge ids are unique. Edges name existing nodes. Sequences are `ACGTN`. Declared feature columns match their types. `length` and `sequence_length` are not stored. Colour and label ids exist in their dictionaries.
- **Current implementation.** `load_cfa` / `dump_cfa` / `validate_cfa` in `formats/cfa`. Dump validates before writing. Round trip of the bubble is tested.
- **Identified gap.** None for identity. Numeric columns stay on the CFA and are not copied into CDBG.
- **Classification.** No gap.
- **Proposed fix.** None.
- **Breaking.** No.

## 2. CFA → CDBG compaction

- **Current invariant.** Merge a forward edge `u → v` when `u` has out-degree 1, `v` has in-degree 1, and `u` is not `v`. Overlap is `k - 1` for `de_bruijn`, otherwise the edge or metadata overlap. A mismatch is an error. `graph_type` is unchanged. Every CFA edge becomes one internal edge or one link. The link id and the internal edge id are the CFA edge id. Unitig ids are `u000001` upward in member-tuple order.
- **Current implementation.** `cfa_to_cdbg` records `internal_edge_ids` in path order, including when the downstream pair is merged first (`test_downstream_merge_keeps_internal_edges`). The chain compacts to the three unitigs in `docs/formats.md`.
- **Identified gap.** None in the compactor. The on-disk synthetic CDBG was stale; see section 10.
- **Classification.** No gap in the live compactor.
- **Proposed fix.** None in the compactor.
- **Breaking.** No.

## 3. CDBG mapping

- **Current invariant.** Each CFA node is on exactly one unitig. Mapping ordinal `i` is member `i` of that unitig. `length` is the CFA sequence length used to split the unitig.
- **Current implementation.** The compactor writes one `NodeMap` per member with `ordinal` equal to the member index. `validate_cdbg` used to compare the set of mapped ids with unitig membership and collapsed duplicate rows in a dict. It did not check ordinals. A swapped ordinal still loaded, and `cdbg_to_cfa` then assigned the wrong sequence and the wrong edge endpoints to those CFA node ids.
- **Identified gap.** Ordinals and duplicate mapping rows were not rejected.
- **Classification.** Bug.
- **Proposed fix.** Reject a duplicate `cfa_node_id`, an ordinal that is not the member index, and a length list that does not tile the unitig sequence. Done in `validate_cdbg`.
- **Breaking.** Breaking for a CDBG whose mapping ordinals do not follow member order, or that repeats a CFA node. Schema-1.0 files written by the compactor already satisfy the check. Schema version stays `1.0`.

## 4. CDBG loading, dumping, and validation

- **Current invariant.** Required files are `metadata.yaml`, `unitigs.fna`, `unitigs.tsv`, `links.tsv`, and `mapping.tsv`. A unitig of N members has N−1 internal edges. Colours on those edges are stored one group per edge. `internal_overlaps` may be omitted; a de Bruijn graph then uses `k - 1`. Links name existing unitigs. Schema version is `1.0`. CDBG does not store dense feature matrices.
- **Current implementation.** `dump_cdbg` writes internal edge ids, colours, and overlaps. `load_cdbg` treats a missing `internal_overlaps` column as empty. Before this change, `validate_cdbg` checked overlap length only when overlaps were present, so a multi-node unitig with an empty `internal_edge_ids` cell still loaded. `cdbg_to_cfa` then omitted those edges and, if the colour list was short, filled the missing groups with an empty set.
- **Identified gap.** Dropped internal edge ids and a short colour list were accepted. Empty internal edge colours were imputed.
- **Classification.** Bug.
- **Proposed fix.** Require `len(internal_edge_ids) == len(members) - 1` and the same length for `internal_edge_colors`. Reject an internal edge id that is empty or that collides with another edge or link. `cdbg_to_cfa` raises `ContractError` if the colour list length disagrees, instead of filling blanks. Done.
- **Breaking.** Breaking for a CDBG that omitted internal edge ids on a multi-node unitig. The synthetic fixture was such a file and was regenerated from the compactor (section 10). Files the current compactor writes still load. Schema version stays `1.0`. Omitting `internal_overlaps` on a de Bruijn graph remains valid.

## 5. CDBG → CGT conversion

- **Current invariant.** Dense ids are `0 .. N-1` in unitig-id order. Edge features, labels, and colours follow CSR order: source dense id, target dense id, link id. There is no dense `N×N` adjacency. CGT mapping stores `dense_id`, `source_id`, and `cfa_node_ids`. It does not become a fourth semantic graph.
- **Current implementation.** `cdbg_to_cgt` permutes link-aligned arrays with `csr_link_order`. Node mapping is written. Link ids are not stored on the tensor. They are recomputed from the CDBG with the same sort. Internal edges are not CSR edges.
- **Identified gap.** Recovery of a CSR slot to a CFA edge id was possible from the CDBG but had no helper, so a permuted link list was easy to misread. The bubble test only covers the case where link order already matches CSR order.
- **Classification.** Missing capability (a reader), not a conversion bug. The permutation itself keeps features on the same link.
- **Proposed fix.** `cgt_edge_cfa_ids` returns the CFA edge id of each CSR slot. `node_lineage` reads the node row. No new on-disk column. Done. A test reverses the bubble links and checks that `edge_features[j]` stays on the recovered edge id.
- **Breaking.** No. Existing CGT directories stay readable.

## 6. CGT validation and serialization

- **Current invariant.** Schema `1.0`, CSR topology, float32 features, int64 labels, uint8 colours, dense ids `0 .. N-1` in order. Node row `i` is dense id `i`. Colour columns follow `color_ids`. Arrays are `.npy` files. That container is not part of the logical contract.
- **Current implementation.** `validate_cgt` checked shapes, dtypes, and dense-id order. It accepted an empty `cfa_node_ids` list, a repeated `source_id`, and a `color_ids` list whose length disagreed with the colour matrices. Dump and load of the bubble round-trip the arrays.
- **Identified gap.** Empty CFA ids and a colour-width mismatch break the identity and colour alignment checks.
- **Classification.** Bug.
- **Proposed fix.** Reject a missing `source_id`, a duplicate `source_id`, an empty or repeated CFA node id, and a colour-id list whose length is not the colour-matrix width. Done.
- **Breaking.** Breaking for a CGT that omitted CFA node ids or disagreed with its colour-id table. Objects produced by `cdbg_to_cgt` already match. Schema version stays `1.0`.

## 7. Colouring

- **Current invariant.** Colouring is an operation on a CFA (`replace`, `merge`, `intersect`, `subtract`). It does not change node ids, edge ids, sequences, or endpoints. Sample colour is not a taxonomic label. Unitig colour is the union of member node colours. Per-node colours stay on the CDBG mapping. CGT colour columns are sorted `color_id`.
- **Current implementation.** `colour_cfa` and `colour_by_reads` copy topology. The compactor unions node colours onto the unitig and keeps per-node colours on `NodeMap`. `cdbg_to_cgt` builds a dense uint8 colour matrix. No bitset is required by schema 1.0.
- **Identified gap.** None in the colouring rules. Colour-matrix width was not checked; that fix is in section 6.
- **Classification.** No gap in colouring. The width check was a CGT validation bug.
- **Proposed fix.** None in `contracts/colouring.py`.
- **Breaking.** No.

## 8. GFA and FASTG import

- **Current invariant.** `gfa_to_cfa` keeps each GFA segment id as a CFA node id, stores `graph_type` (default `repeat`), and stores a CIGAR `NM` overlap. It does not invent `k`. `fastg_to_cfa` keeps each contig once, stores strand as edge orientation, and sets the CFA de Bruijn `k` to assembly `k + 1` so the overlap is the assembly `k`.
- **Current implementation.** GFA edge ids are `e000001` upward in file order. FASTG edge ids are `e000001` upward over the sorted unique oriented pairs. Both fail on a dangling endpoint, a bad sequence, or an overlap mismatch.
- **Identified gap.** GFA `L` lines and FASTG headers do not carry a stable edge id, so the importer assigns one. Segment and contig ids are kept.
- **Classification.** Intentional limitation.
- **Proposed fix.** None. Assigned edge ids are deterministic for a given file. They are not biological names.
- **Breaking.** No.

## 9. DS and GCN

- **Current invariant.** `run_ds` reads a CGT, does not write the caller's arrays, and returns one row per dense id with `dense_id`, `source_id`, `cfa_node_ids`, `predicted_label`, and `probability`. The NumPy model aggregates a node with itself and its outgoing neighbours. It does not allocate a dense `N×N` adjacency. Labels are not concatenated into `X_node`.
- **Current implementation.** `train_numpy_gcn` and `train_pyg_gcn` follow that contract. PyG is an optional backend. The result rows copy the CGT node mapping. There is no per-edge prediction.
- **Identified gap.** Edge-level DS output is not part of contract 7. Node lineage was already on the result row; the new helper is what a caller uses to check it.
- **Classification.** Intentional limitation for edges. No gap for node identity.
- **Proposed fix.** None in the trainer.
- **Breaking.** No.

## 10. Fixtures and round trips

- **Current invariant.** On-disk fixtures under `tests/fixtures/minimal_metagenome` are output of the loaders and the compactor, not a second definition of identity. `CFA → CDBG → CFA` restores sequences, edge endpoints, edge ids, and colours. Numeric CFA feature columns are not restored from CDBG.
- **Current implementation.** The live compactor, run on `coloured_cfa`, writes every internal edge. The checked-in `cdbg/unitigs.tsv` did not. Unitig `u000004` had 7 members and 3 internal edge ids; `u000005` had 8 members and 3; `u000007` had 6 members and 1. Those files still loaded, and `cdbg_to_cfa` restored 20 of 31 edges. Sequences still matched because the split uses node lengths and `k - 1`, not the missing edge ids. `chain/cdbg` already had the internal edge ids (`e000001`, and `e000004`,`e000005`). It omitted `internal_overlaps`; de Bruijn expansion still used `k - 1`. Bubble unitigs are one CFA node each, so they have no internal edges. CGT fixtures store node membership only. That membership matched the compactor even while the synthetic CDBG edge list was short. No test loaded the on-disk CDBG and compared it with `cfa_to_cdbg`.
- **Identified gap.** The synthetic CDBG fixture dropped internal CFA edge ids. Chain and bubble fixtures were missing the overlap column the current dumper writes. Fixtures were not load-tested against the compactor.
- **Classification.** Bug in the synthetic fixture (stale file). The chain file was not missing edge ids. Missing overlaps are an optional column, so that part was an intentional schema allowance, not a dropped id. The lack of a load test was a missing check.
- **Proposed fix.** Regenerated `chain/cdbg`, `bubble/cdbg`, and `cdbg` with `dump_cdbg(cfa_to_cdbg(load_cfa(...)))`. Added internal edge ids that the compactor already emitted (`e000010`, `e000016`, `e000021` on `u000004`; `e000011`, `e000017`, `e000022`, `e000027` on `u000005`; `e000023`, `e000001`, `e000002`, `e000003` on `u000007`) and the `internal_overlaps` column. CGT fixtures were left in place after `assert_cgt_matches_cdbg` succeeded against the regenerated CDBGs. `test_on_disk_fixtures_match_the_compactor` now loads them.
- **Breaking.** The regenerated files are still schema `1.0`. A consumer that required the old, short `internal_edge_ids` cell was reading a file that already violated the compaction rule.

## Identity helpers

`metametro.identity` is the reader for the chain that was already stored:

- `node_lineage(cgt, dense_id)` → CDBG unitig id and CFA node ids
- `assert_cgt_matches_cdbg(cgt, cdbg)` → those CFA ids must equal the unitig member list
- `link_cfa_edge_id(cdbg, link_id)` → the CFA edge id, which is the link id
- `internal_cfa_edge_ids(cdbg, unitig_id)` → CFA edge ids in path order
- `cgt_edge_cfa_ids(cdbg)` → CFA edge id of each CSR slot

No biological name is assigned to a unitig.

## Left as intentional limitations

- CGT does not store DNA, a dense adjacency, or a copy of the CDBG edge table. CSR edge identity is recomputed from the CDBG.
- Internal unitig edges are not CGT edges.
- Numeric CFA feature columns are not stored on the CDBG. Callers pass arrays into `cdbg_to_cgt` in unitig-id order and link order.
- `cdbg_to_cfa` omits the overlap column from every restored edge when any restored edge has no overlap, so the CFA edge schema stays one table. Compactor output sets an overlap on every edge, so that path is not taken for its files.
- GFA and FASTG edge ids are generated. Segment and contig ids are kept.
- Contract 7 predicts a label per node, not per edge.
- `internal_overlaps` may still be absent on a de Bruijn CDBG. Splitting then uses `k - 1`.
- A mapping length that still tiles the unitig but disagrees with a CFA the CDBG does not contain cannot be detected from the CDBG alone.
