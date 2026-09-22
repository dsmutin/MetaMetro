# metametro features

Check a box only after mandatory tests pass.

- [x] Baseline CLI and pipeline (`status=baseline`)
- [x] CFA, CDBG, and CGT schemas, validators, and IO
- [x] CFA → CDBG compaction with CFA mapping
- [x] CDBG → CGT CSR tensor and PyG edge-index adapter
- [x] Colouring by vertex read depth and edge (k+1)-mer density 2
- [x] Unstranded colouring: a read and its reverse complement cover the same node
- [x] Contract 7 NumPy GCN that keeps dense id → CFA id
- [x] Bubble fixture and chain compaction fixture
- [x] MEGAHIT intermediate graph: `contig2fastg` FASTG → CFA with edge orientation
- [x] Branch, join, and bubble counts on an oriented CFA
- [x] Ground-transit CFA: repeat-junction sequences, mode-average edge coverage, route and vehicle-type colours
- [x] Conda environment `metametro` created from `environment.yml` with `conda` (mamba fails here); mandatory tests pass inside it
- [x] Runtime version pins in `environment.yml`, `conda-forge` plus `nodefaults`, verified by a solve
- [x] Optional `environment-pyg.yml` (`metametro-pyg`): PyTorch 2.13.0 CPU and PyG 2.8.0.post1 from conda-forge; `pytest -m optional` passes
- [x] `recipe/meta.yaml` declares the numpy and pyyaml runtime dependencies
- [ ] First GitHub commit and wiki push (only when asked)
