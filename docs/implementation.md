# Implementation

The public entry points are:

```python
cfa = load_cfa(path)
validate_cfa(cfa)
cdbg = cfa_to_cdbg(cfa)
validate_cdbg(cdbg)
cgt = cdbg_to_cgt(cdbg, node_features=..., edge_features=..., node_labels=...)
validate_cgt(cgt)
graph = cgt_to_pyg(cgt)          # requires torch_geometric
result = run_ds(cgt, seed=0)     # does not modify cgt
```

Fixtures: `mock_cfa()`, `mock_cdbg()`, `mock_cgt()`, `chain_cfa()`, `synthetic_genomes()`.

| Contract | Module | What the minimal code does |
| --- | --- | --- |
| 1 | `contracts/assembly.py` `simulate_metagenome` | Deterministic slices of the source genomes. External path: `samovar generate`, then the generated `.generate/generate.sh` Snakemake script, which runs InSilicoSeq. |
| 2 | `dbg_from_sequences`, `contigs_to_dbg`, `fastg_to_cfa` | k-mer de Bruijn graph. External command: MEGAHIT. The T-phage baseline loads MEGAHIT's intermediate FASTG so branches and bubbles survive; `strand_junction_counts` and `directed_bubble_sources` report that structure. |
| 3 | `formats/cfa` | Directory loader, dumper, and validator. The de Bruijn builder already emits CFA. |
| 4 | `contracts/colouring.py` | Explicit set operations and the depth / density rule. k-mer graphs are counted in one pass over the reads; the counts match the general substring rule. |
| 5 | `converters/cfa_to_cdbg.py` | Chain compaction described in `docs/formats.md`. |
| 6 | `converters/cdbg_to_cgt.py` | CSR, dense ids, aligned features and colours. |
| 7 | `contracts/ds.py` | NumPy convolution by default; PyG `GCNConv` when requested. |

Schema files live next to each format (`formats/*/schema.yaml`) and pin schema 1.0. Validators raise `ContractError` with every violation they collected.

## What this tree deliberately does not do

- CFA does not import Bifrost or PyTorch.
- CDBG does not store coverage, GC, entropy, or per-node labels. Those stay on the CFA and are passed into `cdbg_to_cgt` as arrays.
- CGT construction does not allocate an `N×N` adjacency matrix.
- The PyG adapter does not become a fourth on-disk format.
- The NumPy trainer is the tested Contract 7 implementation. PyG is an optional backend, not a second result schema.

## Tests

Mandatory pytest covers, for each format: a valid object, a missing field, a bad dtype, a duplicate id, a dangling endpoint, and a bad schema version. Conversion tests cover the bubble round trip, the chain compaction counts, colour union, feature alignment, CSR edge order, determinism, and the genome → DS id chain on the synthetic metagenome.

`pytest -m optional` builds a PyG `Data` object when `torch_geometric` is installed and skips otherwise.

`examples/mock_pipeline/run.py` trains the NumPy model on the bubble for a few epochs and checks that four predictions come back.

## External tools

| Tool | Role | How it is invoked |
| --- | --- | --- |
| Samovar `generate` | Contract 1 baseline (InSilicoSeq) | `samovar_generate_command` |
| MEGAHIT | Contract 2 baseline | `megahit_command` (`k` odd and ≥ 15, `--keep-tmp-files`) |
| `megahit_toolkit contig2fastg` | Intermediate assembly graph | `contig2fastg_command`; parsed by `fastg_to_cfa` |
| PyTorch Geometric | Contract 7 optional backend | `run_ds(..., backend="pyg")`; install with `environment-pyg.yml` |

Bifrost is a permitted CDBG backend and is not linked in this version. Citations for the tools that are actually invoked are in `cite/tools.bib`. Samovar has no paper entry here.
