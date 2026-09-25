# Benchmarks

`metametro benchbuild` is the only place a benchmark graph is generated. Other tools depend on the directory it writes. They do not keep a second copy of the generator.

Strong100, the samovar10 bundles, and the prebuilt ONT and Illumina CSR dumps are not benchmarks here. They are frozen assemblies without a generator in this repository.

## Command

```bash
metametro benchbuild --list
metametro benchbuild bubble_strain_2
metametro benchbuild bubble_strain_2 --outdir /path/to/empty_dir
metametro benchbuild bubble_strain_2 --colouring as_built
```

The default output directory is

```text
data/bench/{bench_name}/{assembler}/{properties}/
```

That tree is gitignored. A build writes:

| Path | Role |
| --- | --- |
| `cfa/`, `cdbg/`, `cgt/` | The three graph structures |
| `ground_truth/` | Evaluation labels. Not a model input |
| `manifest.yaml` | Bench name, colourings, identity |
| `identity.sha256` | Digest of sequences, topology, and colour sets |

Every colouring registered in `metametro.bench.colourings` that can run is applied, including a colouring added after the benchmark was introduced. `--colouring` restricts that set. `as_built` keeps colours already on the graph. `read_depth` colours nodes by read depth and edges by junction density when the build has reads and an integer `k`.

A column or colour namespace that carries an evaluation target (`truth_taxon_id`, `expected_action`, `genome_abundance`, a `truth` namespace) is rejected. Those values stay under `ground_truth/`.

## Layout

Generation code:

```text
src/metametro/bench/data/universal/     shared builders
src/metametro/bench/data/{bench_name}/  one benchmark, when the name is a Python module
```

Scoring code:

```text
src/metametro/bench/scoring/assembly/universal/
src/metametro/bench/scoring/assembly_binning/universal/
src/metametro/bench/scoring/profiling/universal/
src/metametro/bench/scoring/{type}/{bench_name}/
```

`assembly_binning` is the contig-binning score used by ParaGVAE (majority-label contig F1). `profiling` is the abundance score (L1, Bray–Curtis, presence F1). `assembly` is a length summary such as N50. A scorer reads `ground_truth/` and a prediction table. It does not read a target out of the feature matrix.

## In-process benchmarks

| Name | What it is |
| --- | --- |
| `bubble_strain_2` | Two taxa, one bubble. The BubbleBlower toy. Truth: retain |
| `bubble_error_1` | One taxon, one error bubble. Truth: pop |
| `bubble_nested_2` | Two taxa, outer and inner bubbles |
| `bubble_shared_2` | Two taxa, one shared node, no bubble |
| `bubble_strain_3_n50` | Three strains, 50 bubbles, seed 42 |
| `bubble_reads_2` | Two strains, read length 12, de Bruijn k=5 |
| `mock_bubble` | MetaMetro four-node fixture |
| `mock_chain` | MetaMetro six-node compaction fixture |
| `synthetic_reads_2` | Two synthetic genomes, in-process reads, k=8 |

Legacy names such as `strain_bubble` resolve to the canonical name.

Pinned NCBI communities (family, genus, and phage read-depth benches) are added as contracts in the same registry. A bench that needs `datasets`, Samovar, or MEGAHIT stops with the missing program instead of writing a stand-in graph.
