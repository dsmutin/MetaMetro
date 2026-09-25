# Benchmarks

`metametro benchbuild` is the only place a benchmark graph is generated. Other tools depend on the directory it writes. They do not keep a second copy of the generator.

Strong100, the samovar10 bundles, and the prebuilt ONT and Illumina CSR dumps are not benchmarks here. They are frozen assemblies without a generator in this repository.

## Command

```bash
metametro benchbuild --list
metametro benchbuild --all
metametro benchbuild bubble_strain_2
metametro benchbuild bubble_strain_2 --outdir /path/to/empty_dir
metametro benchbuild bubble_strain_2 --colouring as_built
```

`--all` builds every in-process graph and writes a contract for every community and external benchmark. It does not download genomes or call R. `--all --execute` also downloads and assembles those. A directory that already has `manifest.yaml` and `identity.sha256` is left unchanged.

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

Every colouring registered in `metametro.bench.colourings` that can run is applied, including a colouring added after the benchmark was introduced. `--colouring` restricts that set. `metametro benchbuild --list-colourings` prints the registry.

| Colouring | Namespace | Auto | When it runs |
| --- | --- | --- | --- |
| `as_built` | existing (taxon, sample, route, …) | yes | always; keeps colours already on the graph |
| `read_depth` | `sample` | yes | reads and integer `k` are on the build |
| `composition_kmeans` | `composition` | yes | at least two sequenced nodes |
| `kraken2` | `kraken2` | yes | Kraken2 output or `kraken2` plus a database |
| `kaiju` | `kaiju` | yes | Kaiju output or `kaiju` plus an FM-index |
| `decaying` | `decaying` | yes | the graph already has colours and edges |
| `read_accession` | `accession` | no | named with `--colouring`; ISS-style read ids. Simulated accessions leak truth, so this is not auto |

A half community (`*_half`) still simulates the half metagenome. Its Kraken2 and Kaiju library is the parent pin's full `db` set (the non-synonymous partners), not the half set.

Downstream tools do not recolour. They load the ToCUMG and name namespaces:

```python
from metametro.bench import load_bench_cdbg, namespaces_for

graph = load_bench_cdbg(bench_dir, namespaces=namespaces_for(("kraken2", "decaying")))
```

`filter_colours(graph, namespace="kraken2")` is the same selection on a CFA, CDBG, or (by `color_ids`) a CGT.

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

Pinned NCBI communities and external contracts are in the same registry. `metametro benchbuild NAME` downloads and assembles when `datasets`, Samovar, and MEGAHIT are on `PATH`. `metametro benchbuild NAME --contract-only` writes the accession pin and the lognormal abundance table and does not invent a graph. A second contract build has the same `identity.sha256`.

| Name | Was | Design |
| --- | --- | --- |
| `4domain_family_100` | high100 | 100 families, 4 domains, family rank, 100000 reads |
| `4domain_family_100_half` | half100half | every other family, 50 families |
| `4domain_family_100_x10` | high100_enriched | same families, 1000000 reads |
| `4domain_family_100_half_x10` | half100half_enriched | 50 families, 1000000 reads |
| `3domain_genus_75` | low75 | 75 genera, 3 domains, genus rank |
| `3domain_genus_75_half` | low75half | 38 genera |
| `3domain_genus_75_x10` | low75_enriched | 75 genera, 1000000 reads |
| `3domain_genus_75_half_x10` | low75half_enriched | 38 genera, 1000000 reads |
| `bacteria_species_20_heldout` | heldout_genera | 4 genera, 20 species pairs, species rank |
| `bacteria_strain_10` | half_strains | 10 of those pairs |
| `phage_species_5` | phage_baseline | T1, T3, T4, T5, T7, 400 reads, k=21 |
| `phage_species_5_x10` | phage_x10, phage_10 | same five phages, 4000 reads |
| `roxel` | roxel | sfnetworks street graph |
| `spb_ground_transit` | spb transit | ORGP GTFS, 60 m stop merge; pass `--gtfs` |

There is no ten-species phage benchmark. `phage_10` was the ten-times read budget. Strong100, samovar10, and the prebuilt ONT and Illumina bundles are not in this registry.

