# ONT repeat graph

`samovar` has no `ont` subcommand. ONT reads are `samovar generate --reads_generator ont`, which is NanoSim through CAMISIM. metaFlye is Flye `--nano-raw --meta`. The wrapper `bin/metaflye` replaces an input under 1 MiB with one identity contig, so this run calls `flye` directly. Flye looks up `flye-minimap2` on `PATH`, so the Flye environment `bin` directory has to be on `PATH`.

Genomes: `data/raw/genomes` (T1, T3, T4, T5, T7). Seed 1, one sample, `host_fraction` 0.

A first pass with `--total_reads 20` wrote 18 ONT reads (78,821 bp). Flye 2.9.6-b1802 stopped with mean edge coverage 0 and an empty `20-repeat/graph_before_rr.gfa` (header only, no segments). `gfa_to_cfa` refuses that file.

The recorded pass used `--total_reads 400`. CAMISIM wrote `data/work/ont_repeat_cov/generate/initial/1_full_R1.fastq`: 408 reads, 1,678,235 bp, lengths 241–12,474. `1_full_R2.fastq` is empty. Flye assembly statistics: 168,368 bp, 7 fragments, N50 31,487, largest fragment 38,756, mean coverage 8.

The intermediate repeat graph is `data/work/ont_repeat_cov/flye/20-repeat/graph_before_rr.gfa`: 7 segments and 2 links. Both links are `++` with CIGAR `0M`, and both leave `edge_8` (`edge_8 → edge_9`, `edge_8 → edge_10`). `gfa_to_cfa` stores `graph_type: repeat` and does not add `k`. Compaction keeps that type. The overlap contract is 0, so `compaction` is `chain`, but `edge_8` has out-degree 2, so nothing is merged: 7 unitigs and 2 links. `cdbg_to_cfa` restores the segment sequences and still has no `k`.
