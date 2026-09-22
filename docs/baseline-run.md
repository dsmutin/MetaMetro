# Phage baseline run

**Date:** 2026-09-22
**Command:** `PYTHONPATH=src python3 scripts/phage_baseline.py --genomes data/raw/genomes --work data/work/phage_x10 --k 21 --total-reads 4000 --epochs 20 --samovar /mnt/tank/scratch/dsmutin/tools/my/samovar/samovar/bin/samovar --megahit /mnt/tank/scratch/dsmutin/tools/my/samovar/samovar/.cache/samovar/envs/megahit/bin/megahit`

Samovar `generate` writes `.generate/generate.sh`. That script's Snakemake job calls InSilicoSeq. MEGAHIT 1.2.9 is the binary in the Samovar cache. The graph is not `final.contigs.fa`. It is `intermediate_contigs/k21.contigs.fa` converted with `megahit_toolkit contig2fastg 21`.

An earlier run with `total_reads` 400 assembled `final.contigs.fa` into 41 contigs and 0 inter-contig links. Those final contigs no longer contain the bubbles.

## Reads

`total_reads` 4000 is ten times the earlier setting. InSilicoSeq logged `Generating 16000 reads` because its read-count file counts mates. The metadata table has 16,000 rows and 0 ambiguous reads.

| Genome | Reads | Ambiguous |
| --- | --- | --- |
| T1 | 3200 | 0 |
| T3 | 3200 | 0 |
| T4 | 3200 | 0 |
| T5 | 3200 | 0 |
| T7 | 3200 | 0 |

Sample split from the Samovar file names: 8000 reads in `sample_1`, 8000 in `sample_2`. Abundances are equal. Colouring used vertex depth at least 1 and edge junction density at least 2. A read covers a node from either strand. A `-` endpoint is reverse-complemented before the junction `(k+1)`-mer is taken, and the reverse complement of that junction counts as well. That colours 934 of 983 nodes and 1144 of 1270 edges.

## Assembly graph

MEGAHIT k=21, min count 2, 4 threads, `--keep-tmp-files`, single-end `reads.fastq`:

```text
434 contigs, total 292995 bp, min 200 bp, max 15945 bp, avg 675 bp, N50 861 bp
```

That line is the final contig file. The CFA is the intermediate FASTG:

| Object | Count |
| --- | --- |
| CFA nodes (one per contig, forward sequence) | 983 |
| CFA edges (both strands) | 1270 |
| Same-strand branch nodes | 226 |
| Same-strand join nodes | 226 |
| Bubble sources (two same-strand paths meet within 4 edges) | 191 |

FASTG edges overlap by the assembly k-mer (21). The CFA de Bruijn `k` is 22 so compaction checks that same overlap. Each adjacency is stored twice, once in each direction. Those reciprocal links mean 18 simple `++` chains do not compact. The rest of the intermediate graph is already branched, so the CDBG keeps 983 unitigs and 1270 links. Summary: `data/work/phage_x10/summary.json`.

## Labels and GCN

A node label is the genome that contains a strict majority of that node's 21-mers, on either strand of the derived genome. Otherwise the label is 0. A unitig keeps that label only when every member agrees.

| Label | Meaning | Unitigs |
| --- | --- | --- |
| 0 | unclassified or mixed | 80 |
| 1 | T1 | 18 |
| 2 | T3 | 156 |
| 3 | T4 | 389 |
| 4 | T5 | 177 |
| 5 | T7 | 163 |

The NumPy GCN and the PyTorch Geometric GCN (`torch-geometric` 2.8.0) each returned 983 predictions, one per dense id (`data/work/phage_x10/ds_numpy.tsv` for the NumPy run). That checks that every unitig is addressed. It is not a classification accuracy.
