# Dataset audit

**Date:** 2026-09-22
**Auditor:** coding agent
**Overall status:** Ready with warnings

## 1. Data inventory

| File | Path | Size | Notes |
| --- | --- | --- | --- |
| NCBI FASTA | `data/raw/ncbi/*.fna` | 5 files, 423,874 bytes | original headers and T4 IUPAC bases |
| Pipeline FASTA | `data/raw/genomes/{T1,T3,T4,T5,T7}.fna` | 5 files | record id is the short phage name |
| Manifest | `data/manifests/download_manifest.tsv` | 5 rows | one row per accession |
| Checksums | `data/checksums/checksums.txt` | 10 hashes | NCBI file and derived file |
| Simulated reads | `data/work/phage_baseline/metadata.tsv` | 1,600 rows | produced by Samovar ISS, not downloaded |

No FASTQ was downloaded. Reads are a simulation product.

## 2. Metadata completeness

| Column | Required for genomes | Missing | Issues |
| --- | --- | --- | --- |
| accession | Yes | 0 | matches the file name |
| title | Yes | 0 | from `esummary` |
| length | Yes | 0 | matches FASTA length |

Read metadata columns `read_id`, `sample_id`, `genome_id`, `ambiguous` are present for all 1,600 simulated reads.

## 3. Sample identifiers

Genome ids in the pipeline FASTA are `T1`, `T3`, `T4`, `T5`, `T7`. They are unique. Read ids such as `T1_0_0/1` matched exactly one of those ids (`ambiguous=no` for 1,600/1,600).

## 4. Class balance

One reference genome per phage. Simulated reads are balanced because this `samovar generate` call did not pass an abundance table: 320 reads per genome, 800 reads in `sample_1` and 800 in `sample_2`. That is not a biological abundance design.

## 5. Duplicates and missing values

No duplicate accessions. No empty FASTA. T4 has three ambiguous nucleotides in the official file; the derived file stores `N` at those positions. Those bases are not missing data that were invented: the substitution is recorded in the manifest.

## 6. Outliers

Not applicable to five complete reference chromosomes. No sequencing-depth distribution was computed for the references.

## 7. Metadata and file consistency

| Check | Result |
| --- | --- |
| Every accession has a FASTA | Pass |
| Length equals `esummary` `slen` | Pass, all five |
| Header names the requested phage | Pass |
| Derived id matches `T1`–`T7` | Pass |
| Paired FASTQ from the repository | Not applicable |

## 8. Sequencing depth

No downloaded reads. Depth of the simulated library was not measured beyond the read count above. MEGAHIT, on the concatenated mates, reported 1,600 reads and 41 contigs (12,204 bp, N50 283 bp). That assembly is shallow relative to the phage lengths, especially T4 (168,903 bp) and T5 (121,750 bp).

## 9. Batch effects

One download date (2026-09-22) and one simulator seed (`1`). No batch factor to cross with phage identity.

## 10. Recommendations

| Priority | Issue | Action |
| --- | --- | --- |
| High | T4 IUPAC bases | Keep the NCBI file unchanged. Use the derived `N` copy only with the manifest note. |
| High | Equal simulated abundances | Pass an abundance table to Samovar if the experiment needs 2–3 abundance levels. The in-repository mock already varies read counts. |
| Optional | Shallow 400-pair assembly | The 10× run is `data/work/phage_x10` (`total_reads` 4000). Labels there are a k-mer vote, not a measured classifier score. |

## 11. Readiness

The five reference genomes are usable as simulation input. They are not a sequenced metagenomic cohort. T4's three ambiguous bases must stay visible to anyone who interprets matches against the derived file.
