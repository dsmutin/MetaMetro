# Acquisition report

**Date:** 2026-09-22
**Requested datasets:** RefSeq genomes of coliphages T1, T3, T4, T5, and T7
**Repository:** NCBI nuccore
**Method:** E-utilities `esummary` then `efetch` FASTA (`rettype=fasta`, `retmode=text`, `tool=metametro`)
**Endpoint:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`

## Size

`esummary` `slen` values sum to 417,634 bp. Downloaded FASTA files sum to 423,874 bytes. Both figures are from the API response and the files on disk. No decompression step. Temporary space was the same as the download.

## Commands

Logged in `data/logs/download.log`.

1. `esummary.fcgi?db=nuccore&id=NC_000866.4,NC_001604.1,NC_003298.1,NC_005833.1,NC_005859.1&retmode=json`
2. One `efetch.fcgi` FASTA request per accession.

## Results

| Accession | Title | Length (bp) | Bytes | Technical | Biological |
| --- | --- | --- | --- | --- | --- |
| NC_000866.4 | Enterobacteria phage T4, complete genome | 168903 | 171371 | pass | warn |
| NC_001604.1 | Enterobacteria phage T7, complete genome | 39937 | 40563 | pass | pass |
| NC_003298.1 | Enterobacteria phage T3, complete genome | 38208 | 38809 | pass | pass |
| NC_005833.1 | Escherichia phage T1, complete genome | 48836 | 49586 | pass | pass |
| NC_005859.1 | Enterobacteria phage T5, complete genome | 121750 | 123545 | pass | pass |

5 requested, 5 downloaded, 0 failed downloads. Each file has one FASTA record. Lengths match `esummary`. SHA256 values are in `data/checksums/checksums.txt` and `data/manifests/download_manifest.tsv`.

## Biological warning

NC_000866.4 contains three IUPAC bases that are not `ACGTN`: `M` at 56143, `S` at 70346, `R` at 135175 (1-based). They are kept in `data/raw/ncbi/NC_000866.4.fna`. The pipeline copy `data/raw/genomes/T4.fna` replaces each of them with `N` and sets the record id to `T4`. The other four pipeline copies only change the record id to `T7`, `T3`, `T1`, or `T5`.

## Software

No separate datasets CLI. Download used Python `urllib` and `hashlib.sha256`. NCBI record versions are the accessions above.
