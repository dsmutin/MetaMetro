# Data preparation report

**Date:** 2026-09-22
**Mode:** Full execution for five RefSeq phage genomes, then a local simulation baseline
**Overall status:** Partial

## Executive summary

Five T-phage RefSeq genomes were downloaded from NCBI nuccore and checked against `esummary`. Technical checks passed. T4 is a biological warning because the official sequence has three IUPAC bases; the derived pipeline file replaces them with `N` and says so in the manifest. Reads were not downloaded. Samovar's InSilicoSeq pipeline then simulated 1,600 reads with a recoverable genome id on every read; the assembly from that shallow library is 41 short contigs. A later run with ten times the read budget is recorded in [baseline-run.md](baseline-run.md). The conda environment in `environment.yml` now exists and its tests pass. The repository is still not data-ready as a sequencing study: every read is simulated and the abundances are equal.

## Requested datasets

| Identifier | Source | Required |
| --- | --- | --- |
| NC_000866.4 (T4) | user request, T-phages | yes |
| NC_001604.1 (T7) | user request | yes |
| NC_003298.1 (T3) | user request | yes |
| NC_005833.1 (T1) | user request | yes |
| NC_005859.1 (T5) | user request | yes |

## Phase 0 — Project state

The tree had no `data/` directory before this run. No previous manifest or audit. Nothing was skipped as already validated.

## Phase 1 — Acquisition

**Invoked:** yes, for all five accessions.

| Metric | Value |
| --- | --- |
| Requested | 5 |
| Downloaded | 5 |
| Skipped | 0 |
| Download failures | 0 |
| Validation failures | 0 |
| Biological warnings | 1 (T4 IUPAC) |

Repository: NCBI nuccore via E-utilities. Details: [data/manifests/acquisition_report.md](../data/manifests/acquisition_report.md).

## Phase 2 — Dataset audit

**Status:** Ready with warnings. Report: [dataset-audit.md](dataset-audit.md).

The warning is the three T4 ambiguity codes and the fact that these files are references, not a read cohort.

## Phase 3 — Project audit

**Blocking issues:** none that make the downloaded FASTA unusable. **P1:** the conda env did not install, and PyTorch Geometric was not available. Report: [project-audit.md](project-audit.md). `@verify-methods` was not in scope.

## Phase 4 — Baseline that consumed the genomes

Separate from acquisition. Recorded in [baseline-run.md](baseline-run.md).

| Step | Result |
| --- | --- |
| Samovar ISS | 1,600 reads, 320 per genome, 0 ambiguous |
| MEGAHIT 1.2.9, k=21 | 41 contigs, 12,204 bp, N50 283 bp |
| CFA → CDBG → CGT | 41 unitigs, 0 inter-contig edges |
| NumPy GCN | 41 predictions, ids preserved |
| PyG GCN | not run (`torch_geometric` missing) |

## Not data-ready

| Criterion | Result |
| --- | --- |
| Requested genomes acquired | met |
| Technical validation | met |
| Biological validation | met with a documented T4 warning |
| Dataset audit | Ready with warnings |
| Project audit without P0 | met |
| Documented conda env | not met |
| Read cohort / deep assembly | not requested as a download; the smoke simulation is shallow |

Next step: recreate the conda env when conda-forge answers, then rerun mandatory tests inside it. Raise the ISS read count before treating phage labels as a training set.
