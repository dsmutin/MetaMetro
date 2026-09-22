# Project audit

**Date:** 2026-09-22
**Repository:** `/mnt/tank/scratch/dsmutin/tools/my/metametro`
**Branch / commit:** `main`, no commits yet (empty GitHub clone plus local files)
**Auditor mode:** Read-only relative to the audit; the implementation was written in this session
**Harness:** tool profile (`.cursor/rules/harness.mdc`)
**Properties in scope:** scientific-integrity, validation-first, reproducibility, missing-data-policy, english-docs, no-push, follow-contributing
**Not in scope:** `method_decision`, artifact registry, paper pipeline, SLURM. `@verify-methods` was not run.

## Executive summary

- The tree is a conda-oriented Python package for CFA, CDBG, and CGT.
- The documented environment now exists. `conda env create -f environment.yml` succeeded where `mamba env create` had failed twice; inside `metametro`, `pytest -m mandatory` passed (49 tests) and the full run was 49 passed, 1 skipped (the PyG test).
- `environment.yml` pins every dependency to the versions that solve resolved, and uses `conda-forge` plus `nodefaults`, so no `defaults` Terms-of-Service acceptance is needed.
- `environment-pyg.yml` is the optional PyG environment. Inside `metametro-pyg` the full suite is 50 passed, including `pytest -m optional`, and `run_ds(..., backend="pyg")` returns predictions. That environment cannot take the `setuptools` pin because conda-forge `pytorch` requires `setuptools <82`.
- Format and contract documents are in `docs/formats.md`, `docs/contracts.md`, and `docs/implementation.md`.
- Nothing was pushed.

## Scores

| Dimension | Rating | Summary |
| --- | --- | --- |
| Project completeness | Good | Formats, converters, contracts, fixtures, and tests are in the tree |
| Documentation | Good | English format and contract docs match the tested behaviour |
| Reproducibility | Good | Commands, seeds, and manifests exist; both conda environments were created and tested from the pinned files |
| Methodological rigor | Fair | Contracts are explicit; the phage GCN is a smoke run, not a performance claim |
| Publication readiness | N/A | Tool profile |

## 1. Completeness

| Component | Status | Evidence |
| --- | --- | --- |
| Package and tests | Present | `src/metametro`, `tests/`, 49 mandatory tests passed |
| Reference genomes | Present | `data/manifests/download_manifest.tsv` |
| Format docs | Present | `docs/formats.md`, `docs/contracts.md` |
| Conda env `metametro` | Present | `conda env create -f environment.yml`; `pytest -m mandatory` inside it |
| Conda env `metametro-pyg` | Present | `conda env create -f environment-pyg.yml`; `pytest -m optional` inside it |
| PyG training | Present | `data/work/phage_x10/summary.json` (`pyg_predictions` 983) |

## 2. Documentation gaps

| Document | Gap | Priority |
| --- | --- | --- |
| `method-decision.md` | Absent | Out of scope (`method_decision` disabled) |
| Fresh-install note | `README.md` and `CONTRIBUTING.md` document both environments; `conda` works, `mamba` failed here | Closed |

## 3. Reproducibility

| Issue | Evidence | Priority |
| --- | --- | --- |
| Both environments created and tested from the pinned files | `conda env list`, suite output above | — |
| `mamba env create` still fails on this machine; `conda env create` succeeds | earlier mamba logs versus this run | P3 |
| Earlier results in this session were produced on the base interpreter (Python 3.13.2, NumPy 2.3.5) | session history | P3 |
| Simulator seed and MEGAHIT `k=21` are recorded | `scripts/phage_baseline.py`, summary JSON | — |

## 4. Methods

`method_decision` is disabled. This audit does not reconstruct a method catalogue and does not cite a `method-decision.md`.

## 5. Software versions observed

| Tool | Version | Source |
| --- | --- | --- |
| MEGAHIT | 1.2.9 | baseline log |
| InSilicoSeq | 2.0.1 | `iss --version` on this machine |
| conda | 26.1.1 | `conda --version` |
| Python (env `metametro`) | 3.12.14 | `conda list`, conda-forge |
| NumPy (env) | 2.5.3 | `conda list`, conda-forge |
| PyYAML (env) | 6.0.3 | `conda list`, conda-forge |
| pytest (env) | 9.1.1 | `conda list`, conda-forge |
| PyTorch (env `metametro-pyg`) | 2.13.0, `cpu_mkl_py312` | `conda list`, conda-forge |
| PyTorch Geometric (env `metametro-pyg`) | 2.8.0.post1 | `conda list`, conda-forge |

No claim that these are the newest releases.

## 6. Findings

| Priority | Issue | Evidence | Action |
| --- | --- | --- | --- |
| P2 | The T-phage run is a smoke test | `data/work/phage_x10/summary.json` | 983 predictions confirm identity only. No accuracy is claimed; 80 unitigs are unlabelled and 49 nodes have no sample colour |
| P3 | `mamba` cannot solve these files on this machine | earlier `mamba env create` logs | Use `conda env create`. Both environments were built that way |
| P3 | Session results predate the conda environments | the T-phage runs used the base interpreter | Rerun `scripts/phage_baseline.py` inside `metametro-pyg` if a run must be attributable to the pinned environment |
| P3 | No git commit yet | `git status` reports every path untracked | Commit only when asked |
