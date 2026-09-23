# MetaMetro

[![version](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdsmutin%2FMetaMetro%2Fmain%2FVERSION&search=%5B0-9%5D%2B%5C.%5B0-9%5D%2B%5C.%5B0-9%5D%2B&label=version&color=blue)](VERSION)
[![required tests](https://img.shields.io/github/actions/workflow/status/dsmutin/MetaMetro/required-tests.yml?branch=main&label=required%20tests)](https://github.com/dsmutin/MetaMetro/actions/workflows/required-tests.yml)
[![full tests](https://img.shields.io/github/actions/workflow/status/dsmutin/MetaMetro/full-tests.yml?branch=main&label=full%20tests)](https://github.com/dsmutin/MetaMetro/actions/workflows/full-tests.yml)
[![warning](https://img.shields.io/badge/warning-in%20development-yellow)](https://shields.io/badges/static-badge)

Metagenomic assembly totally coloured graph representations

**Warning: in development.** Interfaces may change. See `VERSION` (single source of truth).

Three graph representations share one identity chain: **CFA** (canonical exchange) → **CDBG** (ToCUMG: totally coloured universal metagenomic graph) → **CGT** (CSR tensor for graph ML). Contracts and the minimal implementation are in [docs/formats.md](docs/formats.md), [docs/contracts.md](docs/contracts.md), and [docs/implementation.md](docs/implementation.md).

```python
from metametro import mock_cgt, run_ds

print(run_ds(mock_cgt(), seed=0)["metadata"])
```

## Install

Conda is the only supported install:

```bash
conda env create -f environment.yml
conda activate metametro
```

`environment.yml` sets `PYTHONPATH=src` and pins every dependency. Do not publish a pip-first install path.

The Contract 7 PyTorch Geometric backend is optional and is not in the required environment, so CI stays light:

```bash
conda env create -f environment-pyg.yml
conda activate metametro-pyg
pytest -m optional
```

## Usage

```bash
metametro --version
metametro
python examples/toy/run.py
python examples/mock_pipeline/run.py
```

## Tests

```bash
pytest -m mandatory    # every commit
pytest                 # mandatory + optional (release / manual CI)
```

## License

MIT. See [CONTRIBUTING.md](CONTRIBUTING.md).
