# Toy example

Runs metametro with the current **baseline** implementation and checks that it works.

## Setup

```bash
conda env create -f ../../environment.yml
conda activate metametro
cd examples/toy
```

## Run

```bash
python run.py
```

Expected: exit code 0 and JSON with `"status": "baseline"` and `"ok": true`.
