"""``metametro benchbuild`` command."""

from __future__ import annotations

import argparse
from pathlib import Path

from metametro.bench.build import build
from metametro.bench.registry import list_specs, resolve


def main(argv: list[str] | None = None) -> int:
    """Build one benchmark, or list the registry."""
    parser = argparse.ArgumentParser(
        prog="metametro benchbuild",
        description="Download or construct a benchmark and write CFA, CDBG, and CGT.",
    )
    parser.add_argument("name", nargs="?", help="benchmark name")
    parser.add_argument("--list", action="store_true", help="list benchmarks and exit")
    parser.add_argument("--outdir", type=Path, default=None, help="output directory (default: data/bench/...)")
    parser.add_argument(
        "--colouring",
        action="append",
        default=None,
        help="colouring to apply; repeat for several. Default: every colouring that can run",
    )
    args = parser.parse_args(argv)
    if args.list or not args.name:
        for spec in list_specs():
            aliases = ""
            if spec.legacy_names:
                aliases = " (was " + ", ".join(spec.legacy_names) + ")"
            print(f"{spec.name}\t{spec.assembler}\t{spec.properties}\t{spec.summary}{aliases}")
        return 0 if args.list or not args.name else 2
    colourings = tuple(args.colouring) if args.colouring else None
    # Resolve first so an alias is accepted before the directory is created.
    resolve(args.name)
    result = build(args.name, outdir=args.outdir, colourings=colourings)
    print(f"bench\t{result.spec.name}")
    print(f"status\t{result.status}")
    print(f"outdir\t{result.outdir}")
    print(f"colourings\t{','.join(result.colourings)}")
    print(f"identity\t{result.identity}")
    return 0
