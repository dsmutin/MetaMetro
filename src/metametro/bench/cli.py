"""``metametro benchbuild`` command."""

from __future__ import annotations

import argparse
from pathlib import Path

from metametro.bench.build import build, build_all
from metametro.bench.registry import list_specs, resolve


def main(argv: list[str] | None = None) -> int:
    """Build one benchmark, or list the registry."""
    parser = argparse.ArgumentParser(
        prog="metametro benchbuild",
        description="Download or construct a benchmark and write CFA, CDBG, and CGT.",
    )
    parser.add_argument("name", nargs="?", help="benchmark name")
    parser.add_argument("--all", action="store_true", help="build every registered benchmark")
    parser.add_argument("--list", action="store_true", help="list benchmarks and exit")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="with --all, also download and assemble community and external benchmarks",
    )
    parser.add_argument("--outdir", type=Path, default=None, help="output directory (default: data/bench/...)")
    parser.add_argument(
        "--colouring",
        action="append",
        default=None,
        help="colouring to apply; repeat for several. Default: every colouring that can run",
    )
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="write the pin and abundance table, and do not download or assemble",
    )
    parser.add_argument("--gtfs", type=Path, default=None, help="GTFS zip for spb_ground_transit")
    args = parser.parse_args(argv)
    if args.all and args.name:
        parser.error("--all does not take a benchmark name")
    if args.all and args.outdir is not None:
        parser.error("--all writes each benchmark under data/bench")
    if args.all:
        results = build_all(execute=args.execute and not args.contract_only, gtfs=args.gtfs)
        failed = [item for item in results if item.status not in {"built", "contract", "present"}]
        for item in results:
            print(f"{item.spec.name}\t{item.status}\t{item.outdir}")
        return 1 if failed else 0
    if args.list or not args.name:
        for spec in list_specs():
            aliases = ""
            if spec.legacy_names:
                aliases = " (was " + ", ".join(spec.legacy_names) + ")"
            print(f"{spec.name}\t{spec.assembler}\t{spec.properties}\t{spec.summary}{aliases}")
        return 0 if args.list or not args.name else 2
    colourings = tuple(args.colouring) if args.colouring else None
    resolve(args.name)
    result = build(
        args.name,
        outdir=args.outdir,
        colourings=colourings,
        execute=not args.contract_only,
        gtfs=args.gtfs,
    )
    print(f"bench\t{result.spec.name}")
    print(f"status\t{result.status}")
    print(f"outdir\t{result.outdir}")
    print(f"colourings\t{','.join(result.colourings)}")
    print(f"identity\t{result.identity}")
    return 0
