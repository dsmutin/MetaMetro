"""Build a CFA from the sfnetworks roxel street graph.

The graph is ``sfnetworks::as_sfnetwork(sfnetworks::roxel)``. Generation lives
in MetaMetro ``benchbuild roxel``. This script only writes one CFA directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from metametro.bench.data.universal.roxel import build_roxel_cfa
from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
from metametro.errors import ContractError
from metametro.formats.cfa.io import dump_cfa


def main(argv: list[str] | None = None) -> int:
    """Write the roxel CFA and compact it once to check overlaps."""
    parser = argparse.ArgumentParser(description="Build a CFA from sfnetworks::roxel.")
    parser.add_argument("--out", type=Path, default=Path("data/work/roxel/cfa"))
    parser.add_argument("--k", type=int, default=21)
    args = parser.parse_args(argv)
    try:
        graph = build_roxel_cfa(k=args.k)
        compacted = cfa_to_cdbg(graph)
    except (ContractError, OSError, UnicodeDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    dump_cfa(graph, args.out)
    print(f"nodes {len(graph.nodes)}")
    print(f"edges {len(graph.edges)}")
    print(f"colors {len(graph.colors or [])}")
    print(f"unitigs {len(compacted.unitigs)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
