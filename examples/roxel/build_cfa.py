"""Build a CFA from the sfnetworks roxel street graph.

The graph is ``sfnetworks::as_sfnetwork(sfnetworks::roxel)``. Every character
attribute becomes a colour namespace. Node sequences are the same repeat-junction
tape used for the transit CFA.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
from metametro.digraph_cfa import coloured_digraph_cfa
from metametro.errors import ContractError
from metametro.formats.cfa.io import dump_cfa
from metametro.transit_cfa import overlap_mismatches

_EXPORT = r"""
suppressPackageStartupMessages({
  library(sf)
  library(sfnetworks)
  library(tidygraph)
})
gr <- as_sfnetwork(roxel)
nodes <- sf::st_as_sf(gr, "nodes")
edges <- sf::st_as_sf(gr, "edges")
nodes$longitude <- sf::st_coordinates(nodes)[, 1]
nodes$latitude <- sf::st_coordinates(nodes)[, 2]
edges <- sf::st_drop_geometry(edges)
nodes <- sf::st_drop_geometry(nodes)
nodes$node_id <- sprintf("n%06d", seq_len(nrow(nodes)))
from_id <- nodes$node_id[edges$from]
to_id <- nodes$node_id[edges$to]
edges$source <- from_id
edges$target <- to_id
edges$from <- NULL
edges$to <- NULL
utils::write.csv(nodes, file.path(Sys.getenv("ROXEL_OUT"), "nodes.csv"), row.names = FALSE)
utils::write.csv(edges, file.path(Sys.getenv("ROXEL_OUT"), "edges.csv"), row.names = FALSE)
cat(paste(packageVersion("sfnetworks")), "\n")
"""


def export_roxel(directory: Path) -> str:
    """Ask R for the roxel nodes and edges. Return the sfnetworks version."""
    directory.mkdir(parents=True, exist_ok=True)
    environment = dict(**{key: value for key, value in __import__("os").environ.items()})
    environment["ROXEL_OUT"] = str(directory)
    completed = subprocess.run(
        ["Rscript", "-e", _EXPORT],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise ContractError([completed.stderr.strip() or "R failed to export roxel"])
    return completed.stdout.strip().splitlines()[-1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    """Write the roxel CFA and compact it once to check overlaps."""
    parser = argparse.ArgumentParser(description="Build a CFA from sfnetworks::roxel.")
    parser.add_argument("--out", type=Path, default=Path("data/work/roxel/cfa"))
    parser.add_argument("--k", type=int, default=21)
    args = parser.parse_args(argv)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            version = export_roxel(Path(temporary))
            nodes = _read_rows(Path(temporary) / "nodes.csv")
            edges = _read_rows(Path(temporary) / "edges.csv")
        colour_columns = [
            column
            for column in edges[0]
            if column not in {"source", "target", "from", "to"}
        ]
        for column in ("longitude", "latitude"):
            if column in nodes[0] and column not in colour_columns:
                pass
        # Coordinates are positions, not colour classes. Colour the street attributes.
        colour_columns = [column for column in colour_columns if column not in {"longitude", "latitude"}]
        if not colour_columns:
            raise ContractError(["roxel edges have no attribute columns to colour"])
        graph = coloured_digraph_cfa(
            nodes,
            edges,
            graph_id="roxel",
            k=args.k,
            colour_columns=colour_columns,
            keep_columns=("longitude", "latitude"),
        )
    except (ContractError, OSError, UnicodeDecodeError, subprocess.SubprocessError) as exc:
        print(exc, file=sys.stderr)
        return 1
    graph.metadata["source"] = {
        "graph": "sfnetworks::as_sfnetwork(sfnetworks::roxel)",
        "sfnetworks_version": version,
        "note": (
            "Node sequences are a repeat-junction tape, not a biological measurement. "
            "Each edge attribute is a colour namespace."
        ),
    }
    mismatches = overlap_mismatches(graph)
    if mismatches:
        print(f"overlap mismatch on {len(mismatches)} edges", file=sys.stderr)
        return 1
    compacted = cfa_to_cdbg(graph)
    dump_cfa(graph, args.out)
    print(f"sfnetworks {version}")
    print(f"nodes {len(graph.nodes)}")
    print(f"edges {len(graph.edges)}")
    print(f"colors {len(graph.colors or [])}")
    print(f"namespaces {sorted({row['namespace'] for row in graph.colors or []})}")
    print(f"unitigs {len(compacted.unitigs)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
