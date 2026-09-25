"""Build the roxel street graph as a coloured CFA.

Sequences are a repeat-junction tape, not a biological measurement. Each edge
attribute is a colour namespace. Coordinates stay as columns, not colours.
"""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from pathlib import Path

from metametro.digraph_cfa import coloured_digraph_cfa
from metametro.errors import ContractError
from metametro.formats.cfa.model import CfaGraph
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
    environment = dict(os.environ)
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
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise ContractError(["R exported roxel but printed no sfnetworks version"])
    return lines[-1]


def build_roxel_cfa(*, k: int = 21) -> CfaGraph:
    """Export roxel with R and return the coloured CFA. Overlap mismatches are an error."""
    if k < 2:
        raise ContractError(["roxel k must be >= 2"])
    with tempfile.TemporaryDirectory() as temporary:
        version = export_roxel(Path(temporary))
        with (Path(temporary) / "nodes.csv").open(newline="", encoding="utf-8") as handle:
            nodes = list(csv.DictReader(handle))
        with (Path(temporary) / "edges.csv").open(newline="", encoding="utf-8") as handle:
            edges = list(csv.DictReader(handle))
    if not edges:
        raise ContractError(["roxel export has no edges"])
    colour_columns = [
        column
        for column in edges[0]
        if column not in {"source", "target", "from", "to", "longitude", "latitude"}
    ]
    if not colour_columns:
        raise ContractError(["roxel edges have no attribute columns to colour"])
    graph = coloured_digraph_cfa(
        nodes,
        edges,
        graph_id="roxel",
        k=k,
        colour_columns=colour_columns,
        keep_columns=("longitude", "latitude"),
    )
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
        raise ContractError([f"overlap mismatch on {len(mismatches)} roxel edges"])
    return graph
