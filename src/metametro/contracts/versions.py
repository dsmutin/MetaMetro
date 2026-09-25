"""Contract version pins. Each contract is versioned independently of VERSION."""

from __future__ import annotations

CONTRACTS: dict[str, str] = {
    "genome_to_metagenome": "1.0",
    "metagenome_to_graph": "1.0",
    "graph_to_cfa": "1.0",
    "colour_cfa": "1.0",
    "cfa_to_cdbg": "1.0",
    "cdbg_to_cgt": "1.0",
    "filter_colours": "1.0",
    "ds_on_cgt": "1.0",
}
