"""Pipeline contracts."""

from metametro.contracts.assembly import (
    contig_overlap_graph,
    contigs_to_dbg,
    dbg_from_sequences,
    directed_bubble_sources,
    fastg_to_cfa,
    gfa_to_cfa,
    load_genomes,
    read_fastq,
    simulate_metagenome,
)
from metametro.contracts.colouring import colour_by_reads, colour_cfa, combine_colors
from metametro.contracts.ds import run_ds
from metametro.contracts.external import contig2fastg_command, megahit_command, samovar_generate_command
from metametro.contracts.versions import CONTRACTS

__all__ = [
    "CONTRACTS",
    "colour_by_reads",
    "colour_cfa",
    "combine_colors",
    "contigs_to_dbg",
    "dbg_from_sequences",
    "load_genomes",
    "megahit_command",
    "read_fastq",
    "run_ds",
    "samovar_generate_command",
    "simulate_metagenome",
    "contig2fastg_command",
    "contig_overlap_graph",
    "directed_bubble_sources",
    "fastg_to_cfa",
    "gfa_to_cfa",
]
