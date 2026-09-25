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
from metametro.contracts.classifiers import (
    colour_from_calls,
    colour_from_kaiju_file,
    colour_from_kraken2_file,
    parse_kaiju_output,
    parse_kraken2_output,
)
from metametro.contracts.colour_filter import filter_colours
from metametro.contracts.colouring import (
    accession_from_read_id,
    colour_by_read_accessions,
    colour_by_reads,
    colour_cfa,
    combine_colors,
    paint_namespace,
)
from metametro.contracts.composition import colour_by_composition, kmeans_onehot
from metametro.contracts.decaying import colour_decaying, decaying_distributions
from metametro.contracts.ds import run_ds
from metametro.contracts.external import contig2fastg_command, megahit_command, samovar_generate_command
from metametro.contracts.versions import CONTRACTS

__all__ = [
    "CONTRACTS",
    "accession_from_read_id",
    "colour_by_composition",
    "colour_by_read_accessions",
    "colour_by_reads",
    "colour_cfa",
    "colour_decaying",
    "colour_from_calls",
    "colour_from_kaiju_file",
    "colour_from_kraken2_file",
    "combine_colors",
    "contigs_to_dbg",
    "dbg_from_sequences",
    "decaying_distributions",
    "kmeans_onehot",
    "load_genomes",
    "megahit_command",
    "paint_namespace",
    "parse_kaiju_output",
    "parse_kraken2_output",
    "read_fastq",
    "run_ds",
    "samovar_generate_command",
    "simulate_metagenome",
    "contig2fastg_command",
    "contig_overlap_graph",
    "directed_bubble_sources",
    "fastg_to_cfa",
    "filter_colours",
    "gfa_to_cfa",
]
