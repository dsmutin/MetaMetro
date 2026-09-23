"""CDBG format package."""

from metametro.formats.cdbg.annotations import (
    AnnotationLayer,
    AnnotationProvenance,
    aggregate_annotations,
    annotate_cdbg,
    get_edge_annotations,
    get_node_annotations,
    transfer_annotations,
)
from metametro.formats.cdbg.io import dump_cdbg, load_cdbg
from metametro.formats.cdbg.model import Cdbg, Link, NodeMap, Unitig
from metametro.formats.cdbg.validator import validate_cdbg

__all__ = [
    "AnnotationLayer",
    "AnnotationProvenance",
    "Cdbg",
    "Link",
    "NodeMap",
    "Unitig",
    "aggregate_annotations",
    "annotate_cdbg",
    "dump_cdbg",
    "get_edge_annotations",
    "get_node_annotations",
    "load_cdbg",
    "transfer_annotations",
    "validate_cdbg",
]
