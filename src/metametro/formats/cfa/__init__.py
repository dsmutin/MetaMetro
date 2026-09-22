"""CFA format: canonical semantic graph."""

from metametro.formats.cfa.io import dump_cfa, load_cfa
from metametro.formats.cfa.model import CfaGraph
from metametro.formats.cfa.validator import validate_cfa

__all__ = ["CfaGraph", "dump_cfa", "load_cfa", "validate_cfa"]
