"""CGT format package."""

from metametro.formats.cgt.io import dump_cgt, load_cgt
from metametro.formats.cgt.model import Cgt
from metametro.formats.cgt.validator import validate_cgt

__all__ = ["Cgt", "dump_cgt", "load_cgt", "validate_cgt"]
