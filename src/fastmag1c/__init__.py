"""MAG1C methane retrieval with an exact rank-2 covariance update."""

from .compare import Comparison, compare
from .emit import orthorectify, read_emit
from .scene import retrieve
from .template import load_template

__version__ = "0.1.0"

__all__ = [
    "Comparison",
    "compare",
    "load_template",
    "orthorectify",
    "read_emit",
    "retrieve",
]
