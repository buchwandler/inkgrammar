"""ScriptStrokes public Python API."""

from .io import load_model
from .model import compile_glyph, compile_text
from .models import WritingModel

try:
    from ._version import __version__
except ImportError:  # source tree before setuptools_scm has generated _version.py
    __version__ = "0+unknown"

__all__ = [
    "WritingModel",
    "__version__",
    "compile_glyph",
    "compile_text",
    "load_model",
]
