"""InkGrammar public Python API."""

from .comparison.api import compare_extracted_letters, compare_letter_images
from .comparison.models import ComparisonIdentity
from .io import load_model
from .model import compile_glyph, compile_text
from .models import WritingModel

try:
    from ._version import __version__
except ImportError:  # source tree before setuptools_scm has generated _version.py
    __version__ = "0+unknown"

__all__ = [
    "WritingModel",
    "ComparisonIdentity",
    "compare_extracted_letters",
    "compare_letter_images",
    "__version__",
    "compile_glyph",
    "compile_text",
    "load_model",
]
