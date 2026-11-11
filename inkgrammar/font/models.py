from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class FontBuildConfig:
    family_name: str | None = None
    style_name: str = "Regular"
    quadratic_error: float = 0.5
    reproducible: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.quadratic_error) or self.quadratic_error <= 0:
            raise ValueError("quadratic_error must be positive")
        if not self.style_name.strip():
            raise ValueError("style_name must not be empty")


@dataclass(frozen=True)
class FontGlyphOutline:
    contours: tuple[tuple[tuple[float, float], ...], ...]
    bounds: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class FontPlan:
    glyph_order: tuple[str, ...]
    variant_names: dict[tuple[str, str], str]
    cmap: dict[int, str]
    horizontal_metrics: dict[str, tuple[int, int]]
    outlines: dict[str, FontGlyphOutline]
    feature_source: str
    issues: tuple[str, ...]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class FontBuildResult:
    path: Path
    glyph_count: int
    tables: tuple[str, ...]
    feature_source_path: Path
    report_path: Path
    font_ready: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class FontValidationResult:
    valid: bool
    glyph_count: int
    tables: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
