from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .geometry import analyze_path
from .model import compile_glyph
from .models import WritingModel


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str
    model: str | None = None
    glyph: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QCReport:
    issues: tuple[Issue, ...]
    glyph_count: int
    component_count: int

    @property
    def errors(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warnings(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "glyphs": self.glyph_count,
                "components": self.component_count,
                "errors": self.errors,
                "warnings": self.warnings,
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _unicode_sequence(text: str) -> list[str]:
    return [f"U+{ord(ch):04X}" for ch in text]


def validate_model(model: WritingModel) -> QCReport:
    issues: list[Issue] = []
    source_ids = {source.id for source in model.sources}

    if not (
        model.metrics.descender
        < model.metrics.baseline
        < model.metrics.x_height
        <= model.metrics.ascender
    ):
        issues.append(
            Issue(
                "error",
                "metrics.vertical-order",
                "expected descender < baseline < x_height <= ascender",
                model.id,
            )
        )

    for name, component in model.components.items():
        try:
            info = analyze_path(component.d)
        except Exception as exc:
            issues.append(
                Issue(
                    "error",
                    "component.invalid-path",
                    f"{name}: {exc}",
                    model.id,
                )
            )
            continue
        if info.move_count != 1 or info.closed:
            issues.append(
                Issue(
                    "error",
                    "component.not-open-trajectory",
                    f"{name}: components must contain one open trajectory",
                    model.id,
                )
            )
        unknown_sources = set(component.provenance.source_ids) - source_ids
        if unknown_sources:
            issues.append(
                Issue(
                    "error",
                    "component.unknown-source",
                    f"{name}: unknown source IDs {sorted(unknown_sources)}",
                    model.id,
                )
            )

    for character, glyph in model.glyphs.items():
        if glyph.character != character:
            issues.append(
                Issue(
                    "error",
                    "glyph.key-mismatch",
                    f"mapping key {character!r} != glyph.character {glyph.character!r}",
                    model.id,
                    character,
                )
            )
        if glyph.unicode != _unicode_sequence(glyph.character):
            issues.append(
                Issue(
                    "error",
                    "glyph.unicode-mismatch",
                    f"expected {_unicode_sequence(glyph.character)!r}",
                    model.id,
                    character,
                )
            )

        unknown_sources = set(glyph.provenance.source_ids) - source_ids
        if unknown_sources:
            issues.append(
                Issue(
                    "error",
                    "glyph.unknown-source",
                    f"unknown source IDs {sorted(unknown_sources)}",
                    model.id,
                    character,
                )
            )

        if glyph.provenance.status in {"source-backed", "reviewed", "verified"}:
            if not glyph.provenance.source_ids:
                issues.append(
                    Issue(
                        "error",
                        "glyph.missing-source",
                        f"{glyph.provenance.status} glyph requires a source",
                        model.id,
                        character,
                    )
                )

        for form_name in glyph.forms:
            try:
                compiled = compile_glyph(model, character, form_name)
            except Exception as exc:
                issues.append(
                    Issue(
                        "error",
                        "glyph.compile",
                        f"{form_name}: {exc}",
                        model.id,
                        character,
                    )
                )
                continue

            stroke_ids = {stroke.id for stroke in compiled.strokes}
            for side, endpoint in (
                ("entry", compiled.entry),
                ("exit", compiled.exit),
            ):
                if endpoint and endpoint[0] not in stroke_ids:
                    issues.append(
                        Issue(
                            "error",
                            "glyph.connection-stroke",
                            f"{form_name}/{side}: unknown stroke {endpoint[0]!r}",
                            model.id,
                            character,
                        )
                    )

    rules = {(rule.from_class, rule.to_class) for rule in model.connection_rules}
    if len(rules) != len(model.connection_rules):
        issues.append(
            Issue(
                "error",
                "connection.duplicate-rule",
                "connection rule classes must be unique",
                model.id,
            )
        )

    return QCReport(
        issues=tuple(issues),
        glyph_count=len(model.glyphs),
        component_count=len(model.components),
    )
