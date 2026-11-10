from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .geometry import analyze_path
from .model import compile_glyph
from .models import ComponentPart, PathStrokeRecipe, TapStrokeRecipe, WritingModel

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

    if len(source_ids) != len(model.sources):
        issues.append(Issue("error", "source.duplicate-id", "source IDs must be unique", model.id))
    if model.id == "sas-1968":
        from .coverage import SAS_SYMBOLS
        missing = set(SAS_SYMBOLS) - set(model.glyphs)
        extra = set(model.glyphs) - set(SAS_SYMBOLS)
        if missing or extra:
            issues.append(Issue("error", "sas.glyph-coverage",
                                f"missing: {sorted(missing)}; unexpected: {sorted(extra)}",
                                model.id))
    used_components = {part.component for glyph in model.glyphs.values()
                       for form in glyph.forms.values() for stroke in form.strokes
                       if isinstance(stroke, PathStrokeRecipe) for part in stroke.parts
                       if isinstance(part, ComponentPart)}
    for name, component in model.components.items():
        if name not in used_components and not component.unused_reason:
            issues.append(Issue("warning", "component.unused",
                                f"{name}: unused component needs an explicit unused_reason",
                                model.id))
    for character, glyph in model.glyphs.items():
        for name in glyph.required_forms:
            if name not in glyph.forms:
                issues.append(Issue("error", "glyph.required-form",
                                    f"required contextual form {name!r} is missing",
                                    model.id, character))
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
        if (component.provenance.status in {"source-backed", "reviewed", "verified"}
                and not component.provenance.source_ids):
            issues.append(Issue("error", "component.missing-source",
                                f"{name}: evidence-backed component requires a source", model.id))

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

        if (
            glyph.provenance.status in {"source-backed", "reviewed", "verified"}
            and not glyph.provenance.source_ids
        ):
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

    entry_classes = {form.connections.entry.class_ for glyph in model.glyphs.values()
                     for form in glyph.forms.values() if form.connections.entry}
    exit_classes = {form.connections.exit.class_ for glyph in model.glyphs.values()
                    for form in glyph.forms.values() if form.connections.exit}
    for class_ in sorted(exit_classes - {rule.from_class for rule in model.connection_rules}):
        issues.append(Issue("error", "connection.missing-exit-rule",
                            f"no outgoing rule for class {class_!r}", model.id))
    for class_ in sorted(entry_classes - {rule.to_class for rule in model.connection_rules}):
        issues.append(Issue("error", "connection.missing-entry-rule",
                            f"no incoming rule for class {class_!r}", model.id))

    for pair in model.pair_overrides:
        if len(pair) != 2 or any(character not in model.glyphs for character in pair):
            issues.append(Issue("error", "connection.invalid-pair",
                                f"override {pair!r} must reference two modeled characters",
                                model.id))

    for character, glyph in model.glyphs.items():
        for form_name, form in glyph.forms.items():
            ids = [stroke.id for stroke in form.strokes]
            if len(set(ids)) != len(ids):
                issues.append(Issue("error", "glyph.duplicate-stroke",
                                    f"{form_name}: duplicate stroke id", model.id, character))
            for endpoint in (form.connections.entry, form.connections.exit):
                if endpoint:
                    stroke = next((s for s in form.strokes if s.id == endpoint.stroke), None)
                    if isinstance(stroke, TapStrokeRecipe):
                        issues.append(Issue("error", "glyph.connection-tap",
                                            f"{form_name}: connection references a tap",
                                            model.id, character))

    # Direct rules must meet for every actual contextual pair to which they apply.
    from .model import compile_text

    direct_rules = {(rule.from_class, rule.to_class) for rule in model.connection_rules
                    if rule.effective_mode == "direct"}
    if direct_rules:
        for left in model.glyphs:
            for right in model.glyphs:
                try:
                    compile_text(model, left + right)
                    compile_text(model, left + right + left + right)
                except Exception as exc:
                    issues.append(Issue("error", "connection.direct-endpoints",
                                        f"{left + right!r}: {exc}", model.id))

    return QCReport(
        issues=tuple(issues),
        glyph_count=len(model.glyphs),
        component_count=len(model.components),
    )
