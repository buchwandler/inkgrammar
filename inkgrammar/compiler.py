from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

from .fitting.family_fit import load_fit_reports, primitive_path
from .fitting.models import (
    GlyphEndpoint,
    GlyphVariantRecipe,
    PrimitiveInstance,
    PrimitiveTemplate,
    TopologyTransform,
)
from .fitting.workspace import FitProjectBundle
from .geometry import analyze_path, path_without_initial_move, sample_path, translate_path
from .models import (
    ConnectionClass as RuntimeConnectionClass,
)
from .models import (
    ConnectionRule,
    GlyphForm,
    GlyphModel,
    Metrics,
    PathStrokeRecipe,
    Provenance,
    RawPathPart,
    RuntimeAnchor,
    RuntimeConnectionRule,
    RuntimeContextRule,
    RuntimeGlyph,
    RuntimeGlyphVariant,
    WritingModelV2,
    WritingModelV3,
)


def glyph_units_to_runtime(xy, *, x_height_units: float) -> np.ndarray:
    """Map x-height-normalized glyph units to runtime/font units.

    The conversion is a named, explicit step: glyph units never implicitly
    mix with runtime metrics.
    """
    if x_height_units <= 0:
        raise ValueError("x_height_units must be positive")
    return np.asarray(xy, dtype=float) * x_height_units


def runtime_units_to_glyph(xy, *, x_height_units: float) -> np.ndarray:
    """Map runtime/font units back to x-height-normalized glyph units."""
    if x_height_units <= 0:
        raise ValueError("x_height_units must be positive")
    return np.asarray(xy, dtype=float) / x_height_units


@dataclass(frozen=True)
class CompileResult:
    model: WritingModelV2 | WritingModelV3
    diagnostics: dict[str, Any]


def _occurrence_transform(instance: PrimitiveInstance) -> TopologyTransform:
    return TopologyTransform(
        scale_x=instance.scale_x,
        scale_y=instance.scale_y,
        shear_x=instance.shear_x,
        dx=instance.dx,
        dy=instance.dy,
    )


def _compile_strokes(strokes, templates: dict[str, PrimitiveTemplate], instances=None):
    compiled: list[PathStrokeRecipe] = []
    instance_paths: dict[str, str] = {}
    source_paths: list[str] = []

    for stroke in strokes:
        current_parts: list[str] = []
        current_end: tuple[float, float] | None = None
        part_number = 0

        def finish(stroke_name: str = stroke.id) -> None:
            nonlocal current_parts, part_number
            if current_parts:
                stroke_id = stroke_name if part_number == 0 else f"{stroke_name}.{part_number}"
                compiled_path = current_parts[0] + "".join(f" {part}" for part in current_parts[1:])
                compiled.append(
                    PathStrokeRecipe(id=stroke_id, parts=[RawPathPart(d=compiled_path)])
                )
                source_paths.append(compiled_path)
                part_number += 1
                current_parts = []

        for occurrence in stroke.parts:
            template = templates[occurrence.primitive]
            transform = (
                occurrence.transform
                if instances is None
                else _occurrence_transform(instances[occurrence.instance])
            )
            path = primitive_path(template, transform)
            info = analyze_path(path)
            if current_parts and not occurrence.attach:
                finish()
                current_end = None
            if current_parts:
                assert current_end is not None
                path = translate_path(
                    path, current_end[0] - info.start[0], current_end[1] - info.start[1]
                )
            current_parts.append(path if not current_parts else path_without_initial_move(path))
            instance_paths[occurrence.instance] = path
            current_end = analyze_path(path).end
        finish()

    return compiled, instance_paths, source_paths


def _compile_anchor(
    endpoint: GlyphEndpoint | None,
    instance_paths: dict[str, str],
    width: float,
) -> RuntimeAnchor | None:
    if endpoint is None:
        return None
    try:
        info = analyze_path(instance_paths[endpoint.instance])
    except KeyError as exc:
        raise ValueError(
            f"endpoint references unmaterialized instance: {endpoint.instance}"
        ) from exc
    if endpoint.endpoint == "start":
        point, tangent = info.start, info.tangent_start
    else:
        point, tangent = info.end, info.tangent_end
    if tangent is None:
        raise ValueError(f"endpoint has no tangent: {endpoint.instance}")
    direction = 1.0 if endpoint.direction == "forward" else -1.0
    return RuntimeAnchor(
        x=point[0],
        y=point[1],
        tangent_x=tangent[0] * direction,
        tangent_y=tangent[1] * direction,
        **{"class": endpoint.class_},
        width=width,
    )


def _compile_variant(
    variant: GlyphVariantRecipe,
    templates: dict[str, PrimitiveTemplate],
    instances: dict[str, PrimitiveInstance],
    stroke_width: float,
) -> tuple[RuntimeGlyphVariant, list[str]]:
    strokes, instance_paths, source_paths = _compile_strokes(
        variant.strokes, templates, instances=instances
    )
    runtime = RuntimeGlyphVariant(
        id=variant.id,
        character=variant.character,
        advance_width=variant.advance_width,
        strokes=strokes,
        entry=_compile_anchor(variant.entry, instance_paths, stroke_width),
        exit=_compile_anchor(variant.exit, instance_paths, stroke_width),
    )
    return runtime, source_paths


def _compile_topology_glyph(glyph, templates: dict[str, PrimitiveTemplate]):
    strokes, _, source_paths = _compile_strokes(glyph.strokes, templates)
    variant_id = f"{glyph.character}.default"
    variant = RuntimeGlyphVariant(
        id=variant_id,
        character=glyph.character,
        advance_width=glyph.advance_width,
        strokes=strokes,
    )
    return RuntimeGlyph(
        character=glyph.character,
        unicode=[f"U+{ord(char):04X}" for char in glyph.character],
        default_variant=variant_id,
        variants={variant_id: variant},
        provenance={"topology_review_status": glyph.review_status},
    ), source_paths


def _geometry_parity(source_paths: list[str], model: WritingModelV3) -> float:
    runtime_paths = [
        part.d
        for glyph in model.glyphs.values()
        for variant in glyph.variants.values()
        for stroke in variant.strokes
        for part in stroke.parts
        if isinstance(part, RawPathPart)
    ]
    if len(source_paths) != len(runtime_paths):
        raise ValueError("compiled path count does not match source geometry")
    maximum = 0.0
    for source, runtime in zip(source_paths, runtime_paths, strict=True):
        source_points = sample_path(source, 64)
        runtime_points = sample_path(runtime, 64)
        if len(source_points) != len(runtime_points):
            raise ValueError("compiled path sampling does not match source geometry")
        for source_point, runtime_point in zip(source_points, runtime_points, strict=True):
            maximum = max(
                maximum,
                (
                    (source_point[0] - runtime_point[0]) ** 2
                    + (source_point[1] - runtime_point[1]) ** 2
                )
                ** 0.5,
            )
    if maximum > 1e-6:
        raise ValueError(f"compiled geometry parity exceeded tolerance: {maximum}")
    return maximum


def _as_v2(model: WritingModelV3) -> WritingModelV2:
    if model.context_rules:
        raise ValueError("runtime schema v2 cannot represent contextual rules")
    glyphs = {}
    for character, glyph in model.glyphs.items():
        if len(glyph.variants) != 1 or glyph.default_variant not in glyph.variants:
            raise ValueError("runtime schema v2 can represent only one default variant per glyph")
        variant = glyph.variants[glyph.default_variant]
        glyphs[character] = GlyphModel(
            character=character,
            unicode=glyph.unicode,
            forms={
                "default": GlyphForm(
                    advance_width=variant.advance_width,
                    strokes=variant.strokes,
                )
            },
            provenance=Provenance(status="inferred"),
        )
    mode_map = {
        "cursive-anchor": "direct",
        "direct": "direct",
        "attached": "attached",
        "none": "none",
        "synthetic-bridge": "continuous-bridge",
    }
    connection_rules = []
    for rule in model.connection_rules:
        if rule.mode == "ligature" or rule.required_variant_left or rule.required_variant_right:
            raise ValueError("runtime schema v2 cannot represent contextual connection rules")
        mode = mode_map[rule.mode]
        connection_rules.append(
            ConnectionRule(
                id=rule.id,
                from_class=rule.from_class,
                to_class=rule.to_class,
                gap=rule.pair_spacing,
                tension=60,
                continuous=mode not in {"attached", "none"},
                mode=mode,
            )
        )
    return WritingModelV2(
        schema_version=2,
        id=model.id,
        name=model.name,
        script=model.script,
        languages=model.languages,
        regions=model.regions,
        status=model.status,
        license=model.license,
        description=model.description,
        metrics=model.metrics,
        sources=model.sources,
        connection_rules=connection_rules,
        glyphs=glyphs,
    )


def _select_fit_templates(
    bundle: FitProjectBundle, *, require_promoted: bool
) -> tuple[dict[str, PrimitiveTemplate], dict[str, str]]:
    reports = {report["primitive_id"]: report for report in load_fit_reports(bundle)}
    templates: dict[str, PrimitiveTemplate] = {}
    promotion: dict[str, str] = {}
    referenced = {
        part.primitive
        for glyph in bundle.topology.glyphs.values()
        for stroke in glyph.strokes
        for part in stroke.parts
    }
    referenced.update(
        occurrence.primitive
        for glyph in bundle.grammar.glyphs.values()
        for variant in glyph.variants.values()
        for stroke in variant.strokes
        for occurrence in stroke.parts
    )

    for primitive_id in sorted(referenced):
        report = reports.get(primitive_id)
        status = report.get("visual_review", {}).get("status") if report else None
        if status == "promoted":
            templates[primitive_id] = PrimitiveTemplate.model_validate(report["primitive"])
            promotion[primitive_id] = "promoted"
        elif require_promoted:
            if report is None:
                raise ValueError(f"missing fit report: {primitive_id}")
            raise ValueError(f"primitive not promoted: {primitive_id}")
        else:
            templates[primitive_id] = bundle.grammar.primitives[primitive_id]
            promotion[primitive_id] = "seed-fallback"
    return templates, promotion


def _compile_fit_model(
    bundle: FitProjectBundle,
    templates: dict[str, PrimitiveTemplate],
    promotion: dict[str, str],
) -> CompileResult:
    glyphs: dict[str, RuntimeGlyph] = {}
    source_paths: list[str] = []
    for character, topology_glyph in bundle.topology.glyphs.items():
        if character in bundle.grammar.glyphs:
            continue
        glyph, paths = _compile_topology_glyph(topology_glyph, templates)
        glyphs[character] = glyph
        source_paths.extend(paths)

    for character, recipe in bundle.grammar.glyphs.items():
        variants: dict[str, RuntimeGlyphVariant] = {}
        for variant_id, variant_recipe in recipe.variants.items():
            variant, paths = _compile_variant(
                variant_recipe,
                templates,
                bundle.grammar.instances,
                bundle.grammar.ductus.stroke_width,
            )
            variants[variant_id] = variant
            source_paths.extend(paths)
        glyphs[character] = RuntimeGlyph(
            character=character,
            unicode=[f"U+{ord(char):04X}" for char in character],
            default_variant=recipe.default_variant,
            variants=variants,
            provenance={"source": "script-grammar"},
        )

    ductus = bundle.grammar.ductus
    model = WritingModelV3(
        schema_version=3,
        id=bundle.project.runtime_model_id,
        name={"en": bundle.project.runtime_model_id},
        script="Latn",
        status="experimental",
        license="NOASSERTION",
        metrics=Metrics(
            units_per_em=ductus.units_per_em,
            baseline=ductus.baseline,
            x_height=ductus.x_height,
            cap_height=ductus.cap_height,
            ascender=ductus.ascender,
            descender=ductus.descender,
            slant_degrees=ductus.slant_degrees,
            default_stroke_width=ductus.stroke_width,
        ),
        glyphs=glyphs,
        connection_classes={
            key: RuntimeConnectionClass.model_validate(value.model_dump())
            for key, value in bundle.grammar.connection_classes.items()
        },
        connection_rules=[
            RuntimeConnectionRule.model_validate(rule.model_dump())
            for rule in bundle.grammar.connection_rules
        ],
        context_rules=[
            RuntimeContextRule.model_validate(rule.model_dump())
            for rule in bundle.grammar.context_rules
        ],
        provenance={
            "fit_project_id": bundle.project.id,
            "primitive_promotion": promotion,
            "occurrence_deltas_compiled": False,
        },
    )
    max_deviation = _geometry_parity(source_paths, model)
    diagnostics = {
        "glyph_count": len(model.glyphs),
        "variant_count": sum(len(glyph.variants) for glyph in model.glyphs.values()),
        "promoted_primitives": sorted(
            primitive_id for primitive_id, status in promotion.items() if status == "promoted"
        ),
        "seed_fallback_primitives": sorted(
            primitive_id for primitive_id, status in promotion.items() if status != "promoted"
        ),
        "maximum_centerline_deviation": max_deviation,
        "font_ready": False,
        "font_readiness_blockers": [
            "font structural validation and OpenType shaping parity have not run",
            *[
                f"synthetic bridge rule is not font-compatible: {rule.id}"
                for rule in model.connection_rules
                if rule.mode == "synthetic-bridge"
            ],
        ],
    }
    compiled_model = _as_v2(model) if bundle.project.compile.runtime_schema_version == 2 else model
    diagnostics["schema_version"] = compiled_model.schema_version
    return CompileResult(model=compiled_model, diagnostics=diagnostics)


def compile_fit_project(
    bundle: FitProjectBundle,
    *,
    require_promoted: bool = True,
) -> CompileResult:
    """Compile promoted fit geometry and topology into a source-free runtime model."""
    templates, promotion = _select_fit_templates(bundle, require_promoted=require_promoted)
    result = _compile_fit_model(bundle, templates, promotion)
    compiled_model = result.model
    diagnostics = dict(result.diagnostics)
    output = (bundle.project_dir / bundle.project.compile.output).resolve()
    if output.is_symlink():
        raise ValueError("compiled runtime output must not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            compiled_model.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    if compiled_model.schema_version == 2:
        reloaded = WritingModelV2.model_validate(payload)
    else:
        reloaded = WritingModelV3.model_validate(payload)
    if reloaded != compiled_model:
        raise ValueError("serialized runtime model differs from compiled model")
    diagnostics["output"] = str(output)
    return CompileResult(model=compiled_model, diagnostics=diagnostics)
