from __future__ import annotations

from collections.abc import Mapping
from math import exp, isfinite
from typing import Literal

from pydantic import Field, model_validator

from ..models import StrictModel

Point = tuple[float, float]


def _path_value(value: object, path: str) -> object:
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current[part]
        elif isinstance(current, (list, tuple)):
            index = {"x": 0, "y": 1}.get(part)
            current = current[int(part) if index is None else index]
        else:
            raise KeyError(path)
    return current


class DuctusModel(StrictModel):
    units_per_em: int = Field(gt=0, le=16384)
    baseline: float = 0.0
    x_height: float
    cap_height: float
    ascender: float
    descender: float
    slant_degrees: float = Field(ge=-45, le=45)
    stroke_width: float = Field(gt=0)

    instance_scale_sigma: float = Field(default=0.05, gt=0)
    instance_shear_sigma: float = Field(default=0.03, gt=0)
    instance_translation_sigma: float = Field(default=12.0, gt=0)


class CubicSegment(StrictModel):
    c1: Point
    c2: Point
    end: Point


class SplineEndpointPolicy(StrictModel):
    start: Literal["locked", "free", "anchored"] = "locked"
    end: Literal["locked", "free", "anchored"] = "locked"


class SplineWidthModel(StrictModel):
    kind: Literal["constant", "bspline"] = "constant"
    value: float | None = Field(default=None, gt=0)
    degree: Literal[2] | None = None
    knots: list[float] | None = None
    coefficients: list[float] | None = None

    @model_validator(mode="after")
    def validate_width(self) -> SplineWidthModel:
        if self.kind == "constant":
            if (
                self.value is None
                or not isfinite(self.value)
                or any(value is not None for value in (self.degree, self.knots, self.coefficients))
            ):
                raise ValueError("constant spline width requires only a positive finite value")
            return self
        if (
            self.value is not None
            or self.degree != 2
            or self.knots is None
            or self.coefficients is None
        ):
            raise ValueError("B-spline width requires degree, knots, and coefficients")
        if len(self.coefficients) < self.degree + 1:
            raise ValueError("B-spline width requires at least degree + 1 coefficients")
        if len(self.knots) != len(self.coefficients) + self.degree + 1:
            raise ValueError("B-spline width knot count does not match coefficient count")
        if not all(isfinite(value) for value in [*self.knots, *self.coefficients]):
            raise ValueError("B-spline width knots and coefficients must be finite")
        if any(value <= 0 for value in self.coefficients):
            raise ValueError("B-spline width coefficients must be positive")
        if self.knots != sorted(self.knots):
            raise ValueError("B-spline width knots must be nondecreasing")
        if self.knots[:3] != [0.0] * 3 or self.knots[-3:] != [1.0] * 3:
            raise ValueError("B-spline width knots must be clamped to [0, 1]")
        interior = self.knots[3:-3]
        if any(not 0.0 < value < 1.0 for value in interior) or len(set(interior)) != len(interior):
            raise ValueError("B-spline width interior knots must be simple and inside (0, 1)")
        return self


class BSplineGeometry(StrictModel):
    degree: Literal[3] = 3
    knots: list[float]
    control_points: list[Point] = Field(min_length=4)
    endpoint_policy: SplineEndpointPolicy = Field(default_factory=SplineEndpointPolicy)
    width: SplineWidthModel | None = None

    @model_validator(mode="after")
    def validate_geometry(self) -> BSplineGeometry:
        if len(self.knots) != len(self.control_points) + self.degree + 1:
            raise ValueError("B-spline knot count must equal control point count + degree + 1")
        if not all(isfinite(value) for value in self.knots):
            raise ValueError("B-spline knots must be finite")
        if not all(isfinite(value) for point in self.control_points for value in point):
            raise ValueError("B-spline control points must be finite")
        if self.knots != sorted(self.knots):
            raise ValueError("B-spline knots must be nondecreasing")
        if self.knots[:4] != [0.0] * 4 or self.knots[-4:] != [1.0] * 4:
            raise ValueError("cubic B-spline knots must be clamped to [0, 1]")
        interior = self.knots[4:-4]
        if any(not 0.0 < value < 1.0 for value in interior) or len(set(interior)) != len(interior):
            raise ValueError("cubic B-spline interior knots must be simple and inside (0, 1)")
        return self


class PrimitiveFrame(StrictModel):
    start: Point
    nominal_end: Point


class PrimitiveProvenance(StrictModel):
    construction: str = "inferred"
    trajectory: str = "inferred"


class PrimitiveTemplate(StrictModel):
    id: str
    kind: Literal["cubic-chain", "bspline-centerline"] = "cubic-chain"
    frame: PrimitiveFrame
    segments: list[CubicSegment] | None = None
    spline: BSplineGeometry | None = None
    free_parameters: list[str] = Field(default_factory=list)
    locked_parameters: list[str] = Field(default_factory=list)
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    provenance: PrimitiveProvenance = Field(default_factory=PrimitiveProvenance)

    @model_validator(mode="after")
    def validate_parameterization(self) -> PrimitiveTemplate:
        if self.kind == "cubic-chain":
            if not self.segments:
                raise ValueError("cubic-chain primitives require at least one segment")
            if self.spline is not None:
                raise ValueError("cubic-chain primitives must not contain spline geometry")
        else:
            if self.spline is None:
                raise ValueError("bspline-centerline primitives require spline geometry")
            if self.segments is not None:
                raise ValueError("bspline-centerline primitives must not contain cubic segments")
            if any(
                path.startswith("spline.")
                for path in [*self.free_parameters, *self.locked_parameters, *self.bounds]
            ):
                raise ValueError("legacy parameterization must not address spline geometry")

        free = self.free_parameters
        locked = self.locked_parameters
        if len(set(free)) != len(free) or len(set(locked)) != len(locked):
            raise ValueError("free and locked parameter names must be unique")
        if set(free) & set(locked):
            raise ValueError("a parameter cannot be both free and locked")
        if set(self.bounds) != set(free):
            raise ValueError("every free parameter must have exactly one bound")

        values = self.model_dump(mode="python")
        for path in [*free, *locked]:
            try:
                value = _path_value(values, path)
            except (KeyError, IndexError, ValueError) as exc:
                raise ValueError(f"unknown primitive parameter path: {path}") from exc
            if path in free:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"free parameter must select a numeric coordinate: {path}")
                lower, upper = self.bounds[path]
                if not lower < upper:
                    raise ValueError(f"parameter bounds must be increasing: {path}")
                if not lower <= value <= upper:
                    raise ValueError(f"initial value is outside bounds: {path}")
        return self


class PrimitiveInstance(StrictModel):
    id: str
    primitive: str
    scale_x: float = Field(default=1.0, gt=0)
    scale_y: float = Field(default=1.0, gt=0)
    shear_x: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    shape: dict[str, float] = Field(default_factory=dict)
    residual: dict[str, float] = Field(default_factory=dict)


class PrimitiveOccurrence(StrictModel):
    instance: str
    primitive: str
    attach: bool = True


class GlyphStroke(StrictModel):
    id: str
    parts: list[PrimitiveOccurrence] = Field(min_length=1)


class GlyphEndpoint(StrictModel):
    instance: str
    endpoint: Literal["start", "end"]
    class_: str = Field(alias="class")
    direction: Literal["forward", "reverse"] = "forward"


class GlyphVariantRecipe(StrictModel):
    id: str
    character: str = Field(min_length=1)
    advance_width: float = Field(gt=0)
    strokes: list[GlyphStroke] = Field(min_length=1)
    entry: GlyphEndpoint | None = None
    exit: GlyphEndpoint | None = None
    provenance: PrimitiveProvenance = Field(default_factory=PrimitiveProvenance)


class GlyphRecipe(StrictModel):
    character: str = Field(min_length=1)
    default_variant: str
    variants: dict[str, GlyphVariantRecipe]

    @model_validator(mode="after")
    def validate_variants(self) -> GlyphRecipe:
        if self.default_variant not in self.variants:
            raise ValueError(f"default variant is missing: {self.default_variant}")
        for key, variant in self.variants.items():
            if key != variant.id:
                raise ValueError(f"variant key does not match id: {key}")
            if variant.character != self.character:
                raise ValueError(f"variant character mismatch: {variant.id}")
        return self


class ConnectionClass(StrictModel):
    id: str
    nominal_height: Literal["baseline", "mid", "upper", "attached", "other"]
    pen_state: Literal["continuous", "lift_before", "lift_after", "detached"]
    notes: str | None = None


class GlyphContext(StrictModel):
    prev_character: str | None = None
    next_character: str | None = None
    prev_exit_class: str | None = None
    next_entry_class: str | None = None
    position: Literal["isolated", "initial", "medial", "final"] | None = None


class ContextRule(StrictModel):
    id: str
    character: str
    variant: str
    prev_characters: list[str] = Field(default_factory=list)
    next_characters: list[str] = Field(default_factory=list)
    prev_exit_classes: list[str] = Field(default_factory=list)
    next_entry_classes: list[str] = Field(default_factory=list)
    positions: list[str] = Field(default_factory=list)
    required: bool = True
    priority: int = 0
    evidence: list[str] = Field(default_factory=list)


class GrammarConnectionRule(StrictModel):
    id: str
    from_class: str
    to_class: str
    mode: Literal[
        "cursive-anchor",
        "direct",
        "attached",
        "ligature",
        "none",
        "synthetic-bridge",
    ]
    x_adjust: float = 0.0
    y_adjust: float = 0.0
    pair_spacing: float = 0.0
    required_variant_left: str | None = None
    required_variant_right: str | None = None
    ligature_variant: str | None = None
    evidence: list[str] = Field(default_factory=list)
    status: Literal["inferred", "reviewed", "verified"] = "inferred"

    @model_validator(mode="after")
    def validate_ligature(self) -> GrammarConnectionRule:
        if self.mode == "ligature" and self.ligature_variant is None:
            raise ValueError("ligature connection rule requires ligature_variant")
        return self


class ConnectionGeometryTarget(StrictModel):
    """Source-free measured geometry for one observed connection boundary."""

    anchor_delta: tuple[float, float]
    exit_tangent: tuple[float, float] | None = None
    entry_tangent: tuple[float, float] | None = None
    exit_width: float | None = Field(default=None, ge=0)
    entry_width: float | None = Field(default=None, ge=0)
    centerline: list[tuple[float, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_geometry(self) -> ConnectionGeometryTarget:
        if self.centerline and len(self.centerline) < 2:
            raise ValueError("connection centerline requires at least two points")
        for tangent in (self.exit_tangent, self.entry_tangent):
            if tangent is not None and tangent == (0.0, 0.0):
                raise ValueError("connection target tangents must be nonzero")
        return self


class ConnectionObservation(StrictModel):
    id: str
    text: str = Field(min_length=2)
    boundary_index: int = Field(ge=1)
    observation_id: str
    split: Literal["fit", "validation", "hold-out"]
    from_character: str = Field(min_length=1)
    to_character: str = Field(min_length=1)
    expected_class: str | None = None
    weight: float = Field(default=1.0, gt=0)
    target_geometry: ConnectionGeometryTarget | None = None

    @model_validator(mode="after")
    def validate_boundary(self) -> ConnectionObservation:
        if self.boundary_index >= len(self.text):
            raise ValueError("connection observation boundary is outside its text")
        if self.text[self.boundary_index - 1] != self.from_character:
            raise ValueError("connection observation from_character does not match boundary")
        if self.text[self.boundary_index] != self.to_character:
            raise ValueError("connection observation to_character does not match boundary")
        return self


class FitAnchors(StrictModel):
    baseline_y: float | None = None
    origin_x: float | None = None
    x_height_y: float | None = None
    ascender_y: float | None = None


class FitLossRegion(StrictModel):
    id: str
    instances: list[str] = Field(min_length=1)
    bbox_source: tuple[float, float, float, float]
    mode: Literal["local-dice", "stroke-track"] = "local-dice"

    @model_validator(mode="after")
    def validate_bbox(self) -> FitLossRegion:
        x0, y0, x1, y1 = self.bbox_source
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError("loss region bbox must have positive dimensions")
        return self


class FitSample(StrictModel):
    id: str
    kind: Literal["glyph", "word"]
    text: str = Field(min_length=1)
    observation_id: str
    split: Literal["fit", "validation", "hold-out"]
    anchors: FitAnchors = Field(default_factory=FitAnchors)
    active_instances: list[str] = Field(default_factory=list)
    loss_regions: list[FitLossRegion] = Field(default_factory=list)
    weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_active_instances(self) -> FitSample:
        if len(set(self.active_instances)) != len(self.active_instances):
            raise ValueError("active instance IDs must be unique")
        if any(not instance.strip() for instance in self.active_instances):
            raise ValueError("active instance IDs must not be empty")
        return self


class FitCorpus(StrictModel):
    format: Literal["inkgrammar-fit-corpus"] = "inkgrammar-fit-corpus"
    format_version: Literal[1] = 1
    samples: list[FitSample] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sample_ids(self) -> FitCorpus:
        ids = [sample.id for sample in self.samples]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate fit sample id")
        return self


class FitCompileConfig(StrictModel):
    output: str
    runtime_schema_version: Literal[2, 3] = 2


class FitProject(StrictModel):
    format: Literal["inkgrammar-fit-project"] = "inkgrammar-fit-project"
    format_version: Literal[1] = 1
    id: str
    runtime_model_id: str
    source_lock: str
    observation_manifest: str
    corpus: str
    families: str
    topology: str
    anchors: str
    fit_config: str
    seed_grammar: str
    private_work_dir: str
    compile: FitCompileConfig


class ScriptGrammar(StrictModel):
    id: str
    ductus: DuctusModel
    primitives: dict[str, PrimitiveTemplate] = Field(default_factory=dict)
    glyphs: dict[str, GlyphRecipe] = Field(default_factory=dict)
    instances: dict[str, PrimitiveInstance] = Field(default_factory=dict)
    connection_classes: dict[str, ConnectionClass] = Field(default_factory=dict)
    connection_rules: list[GrammarConnectionRule] = Field(default_factory=list)
    context_rules: list[ContextRule] = Field(default_factory=list)
    connection_observations: list[ConnectionObservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> ScriptGrammar:
        for key, primitive in self.primitives.items():
            if key != primitive.id:
                raise ValueError(f"primitive key does not match id: {key}")
        for key, instance in self.instances.items():
            if key != instance.id:
                raise ValueError(f"instance key does not match id: {key}")
            if instance.primitive not in self.primitives:
                raise ValueError(f"instance references unknown primitive: {instance.primitive}")
        for key, glyph in self.glyphs.items():
            if key != glyph.character:
                raise ValueError(f"glyph key does not match character: {key}")
            for variant in glyph.variants.values():
                for stroke in variant.strokes:
                    for occurrence in stroke.parts:
                        if occurrence.instance not in self.instances:
                            raise ValueError(
                                f"glyph references unknown instance: {occurrence.instance}"
                            )
                        if self.instances[occurrence.instance].primitive != occurrence.primitive:
                            raise ValueError(
                                f"glyph instance primitive mismatch: {occurrence.instance}"
                            )
                for endpoint in (variant.entry, variant.exit):
                    if endpoint is not None and endpoint.instance not in self.instances:
                        raise ValueError(
                            f"glyph endpoint references unknown instance: {endpoint.instance}"
                        )
                    if endpoint is not None and endpoint.class_ not in self.connection_classes:
                        raise ValueError(
                            f"glyph endpoint references unknown class: {endpoint.class_}"
                        )
        for key, connection_class in self.connection_classes.items():
            if key != connection_class.id:
                raise ValueError(f"connection class key does not match id: {key}")
        variant_ids = {
            variant.id for glyph in self.glyphs.values() for variant in glyph.variants.values()
        }
        context_rule_ids = [rule.id for rule in self.context_rules]
        if len(set(context_rule_ids)) != len(context_rule_ids):
            raise ValueError("duplicate context rule id")
        for rule in self.context_rules:
            if rule.character not in self.glyphs:
                raise ValueError(f"context rule references unknown character: {rule.character}")
            if rule.variant not in self.glyphs[rule.character].variants:
                raise ValueError(f"context rule references unknown variant: {rule.variant}")
            if set(rule.positions) - {"isolated", "initial", "medial", "final"}:
                raise ValueError(f"context rule has an unknown position: {rule.id}")
        connection_rule_ids = [rule.id for rule in self.connection_rules]
        if len(set(connection_rule_ids)) != len(connection_rule_ids):
            raise ValueError("duplicate connection rule id")
        for rule in self.connection_rules:
            if rule.from_class not in self.connection_classes:
                raise ValueError(f"connection rule references unknown class: {rule.from_class}")
            if rule.to_class not in self.connection_classes:
                raise ValueError(f"connection rule references unknown class: {rule.to_class}")
            for variant_id in (
                rule.required_variant_left,
                rule.required_variant_right,
                rule.ligature_variant,
            ):
                if variant_id is not None and variant_id not in variant_ids:
                    raise ValueError(f"connection rule references unknown variant: {variant_id}")
        observation_ids = [item.id for item in self.connection_observations]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("duplicate connection observation id")
        for observation in self.connection_observations:
            if (
                observation.expected_class is not None
                and observation.expected_class not in self.connection_classes
            ):
                raise ValueError(
                    f"connection observation references unknown class: {observation.expected_class}"
                )
        return self


class SourceLockSource(StrictModel):
    id: str
    expected_filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    redistribution: Literal["private-until-reviewed"] = "private-until-reviewed"


class SourceLockFile(StrictModel):
    format: Literal["inkgrammar-source-lock"] = "inkgrammar-source-lock"
    format_version: Literal[1] = 1
    sources: list[SourceLockSource] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_ids(self) -> SourceLockFile:
        ids = [source.id for source in self.sources]
        filenames = [source.expected_filename for source in self.sources]
        if any(
            not filename
            or filename in {".", ".."}
            or "/" in filename
            or chr(92) in filename
            or any(ord(character) < 32 for character in filename)
            for filename in filenames
        ):
            raise ValueError("source filenames must be simple basenames")
        if len(set(ids)) != len(ids) or len(set(filenames)) != len(filenames):
            raise ValueError("source IDs and filenames must be unique")
        return self


class TopologyTransform(StrictModel):
    scale_x: float = Field(default=1.0, gt=0)
    scale_y: float = Field(default=1.0, gt=0)
    shear_x: float = 0.0
    dx: float = 0.0
    dy: float = 0.0


class OccurrenceDelta(StrictModel):
    dx: float = 0.0
    dy: float = 0.0
    log_scale_x: float = 0.0
    log_scale_y: float = 0.0
    shear_x: float = 0.0


class OccurrenceFitBounds(StrictModel):
    dx: tuple[float, float] = (-4.0, 4.0)
    dy: tuple[float, float] = (-4.0, 4.0)
    log_scale_x: tuple[float, float] = (-0.12, 0.12)
    log_scale_y: tuple[float, float] = (-0.12, 0.12)
    shear_x: tuple[float, float] = (-0.08, 0.08)

    @model_validator(mode="after")
    def validate_bounds(self) -> OccurrenceFitBounds:
        for name in ("dx", "dy", "log_scale_x", "log_scale_y", "shear_x"):
            lower, upper = getattr(self, name)
            if not isfinite(lower) or not isfinite(upper) or lower >= upper:
                raise ValueError(f"occurrence bounds must be finite and increasing: {name}")
        return self


class OccurrenceFitState(StrictModel):
    instance_id: str
    nominal_transform: TopologyTransform
    delta: OccurrenceDelta = Field(default_factory=OccurrenceDelta)
    bounds: OccurrenceFitBounds = Field(default_factory=OccurrenceFitBounds)
    free_parameters: list[Literal["dx", "dy", "log_scale_x", "log_scale_y", "shear_x"]] = Field(
        default_factory=lambda: ["dx", "dy", "log_scale_x", "log_scale_y"]
    )

    @model_validator(mode="after")
    def validate_fit_state(self) -> OccurrenceFitState:
        if not self.instance_id:
            raise ValueError("occurrence instance_id must not be empty")
        if len(set(self.free_parameters)) != len(self.free_parameters):
            raise ValueError("occurrence free parameter names must be unique")
        for name in self.free_parameters:
            lower, upper = getattr(self.bounds, name)
            value = getattr(self.delta, name)
            if not lower <= value <= upper:
                raise ValueError(f"occurrence delta is outside bounds: {self.instance_id}.{name}")
        return self


def compose_occurrence_transform(
    nominal: TopologyTransform, delta: OccurrenceDelta
) -> TopologyTransform:
    return TopologyTransform(
        scale_x=nominal.scale_x * exp(delta.log_scale_x),
        scale_y=nominal.scale_y * exp(delta.log_scale_y),
        shear_x=nominal.shear_x + delta.shear_x,
        dx=nominal.dx + delta.dx,
        dy=nominal.dy + delta.dy,
    )


class OccurrenceAlignmentOverride(StrictModel):
    free: list[Literal["dx", "dy", "log_scale_x", "log_scale_y", "shear_x"]] | None = None
    bounds: OccurrenceFitBounds | None = None


class FamilyAlignmentConfig(StrictModel):
    free: list[Literal["dx", "dy", "log_scale_x", "log_scale_y", "shear_x"]] = Field(
        default_factory=lambda: ["dx", "dy", "log_scale_x", "log_scale_y"]
    )
    bounds: OccurrenceFitBounds = Field(default_factory=OccurrenceFitBounds)
    overrides: dict[str, OccurrenceAlignmentOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_alignment(self) -> FamilyAlignmentConfig:
        if len(set(self.free)) != len(self.free):
            raise ValueError("family alignment free parameters must be unique")
        if any(not instance_id for instance_id in self.overrides):
            raise ValueError("alignment override instance IDs must not be empty")
        for override in self.overrides.values():
            if override.free is not None and len(set(override.free)) != len(override.free):
                raise ValueError("instance alignment free parameters must be unique")
        return self


class FamilyTopology(StrictModel):
    path_kind: Literal["open", "loop", "open-with-loop"]
    expected_cycles: int | None = Field(default=None, ge=0)
    expected_self_intersections: int | None = Field(default=None, ge=0)
    endpoint_policy: Literal["supervised", "open", "auto"] = "auto"


class FamilyDefinition(StrictModel):
    id: str
    primitive: str
    topology: FamilyTopology | None = None
    fit: list[str] = Field(default_factory=list)
    validation: list[str] = Field(default_factory=list)
    hold_out: list[str] = Field(default_factory=list)
    alignment: FamilyAlignmentConfig = Field(default_factory=FamilyAlignmentConfig)
    review_status: Literal["pending", "provisional", "reviewed", "uncertain"] = "provisional"
    construction_evidence: str = "inferred"
    trajectory_evidence: str = "inferred"

    @model_validator(mode="after")
    def validate_split_membership(self) -> FamilyDefinition:
        members = [*self.fit, *self.validation, *self.hold_out]
        if len(set(members)) != len(members):
            raise ValueError("family sample splits must be disjoint")
        return self


class FitFamiliesFile(StrictModel):
    format: Literal["inkgrammar-fit-families"] = "inkgrammar-fit-families"
    format_version: Literal[1] = 1
    families: list[FamilyDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_family_ids(self) -> FitFamiliesFile:
        ids = [family.id for family in self.families]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate fit family id")
        return self


class TopologyPart(StrictModel):
    instance: str
    primitive: str
    attach: bool = True
    transform: TopologyTransform = Field(default_factory=TopologyTransform)


class TopologyStroke(StrictModel):
    id: str
    parts: list[TopologyPart] = Field(min_length=1)


class TopologyGlyph(StrictModel):
    character: str = Field(min_length=1)
    advance_width: float = Field(gt=0)
    strokes: list[TopologyStroke] = Field(min_length=1)
    review_status: Literal["pending", "provisional", "reviewed", "uncertain"] = "provisional"
    notes: str | None = None


class FitTopologyFile(StrictModel):
    format: Literal["inkgrammar-fit-topology"] = "inkgrammar-fit-topology"
    format_version: Literal[1] = 1
    glyphs: dict[str, TopologyGlyph] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_glyph_keys(self) -> FitTopologyFile:
        for key, glyph in self.glyphs.items():
            if key != glyph.character:
                raise ValueError(f"topology glyph key does not match character: {key}")
        return self


class FitAnchorRecord(StrictModel):
    source_id: str
    source_bbox: tuple[int, int, int, int]
    origin_x: float
    baseline_y: float
    review_status: Literal["pending", "provisional", "reviewed"] = "pending"
    reviewed_by: str | None = None
    review_method: str | None = None

    @model_validator(mode="after")
    def validate_anchor(self) -> FitAnchorRecord:
        x0, y0, x1, y1 = self.source_bbox
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError("anchor source_bbox must have positive dimensions")
        if not (0 <= self.origin_x <= x1 - x0 and 0 <= self.baseline_y <= y1 - y0):
            raise ValueError("anchor coordinates must be inside the source crop")
        if self.review_status == "reviewed" and not (self.reviewed_by and self.review_method):
            raise ValueError("reviewed anchors require reviewer and method")
        return self


class FitAnchorsFile(StrictModel):
    format: Literal["inkgrammar-fit-anchors"] = "inkgrammar-fit-anchors"
    format_version: Literal[1] = 1
    coordinate_frame: str
    samples: dict[str, FitAnchorRecord] = Field(default_factory=dict)
