# Writing-model architecture

InkGrammar v2 deliberately separates four layers.

## 1. Observation layer

Scans, photographs, crops, and transcriptions are evidence.

An alphabet page can tell us the observed final shape of `A`, `a`, `1`, etc. When the page has a
known row order, InkGrammar can segment the page and assign labels from that declared order.

This does **not** mean the image itself establishes pen order.

Pair/word samples such as `ma`, `Ge`, `Pu`, or `Wa` are particularly valuable because they show
the context-dependent result of joining letters.

## 2. Component layer

Reusable trajectories represent recurring movement/shape building blocks:

- underturn;
- overturn;
- loop;
- oval;
- downstroke;
- crossbar;
- entry/exit movements;
- script-specific components.

A glyph stroke can concatenate multiple components into one uninterrupted pen-down trajectory.

Components have their own provenance. A component inferred from images remains `inferred` even
if it later produces a visually excellent model.

## 3. Glyph-form layer

Each glyph can define:

- `isolated`;
- `initial`;
- `medial`;
- `final`;
- or a `default` fallback.

This is necessary because a cursive letter is not always one invariant drawing. The historical
examples supplied for SAS show both standalone forms and context-dependent connections.

## 4. Connection-model layer

Glyph forms expose entry/exit endpoints with connection classes. A connection rule specifies
how one class joins another.

The MVP uses a cubic bridge parameterized by:

- gap/overlap;
- endpoint tangents;
- connection tension.

Pair-specific overrides are allowed only as explicit data, ideally carrying reference evidence.

`model fit-join` performs a small grid search over gap and tension against a labelled two-letter
reference crop. That is the first genuine "learn from actual output" operation. It does not
pretend to infer historical pen order from pixels.

# Long-term target

For a mature historical writing model:

1. alphabet plates establish isolated/reference shapes;
2. stroke-order diagrams establish movement where explicitly documented;
3. component tables establish recurring construction logic;
4. connection examples establish connection classes;
5. pair/word samples become regression fixtures;
6. the generated model is rendered and compared with all observed examples;
7. QC records both geometric fit and evidence status.

This lets the same model serve Schreibatlas, worksheet generators, font compilation, animation,
or handwriting-comparison software.

## v5 shared shaping and output pipeline

The v5 runtime adds schema version 3 while preserving schema-v2 loading and rendering. Fitting projects compile to a source-free runtime model. Promoted reports are required by default; seed geometry is available only through an explicit development option.

The runtime shaper resolves contextual variants and connection rules into a `ShapedText` plan. Raster and SVG renderers consume that plan. OpenType export uses the same runtime variants, anchor data, and adaptive outline expansion, then encodes representable required substitutions in GSUB `rclt` and cursive attachment in GPOS `curs`. Optional context alternates use `calt`; exceptional ligatures use `rlig`.

The outline stage adaptively samples curves, expands strokes and taps into closed contours, and records bounds. TTF glyph contours are stored as quadratic TrueType outlines. The standalone cubic-to-quadratic helper uses FontTools `cu2qu`; the current stroke expander first produces polygon contours, so the exporter represents their edges as exact quadratic line segments within font-unit rounding tolerance.

`inkgrammar font plan`, `build`, `build-fit`, `validate`, and `compare-runtime` expose planning and export. HarfBuzz parity is optional when `uharfbuzz` is installed. A structurally valid TTF is not automatically font-ready: rights, promotion, glyph and connection coverage, unsupported context rules, and corpus parity remain separate gates. Runtime schema validity and visual output do not establish historical verification.

# Evidence layers and trajectory model

## Four evidence layers

The SAS pipeline separates four layers and never lets one overwrite another:

1. **Source evidence** - immutable facts about a source image (hash, size,
   rights status).
2. **Extracted observations** - deterministic extractions (crop locator,
   source bbox, extraction recipe and artifact hashes).
3. **Reviewed interpretation** - semantic claims with per-field provenance
   (`source-explicit`, `source-layout`, `inferred-reviewed`,
   `automatic-candidate`, ...).
4. **Fitted model** - numerical parameters learned from reviewed evidence;
   replaceable and always carrying provenance hashes.

## Curvature trajectory model

InkGrammar stores how the pen moves; SVG Beziers, raster pixels, and font
outlines are compiled representations of that movement. A pen-down stroke is
a start pose, a normalized length, and a heading-rate profile
`q(u) = d(theta_deg)/du` over normalized progress `u in [0, 1]`:

```text
theta(u) = theta0 + integral_0^u q(v) dv
dx/du = L cos(theta(u)),  dy/du = L sin(theta(u))
```

`q(u)` is scale-invariant: a small and a large letter share one profile and
differ only in the similarity transform (uniform scale, rotation, dx, dy).
Change locations are percentages of drawn stroke length, strictly ordered
with minimum spacing; interpolation is piecewise-linear or shape-preserving
cubic Hermite. The canonical primitive starts at `(0, 0)`; placement belongs
to the occurrence transform. Compilation integrates the trajectory and
approximates it with cubic Beziers before the existing runtime consumes it,
with geometric parity checked in x-height-normalized units.

Fitting takes an ordered centerline (from the dataset index, never ad-hoc
image paths): observed heading and heading-rate estimation, deterministic
event initialization, then bounded robust least squares over the
low-dimensional trajectory parameters with per-occurrence similarity
transforms. Position, tangent, endpoint, total-turn and profile residuals
dominate; raster overlap is only a secondary check.

## Legacy objective-v3 fitting

Objective version 3 remains dispatched to `family_fit_v3` for projects that need fitting reproducibility. It retains its independent centerline and assignment pipeline. This path is frozen. Do not add features or new consumers, and do not migrate its skeleton tracing behavior to SkeletonPaths. New fitting work uses objective version 4.

## Explicit unit conversion

Glyph design units are x-height normalized (`baseline = 0`, `x_height = 1`,
y up). Runtime/font units are reached only through named conversions:

```python
glyph_units_to_runtime(xy, *, x_height_units)
runtime_units_to_glyph(xy, *, x_height_units)
```

Pixel frames and font units never mix implicitly; every fit report records
the units it used.

Source observations are crop-local image pixels (`x` right, `y` down). Reviewed crop anchors store the writing origin and baseline in that frame. `dataset.transforms.ObservationFrame` records source axes and pixels per x-height; `source_px_to_glyph_units` and `glyph_units_to_source_px` make the source-pixel to normalized-glyph conversion explicit.

The current SAS fitting frame uses one fit unit per crop pixel, with `x` right, `y` up, and the baseline at zero. `render_paths` places fit geometry at the reviewed crop origin/baseline when rasterizing an observation. Primitive geometry and topology/occurrence transforms compose in fit units; advances and connection adjustments/spacing use those same units.

`FitCompileConfig.fit_x_height_units` declares the number of fit units per x-height. Compilation maps fit units to normalized glyph units, then uses `glyph_units_to_runtime(..., x_height_units=metrics.x_height)` for paths, advances, anchor positions, and connection offsets. Metrics, pen width, and anchor widths are already in runtime/font units and are not scaled again. Runtime rasterization and OpenType export consume those same runtime units.

The current canonical SAS calibration is 31 fit units per x-height. It is measured from the `a` crop: reviewed baseline y=35 minus the top of the main ink component at y=4. The crop remains private under `work/`; this calibration does not promote or certify the provisional glyph geometry.

## External geometry ownership

| Responsibility                                                 | Owning package | InkGrammar-owned boundary                           |
| -------------------------------------------------------------- | -------------- | --------------------------------------------------- |
| SVG parsing, generic curve and polyline operations, transforms | PlanarCurve    | Coordinate-frame adapters and fitting-domain models |
| Raster skeleton graphs, route enumeration, ranking, ambiguity  | SkeletonPaths  | Topology candidates and evidence provenance         |
| Variable-width expansion, caps, joins, closed contours         | StrokeOutline  | Shaped-text traversal and width-profile selection   |

SkeletonPaths routes remain inferred topology candidates. Only reviewed trajectory annotations support physical stroke-order or direction claims. Pass typed `planarcurve.CubicPath` values to StrokeOutline when available; retain the SVG-string adapter for shaped-text compatibility.
