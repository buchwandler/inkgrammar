# InkGrammar

**Open writing grammars for handwriting and cursive scripts.**

InkGrammar represents a handwriting system as reusable stroke primitives,
glyph recipes, contextual forms, and connection rules, with reproducible
fitting and validation against observed examples.

## Core concepts

InkGrammar models how a handwriting system is constructed instead of treating letters as
unrelated drawings.

1. **Reusable writing primitives** shared across glyphs.
2. **Glyph and connection grammar** for contextual forms and joins.
3. **Reproducible fitting** against observed examples.
4. **Portable runtime data releases** with checksums and compatibility metadata.

## Current capabilities

The project includes:

- parameterized stroke primitives and isolated/initial/medial/final glyph forms;
- typed connection classes, rules, and pair-specific overrides;
- scan-page segmentation with layout-aware detection and review images;
- raster regression and connection fitting against labelled samples;
- separate provenance for observed shapes and claimed pen movement;
- deterministic JSON releases for non-Python consumers.
- schema-v3 contextual shaping with schema-v2 compatibility;
- deterministic SVG and raster rendering from shaped output;
- adaptive stroke outlines and FontTools-based TTF export with GSUB/GPOS features;
- staged glyph, connection, rights, and font-readiness reports.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Because versioning is dynamic, tags produce package versions:

```bash
git tag v0.2.0
python -m build
```

A source tree without Git metadata uses the declared fallback development version.

## Extract a supplied alphabet page

```bash
inkgrammar page extract alphabet.png \
  --layout examples/layouts/sas-1968-alphabet.yaml \
  --output work/alphabet
```

The output includes one crop per declared character, `detections.json`, and `review.png`.

The extractor deliberately does not claim OCR certainty. It detects the image regions and uses
the known alphabet-row sequence to say which crop is `A`, which is `a`, etc.

## Validate a writing model

```bash
inkgrammar model validate data/models/demo-looped.yaml --strict
```

## Render connected writing

```bash
inkgrammar model render data/models/demo-looped.yaml "minimum" \
  --output work/minimum.png
```

## Compare generated output with an observed crop

```bash
inkgrammar model compare data/models/demo-looped.yaml "mi" refs/mi.png
```

Metrics currently include normalized mask IoU, precision, and recall.

## Fit a connection from actual output

```bash
inkgrammar model fit-join data/models/demo-looped.yaml "mi" refs/mi.png \
  --output work/mi-override.yaml
```

The MVP searches pair gap/overlap and cubic-bridge tension. More sophisticated model fitting can
be added without changing the observation/model separation.

## Build the data release

```bash
inkgrammar build-release data/models/demo-looped.yaml \
  --version 0.2.0 \
  --output dist
```

This creates a consumer release with JSON models, schemas, QC reports, checksums, and a ZIP.
Schreibatlas can consume that release without Python.

## Contextual shaping and font export

Render through the shared runtime shaper, or inspect font planning and staged SAS coverage:

```bash
inkgrammar model render-svg data/models/sas-1968.yaml "ab" --output work/ab.svg
inkgrammar font plan data/models/sas-1968.yaml
inkgrammar model coverage data/models/sas-1968.yaml \
  --fit-project data/training/sas-1968/project.yaml
```

Build and validate a schema-v3 runtime font:

```bash
inkgrammar font build model-v3.yaml --output dist/model.ttf
inkgrammar font validate dist/model.ttf --model model-v3.yaml
inkgrammar font compare-runtime model-v3.yaml dist/model.ttf "ab"
```

The HarfBuzz comparison runs when the optional `uharfbuzz` package is installed. Source-derived font builds require rights clearance and promoted geometry by default. Development overrides produce explicitly unready artifacts and do not bypass release gates. The current SAS model remains incomplete, uses seed-fallback geometry, and has unresolved rights, so it is not font-ready.

## Dataset and trajectory tooling

Materialize the private SAS dataset from the source-free recipe and authorized
scans (see `docs/DATA.md`):

```bash
inkgrammar dataset validate-sas-private data/datasets/sas-1968.yaml
inkgrammar dataset status data/datasets/sas-1968.yaml
```

Fit curvature trajectories from dataset observation IDs (see
`docs/ARCHITECTURE.md`):

```bash
inkgrammar fit trajectory data/training/sas-1968/project.yaml \
  --observation sas1968-alphabet-U+0065 --curvature-changes 5
inkgrammar fit trajectory-preview data/training/sas-1968/project.yaml \
  --observation sas1968-alphabet-U+0065 --scales 0.5,1,2
inkgrammar fit trajectory-report work/sas-1968/trajectory/e/parameters.yaml
```

Fit outputs carry provenance hashes and measured gates. Nothing is promoted
without human review.

## Important evidence rule

A scan can establish an observed shape. It does not automatically establish stroke order,
direction, or pen lifts.

The historical material you supplied is more valuable because it contains several kinds of
evidence: an alphabet plate, recurring base forms, explicit stroke-order markings for some
letters, and examples of different connection types. The data model keeps those evidence types
separate so that stronger claims are only made where the source actually supports them.

See:

- `docs/ARCHITECTURE.md`
- `docs/IMAGE-IMPORT.md`
- `docs/CONNECTIONS.md`
- `docs/DATA.md`
- `docs/SAS-1968.md`
- `docs/ROADMAP.md`
