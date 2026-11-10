# ScriptStrokes

**Open writing models for handwriting and cursive scripts.**

ScriptStrokes is not tied to Schreibatlas. It is intended for any project that needs machine-
readable information about how a handwriting system is constructed, written, connected, and
validated against historical or instructional examples.

## What changed in this MVP

This version is organized around a **writing model**, not merely a directory of glyph paths.

It includes:

- no `src/` directory: the package lives directly at `scriptstrokes/`;
- dynamic versions derived from Git tags with `setuptools-scm`;
- reusable stroke components;
- isolated/initial/medial/final glyph forms;
- typed connection classes and connection rules;
- pair-specific connection overrides;
- scan-page segmentation and layout-aware character detection;
- generated review images and detection manifests;
- raster regression between generated writing and observed samples;
- a first `fit-join` command that learns pair spacing/tension from a labelled reference crop;
- provenance separation between observed shape and claimed pen movement;
- deterministic JSON data releases for non-Python consumers.

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
scriptstrokes page extract alphabet.png \
  --layout examples/layouts/sas-1968-alphabet.yaml \
  --output work/alphabet
```

The output includes one crop per declared character, `detections.json`, and `review.png`.

The extractor deliberately does not claim OCR certainty. It detects the image regions and uses
the known alphabet-row sequence to say which crop is `A`, which is `a`, etc.

## Validate a writing model

```bash
scriptstrokes model validate data/models/demo-looped.yaml --strict
```

## Render connected writing

```bash
scriptstrokes model render data/models/demo-looped.yaml "minimum" \
  --output work/minimum.png
```

## Compare generated output with an observed crop

```bash
scriptstrokes model compare data/models/demo-looped.yaml "mi" refs/mi.png
```

Metrics currently include normalized mask IoU, precision, and recall.

## Fit a connection from actual output

```bash
scriptstrokes model fit-join data/models/demo-looped.yaml "mi" refs/mi.png \
  --output work/mi-override.yaml
```

The MVP searches pair gap/overlap and cubic-bridge tension. More sophisticated model fitting can
be added without changing the observation/model separation.

## Build the data release

```bash
scriptstrokes build-release data/models/demo-looped.yaml \
  --version 0.2.0 \
  --output dist
```

This creates a consumer release with JSON models, schemas, QC reports, checksums, and a ZIP.
Schreibatlas can consume that release without Python.

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
