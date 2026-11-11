# SAS-1968 visual validation

This source-free example separates the current runtime state from fitted candidates and missing coverage. It does not render private source images or change the canonical runtime model.

## Geometry modes

- `canonical` uses promoted primitive geometry where available and seed geometry otherwise. It represents the current safe runtime policy.
- `latest-candidate` uses current, successful fit reports only when their project, primitive, objective, optimizer backend, primitive schema, and fit-input hashes match. The result is an explicitly unpromoted preview for visual inspection. Missing or stale reports cause a clear error. Candidate previews never promote geometry or overwrite `data/models/sas-1968.yaml`.
- `both` writes canonical and candidate runs plus side-by-side alphabet and word sheets and a modeled-glyph difference image. Shared cell dimensions and raster scales make geometry changes comparable.

## Run

Install the optional fitting dependencies, then render canonical geometry:

```bash
python -m pip install -e ".[fit]"
python examples/sas1968/render_validation.py \
  --project data/training/sas-1968/project.yaml \
  --geometry canonical \
  --output work/examples/sas-1968/canonical
```

After current compatible fit reports exist, render a candidate or comparison:

```bash
python examples/sas1968/render_validation.py \
  --project data/training/sas-1968/project.yaml \
  --geometry latest-candidate \
  --output work/examples/sas-1968/latest-candidate

python examples/sas1968/render_validation.py \
  --project data/training/sas-1968/project.yaml \
  --geometry both \
  --output work/examples/sas-1968
```

Use `--include-digits` for 63 alphabet targets instead of 53. The default target includes every uppercase and lowercase Latin character and `ß`; unsupported cells remain visible as missing.

Outputs contain `manifest.json`, `01-alphabet.png` and its JSON report, `02-supported-glyphs.png`, `03-words.png`, one diagnostic SVG per supported word under `debug/`, and a local `model-preview.yaml`. The preview YAML is an inspection artifact and never replaces the tracked model. Generated output defaults to ignored `work/examples/sas-1968/` and should not be committed.

The manifest contains source-free report identifiers and fit metadata only. It does not copy source paths, crops, masks, source pixels, or private review images.

## Current limits

The current SAS-1968 runtime model contains 12 glyphs. Most of the 53 alphabet cells are correctly shown as missing. Missing characters are not fabricated, and unsupported words are marked blocked with their missing-character list instead of being partially rendered.

The configured supported words test multi-glyph composition only. The current runtime has zero connection rules, so glyph-boundary joining is not modeled. These images are not validation of connected cursive writing. Future words in `cases.yaml` intentionally exercise missing coverage and are reported as blocked.

The latest fitting configuration is objective v4 with the annealed spline optimizer. Its results remain experimental. No primitive is promoted by this example, and candidate previews are not evidence that promotion gates have passed.
