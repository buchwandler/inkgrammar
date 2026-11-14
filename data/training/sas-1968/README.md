# SAS-1968 fitting project

This directory contains source-free fitting definitions. Historical scans, crops, masks, distance fields, and review images stay under `work/sas-1968/` and are ignored by Git.

`source-lock.json` records the exact private inputs without embedding them. `anchors.json` preserves crop-local seed anchors visually checked against private source crops; the topology and trajectories remain provisional. The `c` and `a` samples are training members for the open-oval vertical slice; `C` is held out. Other family splits preserve the existing seed experiment while later phases revise the glyph topology.

Open-oval loss regions use crop-local pixel boxes and explicitly name the topology instances they supervise. The `c` and `a` fit members drive primitive selection; the `C` hold-out is evaluated only after the training parameters are fixed.

`seed-grammar.yaml` contains typed normalized cubic templates, not optimizer state serialized as SVG text. The topology separately encodes shared underturn, hump, and ascender-loop instances for `w`, `h`, and `k`. Entry/exit connectors and the b/k continuations are provisional and are not claims of complete glyph models.

`target-inventory.yaml` links the 63 alphabet-and-digit targets to the accepted source-label observations and records each runtime status. The 12 seeded topologies are explicitly incomplete and retain their source observation IDs and known gaps. The numbered stroke-order examples and special-form regions are linked to the source layout, but their paths remain untraced and the special-form labels remain pending review. This inventory does not promote or complete any glyph.

`seed-grammar.yaml` also records the baseline, midline, and attached connection headings and their listed pair examples. Only the baseline and midline headings explicitly preserve one-stroke continuity; terminal/entry modifications, pen lifts, attachment points, and tangents remain unrecorded. Six numbered connection pairs link to declared page regions, but review and geometry transcription remain pending. The source-free connection control sheet shows available runtime pair previews or missing-glyph blockers and uses no connection rules.

`corpus.yaml` and `families.yaml` define stable three-way partitions for 15 source-observed glyph samples: five fit, five validation, and five hold-out. Each family has one fit and one validation sample; existing hold-outs remain unchanged. `anchors.json` keeps the reviewed source crop anchors. The generated `work/sas-1968/materialized/materialization.json` records source IDs, pixel boxes, splits, and hashes for crops, masks, distance fields, and the review sheet; a repeat run produced byte-identical output. The full-page dataset materialization under `work/sas-1968/dataset/` validates with warnings but no errors.

## Local workflow

Install fitting support and validate committed metadata without private sources:

```sh
python -m pip install -e '.[fit]'
inkgrammar fit validate-project data/training/sas-1968/project.yaml
```

Keep the five locked source images in a separate private directory, then materialize only after hash and dimension checks pass:

```sh
inkgrammar fit materialize \
  data/training/sas-1968/project.yaml \
  --source-dir /private/sas-1968/sources
```

Generated crops, masks, distance fields, and the review sheet are written under the project's ignored `work/sas-1968/materialized/` directory. The project metadata is not rewritten.

Fit the open-oval primitive on `c`, evaluate `a` as validation, and reserve `C` as hold-out:

```sh
inkgrammar fit optimize-primitive \
  data/training/sas-1968/project.yaml sas-open-oval-01
inkgrammar fit report data/training/sas-1968/project.yaml --format markdown
```

The optimizer selects parameters using only family `fit` members. Hold-outs are evaluated afterward for diagnostics. Checkpoints and reports contain source-free parameters and metrics, not historical imagery.

Fit the compositional families and inspect their member-level reports:

```sh
inkgrammar fit optimize-primitive data/training/sas-1968/project.yaml sas-underturn-01
inkgrammar fit optimize-primitive data/training/sas-1968/project.yaml sas-hump-01
inkgrammar fit optimize-primitive data/training/sas-1968/project.yaml sas-ascender-loop-01
inkgrammar fit report data/training/sas-1968/project.yaml --format markdown
```

Underturn fits `i`, validates `u`, and diagnoses the two-part `w` hold-out. Hump fits `n`, validates `m`, and evaluates `h`'s local hump region as hold-out. Ascender-loop fits `l`, validates `b`, and reports local regions for hold-out `h` and `k`. Source-derived region overlays are generated only under ignored `work/sas-1968/overlays/`; review them before promoting provisional transforms or connectors.

See [phase-h-review.md](phase-h-review.md) for the source-free member metrics and private overlay review findings. The underturn `u` fit member worsens, and the ascender-loop hold-outs remain poor; these fits are diagnostic and provisional, not release-ready.

## Current objective-v4 status

The committed `fit-config.yaml` uses objective version 4 with `spline-annealed-alt-lsq` and 240 annealing steps. This is an experimental configuration, not evidence that the new candidates passed review. The runtime model has 12 modeled glyphs, no promoted primitives, and incomplete alphabet coverage. It has no runtime connection rules or fitted connection trajectories. Three source-labelled connection categories and six numbered pair observations are recorded in the fit project; the pairs remain pending review and held out, and none are mapped to runtime endpoint classes. Word previews only demonstrate composition of supported glyphs; they are not connected-cursive validation. Objective-v3 reports and review results above remain historical and do not describe v4 outcomes. Existing family-fit reports also predate the current fit/validation/hold-out partition and are historical.

Render source-free canonical and candidate examples from the repository root:

```sh
python examples/sas1968/render_validation.py \
  --project data/training/sas-1968/project.yaml \
  --geometry canonical

python examples/sas1968/render_validation.py \
  --project data/training/sas-1968/project.yaml \
  --geometry both \
  --output work/examples/sas-1968
```

Candidate previews require current successful reports with matching project and primitive identity, objective, backend, and fresh fit-input hashes. They remain explicitly unpromoted and inspection-only. If valid reports are unavailable, candidate rendering fails instead of substituting seed geometry. Example outputs are local ignored files and must not be committed.

## Private-work layout and retention

The project config points to the private `work/sas-1968/` workspace. Preserve `sources/` as raw inputs, `observations/` as source-derived measurements and human review, and `vectors/` wherever they contain unique hand-authored geometry or decisions. `materialized/`, `overlays/`, `glyph-fits/`, `word-validation/`, and `control/` are reproducible or review outputs; regenerate them only from preserved inputs and definitions. Keep source images, crops, masks, and private review sheets out of tracked examples and reports.

Before retiring an old experiment, inventory and hash the workspace, classify each file, and preserve unknown or human-authored items. Store checksums and classifications under `_cleanup/`; archive only confirmed superseded experiments under a labelled `_archive/` directory. Do not delete the quarantine until required outputs have been regenerated and reviewed. The source-free example runner writes separately to `work/examples/sas-1968/` by default and does not need private source inputs.
