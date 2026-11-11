# Roadmap

## v5: shared shaping and OpenType export

The current development branch implements schema-v3 runtime models, contextual variant selection, shared shaped output for raster/SVG rendering, adaptive stroke outlines, connection fitting/control sheets, and deterministic TTF planning/building with GSUB/GPOS features. Schema-v2 models remain loadable.

The exporter and `font` CLI are development capabilities, not a release-readiness claim. Source-derived fonts remain blocked by unresolved rights, unpromoted geometry, incomplete glyph/connection evidence, unsupported rules, or missing shaping parity. SAS coverage is reported in stages; no glyph or connection observations are synthesized to make the coverage appear complete.

## v0.2 — this MVP

- writing-model schema;
- reusable components;
- contextual glyph forms;
- connection classes/rules;
- page segmentation with reviewed label assignment;
- pair comparison;
- simple join fitting;
- dynamic package versioning;
- root package layout;
- deterministic release bundle.

## v0.3 — SAS reconstruction workflow

- import all alphabet-page crops;
- record source locators per observation;
- vector-fit isolated forms against the alphabet plate;
- encode documented stroke-order diagrams;
- create component families from recurring base forms;
- encode the documented connection categories;
- import connection/pair examples as regression fixtures;
- add a model coverage/QC dashboard.

## v0.4 — stronger fitting

- automatic baseline/x-height estimation;
- robust deskew and page dewarping;
- pair/word alignment before similarity scoring;
- optimize connector geometry, glyph transforms, spacing, and slant;
- fit component parameters jointly across multiple glyphs;
- cross-validation so one sample is not overfit.

## v0.5: font outputs (partially superseded by v5)

- adaptive round-pen expansion and deterministic TTF generation are available;
- WOFF2 and general overlap cleanup are not implemented;
- representable v3 contextual substitutions and cursive positioning compile to OpenType;
- HarfBuzz parity is optional, and full corpus parity is not yet established.

## Non-goal

Do not reconstruct stroke order from final black shapes and then publish it as historically
verified movement.
