# Roadmap

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

## v0.5 — font outputs

- round-pen stroke expansion;
- outline cleanup;
- TTF/WOFF2 generation;
- contextual OpenType substitutions/positioning;
- render-regression tests against the same observation corpus.

## Non-goal

Do not reconstruct stroke order from final black shapes and then publish it as historically
verified movement.
