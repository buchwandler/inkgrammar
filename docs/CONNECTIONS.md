# Standalone and connected forms

A writing model should answer two different questions:

1. What does the character look like in isolation?
2. What movement/shape does it take when connected to neighbors?

Therefore a glyph may contain `isolated`, `initial`, `medial`, and `final` forms.

Connection endpoints are typed with classes such as:

- `baseline`;
- `midline`;
- `upper`;
- `attached`.

The class vocabulary is model-specific. Historical documentation may justify a small fixed set.

The supplied source examples distinguish, among other things, connections leaving from the
baseline, connections leaving from the middle line, and attached connections. InkGrammar
therefore models connection behavior explicitly rather than assuming one universal join.

## Connection modes

`ConnectionRule.mode` controls compilation independently of its class names:

- `continuous-bridge` adds a cubic connector between the declared endpoints.
- `direct` adds no path and requires the translated endpoints to meet (within
  0.000001 model units). Incompatible endpoints raise an error and fail QC.
- `attached` keeps separate pen-down strokes and adds no synthetic bridge. A
  target-specific reattachment stroke must be encoded in the glyph form.
- `none` adds no connection. Declared spacing still applies.

All modes retain rule gaps and pair spacing overrides. Tension only affects
continuous bridges. An explicit mode takes precedence over legacy `continuous`.
Without `mode`, `continuous: true` means `continuous-bridge` and
`continuous: false` means `attached`. Existing true-valued rules remain compatible.

The SAS class IDs are `baseline-single`, `midline-single`, and `attached`.
Their source labels are respectively “Die einzügige Verbindung von der
Grundlinie aus”, “Die einzügige Verbindung von der Mittellinie aus”, and
“Die angesetzte Verbindung”. Category evidence does not establish the exact
geometry or trajectory of a generated connector.

## Pair regression

Once a model can render a pair, compare it against an observed pair crop:

```bash
inkgrammar model compare data/models/my-model.yaml "ma" refs/ma.png
```

For the MVP, pair fitting can tune spacing and bridge tension:

```bash
inkgrammar model fit-join data/models/my-model.yaml "ma" refs/ma.png \
  --output work/ma-override.yaml
```

The result is a proposed override. It still requires review before being merged into the model.

Future versions should optimize richer parameters and support whole-word fitting.

## Schema-v3 runtime and OpenType connections

Schema v3 keeps connection classes and rules explicit. Runtime modes include `cursive-anchor`, `direct`, `attached`, `ligature`, `none`, and compatibility-only `synthetic-bridge`. Normal connected forms should use contextual variants plus matching entry and exit anchors. The font builder emits required contextual substitutions as GSUB `rclt` and uses class-filtered GPOS `curs` positioning for declared cursive/direct class pairs. Exceptional constructions may use `rlig`.

A rule requiring `synthetic-bridge` is not font-compatible. Context rules that depend on unsupported glyph-position conditions are reported as font blockers rather than silently changed. Build a TTF only from a schema-v3 runtime model; `font build-fit` compiles the fit project first and requires promoted primitive reports unless `--allow-unpromoted` is explicitly selected for development.

Inspect staged evidence without inferring connection classes from character pairs:

```bash
inkgrammar model coverage data/models/sas-1968.yaml \
  --fit-project data/training/sas-1968/project.yaml
```

The report separates source connection observations, fit-project observations by split, runtime class rules, and font readiness. Character-pair coverage does not prove class-pair coverage. Missing observations remain missing.
