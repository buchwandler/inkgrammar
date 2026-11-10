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
baseline, connections leaving from the middle line, and attached connections. ScriptStrokes
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
scriptstrokes model compare data/models/my-model.yaml "ma" refs/ma.png
```

For the MVP, pair fitting can tune spacing and bridge tension:

```bash
scriptstrokes model fit-join data/models/my-model.yaml "ma" refs/ma.png \
  --output work/ma-override.yaml
```

The result is a proposed override. It still requires review before being merged into the model.

Future versions should optimize richer parameters and support whole-word fitting.
