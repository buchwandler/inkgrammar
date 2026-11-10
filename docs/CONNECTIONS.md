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
