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
