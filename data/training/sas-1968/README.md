# SAS-1968 fitting project

This directory contains source-free fitting definitions. Historical scans, crops, masks, distance fields, and review images stay under `work/sas-1968/` and are ignored by Git.

`source-lock.json` records the exact private inputs without embedding them. `anchors.json` preserves crop-local seed anchors visually checked against private source crops; the topology and trajectories remain provisional. The `c` and `a` samples are training members for the open-oval vertical slice; `C` is held out. Other family splits preserve the existing seed experiment while later phases revise the glyph topology.

Open-oval loss regions use crop-local pixel boxes and explicitly name the topology instances they supervise. The `c` and `a` fit members drive primitive selection; the `C` hold-out is evaluated only after the training parameters are fixed.

`seed-grammar.yaml` contains typed normalized cubic templates, not optimizer state serialized as SVG text. The topology separately encodes shared underturn, hump, and ascender-loop instances for `w`, `h`, and `k`. Entry/exit connectors and the b/k continuations are provisional and are not claims of complete glyph models.

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

Fit the shared open-oval primitive from the `c` and `a` training members, then inspect its report:

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

Underturn fits `i` and `u` and diagnoses the two-part `w` hold-out. Hump fits `n` and `m` and evaluates only `h`'s local hump region. Ascender-loop fits `l` and `b`, then reports local regions for `h` and `k`. Source-derived region overlays are generated only under ignored `work/sas-1968/overlays/`; review them before promoting provisional transforms or connectors.

See [phase-h-review.md](phase-h-review.md) for the source-free member metrics and private overlay review findings. The underturn `u` fit member worsens, and the ascender-loop hold-outs remain poor; these fits are diagnostic and provisional, not release-ready.
