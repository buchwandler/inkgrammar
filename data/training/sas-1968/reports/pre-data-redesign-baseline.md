# Pre-data-redesign baseline (Phase D0)

Recorded before any data-first/trajectory redesign changes.
This report contains metadata and hashes only; no source images.

- Date (UTC): 2026-11-11T16:52:20Z
- Git commit: `bdd65033c3a40bb579e82168aac35e0afdbccd25`
- Python: 3.13.15

## Test suite baseline

Command: `python -m pytest -v --tb=no`

- Result: **1 failed, 254 passed** in 51.41s
- Full log: `work/sas-1968/_baseline/pytest-baseline.log` (private workspace)

Failing test (pre-existing, left unfixed at baseline):

- `tests/test_font_builder.py::test_font_cli_plan_build_validate_and_runtime_comparison`
  - font CLI plan/build/validate runs, but runtime comparison reports
    `positions_match: false` for text `ab` (expected origins `[0,0], [240,80]`,
    actual origins `[0,0], [450,0]`, sequence matches, tolerance 1.0 font unit).
  - Consistent with the documented connection-positioning gaps; not introduced
    by this redesign.

## Observation manifest hashes

| File                                       | SHA-256                                                            |
| ------------------------------------------ | ------------------------------------------------------------------ |
| `data/observations/sas-1968/manifest.json` | `ddf77ca78af5e33c6f29b7518aa59ab7bd67878cceaa947b35f07429c5a60e38` |
| `data/observations/sas-1968/sources.json`  | `da497d2a001bea69f3c438b3ce856cb97f8e00780a01266ac0b2c7c2aecaa11b` |

## Training/config hashes

| File                                     | SHA-256                                                            |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `data/training/sas-1968/project.yaml`    | `61a03714197b23bfd50921aa641fb1c162a78f7b8cf4dfa7a6ab6d0fec19ece4` |
| `data/training/sas-1968/fit-config.yaml` | `cafdaa50699eac1f632161723382a9a7805991a0ab535e413d70858a45cd1a8b` |

## Source images (private, hash/dimensions only)

| Source                                           | SHA-256                                                            | Size (px) |
| ------------------------------------------------ | ------------------------------------------------------------------ | --------- |
| `sas-1968-01-alphabet-plate.png`                 | `ad34c4fa0d20357bab77c638d9620bd03402590f8c8f8ef3be35b457cc332a1e` | 754x1180  |
| `sas-1968-02-component-families.png`             | `e510e5d339a23b0336492e93dcaf50da5325b545bbc9bf45ac1180e29c91062f` | 568x469   |
| `sas-1968-03-stroke-order-and-special-forms.png` | `4d903f441956cab467c38dc5fb514964b2d343101926dd490d5516eee1b9a841` | 541x285   |
| `sas-1968-04-numbered-connection-examples.png`   | `2361702b8272cf4c01e5d97e9655f29e51f80103cbee47513cf9d4ea9672a8ee` | 273x223   |
| `sas-1968-05-connection-categories.png`          | `0b804c177500849fd2363d320b7149798fded59eecdaa4706921ef496ac4be3e` | 538x544   |

## Current validation examples

Rendered with `python examples/sas1968/render_validation.py --geometry canonical --output work/examples/sas-1968/pre-data-redesign-baseline` (exit 0).

- `latest-candidate` rendering is blocked by a stale fit report:
  `ValueError: latest candidate for sas-ascender-loop-01 is stale: report format_version=3, project objective version=4`.
- Fixing this would require re-running `inkgrammar fit optimize-primitive`,
  which is fitting work outside D0 scope. Recorded as a baseline finding.

## Legacy evidence preserved (immutable)

The following pre-existing artifacts are frozen as legacy evidence and must not
be overwritten by the new pipeline:

- `work/sas-1968/` (private sources, materialized observations, vectors,
  trajectory/baseline logs)
- `work/examples/sas-1968/current-v4-fit/` (v4 canonical/candidate renders)

Full per-file SHA-256 inventory (59 files): `work/sas-1968/_baseline/legacy-inventory.sha256`.

Current v4 fits are **not promoted**; they remain legacy comparison evidence
only.

## Baseline conclusions

1. The suite is green except one pre-existing font-builder positioning failure.
2. Candidate fit reports are stale against objective version 4.
3. The redesign starts from this exact state; later phases must not show new
   failures relative to these counts.
