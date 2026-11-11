# SAS-1968 objective v3 promotion review

## Run scope

All four documented objective v3 fits completed against the already-materialized private SAS corpus:

| Primitive              |     Fit objective | Hold-out mean | Assignment cycles |
| ---------------------- | ----------------: | ------------: | ----------------: |
| `sas-underturn-01`     |   96.214 → 66.243 |         0.525 |                 4 |
| `sas-open-oval-01`     | 139.733 → 105.046 |         0.474 |                 4 |
| `sas-hump-01`          |   98.568 → 46.937 |         0.285 |                 4 |
| `sas-ascender-loop-01` |   82.289 → 39.343 |         0.481 |                 3 |

The hold-out split was evaluated after fitting and did not participate in restart selection or parameter updates. The geometric gates were reviewed after fitting. A lower aggregate objective alone is not promotion evidence.

## Gates and fit-occurrence results

The review used candidate-to-target p95 ≤ 2.0 px, target-to-candidate p95 ≤ 2.0 px, supervised endpoint RMS ≤ 1.5 px, assigned coverage ≥ 0.90, and visual inspection for branch mismatch and systematic overlays.

| Primitive / fit occurrence  | Candidate p95 | Target p95 | Endpoint RMS | Coverage | Result              |
| --------------------------- | ------------: | ---------: | -----------: | -------: | ------------------- |
| Underturn / `i-underturn`   |          6.50 |       1.70 |          n/a |     0.98 | Fail                |
| Underturn / `u-underturn-a` |          5.59 |       1.97 |         2.62 |     0.97 | Fail                |
| Underturn / `u-underturn-b` |          5.99 |       1.70 |          n/a |     0.99 | Fail                |
| Open oval / `c-oval`        |          6.01 |       3.78 |          n/a |     0.62 | Fail                |
| Open oval / `a-oval`        |          3.70 |       3.34 |         3.66 |     0.76 | Fail                |
| Hump / `n-hump`             |          1.86 |       1.56 |         2.21 |     1.00 | Fail, endpoint gate |
| Hump / `m-hump-a`           |         17.21 |       2.38 |          n/a |     0.96 | Fail                |
| Hump / `m-hump-b`           |          6.51 |       1.28 |          n/a |     1.00 | Fail                |
| Ascender loop / `l-loop`    |          2.15 |       1.84 |         0.90 |     0.97 | Fail, candidate p95 |
| Ascender loop / `b-loop`    |          2.41 |       1.96 |         1.27 |     0.97 | Fail, candidate p95 |

Visual inspection of the private review sheets confirmed broad red/blue disagreement for the underturn, open-oval, and hump families. The ascender-loop fit occurrences are closer, but remain above the candidate p95 gate. Its `k` and `h` hold-out panels still show a substantial branch/continuation mismatch. Those hold-out results are diagnostic only and were not used to tune or select parameters.

## Promotion decision

No primitive is promoted. The four new checkpoints and reports are retained as source-free objective v3 fit candidates, not canonical grammar state. The seed grammar and compiled SAS model were not changed. In particular, the ascender-loop branch mismatch remains unresolved, so promotion is blocked.

The source-free control sheet was regenerated at `work/sas-1968/control/alphabet.png`; its companion JSON reports 12 of 53 target glyphs rendered and explicitly marks missing or provisional glyphs. All review sheets, materialized crops, masks, distance fields, and the control output remain under ignored `work/`.

## Reproduction

```sh
inkgrammar fit optimize-primitive data/training/sas-1968/project.yaml sas-underturn-01
inkgrammar fit optimize-primitive data/training/sas-1968/project.yaml sas-open-oval-01
inkgrammar fit optimize-primitive data/training/sas-1968/project.yaml sas-ascender-loop-01
inkgrammar fit optimize-primitive data/training/sas-1968/project.yaml sas-hump-01
inkgrammar fit control-sheet data/training/sas-1968/project.yaml --output work/sas-1968/control/alphabet.png
```
