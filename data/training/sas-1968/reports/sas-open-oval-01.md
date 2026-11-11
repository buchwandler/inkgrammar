# Fit report: sas-open-oval-01

Project: `sas-1968`

## Training members

| Sample                    | Initial objective | Final objective |
| ------------------------- | ----------------: | --------------: |
| `sas1968-alphabet-U+0063` |          0.390988 |        0.267458 |
| `sas1968-alphabet-U+0061` |          0.299471 |        0.234356 |

Training aggregate: 139.733103 → 105.045900
Training mean: 0.250907; worst member: 0.267458

## Validation diagnostics

Validation members are evaluated after fitting and do not update parameters.

| Sample | Objective |
| ------ | --------: |

## Hold-out diagnostics

Hold-out samples are reported after selection and do not affect parameters.

| Sample                    | Objective |
| ------------------------- | --------: |
| `sas1968-alphabet-U+0043` |  0.474494 |

Training to hold-out mean gap: 0.223587
Model promotion remains pending; inspect overlays in the private work directory.

## Selected parameters

- `occurrence.sas1968-alphabet-U+0061--a-oval.delta.dx`: 1.20631362
- `occurrence.sas1968-alphabet-U+0061--a-oval.delta.dy`: -0.13009659
- `occurrence.sas1968-alphabet-U+0061--a-oval.delta.log_scale_x`: -0.07978608
- `occurrence.sas1968-alphabet-U+0061--a-oval.delta.log_scale_y`: 0.03684571
- `occurrence.sas1968-alphabet-U+0063--c-oval.delta.dx`: 3.04024227
- `occurrence.sas1968-alphabet-U+0063--c-oval.delta.dy`: -1.85828319
- `occurrence.sas1968-alphabet-U+0063--c-oval.delta.log_scale_x`: 0.02181889
- `occurrence.sas1968-alphabet-U+0063--c-oval.delta.log_scale_y`: 0.05314509
- `shared.segments.0.c1.y`: 0.99054878
- `shared.segments.0.c2.x`: 0.09995157
- `shared.segments.1.c1.x`: 0.19249820
