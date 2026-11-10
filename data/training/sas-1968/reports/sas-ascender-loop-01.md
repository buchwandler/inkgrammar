# Fit report: sas-ascender-loop-01

Project: `sas-1968`

## Training members

| Sample                    | Initial objective | Final objective |
| ------------------------- | ----------------: | --------------: |
| `sas1968-alphabet-U+006C` |          0.234744 |        0.146261 |
| `sas1968-alphabet-U+0062` |          0.417911 |        0.319001 |

Training aggregate: 0.436880 → 0.351175
Training mean: 0.232631; worst member: 0.319001

## Validation diagnostics

Validation members are evaluated after fitting and do not update parameters.

| Sample | Objective |
| ------ | --------: |

## Hold-out diagnostics

Hold-out samples are reported after selection and do not affect parameters.

| Sample                    | Objective |
| ------------------------- | --------: |
| `sas1968-alphabet-U+006B` |  0.724181 |
| `sas1968-alphabet-U+0068` |  1.341549 |

Training to hold-out mean gap: 0.800234
Model promotion remains pending; inspect overlays in the private work directory.

## Selected parameters

- `occurrence.sas1968-alphabet-U+0062--b-loop.delta.dx`: -0.46091661
- `occurrence.sas1968-alphabet-U+0062--b-loop.delta.dy`: -0.71875508
- `occurrence.sas1968-alphabet-U+0062--b-loop.delta.log_scale_x`: -0.03658263
- `occurrence.sas1968-alphabet-U+0062--b-loop.delta.log_scale_y`: -0.04157619
- `occurrence.sas1968-alphabet-U+006C--l-loop.delta.dx`: 0.25272450
- `occurrence.sas1968-alphabet-U+006C--l-loop.delta.dy`: 0.46270934
- `occurrence.sas1968-alphabet-U+006C--l-loop.delta.log_scale_x`: -0.00609105
- `occurrence.sas1968-alphabet-U+006C--l-loop.delta.log_scale_y`: 0.01002110
- `shared.segments.0.c1.y`: 0.20380408
- `shared.segments.0.c2.x`: 0.85616579
