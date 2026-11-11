# Fit report: sas-ascender-loop-01

Project: `sas-1968`

## Training members

| Sample                    | Initial objective | Final objective |
| ------------------------- | ----------------: | --------------: |
| `sas1968-alphabet-U+006C` |          0.186204 |        0.073921 |
| `sas1968-alphabet-U+0062` |          0.200446 |        0.098113 |

Training aggregate: 82.289282 → 39.343371
Training mean: 0.086017; worst member: 0.098113

## Validation diagnostics

Validation members are evaluated after fitting and do not update parameters.

| Sample | Objective |
| ------ | --------: |

## Hold-out diagnostics

Hold-out samples are reported after selection and do not affect parameters.

| Sample                    | Objective |
| ------------------------- | --------: |
| `sas1968-alphabet-U+006B` |  0.472535 |
| `sas1968-alphabet-U+0068` |  0.488845 |

Training to hold-out mean gap: 0.394673
Model promotion remains pending; inspect overlays in the private work directory.

## Selected parameters

- `occurrence.sas1968-alphabet-U+0062--b-loop.delta.dx`: 3.99049194
- `occurrence.sas1968-alphabet-U+0062--b-loop.delta.dy`: 2.16471031
- `occurrence.sas1968-alphabet-U+0062--b-loop.delta.log_scale_x`: 0.04983461
- `occurrence.sas1968-alphabet-U+0062--b-loop.delta.log_scale_y`: -0.00190808
- `occurrence.sas1968-alphabet-U+006C--l-loop.delta.dx`: 2.82488018
- `occurrence.sas1968-alphabet-U+006C--l-loop.delta.dy`: 0.37990746
- `occurrence.sas1968-alphabet-U+006C--l-loop.delta.log_scale_x`: -0.04545385
- `occurrence.sas1968-alphabet-U+006C--l-loop.delta.log_scale_y`: 0.06262169
- `shared.segments.0.c1.y`: 0.29764570
- `shared.segments.0.c2.x`: 0.80099619
