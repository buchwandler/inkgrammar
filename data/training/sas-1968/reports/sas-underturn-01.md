# Fit report: sas-underturn-01

Project: `sas-1968`

## Training members

| Sample                    | Initial objective | Final objective |
| ------------------------- | ----------------: | --------------: |
| `sas1968-alphabet-U+0069` |          0.729240 |        0.497801 |
| `sas1968-alphabet-U+0075` |          0.497021 |        0.598690 |

Training aggregate: 0.613130 → 0.548245
Training mean: 0.548245; worst member: 0.598690

## Validation diagnostics

Validation members are evaluated after fitting and do not update parameters.

| Sample | Objective |
| ------ | --------: |

## Hold-out diagnostics

Hold-out samples are reported after selection and do not affect parameters.

| Sample                    | Objective |
| ------------------------- | --------: |
| `sas1968-alphabet-U+0077` |  0.828928 |

Training to hold-out mean gap: 0.280683
Model promotion remains pending; inspect overlays in the private work directory.

## Selected parameters

- `segments.0.c1.x`: 0.78896155
- `segments.0.c1.y`: 0.53344298
- `segments.0.c2.x`: -0.02209149
