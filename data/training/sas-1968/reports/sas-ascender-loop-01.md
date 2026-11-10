# Fit report: sas-ascender-loop-01

Project: `sas-1968`

## Training members

| Sample                    | Initial objective | Final objective |
| ------------------------- | ----------------: | --------------: |
| `sas1968-alphabet-U+006C` |          0.492803 |        0.437086 |
| `sas1968-alphabet-U+0062` |          0.569737 |        0.564999 |

Training aggregate: 0.531270 → 0.501043
Training mean: 0.501043; worst member: 0.564999

## Validation diagnostics

Validation members are evaluated after fitting and do not update parameters.

| Sample | Objective |
| ------ | --------: |

## Hold-out diagnostics

Hold-out samples are reported after selection and do not affect parameters.

| Sample                    | Objective |
| ------------------------- | --------: |
| `sas1968-alphabet-U+006B` |  0.755977 |
| `sas1968-alphabet-U+0068` |  0.791303 |

Training to hold-out mean gap: 0.272597
Model promotion remains pending; inspect overlays in the private work directory.

## Selected parameters

- `segments.0.c1.y`: 0.12450014
- `segments.0.c2.x`: 0.80668448
