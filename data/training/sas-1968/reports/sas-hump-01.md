# Fit report: sas-hump-01

Project: `sas-1968`

## Training members

| Sample                    | Initial objective | Final objective |
| ------------------------- | ----------------: | --------------: |
| `sas1968-alphabet-U+006E` |          0.567215 |        0.511573 |
| `sas1968-alphabet-U+006D` |          0.710538 |        0.632944 |

Training aggregate: 0.638877 → 0.572258
Training mean: 0.572258; worst member: 0.632944

## Validation diagnostics

Validation members are evaluated after fitting and do not update parameters.

| Sample | Objective |
| ------ | --------: |

## Hold-out diagnostics

Hold-out samples are reported after selection and do not affect parameters.

| Sample                    | Objective |
| ------------------------- | --------: |
| `sas1968-alphabet-U+0068` |  0.675105 |

Training to hold-out mean gap: 0.102847
Model promotion remains pending; inspect overlays in the private work directory.

## Selected parameters

- `segments.0.c1.x`: 0.34419850
- `segments.0.c1.y`: 0.26713484
- `segments.1.c1.x`: 1.08964190
