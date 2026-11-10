# Fit report: sas-open-oval-01

Project: `sas-1968`

## Training members

| Sample                    | Initial objective | Final objective |
| ------------------------- | ----------------: | --------------: |
| `sas1968-alphabet-U+0063` |          0.815533 |        0.805092 |
| `sas1968-alphabet-U+0061` |          0.662032 |        0.622759 |

Training aggregate: 0.738783 → 0.713925
Training mean: 0.713925; worst member: 0.805092

## Validation diagnostics

Validation members are evaluated after fitting and do not update parameters.

| Sample | Objective |
| ------ | --------: |

## Hold-out diagnostics

Hold-out samples are reported after selection and do not affect parameters.

| Sample                    | Objective |
| ------------------------- | --------: |
| `sas1968-alphabet-U+0043` |  0.808363 |

Training to hold-out mean gap: 0.094437
Model promotion remains pending; inspect overlays in the private work directory.

## Selected parameters

- `segments.0.c1.y`: 1.03495169
- `segments.0.c2.x`: 0.09423634
- `segments.1.c1.x`: 0.18378333
