# Fit report: sas-underturn-01

Project: `sas-1968`

## Training members

| Sample                    | Initial objective | Final objective |
| ------------------------- | ----------------: | --------------: |
| `sas1968-alphabet-U+0069` |          7.516965 |        0.177563 |
| `sas1968-alphabet-U+0075` |          3.640882 |        0.195667 |

Training aggregate: 5.190299 → 0.274194
Training mean: 0.186615; worst member: 0.195667

## Validation diagnostics

Validation members are evaluated after fitting and do not update parameters.

| Sample | Objective |
| ------ | --------: |

## Hold-out diagnostics

Hold-out samples are reported after selection and do not affect parameters.

| Sample                    | Objective |
| ------------------------- | --------: |
| `sas1968-alphabet-U+0077` |  0.450314 |

Training to hold-out mean gap: 0.263699
Model promotion remains pending; inspect overlays in the private work directory.

## Selected parameters

- `occurrence.sas1968-alphabet-U+0069--i-underturn.delta.dx`: -0.59733702
- `occurrence.sas1968-alphabet-U+0069--i-underturn.delta.dy`: -0.33846704
- `occurrence.sas1968-alphabet-U+0069--i-underturn.delta.log_scale_x`: -0.03060422
- `occurrence.sas1968-alphabet-U+0069--i-underturn.delta.log_scale_y`: -0.01245204
- `occurrence.sas1968-alphabet-U+0075--u-underturn-a.delta.dx`: -0.43358947
- `occurrence.sas1968-alphabet-U+0075--u-underturn-a.delta.dy`: 0.68984315
- `occurrence.sas1968-alphabet-U+0075--u-underturn-a.delta.log_scale_x`: -0.00717719
- `occurrence.sas1968-alphabet-U+0075--u-underturn-a.delta.log_scale_y`: 0.01497046
- `occurrence.sas1968-alphabet-U+0075--u-underturn-b.delta.dx`: -0.84416041
- `occurrence.sas1968-alphabet-U+0075--u-underturn-b.delta.dy`: 0.56306729
- `occurrence.sas1968-alphabet-U+0075--u-underturn-b.delta.log_scale_x`: -0.02331547
- `occurrence.sas1968-alphabet-U+0075--u-underturn-b.delta.log_scale_y`: 0.02088608
- `shared.segments.0.c1.x`: 0.71851647
- `shared.segments.0.c1.y`: 0.50410115
- `shared.segments.0.c2.x`: -0.03411281
