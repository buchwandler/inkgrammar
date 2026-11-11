# Fit report: sas-hump-01

Project: `sas-1968`

## Training members

| Sample                    | Initial objective | Final objective |
| ------------------------- | ----------------: | --------------: |
| `sas1968-alphabet-U+006E` |          0.169775 |        0.077402 |
| `sas1968-alphabet-U+006D` |          0.305102 |        0.135313 |

Training aggregate: 98.567903 → 46.936802
Training mean: 0.106357; worst member: 0.135313

## Validation diagnostics

Validation members are evaluated after fitting and do not update parameters.

| Sample | Objective |
| ------ | --------: |

## Hold-out diagnostics

Hold-out samples are reported after selection and do not affect parameters.

| Sample                    | Objective |
| ------------------------- | --------: |
| `sas1968-alphabet-U+0068` |  0.285394 |

Training to hold-out mean gap: 0.179037
Model promotion remains pending; inspect overlays in the private work directory.

## Selected parameters

- `occurrence.sas1968-alphabet-U+006D--m-hump-a.delta.dx`: 3.99895603
- `occurrence.sas1968-alphabet-U+006D--m-hump-a.delta.dy`: 1.15371147
- `occurrence.sas1968-alphabet-U+006D--m-hump-a.delta.log_scale_x`: 0.04429633
- `occurrence.sas1968-alphabet-U+006D--m-hump-a.delta.log_scale_y`: -0.10901497
- `occurrence.sas1968-alphabet-U+006D--m-hump-b.delta.dx`: 1.55261380
- `occurrence.sas1968-alphabet-U+006D--m-hump-b.delta.dy`: 1.62494318
- `occurrence.sas1968-alphabet-U+006D--m-hump-b.delta.log_scale_x`: 0.04714367
- `occurrence.sas1968-alphabet-U+006D--m-hump-b.delta.log_scale_y`: -0.05319311
- `occurrence.sas1968-alphabet-U+006E--n-hump.delta.dx`: 3.42053388
- `occurrence.sas1968-alphabet-U+006E--n-hump.delta.dy`: 1.40382804
- `occurrence.sas1968-alphabet-U+006E--n-hump.delta.log_scale_x`: -0.04250638
- `occurrence.sas1968-alphabet-U+006E--n-hump.delta.log_scale_y`: -0.03200672
- `shared.segments.0.c1.x`: 0.30422988
- `shared.segments.0.c1.y`: 0.43098072
- `shared.segments.1.c1.x`: 0.95880224
