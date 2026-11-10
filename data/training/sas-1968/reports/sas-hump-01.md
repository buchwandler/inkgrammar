# Fit report: sas-hump-01

Project: `sas-1968`

## Training members

| Sample                    | Initial objective | Final objective |
| ------------------------- | ----------------: | --------------: |
| `sas1968-alphabet-U+006E` |          0.334732 |        0.191148 |
| `sas1968-alphabet-U+006D` |          0.456396 |        0.196119 |

Training aggregate: 0.513316 → 0.353397
Training mean: 0.193634; worst member: 0.196119

## Validation diagnostics

Validation members are evaluated after fitting and do not update parameters.

| Sample | Objective |
| ------ | --------: |

## Hold-out diagnostics

Hold-out samples are reported after selection and do not affect parameters.

| Sample                    | Objective |
| ------------------------- | --------: |
| `sas1968-alphabet-U+0068` |  0.084455 |

Training to hold-out mean gap: -0.109179
Model promotion remains pending; inspect overlays in the private work directory.

## Selected parameters

- `occurrence.sas1968-alphabet-U+006D--m-hump-a.delta.dx`: 3.20550675
- `occurrence.sas1968-alphabet-U+006D--m-hump-a.delta.dy`: -0.93838849
- `occurrence.sas1968-alphabet-U+006D--m-hump-a.delta.log_scale_x`: 0.05525265
- `occurrence.sas1968-alphabet-U+006D--m-hump-a.delta.log_scale_y`: -0.02829242
- `occurrence.sas1968-alphabet-U+006D--m-hump-b.delta.dx`: 1.25204732
- `occurrence.sas1968-alphabet-U+006D--m-hump-b.delta.dy`: 0.36861036
- `occurrence.sas1968-alphabet-U+006D--m-hump-b.delta.log_scale_x`: 0.00739383
- `occurrence.sas1968-alphabet-U+006D--m-hump-b.delta.log_scale_y`: -0.02438364
- `occurrence.sas1968-alphabet-U+006E--n-hump.delta.dx`: 2.04335372
- `occurrence.sas1968-alphabet-U+006E--n-hump.delta.dy`: 0.14138666
- `occurrence.sas1968-alphabet-U+006E--n-hump.delta.log_scale_x`: -0.02834498
- `occurrence.sas1968-alphabet-U+006E--n-hump.delta.log_scale_y`: -0.00459818
- `shared.segments.0.c1.x`: 0.31071160
- `shared.segments.0.c1.y`: 0.31681219
- `shared.segments.1.c1.x`: 0.99084089
