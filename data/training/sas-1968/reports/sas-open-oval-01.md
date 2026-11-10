# Fit report: sas-open-oval-01

Project: `sas-1968`

## Training members

| Sample                    | Initial objective | Final objective |
| ------------------------- | ----------------: | --------------: |
| `sas1968-alphabet-U+0063` |          0.441407 |        0.371804 |
| `sas1968-alphabet-U+0061` |          0.252413 |        0.175896 |

Training aggregate: 0.463325 → 0.392484
Training mean: 0.273850; worst member: 0.371804

## Validation diagnostics

Validation members are evaluated after fitting and do not update parameters.

| Sample | Objective |
| ------ | --------: |

## Hold-out diagnostics

Hold-out samples are reported after selection and do not affect parameters.

| Sample                    | Objective |
| ------------------------- | --------: |
| `sas1968-alphabet-U+0043` |  0.438808 |

Training to hold-out mean gap: 0.164958
Model promotion remains pending; inspect overlays in the private work directory.

## Selected parameters

- `occurrence.sas1968-alphabet-U+0061--a-oval.delta.dx`: 0.69466974
- `occurrence.sas1968-alphabet-U+0061--a-oval.delta.dy`: -0.78443683
- `occurrence.sas1968-alphabet-U+0061--a-oval.delta.log_scale_x`: -0.01920695
- `occurrence.sas1968-alphabet-U+0061--a-oval.delta.log_scale_y`: 0.00771174
- `occurrence.sas1968-alphabet-U+0063--c-oval.delta.dx`: 0.48555956
- `occurrence.sas1968-alphabet-U+0063--c-oval.delta.dy`: 0.17783320
- `occurrence.sas1968-alphabet-U+0063--c-oval.delta.log_scale_x`: 0.00546959
- `occurrence.sas1968-alphabet-U+0063--c-oval.delta.log_scale_y`: -0.01554298
- `shared.segments.0.c1.y`: 1.03051611
- `shared.segments.0.c2.x`: 0.09249544
- `shared.segments.1.c1.x`: 0.10460715
