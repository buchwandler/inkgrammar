# Fit report: sas-underturn-01

Project: `sas-1968`

## Training members

| Sample                    | Initial objective | Final objective |
| ------------------------- | ----------------: | --------------: |
| `sas1968-alphabet-U+0069` |          0.227718 |        0.147112 |
| `sas1968-alphabet-U+0075` |          0.234299 |        0.169319 |

Training aggregate: 96.214436 → 66.242580
Training mean: 0.158216; worst member: 0.169319

## Validation diagnostics

Validation members are evaluated after fitting and do not update parameters.

| Sample | Objective |
| ------ | --------: |

## Hold-out diagnostics

Hold-out samples are reported after selection and do not affect parameters.

| Sample                    | Objective |
| ------------------------- | --------: |
| `sas1968-alphabet-U+0077` |  0.524813 |

Training to hold-out mean gap: 0.366597
Model promotion remains pending; inspect overlays in the private work directory.

## Selected parameters

- `occurrence.sas1968-alphabet-U+0069--i-underturn.delta.dx`: -0.51737605
- `occurrence.sas1968-alphabet-U+0069--i-underturn.delta.dy`: -0.02081295
- `occurrence.sas1968-alphabet-U+0069--i-underturn.delta.log_scale_x`: -0.02177518
- `occurrence.sas1968-alphabet-U+0069--i-underturn.delta.log_scale_y`: -0.07158288
- `occurrence.sas1968-alphabet-U+0075--u-underturn-a.delta.dx`: -2.10912443
- `occurrence.sas1968-alphabet-U+0075--u-underturn-a.delta.dy`: 0.91869817
- `occurrence.sas1968-alphabet-U+0075--u-underturn-a.delta.log_scale_x`: 0.00778112
- `occurrence.sas1968-alphabet-U+0075--u-underturn-a.delta.log_scale_y`: -0.04247373
- `occurrence.sas1968-alphabet-U+0075--u-underturn-b.delta.dx`: -0.55915857
- `occurrence.sas1968-alphabet-U+0075--u-underturn-b.delta.dy`: 1.11870327
- `occurrence.sas1968-alphabet-U+0075--u-underturn-b.delta.log_scale_x`: -0.10114242
- `occurrence.sas1968-alphabet-U+0075--u-underturn-b.delta.log_scale_y`: 0.03423022
- `shared.segments.0.c1.x`: 0.76580495
- `shared.segments.0.c1.y`: 0.54801679
- `shared.segments.0.c2.x`: -0.02295183
