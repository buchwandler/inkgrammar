# Examples

## SAS-1968 validation

Render the current canonical SAS-1968 runtime geometry:

```bash
python examples/sas1968/render_validation.py \
  --project data/training/sas-1968/project.yaml \
  --geometry canonical
```

Use `--geometry latest-candidate` to inspect only current successful reports with matching project, primitive, objective, backend, and fit-input hashes. Candidate output is explicitly unpromoted; stale or unavailable reports cause an error. Use `--geometry both` for common-scale comparisons. Generated images and preview models go to ignored local work output. See [sas1968/README.md](sas1968/README.md).
The current runtime model contains 12 glyphs and no promoted primitives. Objective v4 is experimental. Glyph coverage is incomplete and the runtime has zero connection rules, so supported-word previews show composition only, not connected cursive. They require no private source imagery.


## Source page layouts

`examples/layouts/` contains extraction layouts for privately supplied source pages. These are metadata only and do not contain source imagery.
