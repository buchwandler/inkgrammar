# Contributing

Before submitting model or code changes:

```bash
python -m pip install -e ".[dev]"
ruff check .
python -m pytest
inkgrammar model validate data/models/demo-looped.yaml --strict
```

## Evidence statuses

- `synthetic`: project-created test data;
- `inferred`: reconstructed from indirect evidence such as a final image;
- `source-backed`: the claimed fact is directly supported by a cited source;
- `reviewed`: source-backed and independently reviewed;
- `verified`: accepted after project QC/review.

Do not upgrade inferred trajectories to source-backed merely because their rendered shape fits
a scan.

## Historical scans

Keep copyright/redistribution rights separate from citation. A local scan may be usable for
private model fitting even when it must not be committed or included in public releases.
