# Contributing

Before submitting model or code changes:

```bash
python -m pip install -e ".[dev]"
ruff check .
python -m pytest
inkgrammar model validate data/models/demo-looped.yaml --strict
```

## Legacy objective-v3 fitting

Objective version 3 is a frozen compatibility path for reproducibility. Do not add features or new consumers, or migrate its independent centerline pipeline to SkeletonPaths. New fitting work should use objective version 4.

## Evidence statuses

- `synthetic`: project-created test data;
- `inferred`: reconstructed from indirect evidence such as a final image;
- `source-backed`: the claimed fact is directly supported by a cited source;
- `reviewed`: source-backed and independently reviewed;
- `verified`: accepted after project QC/review.

Do not upgrade inferred trajectories to source-backed merely because their rendered shape fits
a scan.

## External geometry ownership and code review

Review changes against these package boundaries:

- PlanarCurve owns generic curve, SVG, transform, and polyline algorithms. InkGrammar owns coordinate-frame and fitting-domain adapters.
- SkeletonPaths owns raster skeleton graphs and topology-route enumeration and ranking. Routes are inferred candidates; physical order or direction requires reviewed trajectory evidence.
- StrokeOutline owns variable-width expansion, caps, joins, and contour composition. InkGrammar owns shaped-text traversal and width-profile selection. Pass typed `planarcurve.CubicPath` values when available, while retaining the SVG-string compatibility path.
- Preserve connection-fit point-sample distance semantics unless a separately versioned objective change is supported by regression comparison.
- Keep route ambiguity and orientation uncertainty visible. Do not turn topology or geometric fit into unsupported physical-order claims.

## Historical scans

Keep copyright/redistribution rights separate from citation. A local scan may be usable for
private model fitting even when it must not be committed or included in public releases.
