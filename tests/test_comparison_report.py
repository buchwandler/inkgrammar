from __future__ import annotations

import json

from PIL import Image, ImageDraw
from planarcurve.transforms import Similarity2D

from inkgrammar.comparison import (
    AlignmentResult,
    ComparisonResult,
    CropRect,
    ExtractedLetter,
    analyze_skeleton,
    extract_ink_mask,
    recover_ordered_path,
    resample_path,
)


def _extracted_letter(path, color: str) -> ExtractedLetter:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.line((10, 50, 25, 15, 50, 48), fill=color, width=5)
    mask = extract_ink_mask(image)
    diagnostics = analyze_skeleton(mask.mask)
    ordered_path = recover_ordered_path(diagnostics)
    return ExtractedLetter(
        source_path=path,
        crop=CropRect(x=0, y=0, width=64, height=64),
        image=image,
        mask_result=mask,
        diagnostics=diagnostics,
        ordered_path=ordered_path,
        resampled_xy=resample_path(ordered_path.xy),
    )


def test_report_writes_machine_and_human_outputs(tmp_path) -> None:
    reference = _extracted_letter("reference.png", "black")
    sample = _extracted_letter("sample.png", "black")
    alignment = AlignmentResult(
        transform=Similarity2D(),
        direction="forward",
        aligned_sample_xy=reference.resampled_xy,
        mean_squared_error=0.0,
    )
    result = ComparisonResult("e", reference, sample, alignment)

    from inkgrammar.comparison.report import write_comparison_report

    report = write_comparison_report(result, tmp_path / "comparison")

    required = {
        "report.html",
        "report.md",
        "report.json",
        "input-reference.png",
        "input-sample.png",
        "mask-reference.png",
        "mask-sample.png",
        "skeleton-reference.png",
        "skeleton-sample.png",
        "overlay-raster.png",
        "overlay-centerline.png",
        "overlay-error-vectors.png",
        "plot-position-error.png",
        "plot-x-y.png",
        "plot-heading.png",
        "plot-tangent-error.png",
        "plot-curvature.png",
        "plot-curvature-error.png",
        "plot-cumulative-turn.png",
        "plot-width.png",
    }
    assert required <= {item.name for item in report.output_dir.iterdir()}

    payload = json.loads((report.output_dir / "report.json").read_text())
    assert payload["letter"] == "e"
    assert payload["alignment"]["scale"] > 0.0
    assert payload["metrics"]["path"]["mean_error"] < 1e-12
    assert payload["metrics"]["width"]["scored"] is False
    assert len(payload["series"]["progress"]) == 256
    assert "score" not in payload
    html = (report.output_dir / "report.html").read_text()
    assert "overlay-centerline.png" in html
    assert "plot-curvature-error.png" in html
    assert all(name in html for name in required if name.endswith(".png"))
    assert (report.output_dir / "input-reference.png").read_bytes()
