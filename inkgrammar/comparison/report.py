from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .metrics import ComparisonAnalysis, analyze_comparison
from .models import ComparisonResult, ExtractedLetter
from .plots import PlotSeries, write_plot

_REFERENCE = (35, 105, 215)
_SAMPLE = (225, 75, 55)
_PURPLE = (145, 55, 165)

_ARTIFACTS = {
    "input_reference": "input-reference.png",
    "input_sample": "input-sample.png",
    "extraction_reference": "extraction-reference.png",
    "extraction_sample": "extraction-sample.png",
    "mask_reference": "mask-reference.png",
    "mask_sample": "mask-sample.png",
    "skeleton_reference": "skeleton-reference.png",
    "skeleton_sample": "skeleton-sample.png",
    "overlay_raster": "overlay-raster.png",
    "overlay_centerline": "overlay-centerline.png",
    "overlay_error_vectors": "overlay-error-vectors.png",
    "plot_position_error": "plot-position-error.png",
    "plot_xy": "plot-x-y.png",
    "plot_heading": "plot-heading.png",
    "plot_tangent_error": "plot-tangent-error.png",
    "plot_curvature": "plot-curvature.png",
    "plot_curvature_error": "plot-curvature-error.png",
    "plot_cumulative_turn": "plot-cumulative-turn.png",
    "plot_width": "plot-width.png",
    "report_json": "report.json",
    "report_markdown": "report.md",
    "report_html": "report.html",
}


@dataclass(frozen=True)
class WrittenComparisonReport:
    output_dir: Path
    payload: dict[str, Any]

    @property
    def json_path(self) -> Path:
        return self.output_dir / _ARTIFACTS["report_json"]

    @property
    def markdown_path(self) -> Path:
        return self.output_dir / _ARTIFACTS["report_markdown"]

    @property
    def html_path(self) -> Path:
        return self.output_dir / _ARTIFACTS["report_html"]


def _save_rgb_crop(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(path, format="PNG")


def _draw_point(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x, y = xy
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)
    draw.text((x + 5, y - 8), label, fill=color, font=ImageFont.load_default())


def _write_extraction_images(letter: ExtractedLetter, output_dir: Path, side: str) -> None:
    image = letter.image.convert("RGB")
    mask = letter.mask_result.mask
    mask_path = output_dir / _ARTIFACTS[f"mask_{side}"]
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

    extraction = image.copy()
    draw = ImageDraw.Draw(extraction)
    coordinates = np.argwhere(mask)
    if len(coordinates):
        top, left = coordinates.min(axis=0)
        bottom, right = coordinates.max(axis=0)
        draw.rectangle((left, top, right, bottom), outline=(255, 180, 0), width=2)
    else:
        draw.text((4, 4), "No ink detected", fill=(220, 30, 30))
    endpoints = letter.diagnostics.topology.endpoints
    for index, point in enumerate(endpoints[:2]):
        _draw_point(draw, point, _REFERENCE if index == 0 else _SAMPLE, "AB"[index])
    extraction.save(output_dir / _ARTIFACTS[f"extraction_{side}"], format="PNG")

    skeleton_image = image.copy()
    skeleton_draw = ImageDraw.Draw(skeleton_image)
    ys, xs = np.nonzero(letter.diagnostics.skeleton)
    skeleton_draw.point(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=(20, 195, 95))
    for point in letter.diagnostics.topology.endpoints:
        _draw_point(skeleton_draw, point, (20, 110, 230), "E")
    for point in letter.diagnostics.topology.junctions:
        _draw_point(skeleton_draw, point, (220, 40, 180), "J")
    skeleton_image.save(output_dir / _ARTIFACTS[f"skeleton_{side}"], format="PNG")


def _canvas_bounds(
    first: np.ndarray,
    second: np.ndarray,
    padding: int = 12,
) -> tuple[int, int, int, int]:
    points = np.concatenate((first, second), axis=0)
    minimum = np.floor(points.min(axis=0)).astype(int) - padding
    maximum = np.ceil(points.max(axis=0)).astype(int) + padding
    return int(minimum[0]), int(minimum[1]), int(maximum[0]), int(maximum[1])


def _canvas_size(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
    left, top, right, bottom = bounds
    return max(1, right - left + 1), max(1, bottom - top + 1)


def _canvas_points(points: np.ndarray, bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    left, top, _, _ = bounds
    coordinates = np.rint(points - np.array((left, top))).astype(int)
    return list(zip(coordinates[:, 0].tolist(), coordinates[:, 1].tolist(), strict=True))


def _draw_centerline_overlay(
    output: Path,
    reference_xy: np.ndarray,
    sample_xy: np.ndarray,
    *,
    error_vectors: bool = False,
) -> None:
    bounds = _canvas_bounds(reference_xy, sample_xy)
    image = Image.new("RGB", _canvas_size(bounds), "white")
    draw = ImageDraw.Draw(image)
    reference_points = _canvas_points(reference_xy, bounds)
    sample_points = _canvas_points(sample_xy, bounds)
    draw.line(reference_points, fill=_REFERENCE, width=2)
    draw.line(sample_points, fill=_SAMPLE, width=2)
    for points, color, start_label, end_label in (
        (reference_points, _REFERENCE, "R start", "R end"),
        (sample_points, _SAMPLE, "S start", "S end"),
    ):
        for point, label in ((points[0], start_label), (points[-1], end_label)):
            x, y = point
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)
            draw.text((x + 5, y - 8), label, fill=color, font=ImageFont.load_default())
    if error_vectors:
        indexes = np.linspace(0, len(reference_points) - 1, 24).round().astype(int)
        for index in indexes:
            start = reference_points[index]
            end = sample_points[index]
            draw.line((start, end), fill=(55, 55, 55), width=1)
            draw.ellipse((start[0] - 2, start[1] - 2, start[0] + 2, start[1] + 2), fill=_REFERENCE)
            draw.ellipse((end[0] - 2, end[1] - 2, end[0] + 2, end[1] + 2), fill=_SAMPLE)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def _write_raster_overlay(result: ComparisonResult, output: Path) -> None:
    reference_yx = np.argwhere(result.reference.mask_result.mask)
    sample_yx = np.argwhere(result.sample.mask_result.mask)
    reference_xy = reference_yx[:, ::-1].astype(float)
    sample_xy = result.alignment.transform.apply(sample_yx[:, ::-1].astype(float))
    bounds = _canvas_bounds(reference_xy, sample_xy, padding=4)
    width, height = _canvas_size(bounds)
    left, top, _, _ = bounds
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    reference_pixels = np.zeros((height, width), dtype=bool)
    sample_pixels = np.zeros((height, width), dtype=bool)
    for points, target in ((reference_xy, reference_pixels), (sample_xy, sample_pixels)):
        coordinates = np.rint(points - np.array((left, top))).astype(int)
        x = np.clip(coordinates[:, 0], 0, width - 1)
        y = np.clip(coordinates[:, 1], 0, height - 1)
        target[y, x] = True
    image[sample_pixels] = _SAMPLE
    image[reference_pixels] = _REFERENCE
    image[reference_pixels & sample_pixels] = _PURPLE
    Image.fromarray(image).save(output, format="PNG")


def _series(label: str, x: np.ndarray, y: np.ndarray, color: tuple[int, int, int]) -> PlotSeries:
    return label, x, y, color


def _write_plots(analysis: ComparisonAnalysis, output_dir: Path) -> None:
    q = analysis.progress
    data = analysis.series
    plots = [
        (
            "plot_position_error",
            "Normalized position error",
            "Normalized stroke progress q",
            "Error / reference arc length",
            [_series("distance", q, data["position_error"], _PURPLE)],
        ),
        (
            "plot_xy",
            "X/Y coordinates by progress",
            "Normalized stroke progress q",
            "Coordinate / reference arc length",
            [
                _series("reference x", q, data["reference_xy_normalized"][:, 0], _REFERENCE),
                _series("sample x", q, data["sample_xy_normalized"][:, 0], _SAMPLE),
                _series("reference y", q, data["reference_xy_normalized"][:, 1], (25, 155, 145)),
                _series("sample y", q, data["sample_xy_normalized"][:, 1], (225, 145, 30)),
            ],
        ),
        (
            "plot_heading",
            "Heading by progress",
            "Normalized stroke progress q",
            "Heading (degrees)",
            [
                _series("reference", q, data["reference_heading_deg"], _REFERENCE),
                _series("sample", q, data["sample_heading_deg"], _SAMPLE),
            ],
        ),
        (
            "plot_tangent_error",
            "Signed tangent error",
            "Normalized stroke progress q",
            "Sample minus reference (degrees)",
            [_series("tangent error", q, data["tangent_error_deg"], _PURPLE)],
        ),
        (
            "plot_curvature",
            "Normalized turn-rate",
            "Normalized stroke progress q",
            "Turn-rate (radians / progress)",
            [
                _series("reference", q, data["reference_turn_rate"], _REFERENCE),
                _series("sample", q, data["sample_turn_rate"], _SAMPLE),
            ],
        ),
        (
            "plot_curvature_error",
            "Turn-rate error",
            "Normalized stroke progress q",
            "Sample minus reference (radians / progress)",
            [_series("turn-rate error", q, data["turn_rate_error"], _PURPLE)],
        ),
        (
            "plot_cumulative_turn",
            "Cumulative turn",
            "Normalized stroke progress q",
            "Turn from start (radians)",
            [
                _series("reference", q, data["reference_cumulative_turn"], _REFERENCE),
                _series("sample", q, data["sample_cumulative_turn"], _SAMPLE),
            ],
        ),
        (
            "plot_width",
            "Stroke-width diagnostic",
            "Normalized stroke progress q",
            "Estimated width (pixels in each crop)",
            [
                _series("reference", q, data["reference_width_px"], _REFERENCE),
                _series("sample", q, data["sample_width_px"], _SAMPLE),
            ],
        ),
    ]
    for key, title, x_label, y_label, series in plots:
        write_plot(
            output_dir / _ARTIFACTS[key],
            title=title,
            x_label=x_label,
            y_label=y_label,
            series=series,
        )


def _json_series(analysis: ComparisonAnalysis) -> dict[str, list]:
    return {name: np.asarray(values).tolist() for name, values in analysis.series.items()}



def _crop_description(crop: dict[str, int]) -> str:
    return (
        f"x={crop['x']}, y={crop['y']}, "
        f"width={crop['width']}, height={crop['height']}"
    )

def _report_payload(result: ComparisonResult, analysis: ComparisonAnalysis) -> dict[str, Any]:
    transform = result.alignment.transform
    def input_payload(letter: ExtractedLetter, side: str) -> dict[str, Any]:
        return {
            "source_name": letter.source_path.name,
            "crop": letter.crop.model_dump(mode="json"),
            "mask_method": letter.mask_result.method,
            "mask_confidence": letter.mask_result.confidence,
            "warnings": letter.mask_result.warnings,
            "topology": letter.diagnostics.topology.model_dump(mode="json"),
            "path_coverage": letter.ordered_path.coverage,
            "original_artifact": _ARTIFACTS[f"input_{side}"],
            "mask_artifact": _ARTIFACTS[f"mask_{side}"],
            "skeleton_artifact": _ARTIFACTS[f"skeleton_{side}"],
            "extraction_artifact": _ARTIFACTS[f"extraction_{side}"],
        }

    warnings = [
        *result.warnings,
        *result.reference.mask_result.warnings,
        *result.sample.mask_result.warnings,
    ]
    return {
        "format": "inkgrammar-letter-comparison",
        "format_version": 1,
        "letter": result.letter,
        "direction_source": "inferred",
        "inputs": {
            "reference": input_payload(result.reference, "reference"),
            "sample": input_payload(result.sample, "sample"),
        },
        "alignment": {
            "direction": result.alignment.direction,
            "scale": transform.scale,
            "rotation_deg": transform.rotation_deg,
            "translation_xy": [transform.dx, transform.dy],
            "reflection": False,
            "mean_squared_error_px2": result.alignment.mean_squared_error,
            "interpretation": "Nuisance normalization, not a handwriting-quality score.",
        },
        "geometry": {
            "sample_count": len(analysis.progress),
            "progress": "normalized arc-length progress from 0 to 1",
            "position_error_normalization": "reference arc length equals 1.0",
            "smoothing": {
                "algorithm": "Savitzky-Golay",
                "window_length": result.smoothing_window,
                "polynomial_order": result.smoothing_polynomial_order,
            },
        },
        "metrics": analysis.metrics,
        "series": _json_series(analysis),
        "artifacts": {key: value for key, value in _ARTIFACTS.items()},
        "warnings": list(dict.fromkeys(warnings)),
    }


def _markdown(result: ComparisonResult, payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    inputs = payload["inputs"]
    alignment = payload["alignment"]
    rows = [
        f"# Letter comparison: {result.letter}",
        "",
        "## Inputs and extraction",
        "",
        "| | Reference | My writing |",
        "| --- | --- | --- |",
        f"| Source | {inputs['reference']['source_name']} | {inputs['sample']['source_name']} |",
        f"| Crop (x, y, w, h) | {_crop_description(inputs['reference']['crop'])} | "
        f"{_crop_description(inputs['sample']['crop'])} |",
        f"| Ink method | {inputs['reference']['mask_method']} | "
        f"{inputs['sample']['mask_method']} |",
        f"| Confidence | {inputs['reference']['mask_confidence']:.3f} | "
        f"{inputs['sample']['mask_confidence']:.3f} |",
        "",
        "![Reference extraction](extraction-reference.png)",
        "![Sample extraction](extraction-sample.png)",
        "",
        "## Shape comparison",
        "",
        f"Alignment: scale {alignment['scale']:.5g}, "
        f"rotation {alignment['rotation_deg']:.5g} degrees, "
        f"translation {alignment['translation_xy']}; "
        f"sample direction {alignment['direction']} (inferred).",
        "",
        "![Aligned centerlines](overlay-centerline.png)",
        "![Error vectors](overlay-error-vectors.png)",
        "![Position error](plot-position-error.png)",
        "",
        "## Geometry metrics",
        "",
        f"- Position error mean / median / p95 / maximum: {metrics['path']['mean_error']:.5g} / "
        f"{metrics['path']['median_error']:.5g} / {metrics['path']['p95_error']:.5g} / "
        f"{metrics['path']['max_error']:.5g}, normalized by reference arc length.",
        f"- X/Y RMSE: {metrics['path']['x_rmse']:.5g} / {metrics['path']['y_rmse']:.5g}.",
        f"- Heading mean absolute / p95 error: {metrics['heading']['mean_abs_error_deg']:.5g} / "
        f"{metrics['heading']['p95_abs_error_deg']:.5g} degrees.",
        f"- Turn-rate MAE / RMSE / p95 absolute error: {metrics['turn_rate']['mae']:.5g} / "
        f"{metrics['turn_rate']['rmse']:.5g} / {metrics['turn_rate']['p95_abs']:.5g} "
        f"{metrics['turn_rate']['units']}.",
        f"- Endpoint start / end position error: {metrics['endpoints']['start_error']:.5g} / "
        f"{metrics['endpoints']['end_error']:.5g}.",
        "- No composite handwriting score is calculated. Width is an unscored pixel diagnostic.",
        "",
        "![X/Y progress](plot-x-y.png)",
        "![Heading](plot-heading.png)",
        "![Tangent error](plot-tangent-error.png)",
        "![Turn-rate](plot-curvature.png)",
        "![Turn-rate error](plot-curvature-error.png)",
        "![Cumulative turn](plot-cumulative-turn.png)",
        "![Width diagnostic](plot-width.png)",
        "",
        "## Topology",
        "",
        f"Reference: {metrics['topology']['reference']}",
        "",
        f"Sample: {metrics['topology']['sample']}",
        "",
        "## Machine-readable data",
        "",
        "See [report.json](report.json) for full coordinate and progress series.",
        "",
    ]
    return "\n".join(rows)


def _html(result: ComparisonResult, payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    inputs = payload["inputs"]
    transform = payload["alignment"]

    def image(name: str, alt: str) -> str:
        return (
            f'<figure><img src="{html.escape(name)}" alt="{html.escape(alt)}">'
            f"<figcaption>{html.escape(alt)}</figcaption></figure>"
        )

    rows = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>Letter comparison: {html.escape(result.letter)}</title>",
        "<style>body{font:16px sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "img{max-width:100%;height:auto;border:1px solid #bbb}figure{margin:0.5rem}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}"
        "td,th{padding:.35rem .7rem;border:1px solid #ccc;text-align:left}"
        "table{border-collapse:collapse}</style></head><body>",
        f"<h1>Letter comparison: {html.escape(result.letter)}</h1>",
        "<h2>1. Extraction</h2>",
        "<table><thead><tr><th></th><th>Reference</th><th>My writing</th></tr></thead><tbody>",
        f"<tr><th>Source</th><td>{html.escape(inputs['reference']['source_name'])}</td>",
        f"<td>{html.escape(inputs['sample']['source_name'])}</td></tr>",
        f"<tr><th>Crop</th><td>{html.escape(_crop_description(inputs['reference']['crop']))}</td>",
        f"<td>{html.escape(_crop_description(inputs['sample']['crop']))}</td></tr>",
        f"<tr><th>Ink extraction</th><td>{html.escape(inputs['reference']['mask_method'])} "
        f"({inputs['reference']['mask_confidence']:.3f})</td>",
        f"<td>{html.escape(inputs['sample']['mask_method'])} "
        f"({inputs['sample']['mask_confidence']:.3f})</td></tr>",
        "</tbody></table>",
        "<div class=grid>",
        image(_ARTIFACTS["input_reference"], "Reference crop"),
        image(_ARTIFACTS["input_sample"], "My writing crop"),
        image(_ARTIFACTS["extraction_reference"], "Reference ink bounding box and endpoints"),
        image(_ARTIFACTS["extraction_sample"], "Sample ink bounding box and endpoints"),
        image(_ARTIFACTS["mask_reference"], "Reference ink mask"),
        image(_ARTIFACTS["mask_sample"], "Sample ink mask"),
        image(_ARTIFACTS["skeleton_reference"], "Reference skeleton, endpoints and junctions"),
        image(_ARTIFACTS["skeleton_sample"], "Sample skeleton, endpoints and junctions"),
        "</div>",
        "<h2>2. Shape overlay</h2>",
        f"<p>Similarity scale {transform['scale']:.5g}, "
        f"rotation {transform['rotation_deg']:.5g} degrees, "
        f"translation {transform['translation_xy']}, "
        f"direction {html.escape(transform['direction'])} "
        "(inferred from comparison, not measured from the image).</p>",
        "<div class=grid>",
        image(_ARTIFACTS["overlay_raster"], "Aligned raster masks"),
        image(_ARTIFACTS["overlay_centerline"], "Aligned centerlines and endpoints"),
        "</div>",
        "<h2>3. Local position difference</h2><div class=grid>",
        image(_ARTIFACTS["overlay_error_vectors"], "Error vectors"),
        image(_ARTIFACTS["plot_position_error"], "Pointwise normalized position error"),
        "</div>",
        "<h2>4. Writing direction geometry</h2><div class=grid>",
        image(_ARTIFACTS["plot_xy"], "X and Y coordinates by progress"),
        image(_ARTIFACTS["plot_heading"], "Heading by progress"),
        image(_ARTIFACTS["plot_tangent_error"], "Signed tangent error"),
        "</div>",
        "<h2>5. Curvature</h2><div class=grid>",
        image(_ARTIFACTS["plot_curvature"], "Normalized turn-rate"),
        image(_ARTIFACTS["plot_curvature_error"], "Turn-rate error"),
        image(_ARTIFACTS["plot_cumulative_turn"], "Cumulative turn"),
        image(_ARTIFACTS["plot_width"], "Unscored stroke-width diagnostic"),
        "</div>",
        "<h2>6. Endpoints and topology</h2>",
        f"<p>Start/end errors: {metrics['endpoints']['start_error']:.5g} / "
        f"{metrics['endpoints']['end_error']:.5g}, normalized by reference arc length. "
        f"Tangent errors: {metrics['endpoints']['start_tangent_error_deg']:.5g} / "
        f"{metrics['endpoints']['end_tangent_error_deg']:.5g} degrees.</p>",
        "<div class=grid><section><h3>Reference topology</h3><pre>"
        f"{html.escape(json.dumps(metrics['topology']['reference'], indent=2))}</pre></section>",
        "<section><h3>Sample topology</h3><pre>"
        f"{html.escape(json.dumps(metrics['topology']['sample'], indent=2))}</pre></section></div>",
        "<h2>7. Raw metrics</h2><table><tbody>",
        f"<tr><th>Position error mean / median / p95 / maximum</th><td>"
        f"{metrics['path']['mean_error']:.5g} / {metrics['path']['median_error']:.5g} / "
        f"{metrics['path']['p95_error']:.5g} / {metrics['path']['max_error']:.5g}</td></tr>",
        f"<tr><th>X / Y RMSE</th><td>{metrics['path']['x_rmse']:.5g} / "
        f"{metrics['path']['y_rmse']:.5g}</td></tr>",
        f"<tr><th>Heading mean absolute / p95 error</th><td>"
        f"{metrics['heading']['mean_abs_error_deg']:.5g} / "
        f"{metrics['heading']['p95_abs_error_deg']:.5g} degrees</td></tr>",
        f"<tr><th>Turn-rate MAE / RMSE / p95 absolute error</th><td>"
        f"{metrics['turn_rate']['mae']:.5g} / {metrics['turn_rate']['rmse']:.5g} / "
        f"{metrics['turn_rate']['p95_abs']:.5g} "
        f"{html.escape(metrics['turn_rate']['units'])}</td></tr>",
        f"<tr><th>Reference / sample raw path length</th><td>"
        f"{metrics['path']['reference_arc_length_px']:.5g} / "
        f"{metrics['path']['sample_arc_length_px']:.5g} pixels</td></tr>",
        "</tbody></table>",
        "<p>No composite correctness score is calculated. Raw length ratio is not physical size. "
        "Width is an unscored diagnostic.</p>",
        "<h2>8. Machine-readable output</h2><p><a href=\"report.json\">report.json</a>"
        " contains full coordinates, progress series, metrics, and artifact names. "
        "<a href=\"report.md\">report.md</a> provides a portable summary.</p>",
        "</body></html>",
    ]
    return "\n".join(rows)


def write_comparison_report(
    result: ComparisonResult,
    output_dir: str | Path,
) -> WrittenComparisonReport:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    analysis = analyze_comparison(result)

    _save_rgb_crop(result.reference.image, destination / _ARTIFACTS["input_reference"])
    _save_rgb_crop(result.sample.image, destination / _ARTIFACTS["input_sample"])
    _write_extraction_images(result.reference, destination, "reference")
    _write_extraction_images(result.sample, destination, "sample")
    _write_raster_overlay(result, destination / _ARTIFACTS["overlay_raster"])
    _draw_centerline_overlay(
        destination / _ARTIFACTS["overlay_centerline"],
        result.reference.resampled_xy,
        result.alignment.aligned_sample_xy,
    )
    _draw_centerline_overlay(
        destination / _ARTIFACTS["overlay_error_vectors"],
        result.reference.resampled_xy,
        result.alignment.aligned_sample_xy,
        error_vectors=True,
    )
    _write_plots(analysis, destination)

    payload = _report_payload(result, analysis)
    (destination / _ARTIFACTS["report_json"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / _ARTIFACTS["report_markdown"]).write_text(
        _markdown(result, payload),
        encoding="utf-8",
    )
    (destination / _ARTIFACTS["report_html"]).write_text(
        _html(result, payload),
        encoding="utf-8",
    )
    return WrittenComparisonReport(destination, payload)
