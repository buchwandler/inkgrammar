"""Trajectory fitting CLI commands (the brief's sections 50 and 27).

The fitter resolves its input by observation ID through the materialized
dataset index: observation ID -> dataset index -> reviewed/accepted fitting
mask -> stroke annotation -> centerline candidate -> trajectory fit. No
ad-hoc image paths.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer
import yaml
from PIL import Image

from .dataset.project import load_dataset_project
from .dataset.transforms import source_px_to_glyph_units
from .fitting.curvature import evaluate_stroke_at
from .fitting.curvature_fit import (
    CurvatureFitResult,
    estimate_observed_heading,
    fit_curvature_stroke,
    matched_fit_state,
    resample_by_arclength,
)
from .fitting.family_fit_v4 import prepare_topology_target

DEFAULT_DATASET_ROOT = Path("work/sas-1968/dataset")
DEFAULT_FIT_ROOT = Path("work/sas-1968/trajectory")
ANNOTATIONS_DIR = Path("work/sas-1968/annotations/sas-1968")


@dataclass(frozen=True)
class FitContext:
    observation_id: str
    record: dict
    target_xy: np.ndarray
    stroke_annotation: dict | None
    input_hashes: dict[str, str]


def _ductus_polyline(graph) -> np.ndarray:
    """Ordered pen path covering every skeleton edge (small-graph Postman).

    The topology-aware skeleton graph is tiny (ends and junctions only), so
    odd-degree junctions are paired and their connecting edges doubled, then
    an Euler walk reconstructs the ductus order. For a cursive ``e`` the
    middle connector is genuinely traversed twice by the pen, so no spurious
    retrace is introduced.
    """
    incident = {node.id: [] for node in graph.nodes}
    for index, edge in enumerate(graph.edges):
        incident[edge.a].append((index, edge.b))
        if edge.b != edge.a:
            incident[edge.b].append((index, edge.a))

    def edge_path(source: int, target: int) -> list[int]:
        previous = {source: None}
        queue = [source]
        for node in queue:
            if node == target:
                break
            for index, other in incident[node]:
                if other not in previous:
                    previous[other] = (node, index)
                    queue.append(other)
        if target not in previous:
            return []
        indexes = []
        current = target
        while previous[current] is not None:
            node, index = previous[current]
            indexes.append(index)
            current = node
        indexes.reverse()
        return indexes

    nodes_by_id = {node.id: node for node in graph.nodes}
    ends = sorted(
        (node for node in graph.nodes if len(incident[node.id]) == 1),
        key=lambda node: tuple(node.xy),
    )
    if len(ends) >= 2:
        start, goal = ends[0].id, ends[-1].id
    else:
        start = goal = min(graph.nodes, key=lambda node: tuple(node.xy)).id

    odd = sorted(
        (node.id for node in graph.nodes if len(incident[node.id]) % 2 == 1),
        key=lambda node_id: tuple(nodes_by_id[node_id].xy),
    )
    remaining = [node_id for node_id in odd if node_id not in (start, goal)]
    pairs = []
    while len(remaining) >= 2:
        best = None
        for first_index in range(len(remaining)):
            for second_index in range(first_index + 1, len(remaining)):
                distance = len(
                    edge_path(remaining[first_index], remaining[second_index])
                )
                if best is None or distance < best[0]:
                    best = (distance, first_index, second_index)
        _, first_index, second_index = best
        pairs.append((remaining[first_index], remaining[second_index]))
        remaining = [
            node_id
            for index, node_id in enumerate(remaining)
            if index not in (first_index, second_index)
        ]
    multiplicity = [1] * len(graph.edges)
    for source, target in pairs:
        for index in edge_path(source, target):
            multiplicity[index] += 1

    stack_nodes = [start]
    stack_edges: list = [None]
    result_nodes: list = []
    result_edges: list = []
    while stack_nodes:
        current = stack_nodes[-1]
        step = None
        for index, other in incident[current]:
            if multiplicity[index] > 0:
                multiplicity[index] -= 1
                step = (index, other)
                break
        if step is None:
            result_nodes.append(stack_nodes.pop())
            result_edges.append(stack_edges.pop())
        else:
            stack_nodes.append(step[1])
            stack_edges.append(step)
    result_nodes.reverse()
    result_edges.reverse()

    walk: list = []
    previous_node = result_nodes[0]
    previous_index = None
    for node, edge_step in zip(result_nodes[1:], result_edges[1:], strict=True):
        index, _ = edge_step
        if index == previous_index:
            # immediate out-and-back: the second pass is a pure retrace
            previous_node = node
            previous_index = None
            continue
        previous_index = index
        edge = graph.edges[index]
        segment = np.asarray(edge.pixels, dtype=float)
        if edge.a != previous_node:
            segment = segment[::-1]
        walk.extend(segment if not walk else segment[1:])
        previous_node = node
    return np.asarray(walk, dtype=float)


def _scipy_version() -> str:
    import scipy

    return scipy.__version__


def _smooth_polyline(points: np.ndarray, *, window: int) -> np.ndarray:
    """Light edge-preserving moving average for noisy skeleton centerlines."""
    if window <= 1 or len(points) < 3:
        return points
    half = window // 2
    smoothed = np.empty_like(points)
    for axis in range(2):
        padded = np.pad(points[:, axis], (half, half), mode="edge")
        kernel = np.ones(window) / window
        smoothed[:, axis] = np.convolve(padded, kernel, mode="valid")
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fit_context(
    project: Path,
    observation_id: str,
    *,
    dataset_root: Path,
    repository_root: Path | None = None,
) -> FitContext:
    """Resolve an observation ID through the dataset index into fit inputs."""
    root = Path(repository_root or Path.cwd())
    index_path = Path(dataset_root) / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    record = index["observations"].get(observation_id)
    if record is None:
        raise typer.BadParameter(f"unknown observation id: {observation_id}")

    frame_data = record.get("frame")
    if frame_data is None:
        raise typer.BadParameter(
            f"observation {observation_id} has no coordinate frame candidate"
        )
    from .dataset.transforms import ObservationFrame

    frame = ObservationFrame.model_validate(frame_data)
    source_bbox = record["source_bbox"]
    mask_path = Path(dataset_root) / record["artifacts"]["fitting_mask"]
    if not mask_path.is_file():
        raise typer.BadParameter(f"observation {observation_id} has no fitting mask")
    target_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    prepared = prepare_topology_target(target_mask)
    crop_xy = _ductus_polyline(prepared.graph)
    source_xy = crop_xy + np.array([source_bbox[0], source_bbox[1]], dtype=float)
    target_xy = source_px_to_glyph_units(source_xy, frame)
    target_xy = _smooth_polyline(target_xy, window=5)

    annotation_path = root / ANNOTATIONS_DIR / f"{observation_id}.yaml"
    stroke_annotation = None
    if annotation_path.is_file():
        stroke_annotation = yaml.safe_load(annotation_path.read_text(encoding="utf-8"))

    input_hashes = {
        "source_sha256": index["source_hashes"][record["source_id"]],
        "dataset_index_sha256": _sha256_file(index_path),
        "fitting_mask_sha256": record["artifact_hashes"].get(
            "isolation_mask", record["artifact_hashes"].get("clean_mask", "")
        ),
    }
    if stroke_annotation is not None:
        input_hashes["annotation_sha256"] = _sha256_file(annotation_path)
    recipe_path = Path("data/datasets") / f"{index['model_id']}.yaml"
    bundle = load_dataset_project(recipe_path)
    input_hashes["dataset_project_sha256"] = _sha256_file(recipe_path)
    input_hashes["observation_manifest_sha256"] = _sha256_file(bundle.observation_manifest_path)
    input_hashes["fit_project_sha256"] = _sha256_file(Path(project))
    fit_config_path = Path(project).parent / "fit-config.yaml"
    if fit_config_path.is_file():
        input_hashes["fit_config_sha256"] = _sha256_file(fit_config_path)
    return FitContext(
        observation_id=observation_id,
        record=record,
        target_xy=target_xy,
        stroke_annotation=stroke_annotation,
        input_hashes=input_hashes,
    )


def _fit_output_dir(context: FitContext, output: Path | None) -> Path:
    if output is not None:
        return Path(output)
    return DEFAULT_FIT_ROOT / context.record["label"]


def write_fit_outputs(
    context: FitContext, result: CurvatureFitResult, fit_dir: Path
) -> dict[str, Path]:
    """Write parameters, metrics, observation provenance, and fit data."""
    fit_dir.mkdir(parents=True, exist_ok=True)
    target_points = resample_by_arclength(context.target_xy, samples=64)
    fit_xy, fit_heading = matched_fit_state(result.geometry, 64)
    angle = np.deg2rad(result.transform.rotation_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    fit_xy = result.transform.scale * (fit_xy @ rotation.T) + np.array(
        [result.transform.dx, result.transform.dy]
    )
    fit_heading = fit_heading + result.transform.rotation_deg
    target_heading = estimate_observed_heading(target_points)
    parameters = {
        "observation_id": context.observation_id,
        "kind": "curvature-profile",
        "geometry": result.geometry.model_dump(mode="json"),
        "transform": result.transform.model_dump(mode="json"),
    }
    parameters_path = fit_dir / "parameters.yaml"
    parameters_path.write_text(
        yaml.safe_dump(parameters, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    gates = {
        "median_centerline_error_max": 0.01,
        "p95_centerline_error_max": 0.03,
        "endpoint_error_max": 0.02,
        "endpoint_tangent_error_max_deg": 8.0,
    }
    metrics = {
        "observation_id": context.observation_id,
        "optimizer": "scipy-least_squares-soft_l1",
        "optimizer_version": _scipy_version(),
        "success": result.success,
        "cost": result.cost,
        **result.metrics,
        "gates": gates,
        "gates_pass": {
            "median_centerline_error": result.metrics["median_position_error"]
            <= gates["median_centerline_error_max"],
            "p95_centerline_error": result.metrics["p95_position_error"]
            <= gates["p95_centerline_error_max"],
            "endpoint_error": result.metrics["endpoint_error"] <= gates["endpoint_error_max"],
            "endpoint_tangent_error": result.metrics["endpoint_tangent_error_deg"]
            <= gates["endpoint_tangent_error_max_deg"],
        },
    }
    metrics_path = fit_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    observation = {
        "observation_id": context.observation_id,
        "label": context.record["label"],
        "source_id": context.record["source_id"],
        "input_hashes": context.input_hashes,
        "stroke_annotation": context.stroke_annotation,
    }
    observation_path = fit_dir / "observation.json"
    observation_path.write_text(
        json.dumps(observation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fit_data = {
        "u": np.linspace(0.0, 1.0, 64).tolist(),
        "target_xy": target_points.tolist(),
        "fit_xy": fit_xy.tolist(),
        "fit_heading_deg": fit_heading.tolist(),
        "target_heading_deg": target_heading.tolist(),
        "residual_norm": np.linalg.norm(fit_xy - target_points, axis=1).tolist(),
    }
    fit_data_path = fit_dir / "fit-data.json"
    fit_data_path.write_text(
        json.dumps(fit_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "parameters": parameters_path,
        "metrics": metrics_path,
        "observation": observation_path,
        "fit_data": fit_data_path,
    }


def _render_scaled_previews(result: CurvatureFitResult, fit_dir: Path, label: str, scales):
    outputs = []
    for scale in scales:
        name = "small" if scale < 1.0 else ("normal" if scale == 1.0 else "large")
        u = np.linspace(0.0, 1.0, 129)
        xy, _ = evaluate_stroke_at(result.geometry, u)
        points = xy * (result.transform.scale * scale)
        size = 160
        span = max(float(np.abs(points).max()) * 2.2, 1e-6)
        px_per_unit = size / span
        canvas = Image.new("L", (size, size), 255)
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        pixels = [
            (size / 2 + x * px_per_unit, size / 2 - y * px_per_unit) for x, y in points
        ]
        stroke_width = max(2, int(round(0.1 * px_per_unit)))
        draw.line(pixels, fill=0, width=stroke_width)
        path = fit_dir / f"{label}-{name}.png"
        canvas.save(path)
        outputs.append(path)
    return outputs


def _plot_frame(size: tuple[int, int], x_domain, y_domain):
    """Minimal axes frame; returns (image, draw, to_px mapping)."""
    from PIL import ImageDraw

    width, height = size
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    margin = 28
    x0, x1 = x_domain
    y0, y1 = y_domain
    if y1 - y0 < 1e-9:
        y1 = y0 + 1.0

    def to_px(x, y):
        px = margin + (x - x0) / (x1 - x0) * (width - 2 * margin)
        py = height - margin - (y - y0) / (y1 - y0) * (height - 2 * margin)
        return (px, py)

    draw.rectangle(
        (margin, margin, width - margin, height - margin), outline=(120, 120, 120)
    )
    return image, draw, to_px


def _draw_series(draw, to_px, xs, ys, color, *, width=2):
    draw.line([to_px(x, y) for x, y in zip(xs, ys, strict=True)], fill=color, width=width)


def render_fit_diagnostics(
    context, result: CurvatureFitResult, fit_dir: Path, *, dataset_root: Path
) -> list[Path]:
    """Write the pilot diagnostic package (the brief's section 52)."""
    from .fitting.curvature import evaluate_heading_rate

    fit_data = json.loads((fit_dir / "fit-data.json").read_text(encoding="utf-8"))
    u = np.asarray(fit_data["u"])
    target = np.asarray(fit_data["target_xy"])
    fitted = np.asarray(fit_data["fit_xy"])
    residual = np.asarray(fit_data["residual_norm"])
    target_heading = np.asarray(fit_data["target_heading_deg"])
    fit_heading = np.asarray(fit_data["fit_heading_deg"])
    written = []

    target_mask = np.asarray(
        Image.open(
            Path(dataset_root) / context.record["artifacts"]["fitting_mask"]
        ).convert("L")
    ) > 0
    mask_image = Image.fromarray(((~target_mask) * 255).astype(np.uint8))
    mask_path = fit_dir / "target-mask.png"
    mask_image.save(mask_path)
    written.append(mask_path)

    all_points = np.vstack((target, fitted))
    low = all_points.min(axis=0)
    high = all_points.max(axis=0)
    padding = 0.1 * float(np.max(high - low)) + 1e-6
    x_domain = (float(low[0] - padding), float(high[0] + padding))
    y_domain = (float(low[1] - padding), float(high[1] + padding))

    image, draw, to_px = _plot_frame((360, 360), x_domain, y_domain)
    _draw_series(draw, to_px, target[:, 0], target[:, 1], (0, 0, 220))
    start = to_px(target[0, 0], target[0, 1])
    end = to_px(target[-1, 0], target[-1, 1])
    draw.rectangle((start[0] - 4, start[1] - 4, start[0] + 4, start[1] + 4), outline=(0, 160, 0))
    draw.rectangle((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4), outline=(220, 0, 0))
    path = fit_dir / "target-centerline.png"
    image.save(path)
    written.append(path)

    image, draw, to_px = _plot_frame((360, 360), x_domain, y_domain)
    _draw_series(draw, to_px, target[:, 0], target[:, 1], (0, 0, 220))
    _draw_series(draw, to_px, fitted[:, 0], fitted[:, 1], (220, 0, 0))
    path = fit_dir / "fit-overlay.png"
    image.save(path)
    written.append(path)

    image, draw, to_px = _plot_frame(
        (360, 240), (float(u[0]), float(u[-1])), (0.0, float(residual.max()) * 1.1 + 1e-6)
    )
    _draw_series(draw, to_px, u, residual, (220, 0, 0))
    path = fit_dir / "residual-by-u.png"
    image.save(path)
    written.append(path)

    heading_low = float(min(target_heading.min(), fit_heading.min()))
    heading_high = float(max(target_heading.max(), fit_heading.max()))
    span = max(heading_high - heading_low, 1.0)
    image, draw, to_px = _plot_frame(
        (360, 240),
        (float(u[0]), float(u[-1])),
        (heading_low - 0.1 * span, heading_high + 0.1 * span),
    )
    _draw_series(draw, to_px, u, target_heading, (0, 0, 220))
    _draw_series(draw, to_px, u, fit_heading, (220, 0, 0))
    path = fit_dir / "heading-profile.png"
    image.save(path)
    written.append(path)

    rates = evaluate_heading_rate(result.geometry.curvature, u)
    rate_low = float(min(rates.min(), 0.0))
    rate_high = float(max(rates.max(), 0.0))
    span = max(rate_high - rate_low, 1.0)
    image, draw, to_px = _plot_frame(
        (360, 240),
        (float(u[0]), float(u[-1])),
        (rate_low - 0.1 * span, rate_high + 0.1 * span),
    )
    _draw_series(draw, to_px, u, rates, (220, 0, 0))
    for change in result.geometry.curvature.changes:
        top = to_px(change.at, rate_high + 0.1 * span)
        bottom = to_px(change.at, rate_low - 0.1 * span)
        draw.line([top, bottom], fill=(0, 140, 0), width=1)
        draw.text((top[0] + 2, top[1]), f"{change.at:.2f}", fill=(0, 140, 0))
    path = fit_dir / "curvature-profile.png"
    image.save(path)
    written.append(path)
    return written


def register_trajectory_commands(app) -> None:
    """Register trajectory commands on the ``inkgrammar fit`` group."""

    @app.command("trajectory")
    def fit_trajectory(
        project: Path = typer.Argument(..., exists=True),  # noqa: B008
        observation: str = typer.Option(..., "--observation"),
        curvature_changes: int = typer.Option(4, "--curvature-changes"),
        dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, "--dataset-root"),  # noqa: B008
        output: Path = typer.Option(None, "--output"),  # noqa: B008
    ) -> None:
        """Fit a curvature trajectory for one dataset observation ID."""
        context = load_fit_context(project, observation, dataset_root=dataset_root)
        result = fit_curvature_stroke(
            context.target_xy, change_count=curvature_changes, restarts=15, seed=0
        )
        fit_dir = _fit_output_dir(context, output)
        outputs = write_fit_outputs(context, result, fit_dir)
        outputs.update(
            {
                path.stem: path
                for path in render_fit_diagnostics(
                    context, result, fit_dir, dataset_root=dataset_root
                )
            }
        )
        outputs.update(
            {
                path.stem: path
                for path in _render_scaled_previews(
                    result, fit_dir, context.record["label"], (0.5, 1.0, 2.0)
                )
            }
        )
        for name, path in outputs.items():
            typer.echo(f"{name}: {path}")
        typer.echo(f"gates_pass: {json.dumps(_gates_pass(fit_dir), sort_keys=True)}")

    @app.command("trajectory-preview")
    def trajectory_preview(
        project: Path = typer.Argument(..., exists=True),  # noqa: B008
        observation: str = typer.Option(..., "--observation"),
        scales: str = typer.Option("0.5,1,2", "--scales"),
        dataset_root: Path = typer.Option(DEFAULT_DATASET_ROOT, "--dataset-root"),  # noqa: B008
        output: Path = typer.Option(None, "--output"),  # noqa: B008
    ) -> None:
        """Render one fitted trajectory at several scales from one parameter set."""
        context = load_fit_context(project, observation, dataset_root=dataset_root)
        fit_dir = _fit_output_dir(context, output)
        parameters = yaml.safe_load((fit_dir / "parameters.yaml").read_text(encoding="utf-8"))
        from .fitting.models import CurvatureStrokeGeometry, TrajectoryTransform

        result = CurvatureFitResult(
            geometry=CurvatureStrokeGeometry.model_validate(parameters["geometry"]),
            transform=TrajectoryTransform.model_validate(parameters["transform"]),
            success=True,
            cost=0.0,
        )
        scale_values = [float(value) for value in scales.split(",")]
        written = _render_scaled_previews(result, fit_dir, context.record["label"], scale_values)
        for path in written:
            typer.echo(f"preview: {path}")

    @app.command("trajectory-report")
    def trajectory_report(
        parameters_path: Path = typer.Argument(..., exists=True),  # noqa: B008
    ) -> None:
        """Print fitted trajectory parameters and measured gates."""
        fit_dir = parameters_path.parent
        metrics = json.loads((fit_dir / "metrics.json").read_text(encoding="utf-8"))
        parameters = yaml.safe_load(parameters_path.read_text(encoding="utf-8"))
        typer.echo(f"observation: {parameters['observation_id']}")
        typer.echo(f"success: {metrics['success']}")
        typer.echo(f"median_centerline_error: {metrics['median_position_error']:.5f}")
        typer.echo(f"p95_centerline_error: {metrics['p95_position_error']:.5f}")
        typer.echo(f"endpoint_error: {metrics['endpoint_error']:.5f}")
        typer.echo(f"endpoint_tangent_error_deg: {metrics['endpoint_tangent_error_deg']:.3f}")
        typer.echo(f"total_turn_error_deg: {metrics['total_turn_error_deg']:.3f}")
        typer.echo(f"gates_pass: {json.dumps(metrics['gates_pass'], sort_keys=True)}")


def _gates_pass(fit_dir: Path) -> dict:
    metrics = json.loads((fit_dir / "metrics.json").read_text(encoding="utf-8"))
    return metrics["gates_pass"]


