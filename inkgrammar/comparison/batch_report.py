from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .matching import (
    AMBIGUITY_ABSOLUTE_MARGIN,
    AMBIGUITY_RELATIVE_MARGIN,
    ASPECT_WEIGHT,
    FILL_WEIGHT,
    RASTER_WEIGHT,
    TOPOLOGY_WEIGHT,
    MatchScore,
)
from .models import CropRect


@dataclass(frozen=True)
class BatchSampleResult:
    sample_id: str
    sample_bbox: CropRect
    sample_confidence: float
    reference_id: str
    reference_label: str | None
    status: str
    score: MatchScore
    alternatives: tuple[MatchScore, ...]
    report_dir: Path
    report_json: Path


@dataclass(frozen=True)
class BatchComparisonReport:
    output_dir: Path
    payload: dict[str, Any]

    @property
    def json_path(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def markdown_path(self) -> Path:
        return self.output_dir / "summary.md"

    @property
    def html_path(self) -> Path:
        return self.output_dir / "index.html"


def match_score_payload(score: MatchScore) -> dict[str, Any]:
    return {
        "reference_id": score.reference_id,
        "rank": score.rank,
        "cost": score.cost,
        "coarse_cost": score.coarse_cost,
        "refined_cost": score.refined_cost,
        "raster_cost": score.raster_cost,
        "aspect_cost": score.aspect_cost,
        "fill_cost": score.fill_cost,
        "topology_cost": score.topology_cost,
        "centerline_cost": score.centerline_cost,
        "heading_cost": score.heading_cost,
        "turn_rate_cost": score.turn_rate_cost,
    }


def _sample_payload(result: BatchSampleResult, output_dir: Path) -> dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "sample_bbox": result.sample_bbox.model_dump(mode="json"),
        "segmentation_confidence": result.sample_confidence,
        "reference_id": result.reference_id,
        "reference_label": result.reference_label,
        "status": result.status,
        "cost": result.score.cost,
        "alternatives": [match_score_payload(score) for score in result.alternatives],
        "report_dir": result.report_dir.relative_to(output_dir).as_posix(),
        "report_json": result.report_json.relative_to(output_dir).as_posix(),
        "match_json": f"{result.sample_id}/match.json",
        "sample_image": f"{result.sample_id}/sample.png",
        "matched_reference_image": f"{result.sample_id}/matched-reference.png",
        "alternatives_image": f"{result.sample_id}/alternatives.png",
    }


def write_batch_report(
    output_dir: str | Path,
    *,
    reference_path: str | Path,
    sample_path: str | Path,
    reference_sha256: str,
    expected_count: int,
    reference_cache_hit: bool,
    results: tuple[BatchSampleResult, ...],
) -> BatchComparisonReport:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    samples = [_sample_payload(result, root) for result in results]
    ambiguous_count = sum(item["status"] == "ambiguous" for item in samples)
    payload: dict[str, Any] = {
        "status": "ambiguous" if ambiguous_count else "ok",
        "reference": {
            "path": Path(reference_path).name,
            "sha256": reference_sha256,
        },
        "sample": {
            "path": Path(sample_path).name,
            "expected_count": expected_count,
            "detected_count": len(samples),
        },
        "ambiguous_count": ambiguous_count,
        "reference_cache": {"hit": reference_cache_hit, "key": reference_sha256},
        "matching": {
            "cost_interpretation": "lower retrieval cost ranks ahead; it is not a probability",
            "weights": {
                "raster": RASTER_WEIGHT,
                "aspect": ASPECT_WEIGHT,
                "fill": FILL_WEIGHT,
                "topology": TOPOLOGY_WEIGHT,
            },
            "ambiguity_margin": {
                "absolute": AMBIGUITY_ABSOLUTE_MARGIN,
                "relative": AMBIGUITY_RELATIVE_MARGIN,
            },
        },
        "matches": samples,
    }
    (root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    markdown = [
        "# Automatic comparison summary",
        "",
        f"Reference: `{Path(reference_path).name}`",
        f"Sample: `{Path(sample_path).name}` ({expected_count} glyphs requested)",
        f"Status: **{payload['status']}**. Ambiguous matches: {ambiguous_count}.",
        "Retrieval costs rank candidates and are not correctness probabilities.",
        "",
        "| Sample | Best reference | Status | Cost | Pair report |",
        "|---|---|---:|---:|---|",
    ]
    for item in samples:
        markdown.append(
            f"| {item['sample_id']} | {item['reference_id']} | {item['status']} "
            f"| {item['cost']:.4f} | [{item['report_dir']}/report.html]"
            f"({item['report_dir']}/report.html) |"
        )
    markdown.extend(["", "## Ranked alternatives", ""])
    for item in samples:
        markdown.append(f"### {item['sample_id']}")
        for alternative in item["alternatives"]:
            markdown.append(
                f"- #{alternative['rank']} `{alternative['reference_id']}` "
                f"cost {alternative['cost']:.4f}"
            )
        markdown.append("")
    (root / "summary.md").write_text("\n".join(markdown).rstrip() + "\n", encoding="utf-8")

    rows = []
    for item in samples:
        alternatives = "".join(
            "<li>"
            f"#{entry['rank']} {html.escape(entry['reference_id'])}: "
            f"{entry['cost']:.4f}</li>"
            for entry in item["alternatives"]
        )
        report_link = html.escape(item["report_dir"] + "/report.html", quote=True)
        sample_link = html.escape(item["sample_image"], quote=True)
        matched_link = html.escape(item["matched_reference_image"], quote=True)
        alternatives_link = html.escape(item["alternatives_image"], quote=True)
        sample_id = html.escape(item["sample_id"])
        rows.append(
            "<tr>"
            f"<td><a href=\"{sample_link}\"><img src=\"{sample_link}\" width=\"100\" "
            f"alt=\"{sample_id}\"></a></td>"
            f"<td><a href=\"{matched_link}\"><img src=\"{matched_link}\" width=\"100\" "
            f"alt=\"{html.escape(item['reference_id'])}\"></a><br>"
            f"{html.escape(item['reference_id'])}</td>"
            f"<td>{html.escape(item['status'])}</td>"
            f"<td>{item['cost']:.4f}</td>"
            f"<td><a href=\"{report_link}\">pair report</a></td>"
            f"<td><a href=\"{alternatives_link}\"><img src=\"{alternatives_link}\" "
            f"width=\"180\" alt=\"ranked alternatives\"></a>"
            f"<details><summary>scores</summary><ol>{alternatives}</ol></details></td>"
            "</tr>"
        )
    document = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><meta charset="utf-8">',
            f"<title>Automatic comparison: {html.escape(Path(sample_path).name)}</title>",
            "<body>",
            "<h1>Automatic comparison summary</h1>",
            f"<p>Reference: {html.escape(Path(reference_path).name)}</p>",
            f"<p>Sample: {html.escape(Path(sample_path).name)}; "
            f"{expected_count} glyphs requested; status {payload['status']}.</p>",
            f"<p>Ambiguous matches: {ambiguous_count}</p>",
            "<table><thead><tr><th>Sample</th><th>Best reference</th><th>Status</th>"
            "<th>Cost</th><th>Report</th><th>Alternatives</th></tr></thead>",
            f"<tbody>{''.join(rows)}</tbody></table>",
            "</body></html>",
        ]
    )
    (root / "index.html").write_text(document + "\n", encoding="utf-8")
    return BatchComparisonReport(root, payload)
