"""Estimate shared writing metrics from multiple reviewed alphabet observations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from scriptstrokes.raster import threshold_image
from tools.measure_sas_seed import connected_ink

ROOT = Path("work/sas-1968")
UNIT = 12.5
BASELINE_MEMBERS = {
    "uppercase-1": "ABCDEFHIKL",
    "uppercase-2": "MNOPRST",
    "uppercase-3": "UVWXZ",
    "lowercase-1": "abcehik",
    "lowercase-2": "lmnorstu",
    "lowercase-3": "vwxz",
    "digits": "1234567890",
}


def main():
    detections = json.loads((ROOT / "observations/alphabet/detections.json").read_text())
    samples = {}
    for item in detections["detections"]:
        with Image.open(ROOT / "observations/alphabet" / item["crop"]) as image:
            mask = threshold_image(image)
        body = connected_ink(mask)[0]
        ys, xs = zip(*body, strict=True)
        x0, y0, _, _ = item["bbox"]
        samples[item["label"]] = {
            "row": item["row"],
            "top": min(ys) + y0,
            "bottom": max(ys) + y0,
            "left": min(xs) + x0,
            "right": max(xs) + x0,
            "pixels": body,
            "bbox": item["bbox"],
        }
    baselines = {
        row: float(np.median([samples[char]["bottom"] for char in members])) - 1.5
        for row, members in BASELINE_MEMBERS.items()
    }

    def height(char):
        item = samples[char]
        return baselines[item["row"]] - item["top"] - 1.5

    x_members = "aceimnorsuvwxz"
    cap_members = "ABCDEFHIKLMNOPRSTUVWXZ"
    asc_members = "bdhkl"
    desc_members = "gjpqy"
    descents = [
        samples[char]["bottom"] - 1.5 - baselines[samples[char]["row"]] for char in desc_members
    ]
    slants = []
    widths = []
    for char in "iumnh":
        item = samples[char]
        baseline = baselines[item["row"]]
        by_row = {}
        for y, x in item["pixels"]:
            source_y = y + item["bbox"][1]
            if baseline - 25 <= source_y <= baseline - 10:
                by_row.setdefault(source_y, []).append(x)
        ys = sorted(by_row)
        xs = [max(by_row[y]) for y in ys]
        slope = float(np.polyfit(ys, xs, 1)[0])
        angle = math.atan(-slope)
        slants.append(math.degrees(angle))
        for y in ys:
            row = set(by_row[y])
            right = max(row)
            left = right
            while left - 1 in row:
                left -= 1
            widths.append((right - left + 1) * math.cos(angle))
    metrics = {
        "units_per_em": 1000,
        "baseline": 0,
        "x_height": round(float(np.median([height(c) for c in x_members])) * UNIT, 2),
        "cap_height": round(float(np.median([height(c) for c in cap_members])) * UNIT, 2),
        "ascender": round(float(np.median([height(c) for c in asc_members])) * UNIT, 2),
        "descender": -round(float(np.median(descents)) * UNIT, 2),
        "slant_degrees": round(float(np.median(slants)), 2),
        "default_stroke_width": round(float(np.median(widths)) * UNIT, 2),
    }
    payload = {
        "metrics": metrics,
        "source_baselines_px": baselines,
        "x_height_members": x_members,
        "cap_height_members": cap_members,
        "ascender_members": asc_members,
        "descender_members": desc_members,
        "slant_members": "iumnh",
        "slant_samples_degrees": slants,
        "units_per_source_pixel": UNIT,
        "method": "Largest connected body; row medians; stroke-edge correction; median stem fits.",
        "limitations": (
            "Raster estimates, not historical specifications; review contaminated crops."
        ),
    }
    (ROOT / "reports/metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    text = [
        "# SAS 1968 metric estimates",
        "",
        payload["method"],
        "",
        "Exact-byte scans remain unchanged. Measurements use thresholded local derivatives.",
        "Baselines are row medians of non-descending body bottoms minus 1.5 source pixels.",
        "Heights are medians with another 1.5-pixel top-edge correction.",
        "Slant uses rightmost stem runs between 25 and 10 pixels above each baseline.",
        "Stroke width uses perpendicular run thickness corrected by each stem angle.",
        "Model scale is 12.5 units per source pixel; aspect ratio is preserved.",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
        payload["limitations"],
        "",
    ]
    (ROOT / "reports/metrics.md").write_text("\n".join(text))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
