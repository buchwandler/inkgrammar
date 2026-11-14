from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "external_geometry_pipeline.py"


def test_source_free_external_geometry_pipeline_preserves_route_evidence():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    assert report["source_free"] is True
    assert report["evidence_status"] == "inferred"
    assert report["route_role"] == "topology_candidate"
    assert report["physical_direction_claim"] is None
    assert report["route_ambiguous"] is True
    assert report["orientation_ambiguous"] is True
    assert report["feasible_path_count"] > 1
    assert len(report["centerline_xy"]) > 1
    assert report["contours"]
    for contour in report["contours"]:
        assert contour[0] == contour[-1]
        assert all(math.isfinite(value) for point in contour for value in point)
    assert len(report["bounds"]) == 4
    assert all(math.isfinite(value) for value in report["bounds"])
