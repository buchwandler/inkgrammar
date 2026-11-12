from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from examples.sas1968 import reproduce_input as reproduction

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data/models/sas-1968.yaml"


def _file_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_reproduction_example_reports_preflight_and_is_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest = reproduction.render_input(MODEL_PATH, first)
    reproduction.render_input(MODEL_PATH, second)

    assert manifest["coverage"]["target_count"] == 63
    assert manifest["coverage"]["modeled_count"] == 12
    assert manifest["coverage"]["missing_count"] == 51
    assert manifest["connection_rule_count"] == 0
    assert manifest["context_rule_count"] == 0
    assert manifest["words"]["blocked"] == [
        "Schulausgangsschrift",
        "Schule",
        "schreiben",
        "Ausgangsschrift",
    ]
    assert manifest["words"]["supported"] == [
        "minimum",
        "banana",
        "Cabin",
        "buch",
        "whim",
        "calcium",
    ]
    assert set(manifest["connection_targets"]["supported"]) == {
        "nu",
        "na",
        "nl",
        "bu",
        "bl",
        "ba",
    }
    assert all(
        row["connection_rules_used"] == 0
        for row in manifest["connection_targets"]["targets"]
        if row["rendered"]
    )
    assert manifest["reproduction_claim"] == "not_claimed"
    assert manifest["per_glyph"]["a"]["a.default"]["status"] == "ok"
    assert manifest["outputs"] == list(reproduction.OUTPUT_NAMES[:-1])
    assert _file_bytes(first) == _file_bytes(second)


def test_script_runs_from_snapshot_without_git_or_training_workspace(tmp_path):
    snapshot = tmp_path / "snapshot"
    script = snapshot / "examples/sas1968/reproduce_input.py"
    script.parent.mkdir(parents=True)
    shutil.copy(ROOT / "examples/sas1968/reproduce_input.py", script)
    shutil.copytree(
        ROOT / "inkgrammar",
        snapshot / "inkgrammar",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    model = snapshot / "data/models/sas-1968.yaml"
    model.parent.mkdir(parents=True)
    shutil.copy(MODEL_PATH, model)
    assert not (snapshot / ".git").exists()
    assert not (snapshot / "data/training").exists()
    assert not (snapshot / "data/observations").exists()

    output = snapshot / "review-output"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            "data/models/sas-1968.yaml",
            "--output",
            str(output),
        ],
        cwd=snapshot,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["coverage"]["modeled_count"] == 12
    assert len(manifest["outputs"]) == 6
    assert all((output / filename).is_file() for filename in manifest["outputs"])
