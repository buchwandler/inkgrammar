"""Source-free SAS dataset checks: stable IDs, five-page representation,
release-safe metadata hygiene. These tests never touch private scan pixels.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from inkgrammar.dataset.project import load_dataset_project
from inkgrammar.io import load_layout

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "data" / "datasets" / "sas-1968.yaml"
MANIFEST = ROOT / "data" / "observations" / "sas-1968" / "manifest.json"
LAYOUTS = ROOT / "examples" / "layouts"


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_alphabet_ids_derive_from_declared_layout_sequence():
    layout = load_layout(LAYOUTS / "sas-1968-alphabet.yaml")
    expected = []
    for row in layout.rows:
        for index, label in enumerate(row.labels):
            codepoints = "-".join(f"U+{ord(ch):04X}" for ch in label)
            expected.append(
                {
                    "id": f"sas1968-alphabet-{codepoints}",
                    "label": label,
                    "crop": f"{row.id}-{index:02d}-{codepoints}.png",
                }
            )

    items = {item["id"]: item for item in _manifest()["items"]}
    assert len(expected) == 63
    for entry in expected:
        item = items[entry["id"]]
        assert item["label"] == entry["label"]
        assert item["crop_locator"].endswith(entry["crop"])
    assert len(items) == 63


def test_recipe_represents_all_five_sources():
    bundle = load_dataset_project(RECIPE)
    manifest = _manifest()

    source_ids = [source["source_id"] for source in manifest["sources"]]
    page_ids = [page.source_id for page in bundle.project.pages]

    assert sorted(page_ids) == sorted(source_ids)
    assert len(page_ids) == 5
    assert {page.extractor for page in bundle.project.pages} == {
        "labeled-rows-v1",
        "region-layout-v1",
        "annotated-region-v1",
    }
    assert len({page.layout for page in bundle.project.pages}) == 5


def test_release_safe_metadata_contains_no_private_leakage():
    files = [
        *sorted((ROOT / "data" / "observations").rglob("*.json")),
        *sorted((ROOT / "data" / "datasets").rglob("*.yaml")),
        *sorted(LAYOUTS.glob("*.yaml")),
    ]
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"(/home/|/Users/|[A-Za-z]:\\)", text), path
        assert max((len(line) for line in text.splitlines()), default=0) < 1000, path

    for item in _manifest()["items"]:
        locator = item["crop_locator"]
        assert locator.startswith("local-only:")
        assert ".." not in locator


def test_private_recipe_paths_point_outside_release_data():
    bundle = load_dataset_project(RECIPE)

    assert "work" in bundle.private_root.parts
    assert not bundle.observation_manifest_path.is_relative_to(bundle.private_root)
