from __future__ import annotations

import json
from pathlib import Path

import yaml

from inkgrammar.fitting.connection_fit import connection_observation_coverage
from inkgrammar.fitting.workspace import load_fit_project

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "data/training/sas-1968/project.yaml"
CATEGORY_LAYOUT = ROOT / "examples/layouts/sas-1968-connection-categories.yaml"
PAIR_LAYOUT = ROOT / "examples/layouts/sas-1968-connections-numbered.yaml"
OBSERVATION_MANIFEST = ROOT / "data/observations/sas-1968/manifest.json"


def test_connection_categories_preserve_source_labels_examples_and_unknown_behavior():
    bundle = load_fit_project(PROJECT)
    layout = yaml.safe_load(CATEGORY_LAYOUT.read_text(encoding="utf-8"))
    regions = {region["id"]: region for region in layout["regions"]}
    categories = {category.id: category for category in bundle.grammar.connection_categories}

    assert set(categories) == set(regions)
    assert bundle.grammar.connection_rules == []
    for category_id, category in categories.items():
        region = regions[category_id]
        assert category.source_id == "sas-1968-connection-categories"
        assert category.layout_region_id == category_id
        assert category.source_label == region["label"]
        assert category.example_pairs == region["members"]
        assert category.runtime_mapping_status == "pending-review"
        assert category.behavior.left_terminal_modification == "not-recorded"
        assert category.behavior.right_entry_modification == "not-recorded"
        assert category.behavior.pen_lift == "not-recorded"
        assert category.behavior.attachment_point == "not-recorded"
        assert category.behavior.exit_tangent == "not-recorded"
        assert category.behavior.entry_tangent == "not-recorded"

    assert categories["connection-category-baseline"].origin_level == "baseline"
    assert categories["connection-category-baseline"].behavior.continuity == "one-stroke"
    assert categories["connection-category-midline"].origin_level == "midline"
    assert categories["connection-category-midline"].behavior.continuity == "one-stroke"
    assert categories["connection-category-attached"].origin_level == "attached"
    assert categories["connection-category-attached"].behavior.continuity == "not-recorded"


def test_numbered_pair_observations_are_source_linked_and_held_out():
    bundle = load_fit_project(PROJECT)
    layout = yaml.safe_load(PAIR_LAYOUT.read_text(encoding="utf-8"))
    regions = {region["id"]: region for region in layout["regions"]}
    manifest = json.loads(OBSERVATION_MANIFEST.read_text(encoding="utf-8"))
    observations = {item.id: item for item in bundle.observations.items}
    pairs = bundle.grammar.connection_observations

    assert len(pairs) == 6
    assert {observation.id for observation in pairs} == {
        "connection-02-ba",
        "connection-02-be",
        "connection-02-He",
        "connection-03-Do",
        "connection-03-Jg",
        "connection-03-Be",
    }
    assert all(pair.split == "hold-out" for pair in pairs)
    assert all(pair.expected_class is None and pair.target_geometry is None for pair in pairs)
    assert connection_observation_coverage(pairs)["split_counts"] == {
        "fit": 0,
        "validation": 0,
        "hold-out": 6,
    }

    source_rows = {item["id"]: item for item in manifest["items"]}
    for pair in pairs:
        source = observations[pair.observation_id]
        row = source_rows[pair.observation_id]
        layout_id = row["extraction"]["layout_region_id"]
        region = regions[layout_id]
        width, height = 273, 223
        expected_bbox = tuple(
            round(value * (width if index % 2 == 0 else height))
            for index, value in enumerate(region["bbox"])
        )
        assert source.kind == "connection-pair"
        assert source.review_status == "pending"
        assert source.fitting_status == "pending"
        assert source.source_bbox == expected_bbox
        assert pair.text == region["label"]
        assert pair.observation_id == f"sas1968-{layout_id}"
