from __future__ import annotations

import json
from pathlib import Path

import yaml

from inkgrammar.fitting.models import FitTopologyFile
from inkgrammar.io import load_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/training/sas-1968"
INVENTORY_PATH = DATA_DIR / "target-inventory.yaml"
OBSERVATIONS_PATH = ROOT / "data/observations/sas-1968/manifest.json"
MODEL_PATH = ROOT / "data/models/sas-1968.yaml"
TOPOLOGY_PATH = DATA_DIR / "topology.yaml"
STROKE_LAYOUT_PATH = ROOT / "examples/layouts/sas-1968-stroke-order.yaml"


def test_target_inventory_covers_source_observations_and_runtime_statuses():
    inventory = yaml.safe_load(INVENTORY_PATH.read_text(encoding="utf-8"))
    observations = json.loads(OBSERVATIONS_PATH.read_text(encoding="utf-8"))
    assert (DATA_DIR / inventory["source_observations"]["manifest"]).resolve() == OBSERVATIONS_PATH
    runtime_model = load_model(MODEL_PATH)
    topology = FitTopologyFile.model_validate(
        yaml.safe_load(TOPOLOGY_PATH.read_text(encoding="utf-8"))
    )

    target_sets = inventory["source_observations"]["target_sets"]
    targets = [character for characters in target_sets.values() for character in characters]
    source_items = {
        item["label"]: item
        for item in observations["items"]
        if item["source_id"] == inventory["source_observations"]["source_id"]
        and item["kind"] == inventory["source_observations"]["item_kind"]
    }
    modeled = set(inventory["runtime_status"]["modeled_incomplete"])
    unsupported_groups = inventory["runtime_status"]["unsupported"]
    unsupported = {
        character
        for reason, characters in unsupported_groups.items()
        if reason != "reasons"
        for character in characters
    }
    assert set(unsupported_groups["reasons"]) == {
        "modeled_incomplete",
        "needs_crop_isolation",
        "no_runtime_trajectory",
    }

    assert len(targets) == 63
    assert len(set(targets)) == 63
    assert set(source_items) == set(targets)
    assert all(item["review_status"] == "accepted" for item in source_items.values())
    assert modeled == set(runtime_model.glyphs) == set(topology.glyphs)
    assert not modeled & unsupported
    assert modeled | unsupported == set(targets)
    assert all(
        source_items[character]["fitting_status"] == "needs-isolation"
        for character in unsupported_groups["needs_crop_isolation"]
    )
    assert all(character not in runtime_model.glyphs for character in unsupported)
    assert all(
        topology.glyphs[character].trajectory_status == "incomplete"
        and topology.glyphs[character].stroke_order_status == "unannotated"
        and topology.glyphs[character].known_gaps
        and topology.glyphs[character].composition_status == "unrecorded"
        and not topology.glyphs[character].composition_constraints
        for character in modeled
    )
    assert all(
        topology.glyphs[character].source_observation_id == source_items[character]["id"]
        for character in modeled
    )


def test_special_forms_and_stroke_order_are_linked_to_source_layout_regions():
    inventory = yaml.safe_load(INVENTORY_PATH.read_text(encoding="utf-8"))
    layout = yaml.safe_load(STROKE_LAYOUT_PATH.read_text(encoding="utf-8"))
    assert (DATA_DIR / inventory["special_forms"]["layout"]).resolve() == STROKE_LAYOUT_PATH
    assert (
        DATA_DIR / inventory["stroke_order_observations"]["layout"]
    ).resolve() == STROKE_LAYOUT_PATH
    regions = {region["id"]: region for region in layout["regions"]}

    special_forms = inventory["special_forms"]["observations"]
    assert {row["region_id"] for row in special_forms} == {
        region_id for region_id, region in regions.items() if region["kind"] == "glyph-isolated"
    }
    assert all(row["status"] != "reviewed" for row in special_forms)
    assert any(row["status"] == "transcription-uncertain" for row in special_forms)

    stroke_order = inventory["stroke_order_observations"]
    examples = stroke_order["examples"]
    assert stroke_order["path_status"] == "not-traced"
    assert {row["region_id"] for row in examples} == {
        region_id
        for region_id, region in regions.items()
        if region["kind"] == "stroke-order-demonstration"
    }
    assert all(
        row["sequence"] == [int(number) for number in regions[row["region_id"]]["sequence"]]
        for row in examples
    )
    assert (
        next(row for row in examples if row["region_id"] == "stroke-order-K")[
            "annotation_isolation"
        ]
        == "pending"
    )
