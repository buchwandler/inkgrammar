import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from inkgrammar.io import load_layout
from inkgrammar.observations import ObservationManifest


def test_sas_label_sequences():
    layout = load_layout("examples/layouts/sas-1968-alphabet.yaml")
    labels = [label for row in layout.rows for label in row.labels]
    assert len(labels) == 63
    assert labels[:26] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert set(labels[26:53]) == set("abcdefghijklmnopqrstuvwxyzß")
    assert labels[53:] == list("1234567890")


def manifest_data():
    return json.loads(Path("data/observations/sas-1968/manifest.json").read_text())


def test_sas_observation_manifest():
    manifest = ObservationManifest.model_validate(manifest_data())
    assert len(manifest.sources) == 5
    assert len(manifest.items) == 69
    glyph_items = [item for item in manifest.items if item.source_id == "sas-1968-alphabet"]
    connection_items = [item for item in manifest.items if item.kind == "connection-pair"]
    assert all(item.review_status == "accepted" for item in glyph_items)
    assert len(connection_items) == 6
    assert all(item.review_status == "pending" for item in connection_items)
    assert all(item.crop_locator.startswith("local-only:") for item in manifest.items)
    assert all(source.rights_status == "unknown-needs-review" for source in manifest.sources)
    assert len({item.id for item in manifest.items}) == 69


@pytest.mark.parametrize("mutation", ["duplicate", "unknown-source", "bbox", "bounds"])
def test_reject_invalid_observation_manifest(mutation):
    data = manifest_data()
    if mutation == "duplicate":
        data["items"].append(data["items"][0])
    elif mutation == "unknown-source":
        data["items"][0]["source_id"] = "missing"
    elif mutation == "bbox":
        data["items"][0]["source_bbox"] = [5, 5, 1, 1]
    else:
        data["items"][0]["source_bbox"] = [0, 0, 10000, 10000]
    with pytest.raises(ValidationError):
        ObservationManifest.model_validate(data)
