from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from inkgrammar.dataset.preprocess import (
    CleanupRecord,
    PreprocessParams,
    clean_mask,
    detect_speck_candidates,
    estimate_metric_guides,
    label_components,
    normalize_background,
    pad_bbox,
    preprocess_crop,
    threshold_mask,
)
from inkgrammar.dataset.project import load_dataset_project, verify_sources


def test_clean_mask_removes_specks_and_preserves_raw():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:8, 2:8] = True  # large ink component
    mask[15, 15] = True  # single-pixel dust

    cleaned, record = clean_mask(mask, PreprocessParams(speck_min_area=4))

    assert cleaned[4, 4]
    assert not cleaned[15, 15]
    assert mask[15, 15]  # raw mask untouched
    assert record == CleanupRecord(
        removed_components=[
            {
                "bbox": (15, 15, 16, 16),
                "area": 1,
                "reason": "automatic-speck-candidate",
                "reviewed": False,
            }
        ]
    )


def test_detect_speck_candidates_only_reports_small_components():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:2, 0:2] = True  # area 4
    mask[5, 5] = True  # area 1

    candidates = detect_speck_candidates(mask, min_area=4)

    assert [candidate.area for candidate in candidates] == [1]
    assert candidates[0].reviewed is False


def test_label_components_finds_8_connected_groups():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True  # diagonal neighbor of the first
    mask[4, 0] = True

    components = label_components(mask)

    assert sorted(component[0].size for component in components) == [1, 2]


def test_estimate_metric_guides_lowercase_and_uppercase():
    mask = np.zeros((100, 50), dtype=bool)
    mask[40:90, :] = True

    lowercase = estimate_metric_guides(mask, glyph_class="lowercase")
    assert lowercase.x_height_y == pytest.approx(0.4, abs=0.02)
    assert lowercase.baseline_y == pytest.approx(0.89, abs=0.02)
    assert lowercase.metric_status == "automatic-candidate"
    assert lowercase.cap_height_y is None

    uppercase = estimate_metric_guides(mask, glyph_class="uppercase")
    assert uppercase.cap_height_y == pytest.approx(0.4, abs=0.02)
    assert uppercase.x_height_y is None

    mixed = estimate_metric_guides(mask, glyph_class="mixed")
    assert mixed.ascender_y == pytest.approx(0.4, abs=0.02)
    assert mixed.x_height_y is None


def test_estimate_metric_guides_rejects_empty_mask():
    with pytest.raises(ValueError, match="empty mask"):
        estimate_metric_guides(np.zeros((10, 10), dtype=bool))


def test_normalize_background_brightens_dim_scans():
    gray = np.full((10, 10), 100, dtype=np.uint8)
    gray[5, 5] = 10

    normalized = normalize_background(gray, percentile=95)

    assert normalized[0, 0] == 255
    assert normalized[5, 5] < 100


def test_threshold_mask_separates_ink_from_background():
    gray = np.full((20, 20), 200, dtype=np.uint8)
    gray[5:15, 5:15] = 20

    mask = threshold_mask(gray)

    assert mask[10, 10]
    assert not mask[0, 0]


def test_pad_bbox_clamps_to_image_shape():
    assert pad_bbox((0, 0, 10, 10), pad=3, shape=(12, 12)) == (0, 0, 12, 12)
    assert pad_bbox((5, 5, 6, 6), pad=2, shape=(100, 100)) == (3, 3, 8, 8)


def test_preprocess_crop_returns_versioned_result():
    image = Image.new("L", (30, 30), 255)
    pixels = np.asarray(image).copy()
    pixels[5:20, 5:20] = 0
    pixels[25, 25] = 0  # dust
    image = Image.fromarray(pixels)
    params = PreprocessParams(speck_min_area=4)

    result = preprocess_crop(image, params, glyph_class="lowercase")

    assert result.params == params
    assert result.params.version == 1
    assert result.raw_mask[25, 25]
    assert not result.clean_mask[25, 25]
    assert [removed.area for removed in result.cleanup.removed_components] == [1]
    assert result.metrics.metric_status == "automatic-candidate"


def _synthetic_project(tmp_path, *, source_bytes=None, width=4, height=3):

    private_root = tmp_path / "private"
    private_root.mkdir()
    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).point((0, 0), fill="black")
    image.save(private_root / "source.png")
    digest = hashlib.sha256((private_root / "source.png").read_bytes()).hexdigest()
    if source_bytes is not None:
        (private_root / "source.png").write_bytes(source_bytes)
    manifest = {
        "model_id": "synthetic",
        "sources": [
            {
                "source_id": "src-1",
                "filename": "source.png",
                "sha256": digest,
                "width_px": width,
                "height_px": height,
                "rights_status": "unknown-needs-review",
            }
        ],
        "items": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "project.yaml").write_text(
        "schema_version: 1\n"
        "id: synthetic\n"
        "observation_manifest:\n"
        "  path: manifest.json\n"
        "sources:\n"
        "  private_root: private\n"
        "pages:\n"
        "  - source_id: src-1\n"
        "    layout: layout.yaml\n"
        "    extractor: region-layout-v1\n",
        encoding="utf-8",
    )
    return tmp_path / "project.yaml"


def test_load_dataset_project_resolves_paths(tmp_path):
    bundle = load_dataset_project(_synthetic_project(tmp_path))

    assert bundle.observation_manifest_path == tmp_path / "manifest.json"
    assert bundle.private_root == tmp_path / "private"


def test_verify_sources_accepts_matching_files(tmp_path):

    bundle = load_dataset_project(_synthetic_project(tmp_path))
    verification = verify_sources(bundle)

    assert verification.ok
    assert verification.checks[0].status == "verified"


def test_verify_sources_reports_hash_and_dimension_and_missing(tmp_path):

    tampered = tmp_path / "tampered"
    tampered.mkdir()
    bundle = load_dataset_project(_synthetic_project(tampered, source_bytes=b"not-the-source"))
    verification = verify_sources(bundle)
    assert verification.checks[0].status == "hash-mismatch"

    wrong_size = tmp_path / "wrong-size"
    wrong_size.mkdir()
    bundle = load_dataset_project(_synthetic_project(wrong_size, width=9, height=9))
    manifest = bundle.observation_manifest_path
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('"width_px": 9', '"width_px": 4'),
        encoding="utf-8",
    )
    verification = verify_sources(bundle)
    assert verification.checks[0].status == "dimension-mismatch"

    missing = tmp_path / "missing"
    missing.mkdir()
    bundle = load_dataset_project(_synthetic_project(missing))
    bundle.private_root.joinpath("source.png").unlink()
    verification = verify_sources(bundle)
    assert verification.checks[0].status == "missing"


def test_committed_sas_recipe_parses_without_private_data():

    recipe = Path(__file__).resolve().parents[1] / "data" / "datasets" / "sas-1968.yaml"
    bundle = load_dataset_project(recipe)

    assert bundle.project.id == "sas-1968"
    assert [page.extractor for page in bundle.project.pages] == [
        "labeled-rows-v1",
        "region-layout-v1",
        "annotated-region-v1",
        "region-layout-v1",
        "region-layout-v1",
    ]
    assert bundle.observation_manifest_path.is_file()


def _materialize_fixture(root):
    """Synthetic region-layout dataset: private source, manifest, layout, recipe."""
    root.mkdir(parents=True, exist_ok=True)
    private = root / "private"
    private.mkdir()
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 40, 60), fill="black")
    draw.rectangle((55, 20, 85, 50), fill="black")
    image.save(private / "source.png")
    digest = hashlib.sha256((private / "source.png").read_bytes()).hexdigest()

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "synthetic",
                "sources": [
                    {
                        "source_id": "src-1",
                        "filename": "source.png",
                        "sha256": digest,
                        "width_px": 100,
                        "height_px": 80,
                        "rights_status": "unknown-needs-review",
                    }
                ],
                "items": [
                    {
                        "id": "sas1968-alpha-01",
                        "kind": "glyph-isolated",
                        "label": "a",
                        "unicode": ["U+0061"],
                        "source_id": "src-1",
                        "source_bbox": [10, 10, 40, 60],
                        "crop_locator": "local-only:alpha-01.png",
                        "label_source": "declared-layout-sequence",
                        "review_status": "accepted",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "layout.yaml").write_text(
        "schema_version: 1\n"
        "id: synthetic-page\n"
        "regions:\n"
        "  - id: alpha-01\n"
        "    kind: glyph-isolated\n"
        "    label: a\n"
        "    bbox: [0.1, 0.125, 0.4, 0.75]\n"
        "  - id: pair-01\n"
        "    kind: connection-pair\n"
        "    label: ab\n"
        "    bbox: [0.55, 0.25, 0.85, 0.625]\n"
        "    sequence: [a, b]\n"
        "    category_hint: baseline-origin\n",
        encoding="utf-8",
    )
    (root / "project.yaml").write_text(
        "schema_version: 1\n"
        "id: synthetic\n"
        "observation_manifest:\n"
        "  path: manifest.json\n"
        "sources:\n"
        "  private_root: private\n"
        "pages:\n"
        "  - source_id: src-1\n"
        "    layout: layout.yaml\n"
        "    extractor: region-layout-v1\n",
        encoding="utf-8",
    )
    return root / "project.yaml"


def test_materialize_is_deterministic(tmp_path):
    from inkgrammar.dataset.materialize import materialize_dataset
    from inkgrammar.dataset.project import load_dataset_project

    bundle_a = load_dataset_project(_materialize_fixture(tmp_path / "run-a"))
    bundle_b = load_dataset_project(_materialize_fixture(tmp_path / "run-b"))
    index_a = materialize_dataset(bundle_a, tmp_path / "run-a" / "out")
    index_b = materialize_dataset(bundle_b, tmp_path / "run-b" / "out")

    assert (tmp_path / "run-a" / "out" / "index.json").read_bytes() == (
        tmp_path / "run-b" / "out" / "index.json"
    ).read_bytes()
    assert index_a["observations"].keys() == index_b["observations"].keys()
    for obs_id, record in index_a["observations"].items():
        assert record["artifact_hashes"] == index_b["observations"][obs_id]["artifact_hashes"]
        assert record["derived"] == index_b["observations"][obs_id]["derived"]


def test_materialize_records_artifacts_and_derived(tmp_path):
    from inkgrammar.dataset.materialize import materialize_dataset
    from inkgrammar.dataset.project import load_dataset_project

    bundle = load_dataset_project(_materialize_fixture(tmp_path / "run"))
    index = materialize_dataset(bundle, tmp_path / "run" / "out")

    matched = index["observations"]["sas1968-alpha-01"]
    generated = index["observations"]["synthetic-pair-01"]
    assert matched["kind"] == "glyph-isolated"
    assert matched["review_status"] == "accepted"
    assert generated["relations"]["left_label"] == "a"
    assert generated["relations"]["right_label"] == "b"
    assert generated["relations"]["connection_category"] == "baseline-origin"
    for record in (matched, generated):
        assert set(record["artifact_hashes"]) == {
            "crop",
            "raw_mask",
            "clean_mask",
            "cleanup",
            "normalized",
        }
        assert record["derived"]["skeleton"]["algorithm"].startswith("zhang-suen")
        assert record["derived"]["centerline"]["algorithm"] == "edge-covering-walk-v1"
        assert record["derived"]["topology"]["algorithm"] == "pixel-graph-v1"
        for name in ("skeleton", "centerline", "topology"):
            assert (tmp_path / "run" / "out" / record["derived"][name]["path"]).is_file()
    assert (
        index["source_hashes"]["src-1"]
        == hashlib.sha256((tmp_path / "run" / "private" / "source.png").read_bytes()).hexdigest()
    )


def test_materialize_composes_review_patches(tmp_path):
    from inkgrammar.dataset.materialize import materialize_dataset
    from inkgrammar.dataset.project import load_dataset_project
    from inkgrammar.dataset.review import extraction_sha256

    project = _materialize_fixture(tmp_path / "run")
    bundle = load_dataset_project(project)
    review_dir = tmp_path / "run" / "reviews"
    index = materialize_dataset(bundle, tmp_path / "run" / "out", review_dir=review_dir)
    record = index["observations"]["sas1968-alpha-01"]
    automatic = {key: value for key, value in record.items() if key != "review_patch"}

    patch = {
        "observation_id": "sas1968-alpha-01",
        "input_hash": {
            "source_sha256": index["source_hashes"]["src-1"],
            "extraction_sha256": extraction_sha256(automatic),
        },
        "set": {"isolation_status": "usable-reviewed", "reviewer_notes": "checked"},
    }
    review_dir.mkdir(parents=True)
    (review_dir / "sas1968-alpha-01.yaml").write_text(json.dumps(patch), encoding="utf-8")

    index = materialize_dataset(bundle, tmp_path / "run" / "out2", review_dir=review_dir)
    applied = index["observations"]["sas1968-alpha-01"]
    assert applied["review_patch"]["status"] == "applied"
    assert applied["isolation_status"] == "usable-reviewed"
    assert applied["annotation"]["reviewer_notes"] == "checked"

    patch["input_hash"]["extraction_sha256"] = "f" * 64
    (review_dir / "sas1968-alpha-01.yaml").write_text(json.dumps(patch), encoding="utf-8")
    index = materialize_dataset(bundle, tmp_path / "run" / "out3", review_dir=review_dir)
    stale = index["observations"]["sas1968-alpha-01"]
    assert stale["review_patch"]["status"] == "stale"
    assert stale["isolation_status"] is None


def test_materialize_flags_neighboring_ink_and_maps_word_relations(tmp_path):
    from inkgrammar.dataset.materialize import materialize_dataset
    from inkgrammar.dataset.project import verify_sources
    from inkgrammar.dataset.status import validate_dataset

    project = _materialize_fixture(tmp_path / "run")
    layout = tmp_path / "run" / "layout.yaml"
    layout.write_text(
        layout.read_text(encoding="utf-8")
        + "  - id: wide-01\n    kind: word\n    label: ab\n    bbox: [0.05, 0.1, 0.9, 0.8]\n",
        encoding="utf-8",
    )
    bundle = load_dataset_project(project)
    root = tmp_path / "run" / "out"
    index = materialize_dataset(bundle, root)

    wide = index["observations"]["synthetic-wide-01"]
    assert wide["relations"]["sequence_text"] == "ab"

    validation = validate_dataset(
        bundle, index, dataset_root=root, verification=verify_sources(bundle)
    )
    rules = {issue.rule for issue in validation.warnings}
    assert "neighboring-ink-likely-present" in rules


def test_isolate_target_ink_picks_component_nearest_center():
    import numpy as np

    from inkgrammar.dataset.preprocess import isolate_target_ink

    mask = np.zeros((20, 40), dtype=bool)
    mask[2:8, 2:10] = True  # left component
    mask[2:8, 25:35] = True  # right component

    candidate, proposal = isolate_target_ink(mask, center=(28.0, 5.0))

    assert candidate[5, 28] and not candidate[5, 5]
    assert [c.kept for c in proposal.components] == [False, True]
    assert proposal.components[0].reason == "not-nearest-component"


def test_materialize_labeled_rows_get_isolation_frame_and_dot_exclusion(tmp_path):
    from inkgrammar.dataset.materialize import materialize_dataset

    root = tmp_path / "run"
    root.mkdir(parents=True)
    private = root / "private"
    private.mkdir()
    image = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, 70, 85), fill="black")  # letter body
    draw.rectangle((45, 15, 55, 25), fill="black")  # dot above the body
    draw.rectangle((130, 40, 170, 85), fill="black")  # second letter
    image.save(private / "source.png")
    digest = hashlib.sha256((private / "source.png").read_bytes()).hexdigest()

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "synthetic",
                "sources": [
                    {
                        "source_id": "src-1",
                        "filename": "source.png",
                        "sha256": digest,
                        "width_px": 200,
                        "height_px": 100,
                        "rights_status": "unknown-needs-review",
                    }
                ],
                "items": [],
            }
        ),
        encoding="utf-8",
    )
    (root / "layout.yaml").write_text(
        "schema_version: 1\n"
        "id: synthetic-page\n"
        "rows:\n"
        "  - id: row\n"
        "    y: [0.1, 0.9]\n"
        "    x: [0.0, 1.0]\n"
        "    labels: [a, b]\n"
        "    centers: [0.25, 0.75]\n",
        encoding="utf-8",
    )
    (root / "project.yaml").write_text(
        "schema_version: 1\n"
        "id: synthetic\n"
        "observation_manifest:\n"
        "  path: manifest.json\n"
        "sources:\n"
        "  private_root: private\n"
        "pages:\n"
        "  - source_id: src-1\n"
        "    layout: layout.yaml\n"
        "    extractor: labeled-rows-v1\n",
        encoding="utf-8",
    )
    bundle = load_dataset_project(root / "project.yaml")
    out = root / "out"
    index = materialize_dataset(bundle, out)

    records = list(index["observations"].values())
    assert len(records) == 2
    for record in records:
        assert record["isolation"]["status"] == "pending"
        assert record["isolation_status"] == "pending"
        assert record["frame"] is not None
        assert record["frame"]["x_height_px"] > 0
        assert record["artifacts"]["fitting_mask"].endswith("__isolation.png")
        assert (out / record["artifacts"]["isolation_mask"]).is_file()
        assert (out / record["artifacts"]["isolation_overlay"]).is_file()
        assert record["derived"]["centerline"]["sha256"]
        assert record["artifact_hashes"]["isolation_mask"]

    first = next(r for r in records if r["label"] == "a")
    excluded = [c for c in first["isolation"]["components"] if not c["kept"]]
    assert excluded, "the dot above the body must be excluded from the target ink"
    assert excluded[0]["reason"] == "not-nearest-component"
