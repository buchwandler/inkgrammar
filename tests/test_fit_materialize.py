import hashlib
import json

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from scriptstrokes.cli import app
from scriptstrokes.fitting.workspace import load_fit_project, materialize_fit_project


def _write_fixture(tmp_path):
    repository = tmp_path / "repo"
    (repository / ".git").mkdir(parents=True)
    project_dir = repository / "data" / "training" / "toy"
    project_dir.mkdir(parents=True)
    observation_dir = repository / "data" / "observations" / "toy"
    observation_dir.mkdir(parents=True)
    source_dir = tmp_path / "private-sources"
    source_dir.mkdir()

    image = Image.new("L", (24, 20), 255)
    draw = ImageDraw.Draw(image)
    draw.line((4, 4, 17, 15), fill=0, width=2)
    source_path = source_dir / "reference.png"
    image.save(source_path, format="PNG", optimize=False, compress_level=9)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source_lock = {
        "format": "scriptstrokes-source-lock",
        "format_version": 1,
        "sources": [
            {
                "id": "toy-source",
                "expected_filename": "reference.png",
                "sha256": digest,
                "width_px": 24,
                "height_px": 20,
                "redistribution": "private-until-reviewed",
            }
        ],
    }
    observation_manifest = {
        "format": "scriptstrokes-observations",
        "format_version": 1,
        "model_id": "toy",
        "sources": [
            {
                "source_id": "toy-source",
                "filename": "reference.png",
                "sha256": digest,
                "width_px": 24,
                "height_px": 20,
                "rights_status": "unknown-needs-review",
            }
        ],
        "items": [
            {
                "id": "observation-a",
                "kind": "glyph-isolated",
                "label": "a",
                "unicode": ["U+0061"],
                "source_id": "toy-source",
                "source_bbox": [2, 2, 22, 18],
                "crop_locator": "local-only:a.png",
                "label_source": "synthetic test",
                "review_status": "accepted",
            }
        ],
    }
    corpus = {
        "format": "scriptstrokes-fit-corpus",
        "format_version": 1,
        "samples": [
            {
                "id": "sample-a",
                "kind": "glyph",
                "text": "a",
                "observation_id": "observation-a",
                "split": "fit",
            }
        ],
    }
    families = {
        "format": "scriptstrokes-fit-families",
        "format_version": 1,
        "families": [
            {
                "id": "family-a",
                "primitive": "primitive-a",
                "fit": ["sample-a"],
                "review_status": "reviewed",
                "construction_evidence": "synthetic",
                "trajectory_evidence": "synthetic",
            }
        ],
    }
    topology = {
        "format": "scriptstrokes-fit-topology",
        "format_version": 1,
        "glyphs": {
            "a": {
                "character": "a",
                "advance_width": 20,
                "review_status": "reviewed",
                "strokes": [
                    {
                        "id": "body",
                        "parts": [
                            {
                                "instance": "a-body",
                                "primitive": "primitive-a",
                                "attach": False,
                            }
                        ],
                    }
                ],
            }
        },
    }
    anchors = {
        "format": "scriptstrokes-fit-anchors",
        "format_version": 1,
        "coordinate_frame": "observation crop pixels",
        "samples": {
            "sample-a": {
                "source_id": "toy-source",
                "source_bbox": [2, 2, 22, 18],
                "origin_x": 0,
                "baseline_y": 15,
                "review_status": "provisional",
            }
        },
    }
    fit_config = {
        "format": "scriptstrokes-fit-config",
        "format_version": 1,
        "objective": {"version": 1, "weights": {}, "residual_sigma": 0.05},
        "optimizer": {},
        "split_policy": {
            "fit": "fit only",
            "validation": "selection only",
            "hold-out": "diagnostic only",
        },
    }
    grammar = {
        "id": "toy-seed",
        "ductus": {
            "units_per_em": 1000,
            "baseline": 0,
            "x_height": 400,
            "cap_height": 700,
            "ascender": 700,
            "descender": -200,
            "slant_degrees": 0,
            "stroke_width": 50,
        },
        "primitives": {
            "primitive-a": {
                "id": "primitive-a",
                "kind": "cubic-chain",
                "frame": {"start": [0, 0], "nominal_end": [1, 1]},
                "segments": [{"c1": [0.2, 0.3], "c2": [0.8, 0.7], "end": [1, 1]}],
                "provenance": {"construction": "synthetic", "trajectory": "synthetic"},
            }
        },
    }
    project = {
        "format": "scriptstrokes-fit-project",
        "format_version": 1,
        "id": "toy",
        "runtime_model_id": "toy",
        "source_lock": "source-lock.json",
        "observation_manifest": "../../observations/toy/manifest.json",
        "corpus": "corpus.yaml",
        "families": "families.yaml",
        "topology": "topology.yaml",
        "anchors": "anchors.json",
        "fit_config": "fit-config.yaml",
        "seed_grammar": "seed-grammar.yaml",
        "private_work_dir": "../../../work/toy",
        "compile": {"output": "../../models/toy.yaml", "runtime_schema_version": 2},
    }

    (project_dir / "source-lock.json").write_text(json.dumps(source_lock), encoding="utf-8")
    (observation_dir / "manifest.json").write_text(
        json.dumps(observation_manifest), encoding="utf-8"
    )
    for name, document in (
        ("corpus.yaml", corpus),
        ("families.yaml", families),
        ("topology.yaml", topology),
        ("fit-config.yaml", fit_config),
        ("seed-grammar.yaml", grammar),
        ("project.yaml", project),
    ):
        (project_dir / name).write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    (project_dir / "anchors.json").write_text(json.dumps(anchors), encoding="utf-8")
    return repository, project_dir / "project.yaml", source_dir, source_path


def _tree_hashes(path):
    return {
        item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_project_validation_cli_resolves_schemas_and_references(tmp_path):
    repository, project_file, _, _ = _write_fixture(tmp_path)
    result = CliRunner().invoke(app, ["fit", "validate-project", str(project_file)])

    assert result.exit_code == 0, result.output
    assert "Validated toy: 1 source locks, 1 samples" in result.output
    assert repository in project_file.parents


def test_materialization_verifies_sources_and_regenerates_deterministically(tmp_path):
    repository, project_file, source_dir, _ = _write_fixture(tmp_path)
    bundle = load_fit_project(project_file)
    project_hashes = _tree_hashes(project_file.parent)

    first = materialize_fit_project(bundle, source_dir)
    first_hashes = _tree_hashes(first.output_dir)
    second = materialize_fit_project(bundle, source_dir)

    assert _tree_hashes(second.output_dir) == first_hashes
    assert _tree_hashes(project_file.parent) == project_hashes
    assert first.source_count == 1
    assert first.sample_count == 1
    assert first.artifact_count == 5
    with Image.open(first.output_dir / "crops" / "sample-a.png") as crop:
        assert crop.size == (20, 16)
    with Image.open(first.output_dir / "masks" / "sample-a.png") as mask:
        assert mask.size == (20, 16)
        assert set(np.asarray(mask).ravel()) == {0, 255}
    distance = np.load(first.output_dir / "distance-fields" / "sample-a.npy", allow_pickle=False)
    assert distance.shape == (16, 20)
    assert distance.dtype == np.float32
    assert (first.output_dir / "review-sheet.png").is_file()
    materialization = json.loads(
        (first.output_dir / "materialization.json").read_text(encoding="utf-8")
    )
    assert materialization["project_id"] == "toy"
    assert repository / "work" / "toy" == second.output_dir.parent


def test_source_hash_and_dimensions_fail_before_output_is_created(tmp_path):
    repository, project_file, source_dir, source_path = _write_fixture(tmp_path)
    bundle = load_fit_project(project_file)
    source_path.write_bytes(b"changed after lock")

    with pytest.raises(ValueError, match="source hash mismatch"):
        materialize_fit_project(bundle, source_dir)
    assert not (repository / "work" / "toy").exists()


def test_source_dimension_mismatch_is_rejected_even_with_matching_metadata(tmp_path):
    repository, project_file, source_dir, _ = _write_fixture(tmp_path)
    source_lock_path = project_file.parent / "source-lock.json"
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    source_lock["sources"][0]["width_px"] = 25
    source_lock_path.write_text(json.dumps(source_lock), encoding="utf-8")
    manifest_path = repository / "data" / "observations" / "toy" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["width_px"] = 25
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle = load_fit_project(project_file)

    with pytest.raises(ValueError, match="source dimensions mismatch"):
        materialize_fit_project(bundle, source_dir)
    assert not (repository / "work" / "toy").exists()


def test_materialization_rejects_unsafe_paths_and_source_filenames(tmp_path):
    repository, project_file, source_dir, _ = _write_fixture(tmp_path)
    bundle = load_fit_project(project_file)

    with pytest.raises(ValueError, match="must stay inside"):
        materialize_fit_project(bundle, source_dir, work_dir=tmp_path / "outside")

    source_lock_path = project_file.parent / "source-lock.json"
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    source_lock["sources"][0]["expected_filename"] = "../reference.png"
    source_lock_path.write_text(json.dumps(source_lock), encoding="utf-8")
    with pytest.raises(ValueError, match="simple basenames"):
        load_fit_project(project_file)

    assert repository in project_file.parents


def test_materialization_refuses_output_symlinks(tmp_path):
    repository, project_file, source_dir, _ = _write_fixture(tmp_path)
    bundle = load_fit_project(project_file)
    work_dir = repository / "work" / "toy"
    work_dir.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (work_dir / "materialized").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        materialize_fit_project(bundle, source_dir)
    assert list(external.iterdir()) == []


def test_materialization_rejects_source_symlinks(tmp_path):
    repository, project_file, source_dir, source_path = _write_fixture(tmp_path)
    bundle = load_fit_project(project_file)
    external_source = tmp_path / "external-source.png"
    source_path.replace(external_source)
    source_path.symlink_to(external_source)

    with pytest.raises(ValueError, match="must not be a symlink"):
        materialize_fit_project(bundle, source_dir)
    assert not (repository / "work" / "toy").exists()


def test_project_rejects_escaping_config_paths_and_artifact_ids(tmp_path):
    repository, project_file, _, _ = _write_fixture(tmp_path)
    project = yaml.safe_load(project_file.read_text(encoding="utf-8"))
    project["source_lock"] = "../../../../outside-source-lock.json"
    project_file.write_text(yaml.safe_dump(project), encoding="utf-8")
    (tmp_path / "outside-source-lock.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must stay inside"):
        load_fit_project(project_file)

    project["source_lock"] = "source-lock.json"
    project_file.write_text(yaml.safe_dump(project), encoding="utf-8")
    corpus_path = project_file.parent / "corpus.yaml"
    corpus = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
    corpus["samples"][0]["id"] = "../unsafe"
    corpus_path.write_text(yaml.safe_dump(corpus), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe fitting artifact identifier"):
        load_fit_project(project_file)
    assert repository in project_file.parents
