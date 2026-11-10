import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt
from typer.testing import CliRunner

from inkgrammar.cli import app
from inkgrammar.fitting.checkpoint import load_checkpoint
from inkgrammar.fitting.family_fit import (
    optimize_primitive_family,
    primitive_path,
    refresh_primitive_artifact_hashes,
)
from inkgrammar.fitting.models import (
    CubicSegment,
    PrimitiveFrame,
    PrimitiveTemplate,
    TopologyTransform,
)
from inkgrammar.fitting.render import render_paths
from inkgrammar.fitting.workspace import load_fit_project, materialize_fit_project

PRIMITIVE_ID = "sas-open-oval-01"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _template() -> PrimitiveTemplate:
    return PrimitiveTemplate(
        id=PRIMITIVE_ID,
        frame=PrimitiveFrame(start=(0.0, 0.0), nominal_end=(1.0, 0.0)),
        segments=[
            CubicSegment(
                c1=(0.25, 0.32),
                c2=(0.75, 0.68),
                end=(1.0, 0.0),
            )
        ],
        free_parameters=["segments.0.c1.y"],
        bounds={"segments.0.c1.y": (0.1, 0.9)},
    )


def _source_image(template: PrimitiveTemplate) -> np.ndarray:
    transform = TopologyTransform(scale_x=45.0, scale_y=40.0)
    coverage = render_paths(
        {"oval": primitive_path(template, transform)},
        canvas_size=(64, 64),
        origin=(8.0, 55.0),
        pixels_per_unit=1.0,
        stroke_width=3.0,
        supersampling=4,
    ).coverage
    return np.clip(255.0 - coverage * 255.0, 0, 255).astype(np.uint8)


def _write_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repo"
    (repository / ".git").mkdir(parents=True)
    project_dir = repository / "data" / "training" / "toy"
    project_dir.mkdir(parents=True)
    observation_dir = repository / "data" / "observations" / "toy"
    observation_dir.mkdir(parents=True)
    source_dir = repository / "private-sources"
    source_dir.mkdir()

    seed = _template()
    true_template = seed.model_copy(
        update={
            "segments": [
                CubicSegment(
                    c1=(0.25, 0.74),
                    c2=(0.75, 0.68),
                    end=(1.0, 0.0),
                )
            ]
        }
    )
    source_records = []
    observations = []
    samples = []
    anchors = {}
    topology_glyphs = {}
    for character, sample_id, split in (
        ("c", "sample-c", "fit"),
        ("a", "sample-a", "fit"),
        ("C", "sample-C", "hold-out"),
    ):
        source_id = f"source-{sample_id}"
        filename = f"{sample_id}.png"
        source_path = source_dir / filename
        Image.fromarray(_source_image(true_template), mode="L").save(
            source_path, format="PNG", optimize=False, compress_level=9
        )
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        source_records.append(
            {
                "id": source_id,
                "expected_filename": filename,
                "sha256": digest,
                "width_px": 64,
                "height_px": 64,
            }
        )
        observations.append(
            {
                "id": f"observation-{sample_id}",
                "kind": "glyph-isolated",
                "label": character,
                "unicode": [f"U+{ord(character):04X}"],
                "source_id": source_id,
                "source_bbox": [0, 0, 64, 64],
                "label_source": "synthetic known-answer fixture",
                "review_status": "accepted",
                "fitting_status": "usable",
            }
        )
        samples.append(
            {
                "id": sample_id,
                "kind": "glyph",
                "text": character,
                "observation_id": f"observation-{sample_id}",
                "split": split,
                "loss_regions": [
                    {
                        "id": "oval-core",
                        "instances": [f"{character}-oval"],
                        "bbox_source": [0, 0, 64, 64],
                    }
                ],
            }
        )
        anchors[sample_id] = {
            "source_id": source_id,
            "source_bbox": [0, 0, 64, 64],
            "origin_x": 8.0,
            "baseline_y": 55.0,
            "review_status": "reviewed",
            "reviewed_by": "synthetic-test",
            "review_method": "generated known-answer geometry",
        }
        topology_glyphs[character] = {
            "character": character,
            "advance_width": 60.0,
            "review_status": "reviewed",
            "strokes": [
                {
                    "id": "oval-stroke",
                    "parts": [
                        {
                            "instance": f"{character}-oval",
                            "primitive": PRIMITIVE_ID,
                            "attach": False,
                            "transform": {"scale_x": 45.0, "scale_y": 40.0},
                        }
                    ],
                }
            ],
        }

    source_manifest = {
        "format": "inkgrammar-observations",
        "format_version": 1,
        "model_id": "toy",
        "sources": [
            {
                "source_id": record["id"],
                "filename": record["expected_filename"],
                "sha256": record["sha256"],
                "width_px": record["width_px"],
                "height_px": record["height_px"],
                "rights_status": "private synthetic fixture",
            }
            for record in source_records
        ],
        "items": observations,
    }
    _write_json(
        project_dir / "source-lock.json",
        {"format": "inkgrammar-source-lock", "format_version": 1, "sources": source_records},
    )
    _write_json(observation_dir / "manifest.json", source_manifest)
    _write_yaml(
        project_dir / "corpus.yaml",
        {"format": "inkgrammar-fit-corpus", "format_version": 1, "samples": samples},
    )
    _write_yaml(
        project_dir / "families.yaml",
        {
            "format": "inkgrammar-fit-families",
            "format_version": 1,
            "families": [
                {
                    "id": PRIMITIVE_ID,
                    "primitive": PRIMITIVE_ID,
                    "fit": ["sample-c", "sample-a"],
                    "hold_out": ["sample-C"],
                    "review_status": "reviewed",
                    "construction_evidence": "synthetic known answer",
                    "trajectory_evidence": "synthetic known answer",
                }
            ],
        },
    )
    _write_yaml(
        project_dir / "topology.yaml",
        {
            "format": "inkgrammar-fit-topology",
            "format_version": 1,
            "glyphs": topology_glyphs,
        },
    )
    _write_json(
        project_dir / "anchors.json",
        {
            "format": "inkgrammar-fit-anchors",
            "format_version": 1,
            "coordinate_frame": "crop-local pixels",
            "samples": anchors,
        },
    )
    _write_yaml(
        project_dir / "fit-config.yaml",
        {
            "format": "inkgrammar-fit-config",
            "format_version": 1,
            "objective": {"version": 1, "residual_sigma": 0.05},
            "optimizer": {"max_iterations": 100, "tolerance": 1e-6},
            "split_policy": {
                "fit": "parameter selection",
                "validation": "model selection only",
                "hold-out": "diagnostic only",
            },
            "render": {"stroke_width_px": 3.0, "supersampling": 4},
        },
    )
    _write_yaml(
        project_dir / "seed-grammar.yaml",
        {
            "id": "toy-seed",
            "ductus": {
                "units_per_em": 1000,
                "baseline": 0.0,
                "x_height": 40.0,
                "cap_height": 70.0,
                "ascender": 70.0,
                "descender": -20.0,
                "slant_degrees": 0.0,
                "stroke_width": 3.0,
            },
            "primitives": {PRIMITIVE_ID: seed.model_dump(mode="json")},
        },
    )
    project = {
        "format": "inkgrammar-fit-project",
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
    project_path = project_dir / "project.yaml"
    _write_yaml(project_path, project)
    return repository, project_path, source_dir


def test_refresh_primitive_artifact_hashes_preserves_fit_results(tmp_path):
    repository, project_path, source_dir = _write_project(tmp_path)
    bundle = load_fit_project(project_path, repository_root=repository)
    materialize_fit_project(bundle, source_dir)
    optimize_primitive_family(bundle, PRIMITIVE_ID)

    checkpoint_path = bundle.project_dir / "checkpoints" / f"{PRIMITIVE_ID}.json"
    report_path = bundle.project_dir / "reports" / f"{PRIMITIVE_ID}.json"
    before_checkpoint = load_checkpoint(checkpoint_path)
    before_report = json.loads(report_path.read_text(encoding="utf-8"))
    project_path.write_text(project_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    refreshed = refresh_primitive_artifact_hashes(bundle, PRIMITIVE_ID)

    after_checkpoint = load_checkpoint(checkpoint_path)
    after_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert refreshed.project_sha256 != before_checkpoint.inputs.project_sha256
    assert after_checkpoint.inputs == refreshed
    assert after_checkpoint.parameters == before_checkpoint.parameters
    assert after_checkpoint.history == before_checkpoint.history
    assert after_report.pop("input_hashes") == refreshed.model_dump(mode="json")
    before_report.pop("input_hashes")
    assert after_report == before_report


def test_hold_out_changes_cannot_change_primitive_parameters(tmp_path):
    repository, project_path, source_dir = _write_project(tmp_path)
    bundle = load_fit_project(project_path, repository_root=repository)
    materialize_fit_project(bundle, source_dir)

    first = optimize_primitive_family(bundle, PRIMITIVE_ID)
    first_parameters = first.report["parameters"]
    first_holdout_loss = first.report["hold_out"]["samples"][0]["objective"]
    assert first.private_review_sheet_path.is_file()
    assert first.report["visual_review"]["scope"] == "private work directory"
    assert first.report["visual_review"]["status"] == "promotion_pending"
    first_sample = first.report["training"]["samples"][0]
    assert first_sample["instances"][0]["transform"]["scale_x"] == 45.0
    assert first_sample["instances"][0]["residuals"] == {}
    assert first_sample["terms"]["anchor"] == 0.0
    assert first_sample["terms"]["endpoint"] == 0.0
    assert first.report["validation"]["samples"] == []
    assert (
        first.report["training"]["final_objective"] < first.report["training"]["initial_objective"]
    )

    materialized = bundle.private_work_dir / "materialized"
    changed_mask = np.zeros((64, 64), dtype=np.uint8)
    changed_mask[31:34, 12:52] = 255
    Image.fromarray(changed_mask, mode="L").save(
        materialized / "masks" / "sample-C.png", format="PNG", optimize=False
    )
    distance = distance_transform_edt(changed_mask == 0).astype(np.float32)
    with (materialized / "distance-fields" / "sample-C.npy").open("wb") as stream:
        np.save(stream, distance, allow_pickle=False)

    cli_result = CliRunner().invoke(
        app,
        ["fit", "optimize-primitive", str(project_path), PRIMITIVE_ID, "--resume"],
    )
    assert cli_result.exit_code == 0, cli_result.output
    report_path = repository / "data" / "training" / "toy" / "reports" / f"{PRIMITIVE_ID}.json"
    second_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert second_report["parameters"] == pytest.approx(first_parameters, abs=1e-6)
    assert second_report["training"]["final_objective"] == pytest.approx(
        first.report["training"]["final_objective"], abs=1e-8
    )
    assert second_report["hold_out"]["samples"][0]["objective"] != first_holdout_loss
    assert second_report["training"]["selection_split"] == "fit"
    assert "diagnostic-only" in second_report["hold_out"]["policy"]
    checkpoint_path = (
        repository / "data" / "training" / "toy" / "checkpoints" / f"{PRIMITIVE_ID}.json"
    )
    assert checkpoint_path.is_file()

    report_result = CliRunner().invoke(
        app,
        ["fit", "report", str(project_path), "--format", "markdown"],
    )
    assert report_result.exit_code == 0, report_result.output
    assert "Hold-out diagnostics" in report_result.output


@pytest.mark.parametrize("output_name", ["checkpoints", "reports"])
def test_fit_rejects_output_directory_symlinks(tmp_path, output_name):
    repository, project_path, source_dir = _write_project(tmp_path)
    bundle = load_fit_project(project_path, repository_root=repository)
    materialize_fit_project(bundle, source_dir)
    external = tmp_path / "external"
    external.mkdir()
    (project_path.parent / output_name).symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        optimize_primitive_family(bundle, PRIMITIVE_ID)
    assert list(external.iterdir()) == []


def test_objective_v2_fit_keeps_hold_out_out_of_parameter_selection(tmp_path):
    repository, project_path, source_dir = _write_project(tmp_path)
    project_dir = project_path.parent
    families_path = project_dir / "families.yaml"
    families = yaml.safe_load(families_path.read_text(encoding="utf-8"))
    families["families"][0]["alignment"] = {"free": []}
    _write_yaml(families_path, families)

    config_path = project_dir / "fit-config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["objective"]["version"] = 2
    config["optimizer"].update(
        {
            "method": "alternating-least-squares",
            "max_cycles": 1,
            "max_nfev": 30,
            "restarts": 2,
        }
    )
    _write_yaml(config_path, config)

    bundle = load_fit_project(project_path, repository_root=repository)
    materialize_fit_project(bundle, source_dir)
    first = optimize_primitive_family(bundle, PRIMITIVE_ID)
    first_parameters = first.report["parameters"]
    first_holdout = first.report["hold_out"]["samples"][0]["objective"]
    assert first.report["format_version"] == 2
    assert first.report["training"]["selection_split"] == "fit"
    assert first.report["validation"]["samples"] == []
    blocks = first.report["selection_objective"]["blocks"]
    assert {"mean", "family_worst", "family_spread", "transform_prior", "shape_prior"} <= set(
        blocks
    )
    assert first.report["optimizer"]["strategy"] == "alternating-least-squares"
    assert first.private_review_sheet_path.is_file()
    overlay_dir = bundle.private_work_dir / "overlays" / PRIMITIVE_ID
    assert (overlay_dir / "sample-C--oval-core--nominal.png").is_file()
    assert (overlay_dir / "sample-C--oval-core--aligned.png").is_file()

    changed_mask = np.zeros((64, 64), dtype=np.uint8)
    changed_mask[31:34, 12:52] = 255
    Image.fromarray(changed_mask, mode="L").save(
        bundle.private_work_dir / "materialized" / "masks" / "sample-C.png",
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    distance = distance_transform_edt(changed_mask == 0).astype(np.float32)
    with (bundle.private_work_dir / "materialized" / "distance-fields" / "sample-C.npy").open(
        "wb"
    ) as stream:
        np.save(stream, distance, allow_pickle=False)

    second = optimize_primitive_family(bundle, PRIMITIVE_ID)
    assert second.report["parameters"] == pytest.approx(first_parameters, abs=1e-8)
    assert second.report["hold_out"]["samples"][0]["objective"] != first_holdout


def test_objective_v2_recovers_shared_shape_with_occurrence_alignment(tmp_path):
    repository, project_path, source_dir = _write_project(tmp_path)
    project_dir = project_path.parent
    topology_path = project_dir / "topology.yaml"
    topology = yaml.safe_load(topology_path.read_text(encoding="utf-8"))
    topology["glyphs"]["a"]["strokes"][0]["parts"][0]["transform"] = {
        "scale_x": 50.0,
        "scale_y": 40.0,
        "dx": -3.0,
    }
    _write_yaml(topology_path, topology)

    families_path = project_dir / "families.yaml"
    families = yaml.safe_load(families_path.read_text(encoding="utf-8"))
    families["families"][0]["alignment"] = {"free": ["dx", "dy", "log_scale_x", "log_scale_y"]}
    _write_yaml(families_path, families)
    config_path = project_dir / "fit-config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["objective"]["version"] = 2
    config["optimizer"].update(
        {
            "method": "alternating-least-squares",
            "max_cycles": 2,
            "max_nfev": 60,
            "restarts": 2,
        }
    )
    _write_yaml(config_path, config)

    bundle = load_fit_project(project_path, repository_root=repository)
    materialize_fit_project(bundle, source_dir)
    result = optimize_primitive_family(bundle, PRIMITIVE_ID)
    fitted = result.report["primitive"]["segments"][0]["c1"][1]
    assert fitted > 0.6
    aligned = result.report["occurrences"]["sample-a--a-oval"]["effective_transform"]
    assert abs(aligned["scale_x"] - 45.0) < abs(50.0 - 45.0)
    assert abs(aligned["dx"]) < 3.0


def test_objective_v2_reports_occurrence_bound_saturation(tmp_path):
    repository, project_path, source_dir = _write_project(tmp_path)
    project_dir = project_path.parent
    families_path = project_dir / "families.yaml"
    families = yaml.safe_load(families_path.read_text(encoding="utf-8"))
    families["families"][0]["alignment"] = {
        "free": ["dx"],
        "bounds": {"dx": [0.0, 1.0]},
    }
    _write_yaml(families_path, families)

    config_path = project_dir / "fit-config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["objective"]["version"] = 2
    config["optimizer"].update(
        {"max_cycles": 1, "max_nfev": 30, "restarts": 1, "joint_refine": False}
    )
    _write_yaml(config_path, config)

    bundle = load_fit_project(project_path, repository_root=repository)
    materialize_fit_project(bundle, source_dir)
    result = optimize_primitive_family(bundle, PRIMITIVE_ID)
    assert any("sample-a--a-oval.dx" in item for item in result.report["bound_saturation_warnings"])


def test_compositional_glyph_fit_uses_only_active_instances_and_fit_data(tmp_path):
    from inkgrammar.fitting.glyph_fit import fit_compositional_glyphs

    repository, project_path, source_dir = _write_project(tmp_path)
    project_dir = project_path.parent
    corpus_path = project_dir / "corpus.yaml"
    corpus = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
    sample = next(item for item in corpus["samples"] if item["id"] == "sample-a")
    sample["active_instances"] = ["a-oval"]
    _write_yaml(corpus_path, corpus)

    topology_path = project_dir / "topology.yaml"
    topology = yaml.safe_load(topology_path.read_text(encoding="utf-8"))
    topology["glyphs"]["a"]["strokes"].append(
        {
            "id": "inactive-extra",
            "parts": [
                {
                    "instance": "a-unused",
                    "primitive": PRIMITIVE_ID,
                    "attach": False,
                    "transform": {"scale_x": 8.0, "scale_y": 8.0},
                }
            ],
        }
    )
    _write_yaml(topology_path, topology)

    config_path = project_dir / "fit-config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["objective"]["version"] = 2
    _write_yaml(config_path, config)

    bundle = load_fit_project(project_path, repository_root=repository)
    materialize_fit_project(bundle, source_dir)
    first = fit_compositional_glyphs(bundle, sample_ids={"sample-a"})[0]
    assert first.report["active_instances"] == ["a-oval"]
    assert first.report["composed_instances"] == ["a-oval"]
    assert first.report["shared_geometry_updated"] is False
    assert set(first.report["occurrences"]) == {"a-oval"}
    assert first.overlay_path.is_file()

    holdout_mask = np.zeros((64, 64), dtype=np.uint8)
    holdout_mask[30:34, 10:54] = 255
    Image.fromarray(holdout_mask, mode="L").save(
        bundle.private_work_dir / "materialized" / "masks" / "sample-C.png",
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    holdout_distance = distance_transform_edt(holdout_mask == 0).astype(np.float32)
    with (bundle.private_work_dir / "materialized" / "distance-fields" / "sample-C.npy").open(
        "wb"
    ) as stream:
        np.save(stream, holdout_distance, allow_pickle=False)

    second = fit_compositional_glyphs(bundle, sample_ids={"sample-a"})[0]
    assert second.report["occurrences"]["a-oval"]["delta"] == pytest.approx(
        first.report["occurrences"]["a-oval"]["delta"], abs=1e-8
    )


def test_word_validation_is_diagnostic_only_for_holdout_words(tmp_path):
    from inkgrammar.fitting.glyph_fit import fit_compositional_glyphs
    from inkgrammar.fitting.word_validation import validate_composed_words

    repository, project_path, source_dir = _write_project(tmp_path)
    project_dir = project_path.parent
    corpus_path = project_dir / "corpus.yaml"
    corpus = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
    sample = next(item for item in corpus["samples"] if item["id"] == "sample-a")
    sample["active_instances"] = ["a-oval"]
    word_sample = dict(sample)
    word_sample.update(
        {
            "id": "sample-word",
            "kind": "word",
            "text": "aa",
            "split": "hold-out",
            "active_instances": ["0:a-oval", "1:a-oval"],
            "loss_regions": [],
        }
    )
    corpus["samples"].append(word_sample)
    _write_yaml(corpus_path, corpus)

    anchors_path = project_dir / "anchors.json"
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    anchors["samples"]["sample-word"] = dict(anchors["samples"]["sample-a"])
    _write_json(anchors_path, anchors)

    config_path = project_dir / "fit-config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["objective"]["version"] = 2
    _write_yaml(config_path, config)

    bundle = load_fit_project(project_path, repository_root=repository)
    materialize_fit_project(bundle, source_dir)
    fit_compositional_glyphs(bundle, sample_ids={"sample-a"})
    first = validate_composed_words(bundle, sample_ids={"sample-word"})[0]
    assert first.report["parameter_updates"] is False
    assert first.report["selection_split"] is None
    assert first.report["occurrence_fit_sources"] == {"a": "sample-a"}
    assert first.report["connection_checks"][-1]["status"] == "not-specified"
    assert first.overlay_path.is_file()

    holdout_mask = np.zeros((64, 64), dtype=np.uint8)
    Image.fromarray(holdout_mask, mode="L").save(
        bundle.private_work_dir / "materialized" / "masks" / "sample-word.png",
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    holdout_distance = distance_transform_edt(holdout_mask == 0).astype(np.float32)
    with (bundle.private_work_dir / "materialized" / "distance-fields" / "sample-word.npy").open(
        "wb"
    ) as stream:
        np.save(stream, holdout_distance, allow_pickle=False)

    second = validate_composed_words(bundle, sample_ids={"sample-word"})[0]
    assert second.report["metrics"] != first.report["metrics"]
    assert second.report["occurrence_fit_sources"] == first.report["occurrence_fit_sources"]
    assert second.report["parameter_updates"] is False
