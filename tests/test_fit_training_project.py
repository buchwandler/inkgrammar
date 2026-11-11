import json
from pathlib import Path

import pytest
import yaml

from inkgrammar.fitting.models import (
    FitCorpus,
    FitFamiliesFile,
    FitProject,
    FitTopologyFile,
    ScriptGrammar,
    SourceLockFile,
)
from inkgrammar.fitting.objective import ObjectiveConfig
from inkgrammar.fitting.optimizer import OptimizerConfig
from inkgrammar.fitting.workspace import FitRunConfig
from inkgrammar.observations import ObservationManifest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT / "data" / "training" / "sas-1968"


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_training_project_files_validate_and_resolve_relative_paths():
    project = FitProject.model_validate(read_yaml(PROJECT_DIR / "project.yaml"))
    source_lock = SourceLockFile.model_validate_json(
        (PROJECT_DIR / project.source_lock).read_text(encoding="utf-8")
    )
    corpus = FitCorpus.model_validate(read_yaml(PROJECT_DIR / project.corpus))
    topology = FitTopologyFile.model_validate(read_yaml(PROJECT_DIR / project.topology))
    grammar = ScriptGrammar.model_validate(read_yaml(PROJECT_DIR / "seed-grammar.yaml"))
    fit_config = read_yaml(PROJECT_DIR / project.fit_config)
    objective = ObjectiveConfig.model_validate(fit_config["objective"])
    optimizer = OptimizerConfig.model_validate(fit_config["optimizer"])

    assert project.runtime_model_id == "sas-1968"
    assert project.compile.runtime_schema_version == 2
    assert (PROJECT_DIR / project.private_work_dir).resolve() == (ROOT / "work" / "sas-1968")
    assert (PROJECT_DIR / project.compile.output).resolve() == (
        ROOT / "data" / "models" / "sas-1968.yaml"
    )
    assert len(source_lock.sources) == 5
    assert len(corpus.samples) == 15
    assert objective.version == 3
    assert optimizer.seed == 1968
    assert set(topology.glyphs) <= {sample.text for sample in corpus.samples}
    assert all(
        part.primitive in grammar.primitives
        for glyph in topology.glyphs.values()
        for stroke in glyph.strokes
        for part in stroke.parts
    )


def test_source_lock_splits_and_anchors_match_release_safe_observation_metadata():
    source_lock = SourceLockFile.model_validate_json(
        (PROJECT_DIR / "source-lock.json").read_text(encoding="utf-8")
    )
    corpus = FitCorpus.model_validate(read_yaml(PROJECT_DIR / "corpus.yaml"))
    families = FitFamiliesFile.model_validate(read_yaml(PROJECT_DIR / "families.yaml"))
    anchors = json.loads((PROJECT_DIR / "anchors.json").read_text(encoding="utf-8"))
    observations = ObservationManifest.model_validate_json(
        (ROOT / "data" / "observations" / "sas-1968" / "manifest.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads(
        (ROOT / "data" / "observations" / "sas-1968" / "sources.json").read_text(encoding="utf-8")
    )
    samples = {sample.id: sample for sample in corpus.samples}
    observations_by_id = {item.id: item for item in observations.items}
    sources_by_id = {source["id"]: source for source in source_manifest}

    assert set(anchors["samples"]) == set(samples)
    for source in source_lock.sources:
        source_record = sources_by_id[source.id]
        source_image = next(item for item in observations.sources if item.source_id == source.id)
        assert source.expected_filename == source_image.filename
        assert source_record["locator"] == f"prepared-image:{source.expected_filename}"
        assert "Do not redistribute" in source_record["rights"]
        assert source.sha256 == source_image.sha256
        assert (source.width_px, source.height_px) == (
            source_image.width_px,
            source_image.height_px,
        )

    for family in families.families:
        for sample_id in family.fit:
            assert samples[sample_id].split == "fit"
        for sample_id in family.validation:
            assert samples[sample_id].split == "validation"
        for sample_id in family.hold_out:
            assert samples[sample_id].split == "hold-out"
        if family.review_status != "uncertain":
            assert (
                family.primitive
                in ScriptGrammar.model_validate(
                    read_yaml(PROJECT_DIR / "seed-grammar.yaml")
                ).primitives
            )

    for sample_id, anchor in anchors["samples"].items():
        observation = observations_by_id[sample_id]
        assert anchor["source_id"] == observation.source_id
        assert anchor["source_bbox"] == list(observation.source_bbox)
        assert anchor["review_status"] == "reviewed"
        assert anchor["reviewed_by"] == "agent:coding-assistant"
        assert anchor["review_method"] == "visual review of source crop"
        x0, y0, x1, y1 = observation.source_bbox
        assert 0 <= anchor["origin_x"] <= x1 - x0
        assert 0 <= anchor["baseline_y"] <= y1 - y0


def test_committed_training_tree_contains_only_source_free_documents():
    allowed = {".json", ".md", ".yaml", ".yml"}

    assert all(path.suffix in allowed for path in PROJECT_DIR.rglob("*") if path.is_file())


def test_compositional_families_isolate_w_h_k_and_local_responsibility():
    corpus = FitCorpus.model_validate(read_yaml(PROJECT_DIR / "corpus.yaml"))
    families = FitFamiliesFile.model_validate(read_yaml(PROJECT_DIR / "families.yaml"))
    topology = FitTopologyFile.model_validate(read_yaml(PROJECT_DIR / "topology.yaml"))
    grammar = ScriptGrammar.model_validate(read_yaml(PROJECT_DIR / "seed-grammar.yaml"))
    samples = {sample.text: sample for sample in corpus.samples}
    family_by_id = {family.id: family for family in families.families}

    def primitive_ids(character):
        glyph = topology.glyphs[character]
        return [part.primitive for stroke in glyph.strokes for part in stroke.parts]

    assert primitive_ids("w").count("sas-underturn-01") == 2
    assert primitive_ids("n") == ["sas-entry-01", "sas-hump-01", "sas-exit-01"]
    assert primitive_ids("m") == [
        "sas-entry-01",
        "sas-hump-01",
        "sas-hump-01",
        "sas-exit-01",
    ]
    assert primitive_ids("h") == [
        "sas-ascender-loop-01",
        "sas-hump-01",
        "sas-exit-01",
    ]
    assert primitive_ids("k") == [
        "sas-ascender-loop-01",
        "sas-k-continuation-01",
    ]
    assert all(
        primitive in grammar.primitives
        for character in ("w", "n", "m", "h", "k")
        for primitive in primitive_ids(character)
    )

    assert family_by_id["sas-underturn-01"].fit == [
        "sas1968-alphabet-U+0069",
        "sas1968-alphabet-U+0075",
    ]
    assert family_by_id["sas-underturn-01"].hold_out == [
        "sas1968-alphabet-U+0077",
    ]
    assert family_by_id["sas-hump-01"].hold_out == [
        "sas1968-alphabet-U+0068",
    ]
    assert family_by_id["sas-ascender-loop-01"].hold_out == [
        "sas1968-alphabet-U+006B",
        "sas1968-alphabet-U+0068",
    ]
    assert {region.id for region in samples["h"].loss_regions} == {
        "ascender-loop",
        "hump-core",
    }


def test_objective_v3_rejects_unsupported_optimizer_settings():
    config = {
        "format": "inkgrammar-fit-config",
        "format_version": 1,
        "objective": {"version": 3},
        "optimizer": {
            "method": "centerline-em-trf",
            "max_iterations": 20,
        },
        "split_policy": {
            "fit": "fit",
            "validation": "diagnostic",
            "hold-out": "report",
        },
    }

    with pytest.raises(ValueError, match="does not support optimizer settings"):
        FitRunConfig.model_validate(config)
