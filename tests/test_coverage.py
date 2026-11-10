import json

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from scriptstrokes.cli import app
from scriptstrokes.coverage import coverage_markdown, model_coverage
from scriptstrokes.io import load_model
from scriptstrokes.regression import RegressionManifest


def test_coverage_does_not_confuse_schema_validity_with_sas_completeness():
    model = load_model("data/models/demo-looped.yaml")
    model.id = "sas-1968"
    report = model_coverage(model)
    assert len(report["expected_glyphs"]) == 63
    assert report["missing_glyphs"]
    assert report["coverage_blockers"]
    assert not report["complete"]
    assert "Coverage and schema validity" in coverage_markdown(report)


def test_regression_manifest_integrity():
    manifest = RegressionManifest(model_id="test", fixtures=[{
        "id": "a", "group": "isolated-lowercase", "text": "a", "split": "public-snapshot",
    }])
    payload = manifest.model_dump()
    payload["results"] = [{"fixture_id": "unknown", "status": "pass"}]
    with pytest.raises(ValidationError, match="unknown fixture"):
        RegressionManifest.model_validate(payload)
    payload["results"] = []
    payload["fixtures"] *= 2
    with pytest.raises(ValidationError, match="duplicate"):
        RegressionManifest.model_validate(payload)


def test_regression_missing_and_failed_results_are_not_complete():
    model = load_model("data/models/demo-looped.yaml")
    regressions = RegressionManifest(model_id=model.id, fixtures=[{
        "id": "sample", "group": "isolated-lowercase", "text": "i", "split": "public-snapshot",
    }])
    report = model_coverage(model, regressions=regressions)
    assert not report["complete"]
    assert report["regression_not_run_count"] == 1
    assert report["regression_missing_results"] == ["sample"]


def test_coverage_cli_formats():
    runner = CliRunner()
    result = runner.invoke(app, ["model", "coverage", "data/models/demo-looped.yaml"])
    assert result.exit_code == 0
    assert json.loads(result.output)["model_id"] == "demo-looped"
    result = runner.invoke(app, ["model", "coverage", "data/models/demo-looped.yaml",
                                 "--format", "markdown"])
    assert result.exit_code == 0
    assert "# Coverage: demo-looped" in result.output


@pytest.mark.parametrize("problem,code", [
    ("missing-form", "glyph.required-form"),
    ("unused-component", "component.unused"),
    ("missing-rule", "connection.missing-exit-rule"),
    ("missing-glyph", "sas.glyph-coverage"),
])
def test_additional_release_qc(problem, code):
    from scriptstrokes.qc import validate_model
    model = load_model("data/models/demo-looped.yaml")
    if problem == "missing-form":
        model.glyphs["i"].required_forms = ["medial"]
    elif problem == "unused-component":
        model.components["unused"] = next(iter(model.components.values())).model_copy()
    elif problem == "missing-rule":
        model.connection_rules = []
    else:
        model.id = "sas-1968"
    assert code in {issue.code for issue in validate_model(model).issues}


def test_explicit_unused_component_is_allowed():
    from scriptstrokes.qc import validate_model
    model = load_model("data/models/demo-looped.yaml")
    component = next(iter(model.components.values())).model_copy()
    component.unused_reason = "Reserved for documented family member not yet fitted."
    model.components["unused"] = component
    assert "component.unused" not in {issue.code for issue in validate_model(model).issues}
