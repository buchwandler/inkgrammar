import json
import zipfile
from pathlib import Path

from scriptstrokes.release import build_release


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data" / "models" / "demo-looped.yaml"


def test_release(tmp_path):
    release_root, archive = build_release([MODEL], tmp_path, "test")
    assert (release_root / "manifest.json").is_file()
    assert (release_root / "models" / "demo-looped.json").is_file()
    assert (release_root / "schemas" / "writing-model.schema.json").is_file()
    assert archive.is_file()

    catalog = json.loads((release_root / "catalog.json").read_text())
    assert catalog["models"][0]["id"] == "demo-looped"

    with zipfile.ZipFile(archive) as handle:
        assert "scriptstrokes-data-test/manifest.json" in handle.namelist()
