import functools
import hashlib
import http.server
import json
import stat
import threading
import zipfile

import pytest
from typer.testing import CliRunner

from scriptstrokes import data
from scriptstrokes.cli import app
from scriptstrokes.release import build_release


@pytest.fixture
def server(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.mkdir()
    _, archive = build_release(["data/models/demo-looped.yaml"], public / "v0.1.0", "0.1.0")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(public))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    monkeypatch.setenv("SCRIPTSTROKES_DATA_BASE_URL", base)
    monkeypatch.setenv("SCRIPTSTROKES_DATA_DIR", str(tmp_path / "cache"))
    (public / "latest.json").write_text(
        json.dumps(
            {
                "tag_name": "v0.1.0",
                "assets": [
                    {"name": archive.name, "browser_download_url": f"{base}/v0.1.0/{archive.name}"}
                ],
            }
        )
    )
    yield archive
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def rewrite_archive(archive, payloads):
    with zipfile.ZipFile(archive, "w") as output:
        for name, content in payloads.items():
            output.writestr(name, content)
    archive.with_suffix(".zip.sha256").write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n"
    )


def test_install_and_load_offline(server, monkeypatch):
    root = data.install_release("0.1.0")
    assert root.is_dir()
    assert data.list_installed_releases() == ["0.1.0"]
    monkeypatch.setattr(
        data.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected network")
    )
    assert data.verify_release("0.1.0") == root
    assert data.load_released_model("demo-looped", release="0.1.0").id == "demo-looped"
    assert data.load_released_model("demo-looped", release="latest").id == "demo-looped"
    assert data.install_release("0.1.0") == root
    with pytest.raises(KeyError):
        data.load_released_model("missing", release="0.1.0")
    data.remove_release("0.1.0")
    assert data.list_installed_releases() == []


def test_latest_metadata_and_version_override(server, monkeypatch):
    monkeypatch.setenv("SCRIPTSTROKES_DATA_VERSION", "0.1.0")
    assert data.resolve_release()[0] == "0.1.0"
    assert data.install_release("latest").name == "0.1.0"


def test_latest_uses_asset_url_not_guessed_path(server):
    alternate = server.parent.parent / "alternate.zip"
    alternate.write_bytes(server.read_bytes())
    alternate.with_suffix(".zip.sha256").write_bytes(server.with_suffix(".zip.sha256").read_bytes())
    latest = server.parent.parent / "latest.json"
    metadata = json.loads(latest.read_text())
    url = metadata["assets"][0]["browser_download_url"]
    metadata["assets"][0]["browser_download_url"] = url.split("/v0.1.0/")[0] + "/alternate.zip"
    latest.write_text(json.dumps(metadata))
    server.unlink()
    assert data.install_release("latest").name == "0.1.0"


def test_archive_corruption_rejected_before_extraction(server):
    server.write_bytes(server.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="archive SHA-256"):
        data.install_release("0.1.0")
    assert not (data.data_cache_dir() / "releases" / "0.1.0").exists()
    assert not list((data.data_cache_dir() / "downloads").iterdir())


def test_internal_corruption_rejected_atomically(server):
    with zipfile.ZipFile(server) as source:
        payloads = {name: source.read(name) for name in source.namelist()}
    payloads["scriptstrokes-data-0.1.0/models/demo-looped.json"] += b" "
    rewrite_archive(server, payloads)
    with pytest.raises(ValueError, match="internal checksum"):
        data.install_release("0.1.0")
    assert not list((data.data_cache_dir() / "releases").iterdir())


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/absolute",
        "root/../../escape",
        "root\\escape",
        "scriptstrokes-data-0.1.0/../escape",
        "unexpected-root/file",
    ],
)
def test_unsafe_paths_rejected(server, name):
    rewrite_archive(server, {name: b"unsafe"})
    with pytest.raises(ValueError, match="unsafe|unexpected"):
        data.install_release("0.1.0")
    assert not (data.data_cache_dir() / "releases" / "0.1.0").exists()


def test_symlink_member_rejected(server):
    with zipfile.ZipFile(server, "w") as archive:
        member = zipfile.ZipInfo("scriptstrokes-data-0.1.0/link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, "../../escape")
    server.with_suffix(".zip.sha256").write_text(
        f"{hashlib.sha256(server.read_bytes()).hexdigest()}  {server.name}\n"
    )
    with pytest.raises(ValueError, match="member type"):
        data.install_release("0.1.0")


def test_installed_corruption_rejected(server):
    root = data.install_release("0.1.0")
    (root / "models/demo-looped.json").write_text("{}")
    with pytest.raises(ValueError, match="internal checksum"):
        data.verify_release("0.1.0")


def test_incompatible_release_rejected(server):
    build_release(
        ["data/models/demo-looped.yaml"], server.parent, "0.1.0", requires_scriptstrokes=">=999"
    )
    with pytest.raises(ValueError, match="incompatible"):
        data.install_release("0.1.0")
    assert not (data.data_cache_dir() / "releases" / "0.1.0").exists()


@pytest.mark.parametrize("version", ["../bad", "/absolute", "v0.1.0", "0/1", ""])
def test_bad_version_rejected(version, monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIPTSTROKES_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        data.verify_release(version)


def test_cli(server):
    runner = CliRunner()
    result = runner.invoke(app, ["data", "install", "0.1.0"])
    assert result.exit_code == 0, result.output
    assert "Release URL:" in result.output
    assert "checksums verified" in result.output
    for args in (
        ["list"],
        ["path", "0.1.0"],
        ["verify", "0.1.0"],
        ["model", "demo-looped", "--release", "0.1.0"],
        ["remove", "0.1.0"],
    ):
        result = runner.invoke(app, ["data", *args])
        assert result.exit_code == 0, result.output


def test_v3_release_deterministic_and_self_contained(tmp_path):
    root, first = build_release(["data/models/demo-looped.yaml"], tmp_path / "a", "0.1.0")
    _, second = build_release(["data/models/demo-looped.yaml"], tmp_path / "b", "0.1.0")
    assert first.read_bytes() == second.read_bytes()
    assert json.loads((root / "manifest.json").read_text())["format_version"] == 3
    assert (root / "schemas/observation.schema.json").is_file()
    assert (root / "schemas/regression.schema.json").is_file()
    assert (
        first.with_suffix(".zip.sha256")
        .read_text()
        .startswith(hashlib.sha256(first.read_bytes()).hexdigest())
    )
    assert not list(root.rglob("*.png"))


@pytest.mark.parametrize("problem", ["wrong-name", "malformed"])
def test_bad_external_sidecar(server, problem):
    payload = ("0" * 64 + "  wrong.zip\n") if problem == "wrong-name" else "bad checksum"
    server.with_suffix(".zip.sha256").write_text(payload)
    with pytest.raises(ValueError):
        data.install_release("0.1.0")


def test_unchecksummed_payload_rejected(server):
    with zipfile.ZipFile(server) as source:
        payloads = {name: source.read(name) for name in source.namelist()}
    payloads["scriptstrokes-data-0.1.0/extra.txt"] = b"unlisted"
    rewrite_archive(server, payloads)
    with pytest.raises(ValueError, match="exactly all"):
        data.install_release("0.1.0")


def test_duplicate_member_rejected(server):
    with zipfile.ZipFile(server, "w") as archive:
        archive.writestr("scriptstrokes-data-0.1.0/file", "first")
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("scriptstrokes-data-0.1.0/file", "second")
    server.with_suffix(".zip.sha256").write_text(
        f"{hashlib.sha256(server.read_bytes()).hexdigest()}  {server.name}\n"
    )
    with pytest.raises(ValueError, match="duplicate member"):
        data.install_release("0.1.0")
