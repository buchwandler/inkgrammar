"""Explicit, verified installation of versioned runtime data bundles."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from importlib.metadata import version as package_version
from pathlib import Path, PurePosixPath

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from .models import WritingModel

GITHUB = "https://github.com/buchwandler/scriptstrokes/releases/download"
LATEST = "https://api.github.com/repos/buchwandler/scriptstrokes/releases/latest"


def data_cache_dir() -> Path:
    if override := os.environ.get("SCRIPTSTROKES_DATA_DIR"):
        return Path(override).expanduser()
    if os.name == "nt":
        return Path(os.environ["LOCALAPPDATA"]) / "scriptstrokes"
    import sys
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "scriptstrokes"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "scriptstrokes"


def _version(value: str) -> str:
    if not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", value):
        raise ValueError("invalid release version")
    Version(value)
    return value


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "scriptstrokes-data"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def resolve_release(release: str | None = None) -> tuple[str, str]:
    release = release or os.environ.get("SCRIPTSTROKES_DATA_VERSION", "latest")
    base = os.environ.get("SCRIPTSTROKES_DATA_BASE_URL", GITHUB).rstrip("/")
    if release == "latest":
        metadata_url = LATEST if base == GITHUB else f"{base}/latest.json"
        metadata = json.loads(_fetch(metadata_url))
        release = _version(metadata["tag_name"].removeprefix("v"))
        filename = f"scriptstrokes-data-{release}.zip"
        assets = [asset for asset in metadata["assets"] if asset["name"] == filename]
        if len(assets) != 1:
            raise ValueError("latest release metadata must identify exactly one data archive")
        return release, assets[0]["browser_download_url"]
    release = _version(release)
    return release, f"{base}/v{release}/scriptstrokes-data-{release}.zip"


def _checksum_rows(payload: str) -> dict[str, str]:
    rows = {}
    for line in payload.splitlines():
        digest, name = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or name in rows:
            raise ValueError("invalid or duplicate checksum entry")
        _safe_path(name)
        rows[name] = digest
    return rows


def _safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ".." in path.parts
            or ":" in name or not path.parts or path.as_posix() != name.rstrip("/")):
        raise ValueError(f"unsafe archive path: {name!r}")
    return path


def download_release(release: str | None = None) -> Path:
    resolved, url = resolve_release(release)
    return _download(resolved, url)


def _download(resolved: str, url: str) -> Path:
    filename = f"scriptstrokes-data-{resolved}.zip"
    checksums = _checksum_rows(_fetch(url + ".sha256").decode("utf-8"))
    if set(checksums) != {filename}:
        raise ValueError("archive checksum sidecar does not name the resolved archive")
    downloads = data_cache_dir() / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=downloads, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "scriptstrokes-data"})
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    digest.update(block)
                    handle.write(block)
            handle.close()
            if digest.hexdigest() != checksums[filename]:
                raise ValueError("archive SHA-256 mismatch")
            destination = downloads / filename
            temporary.replace(destination)
            return destination
        finally:
            handle.close()
            temporary.unlink(missing_ok=True)


def verify_release(release: str) -> Path:
    root = data_cache_dir() / "releases" / _version(release)
    _verify_directory(root, release)
    return root


def _verify_directory(root: Path, release: str) -> None:
    if root.is_symlink():
        raise ValueError("symlink release directory")
    checksum_path = root / "checksums.sha256"
    if checksum_path.is_symlink():
        raise ValueError("symlink checksum file")
    rows = _checksum_rows(checksum_path.read_text(encoding="utf-8"))
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlinks are not allowed in installed data")
        if path.is_file() and path != checksum_path:
            actual.add(path.relative_to(root).as_posix())
    if set(rows) != actual:
        raise ValueError("internal checksums must cover exactly all installed payload files")
    for name, expected in rows.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"internal checksum mismatch: {name}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != "scriptstrokes-release"
            or manifest.get("format_version") != 3
            or manifest.get("release_version") != release
            or manifest.get("catalog") != "catalog.json"
            or manifest.get("checksums") != "checksums.sha256"):
        raise ValueError("unsupported or inconsistent release manifest")
    requirement = manifest.get("requires_scriptstrokes")
    if (not requirement
            or Version(package_version("scriptstrokes")) not in SpecifierSet(requirement)):
        raise ValueError("data release is incompatible with installed scriptstrokes")
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    if (catalog.get("format") != "scriptstrokes-catalog"
            or catalog.get("release_version") != release):
        raise ValueError("inconsistent catalog")
    ids = set()
    for item in catalog["models"]:
        if item["id"] in ids:
            raise ValueError("duplicate catalog model")
        ids.add(item["id"])
        path = _safe_path(item["model"]).as_posix()
        if path not in rows:
            raise ValueError("catalog model is not checksummed")
        model = WritingModel.model_validate_json((root / path).read_text(encoding="utf-8"))
        if model.id != item["id"]:
            raise ValueError("catalog model ID mismatch")


def install_release(release: str | None = None) -> Path:
    resolved, url = resolve_release(release)
    target = data_cache_dir() / "releases" / resolved
    if target.exists():
        return verify_release(resolved)
    archive_path = _download(resolved, url)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        temporary = Path(temporary)
        expected_root = f"scriptstrokes-data-{resolved}"
        seen = set()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                path = _safe_path(member.filename)
                kind = stat.S_IFMT(member.external_attr >> 16)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError("unsafe archive member type")
                if path.parts[0] != expected_root or member.filename in seen:
                    raise ValueError("unexpected archive root or duplicate member")
                seen.add(member.filename)
                destination = temporary.joinpath(*path.parts)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
        root = temporary / expected_root
        _verify_directory(root, resolved)
        root.rename(target)
    return target


def list_installed_releases() -> list[str]:
    root = data_cache_dir() / "releases"
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir()
                  if path.is_dir() and (path / "manifest.json").is_file())


def released_model_path(model_id: str, *, release: str,
                        auto_install: bool = False) -> Path:
    # Offline by default, including when selecting the latest installed release.
    if release == "latest" and not auto_install:
        installed = list_installed_releases()
        if not installed:
            raise FileNotFoundError("no installed data releases")
        release = max(installed, key=Version)
    root = install_release(release) if auto_install else verify_release(release)
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    for item in catalog["models"]:
        if item["id"] == model_id:
            return root / _safe_path(item["model"])
    raise KeyError(f"model {model_id!r} not found in release {release}")


def load_released_model(model_id: str, *, release: str,
                        auto_install: bool = False) -> WritingModel:
    path = released_model_path(model_id, release=release, auto_install=auto_install)
    return WritingModel.model_validate_json(path.read_text(encoding="utf-8"))


def remove_release(release: str) -> None:
    root = data_cache_dir() / "releases" / _version(release)
    if root.is_symlink():
        raise ValueError("refusing to remove a symlink release")
    shutil.rmtree(root)
