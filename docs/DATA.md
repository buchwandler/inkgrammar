# Versioned runtime data

Runtime data is installed explicitly. `load_model(path)` never downloads data.
The historical scans and crops are not needed for loading a released model.

## Commands

```sh
scriptstrokes data install 0.3.0
scriptstrokes data list
scriptstrokes data path 0.3.0
scriptstrokes data verify 0.3.0
scriptstrokes data model sas-1968 --release 0.3.0
scriptstrokes data remove 0.3.0
```

These are intended release usage examples, not a claim that version 0.3.0 or
the SAS model has been published. SAS fitting remains in progress.

```python
from scriptstrokes.data import install_release, load_released_model

install_release("0.3.0")
model = load_released_model("sas-1968", release="0.3.0")
```

`load_released_model(..., release="latest")` selects the highest installed
version offline. Set `auto_install=True` explicitly to resolve/install a remote
release. Missing installed data raises an error rather than triggering a
network request. Installed payload integrity and compatibility are checked
when loading a released model.

## Cache and mirrors

`SCRIPTSTROKES_DATA_DIR` overrides the cache directory. Defaults are the XDG
cache directory on Linux, `~/Library/Caches/scriptstrokes` on macOS and
`%LOCALAPPDATA%/scriptstrokes` on Windows. Layout:

```text
<cache>/downloads/scriptstrokes-data-<version>.zip
<cache>/releases/<version>/manifest.json
<cache>/releases/<version>/catalog.json
<cache>/releases/<version>/models/<model-id>.json
```

`SCRIPTSTROKES_DATA_VERSION` supplies the default version for the Python
installation/download API. `SCRIPTSTROKES_DATA_BASE_URL` overrides the download
base. Explicit versions use:

```text
<base>/v<version>/scriptstrokes-data-<version>.zip
<base>/v<version>/scriptstrokes-data-<version>.zip.sha256
```

For `latest`, the default client reads GitHub's latest release metadata. A
mirror must provide `<base>/latest.json` with GitHub-compatible `tag_name` and
`assets` fields. The matching archive's `browser_download_url` is used rather
than guessing its location. The checksum sidecar is fetched at that URL plus
`.sha256`. Production mirrors should use HTTPS. HTTP is supported for local
tests. Checksums detect corruption; they are not publisher signatures and
cannot protect against a compromised mirror serving both archive and checksum.

## Verification and atomic installation

The archive SHA-256 is checked before extraction. Absolute paths, traversal,
backslash/drive paths, duplicate members, symlinks and unexpected archive roots
are rejected. Extraction occurs in a temporary directory beside the final
installation. Internal checksums must cover exactly every payload file. The
manifest format, version, package compatibility and catalog/model identity are
checked before the directory is renamed into place. Failed installation leaves
no partially installed release. Existing installations are verified, not
silently overwritten. Removal leaves the cached download intact.

## Building bundles

`build-release` emits deterministic format-v3 bundles and an external
`.zip.sha256` sidecar. Bundles contain model/catalog data, structural QC,
schemas, model/source rights notices and any available observation/regression
manifests. It never copies referenced scans or crops. The
`requires_scriptstrokes` compatibility range defaults to the builder's public
package version through the next minor boundary. Python callers can set an
explicit compatibility range and `git_commit` through `build_release`.

Rights notices are not a rights clearance. Do not publish a reconstructed SAS
model while its model-data license or derivation rights remain unresolved.
