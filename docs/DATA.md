# Versioned runtime data

Runtime data is installed explicitly. `load_model(path)` never downloads data.
The historical scans and crops are not needed for loading a released model.

## Commands

```sh
inkgrammar data install 0.3.0
inkgrammar data list
inkgrammar data path 0.3.0
inkgrammar data verify 0.3.0
inkgrammar data model sas-1968 --release 0.3.0
inkgrammar data remove 0.3.0
```

These are intended release usage examples, not a claim that version 0.3.0 or
the SAS model has been published. SAS fitting remains in progress.

```python
from inkgrammar.data import install_release, load_released_model

install_release("0.3.0")
model = load_released_model("sas-1968", release="0.3.0")
```

`load_released_model(..., release="latest")` selects the highest installed
version offline. Set `auto_install=True` explicitly to resolve/install a remote
release. Missing installed data raises an error rather than triggering a
network request. Installed payload integrity and compatibility are checked
when loading a released model.

## Cache and mirrors

`INKGRAMMAR_DATA_DIR` overrides the cache directory. Defaults are the XDG
cache directory on Linux, `~/Library/Caches/inkgrammar` on macOS and
`%LOCALAPPDATA%/inkgrammar` on Windows. Layout:

```text
<cache>/downloads/inkgrammar-data-<version>.zip
<cache>/releases/<version>/manifest.json
<cache>/releases/<version>/catalog.json
<cache>/releases/<version>/models/<model-id>.json
```

`INKGRAMMAR_DATA_VERSION` supplies the default version for the Python
installation/download API. `INKGRAMMAR_DATA_BASE_URL` overrides the download
base. Explicit versions use:

```text
<base>/v<version>/inkgrammar-data-<version>.zip
<base>/v<version>/inkgrammar-data-<version>.zip.sha256
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
`requires_inkgrammar` compatibility range defaults to the builder's public
package version through the next minor boundary. Python callers can set an
explicit compatibility range and `git_commit` through `build_release`.

Rights notices are not a rights clearance. Do not publish a reconstructed SAS
model while its model-data license or derivation rights remain unresolved.

## SAS-1968 dataset pipeline

The private SAS workspace is organized as a four-layer evidence chain (source
evidence, extracted observations, reviewed interpretation, fitted model). The
dataset pipeline materializes the first two layers deterministically from a
source-free recipe plus authorized private scans:

```sh
inkgrammar dataset verify-sources data/datasets/sas-1968.yaml
inkgrammar dataset materialize data/datasets/sas-1968.yaml
inkgrammar dataset review-sheet data/datasets/sas-1968.yaml --output work/sas-1968/dataset/review
inkgrammar dataset validate data/datasets/sas-1968.yaml
inkgrammar dataset status data/datasets/sas-1968.yaml
inkgrammar dataset validate-sas-private data/datasets/sas-1968.yaml   # one-shot local run
```

The recipe `data/datasets/sas-1968.yaml` contains no scan bytes; private
scans live under `work/sas-1968/sources/` and are verified by SHA-256 and
dimensions. Materialized output goes to `work/sas-1968/dataset/` (crops,
raw/clean masks with cleanup records, normalized previews, derived
skeleton/centerline/topology artifacts with algorithm versions, and a
canonical `index.json`). Two clean runs produce byte-identical semantic
artifacts.

### Annotation and review-patch policy

Reviewed semantic annotations and review patches live in the private tree
`work/sas-1968/annotations/sas-1968/` until source rights are reviewed. Review
patches carry the input hashes they were reviewed against; stale patches are
reported and never silently applied to new crops. Automatic outputs are
candidates (`review_status: candidate` / `automatic-candidate`) and no dataset
or primitive is promoted without human review.

### Source-free CI

Public CI runs without the historical scans: schema validation, layout
parsing, ID/relation checks, synthetic materialization, trajectory synthetic
tests, and compiler tests. Source hashes, real materialization, SAS review
outputs, and real fits run only as authorized local validation
(`validate-sas-private`). Missing private source images must not make ordinary
package CI unusable.

### Trajectory fit outputs

Trajectory experiments write to `work/sas-1968/trajectory/<label>/` (see
`docs/ARCHITECTURE.md` for the trajectory model). Every fit report records
provenance hashes (dataset project, observation manifest, annotation,
materialized artifact, fit config, fit project, optimizer version); a fit is
invalidated when reviewed evidence changes.
