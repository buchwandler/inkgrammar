"""Verify the prepared SAS bundle and generate release-safe source metadata.

Run from the repository root after page extraction and visual label review.
No scans or crops are copied into data/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from scriptstrokes.models import SourceRecord
from scriptstrokes.observations import Observation, ObservationManifest, SourceImage

SOURCE_IDS = [
    "sas-1968-alphabet", "sas-1968-components", "sas-1968-stroke-order",
    "sas-1968-connections-numbered", "sas-1968-connection-categories",
]


def main() -> None:
    bundle = Path("sas-1968-reference-images")
    metadata = json.loads((bundle / "manifest.json").read_text())
    sources = []
    records = []
    for source_id, item in zip(SOURCE_IDS, metadata["items"], strict=True):
        image_path = bundle / "images" / item["filename"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"hash mismatch: {image_path}")
        with Image.open(image_path) as image:
            if image.size != (item["width_px"], item["height_px"]):
                raise ValueError(f"dimension mismatch: {image_path}")
        sources.append(SourceImage(source_id=source_id, **{
            key: item[key] for key in (
                "filename", "sha256", "width_px", "height_px", "rights_status"
            )
        }))
        records.append(SourceRecord(
            id=source_id, title=f"Schulausgangsschrift 1968: {item['role']}",
            kind="scan", year=1968, locator=f"prepared-image:{item['filename']}",
            rights=item["public_release_policy"],
        ).model_dump(mode="json", exclude_none=True))
    detections = json.loads(Path(
        "work/sas-1968/observations/alphabet/detections.json"
    ).read_text())["detections"]
    expected = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzß0123456789")
    if len(detections) != 63 or {item["label"] for item in detections} != expected:
        raise ValueError("expected exactly the 63 alphabet plate symbols")
    observations = [Observation(
        id=f"sas1968-alphabet-{item['unicode'][0]}", kind="glyph-isolated",
        label=item["label"], unicode=item["unicode"], source_id=SOURCE_IDS[0],
        source_bbox=item["bbox"], crop_locator=f"local-only:alphabet/{item['crop']}",
        label_source="declared-layout-sequence", review_status="accepted",
        fitting_status="needs-isolation" if item["label"] in "OPQopq" else "pending",
        notes=("Label visually checked against the page review. "
               "Crop acceptance for fitting is separate."),
    ) for item in detections]
    manifest = ObservationManifest(model_id="sas-1968", sources=sources, items=observations)
    output = Path("data/observations/sas-1968")
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    (output / "sources.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(f"Verified {len(sources)} sources; wrote {len(observations)} observations to {output}")


if __name__ == "__main__":
    main()
