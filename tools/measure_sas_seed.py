"""Report connected-ink measurements without altering the supplied scans."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scriptstrokes.raster import threshold_image


def connected_ink(mask):
    remaining = {tuple(point) for point in np.argwhere(mask)}
    result = []
    while remaining:
        point = min(remaining)
        remaining.remove(point)
        stack = [point]
        pixels = [point]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
                        pixels.append(neighbor)
        result.append(pixels)
    return sorted(result, key=lambda pixels: -len(pixels))


def ink_components(mask):
    result = []
    for pixels in connected_ink(mask):
        ys, xs = zip(*pixels, strict=True)
        result.append({"area": len(pixels), "bbox": [int(min(xs)), int(min(ys)),
                                                   int(max(xs)) + 1, int(max(ys)) + 1]})
    return result


def main():
    manifest = json.loads(Path("data/observations/sas-1968/manifest.json").read_text())
    rows = []
    for item in manifest["items"]:
        if item["label"] not in "iuw":
            continue
        path = Path("work/sas-1968/observations") / item["crop_locator"].split(":", 1)[1]
        with Image.open(path) as image:
            components = ink_components(threshold_image(image))
        rows.append({"character": item["label"], "source_bbox": item["source_bbox"],
                     "ink_components": components})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
