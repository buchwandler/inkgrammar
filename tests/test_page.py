import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from inkgrammar.dataset.layouts import LayoutRegion, RegionLayout, load_region_layout
from inkgrammar.models import LayoutRow, PageLayout
from inkgrammar.page import (
    extract_annotated_region_page,
    extract_labeled_page,
    extract_region_page,
)


def test_layout_aware_page_extraction(tmp_path):
    image = Image.new("RGB", (600, 160), "white")
    draw = ImageDraw.Draw(image)
    positions = [40, 180, 320, 460]
    for x, label in zip(positions, ["A", "B", "a", "b"], strict=False):
        draw.text((x, 50), label, fill="black", stroke_width=2)

    source = tmp_path / "page.png"
    image.save(source)

    layout = PageLayout(
        schema_version=1,
        id="test",
        rows=[
            LayoutRow(
                id="row",
                y=(0.2, 0.8),
                x=(0.0, 1.0),
                labels=["A", "B", "a", "b"],
            )
        ],
    )
    output = tmp_path / "out"
    manifest = extract_labeled_page(source, layout, output)

    assert [item["label"] for item in manifest["detections"]] == [
        "A",
        "B",
        "a",
        "b",
    ]
    assert (output / "review.png").is_file()
    assert (output / "detections.json").is_file()




def test_region_layout_extraction(tmp_path):
    image = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 20, 50, 80), fill="black")
    draw.rectangle((80, 20, 120, 80), fill="black")
    source = tmp_path / "page.png"
    image.save(source)

    layout = RegionLayout(
        schema_version=1,
        id="test-regions",
        regions=[
            LayoutRegion(
                id="pair-01",
                kind="connection-pair",
                bbox=(0.05, 0.2, 0.3, 0.8),
                label="ab",
                sequence=["a", "b"],
                category_hint="baseline-origin",
                subregions={"left": (0.05, 0.2, 0.15, 0.8), "right": (0.2, 0.2, 0.3, 0.8)},
                notes="declared pair",
            )
        ],
    )
    output = tmp_path / "out"
    manifest = extract_region_page(source, layout, output)

    record = manifest["detections"][0]
    assert manifest["extractor"] == "region-layout-v1"
    assert record["region"] == "pair-01"
    assert record["sequence"] == ["a", "b"]
    assert record["category_hint"] == "baseline-origin"
    assert set(record["subregions"]) == {"left", "right"}
    assert record["bbox"] == [10, 20, 60, 80]
    assert (output / "crops" / "pair-01.png").is_file()
    assert (output / "crops" / "pair-01__left.png").is_file()
    assert (output / "review.png").is_file()
    assert (output / "detections.json").is_file()


def test_annotated_region_extraction_separates_ink_and_annotations(tmp_path):
    image = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 20, 100, 80), fill="black")
    draw.rectangle((120, 10, 150, 40), fill="black")
    source = tmp_path / "page.png"
    image.save(source)

    layout = RegionLayout(
        schema_version=1,
        id="test-annotated",
        regions=[
            LayoutRegion(
                id="stroke-order-01",
                kind="stroke-order-demonstration",
                bbox=(0.1, 0.1, 0.8, 0.9),
                label="a",
                sequence=["1"],
                subregions={
                    "ink": (0.15, 0.2, 0.5, 0.8),
                    "annotation-1": (0.6, 0.1, 0.75, 0.4),
                },
            )
        ],
    )
    output = tmp_path / "out"
    manifest = extract_annotated_region_page(source, layout, output)

    record = manifest["detections"][0]
    assert manifest["extractor"] == "annotated-region-v1"
    assert record["ink_region"]["crop"] == "crops/stroke-order-01__ink.png"
    assert record["annotation_regions"] == [
        {
            "name": "annotation-1",
            "bbox": [120, 10, 150, 40],
            "crop": "crops/stroke-order-01__annotation-1.png",
        }
    ]
    assert "ink" not in record["subregions"]
    assert "annotation-1" not in record["subregions"]
    assert (output / "crops" / "stroke-order-01__ink.png").is_file()
    assert (output / "crops" / "stroke-order-01__annotation-1.png").is_file()


def test_annotated_region_requires_ink_subregion(tmp_path):
    image = Image.new("RGB", (200, 100), "white")
    source = tmp_path / "page.png"
    image.save(source)

    layout = RegionLayout(
        schema_version=1,
        id="test-invalid",
        regions=[
            LayoutRegion(
                id="stroke-order-02",
                kind="stroke-order-demonstration",
                subregions={"annotation-1": (0.1, 0.1, 0.2, 0.2)},
            )
        ],
    )
    with pytest.raises(ValueError, match="without an 'ink' subregion"):
        extract_annotated_region_page(source, layout, tmp_path / "out")


def test_region_layout_rejects_duplicate_ids_and_bad_boxes():
    region = LayoutRegion(id="r", kind="k")
    with pytest.raises(ValidationError, match="duplicate layout region id"):
        RegionLayout(schema_version=1, id="bad", regions=[region, region])
    with pytest.raises(ValidationError):
        LayoutRegion(id="r", kind="k", bbox=(0.5, 0.5, 0.4, 0.6))
    with pytest.raises(ValidationError):
        LayoutRegion(id="r", kind="k", subregions={"s": (0.0, 0.0, 1.5, 0.5)})


def test_load_region_layout_parses_yaml(tmp_path):
    path = tmp_path / "layout.yaml"
    path.write_text(
        "schema_version: 1\n"
        "id: test\n"
        "regions:\n"
        "  - id: family-01\n"
        "    kind: component-family\n"
        "    bbox: [0.1, 0.1, 0.9, 0.3]\n"
        "    members: [c, e]\n",
        encoding="utf-8",
    )

    layout = load_region_layout(path)

    assert layout.id == "test"
    assert layout.regions[0].members == ["c", "e"]
