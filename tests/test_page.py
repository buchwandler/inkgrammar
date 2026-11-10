from PIL import Image, ImageDraw

from scriptstrokes.models import LayoutRow, PageLayout
from scriptstrokes.page import extract_labeled_page


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
