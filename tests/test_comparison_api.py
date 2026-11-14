from __future__ import annotations

from PIL import Image, ImageChops, ImageDraw
from typer.testing import CliRunner

from inkgrammar import compare_letter_images as root_compare_letter_images
from inkgrammar.cli import app
from inkgrammar.comparison import compare_letter_images


def _write_letter_images(directory):
    reference_path = directory / "reference.png"
    sample_path = directory / "sample.png"
    for path, ink in ((reference_path, "black"), (sample_path, (20, 70, 220))):
        image = Image.new("RGB", (80, 72), "white")
        draw = ImageDraw.Draw(image)
        if path == sample_path:
            draw.line((2, 40, 77, 40), fill=(180, 180, 180), width=1)
        draw.line(
            [(16, 36), (30, 36), (43, 36), (41, 31), (37, 27), (31, 25), (25, 26),
             (20, 30), (18, 36), (20, 43), (25, 48), (32, 50), (39, 49), (45, 45)],
            fill=ink,
            width=5,
        )
        image.save(path)
    return reference_path, sample_path


def _compare_options(output_dir):
    return {
        "letter": "e",
        "reference_crop": (5, 5, 70, 62),
        "sample_crop": (5, 5, 70, 62),
        "output_dir": output_dir,
    }


def test_compare_letter_images_api_writes_complete_report_and_keeps_crop_pixels(tmp_path):
    reference_path, sample_path = _write_letter_images(tmp_path)
    output_dir = tmp_path / "api-report"

    report = compare_letter_images(
        reference_path,
        sample_path,
        **_compare_options(output_dir),
    )

    assert report.html_path.is_file()
    assert report.json_path.is_file()
    assert report.markdown_path.is_file()
    assert report.payload["letter"] == "e"
    assert report.payload["inputs"]["reference"]["mask_method"] == "grayscale"
    assert report.payload["inputs"]["sample"]["mask_method"] == "color"
    assert report.payload["inputs"]["sample"]["topology"]["component_count"] == 1
    assert report.payload["geometry"]["sample_count"] == 256
    assert report.payload["metrics"]["path"]["mean_error"] < 1e-12

    with Image.open(reference_path) as source, Image.open(
        output_dir / "input-reference.png"
    ) as retained:
        expected = source.crop((5, 5, 75, 67))
        assert ImageChops.difference(expected, retained).getbbox() is None


def test_compare_letter_images_is_exported_from_package_root(tmp_path):
    reference_path, sample_path = _write_letter_images(tmp_path)

    report = root_compare_letter_images(
        reference_path,
        sample_path,
        **_compare_options(tmp_path / "root-report"),
    )

    assert report.html_path.is_file()


def test_compare_letter_cli_writes_report_for_manual_crops(tmp_path):
    reference_path, sample_path = _write_letter_images(tmp_path)
    output_dir = tmp_path / "cli-report"

    result = CliRunner().invoke(
        app,
        [
            "compare-letter",
            str(reference_path),
            str(sample_path),
            "--letter",
            "e",
            "--reference-crop",
            "5,5,70,62",
            "--sample-crop",
            "5,5,70,62",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "report.html").is_file()
    assert "HTML report:" in result.output
    assert "JSON report:" in result.output


def test_compare_letter_cli_rejects_malformed_crop_coordinates(tmp_path):
    reference_path, sample_path = _write_letter_images(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "compare-letter",
            str(reference_path),
            str(sample_path),
            "--letter",
            "e",
            "--reference-crop",
            "5,5,70",
            "--sample-crop",
            "5,5,70,62",
            "--output",
            str(tmp_path / "invalid-report"),
        ],
    )

    assert result.exit_code == 2
    assert "--reference-crop must be x,y,width,height" in result.output
