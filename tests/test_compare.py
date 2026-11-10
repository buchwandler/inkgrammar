from pathlib import Path

from inkgrammar.fit import fit_join
from inkgrammar.io import load_model
from inkgrammar.raster import compare_text_to_image, render_text_mask

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data" / "models" / "demo-looped.yaml"


def test_compare_and_fit_join(tmp_path):
    model = load_model(MODEL)
    reference = tmp_path / "mi.png"
    render_text_mask(model, "mi").save(reference)

    metrics = compare_text_to_image(model, "mi", reference)
    assert metrics["iou"] > 0.95

    fit = fit_join(
        model,
        "mi",
        reference,
        gap_min=-20,
        gap_max=20,
        gap_step=20,
        tensions=(55,),
    )
    assert fit.score > 0.90
