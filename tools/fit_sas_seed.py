"""Build and jointly fit SAS construction-family seeds in the private workspace."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from scriptstrokes.component_fit import fit_component_candidates
from scriptstrokes.geometry import transform_path
from scriptstrokes.models import Transform, WritingModel
from scriptstrokes.raster import compare_anchored_masks, render_text_frame, threshold_image
from tools.measure_sas_seed import connected_ink

ROOT = Path("work/sas-1968")
UNIT = 12.5


def _part(part):
    result = (
        {"component": part["component"]}
        if "component" in part
        else {"d": transform_path(part["d"], Transform(scale_x=UNIT, scale_y=UNIT))}
    )
    if "transform" in part:
        values = part["transform"].copy()
        for name in ("dx", "dy"):
            if name in values:
                values[name] *= UNIT
        result["transform"] = values
    return result


def build_seed(recipes):
    source_records = json.loads(Path("data/observations/sas-1968/sources.json").read_text())
    provenance = {
        "status": "inferred",
        "source_ids": ["sas-1968-components", "sas-1968-alphabet"],
        "notes": "Recurring construction is documented; exact vector trajectory is reconstructed.",
    }
    components = {
        name: {
            "d": transform_path(value["d"], Transform(scale_x=UNIT, scale_y=UNIT)),
            "description": value["description"],
            "provenance": provenance,
        }
        for name, value in recipes["components"].items()
    }
    glyphs = {}
    for character, recipe in recipes["glyphs"].items():
        strokes = []
        for stroke in recipe["strokes"]:
            if "at" in stroke:
                strokes.append({"id": stroke["id"], "at": [x * UNIT for x in stroke["at"]]})
            else:
                strokes.append(
                    {
                        "id": stroke["id"],
                        "role": stroke.get("role", "body"),
                        "parts": [_part(part) for part in stroke["parts"]],
                    }
                )
        glyphs[character] = {
            "character": character,
            "unicode": [f"U+{ord(character):04X}"],
            "provenance": provenance,
            "forms": {"isolated": {"advance_width": recipe["advance"] * UNIT, "strokes": strokes}},
        }
    return WritingModel.model_validate(
        {
            "schema_version": 2,
            "id": "sas-1968-seed",
            "name": {"en": "SAS 1968 local seed"},
            "script": "Latn",
            "languages": ["de"],
            "regions": ["DE"],
            "status": "experimental",
            "license": "REPLACE-AFTER-RIGHTS-REVIEW",
            "metrics": recipes["metrics"],
            "sources": source_records,
            "components": components,
            "glyphs": glyphs,
        }
    )


def fit_family(model, component_id, family, references):
    candidates = [model.components[component_id].d]
    for path in family.get("candidates", []):
        candidates.append(transform_path(path, Transform(scale_x=UNIT, scale_y=UNIT)))
    result = fit_component_candidates(
        model,
        component_id,
        candidates,
        references,
        training=family["training"],
        holdout=family["holdout"],
    )
    history = [{"training": result.training_scores, "holdout": result.holdout_scores}]
    for step in (UNIT * 2, UNIT, UNIT / 2):
        tokens = result.model.components[component_id].d.split()
        numeric = [
            index for index, value in enumerate(tokens) if re.fullmatch(r"[-+0-9.eE]+", value)
        ]
        neighborhood = [" ".join(tokens)]
        # Endpoints stay fixed; only cubic control points are jointly refined.
        for operator_index, token in enumerate(tokens):
            if token != "C":
                continue
            for index in range(operator_index + 1, operator_index + 5):
                assert index in numeric
                for direction in (-1, 1):
                    changed = tokens.copy()
                    changed[index] = str(float(tokens[index]) + step * direction)
                    neighborhood.append(" ".join(changed))
        result = fit_component_candidates(
            model,
            component_id,
            neighborhood,
            references,
            training=family["training"],
            holdout=family["holdout"],
        )
        history.append({"training": result.training_scores, "holdout": result.holdout_scores})
    return result, history


def write_review(model, recipes, alphabet, references, family_reports):
    rows = []
    sheet = Image.new("RGB", (900, len(references) * 360), "white")
    draw = ImageDraw.Draw(sheet)
    training = {
        character for family in recipes["families"].values() for character in family["training"]
    }
    for index, (character, reference) in enumerate(references.items()):
        with Image.open(reference) as image:
            ref = image.convert("RGB")
        frame = render_text_frame(model, character, scale=1 / UNIT)
        generated = frame.image
        observation = alphabet[character]
        recipe = recipes["glyphs"][character]
        baseline = recipe["source_baseline"] - observation["source_bbox"][1]
        origin = recipe.get("source_origin", observation["source_bbox"][0] + 4)
        anchor_x = origin - observation["source_bbox"][0]
        metrics = compare_anchored_masks(
            threshold_image(ref),
            threshold_image(generated),
            reference_anchor=(anchor_x, baseline),
            generated_anchor=frame.origin,
        )
        rows.append(
            {
                "character": character,
                "split": "fit" if character in training else "hold-out",
                "metrics": metrics,
                "visual_review": "pending",
            }
        )
        draw.text(
            (10, index * 360 + 5), f"{character}: reference | generated | overlay", fill="black"
        )
        sheet.paste(ref.resize((ref.width * 3, ref.height * 3)), (10, index * 360 + 25))
        sheet.paste(
            generated.resize((generated.width * 3, generated.height * 3)), (300, index * 360 + 25)
        )
        reference_mask = threshold_image(ref)
        generated_mask = threshold_image(generated)
        aligned = Image.fromarray(generated_mask.astype(np.uint8) * 255).transform(
            ref.size,
            Image.Transform.AFFINE,
            (1, 0, frame.origin[0] - anchor_x, 0, 1, frame.origin[1] - baseline),
            resample=Image.Resampling.NEAREST,
        )
        overlay = np.full((ref.height, ref.width, 3), 255, dtype=np.uint8)
        overlay[reference_mask] = (220, 40, 40)
        generated_mask = np.asarray(aligned) > 0
        overlay[generated_mask] = (40, 70, 220)
        overlay[reference_mask & generated_mask] = (30, 30, 30)
        sheet.paste(
            Image.fromarray(overlay).resize((ref.width * 3, ref.height * 3)),
            (600, index * 360 + 25),
        )
    sheet.save(ROOT / "reports/seed-review.png")
    for start in range(0, len(references), 3):
        sheet.crop((0, start * 360, 900, min(len(references), start + 3) * 360)).save(
            ROOT / "reports" / f"seed-review-{start // 3 + 1:02d}.png"
        )
    report = {
        "families": family_reports,
        "items": rows,
        "notes": "Local seed proposals, not accepted glyph geometry.",
    }
    (ROOT / "reports/seed-fit.json").write_text(json.dumps(report, indent=2) + "\n")


def isolate_references(references, recipes, alphabet):
    output = ROOT / "observations/seed-clean"
    output.mkdir(parents=True, exist_ok=True)
    isolated = {}
    records = []
    for character, path in references.items():
        with Image.open(path) as image:
            mask = threshold_image(image)
        components = connected_ink(mask)
        keep = recipes["glyphs"][character].get("keep_ink_components", 1)
        clean = np.zeros_like(mask)
        for component in components[:keep]:
            for y, x in component:
                clean[y, x] = True
        destination = output / f"U+{ord(character):04X}.png"
        Image.fromarray(np.where(clean, 0, 255).astype(np.uint8)).save(destination)
        isolated[character] = destination
        records.append(
            {
                "character": character,
                "source_bbox": alphabet[character]["source_bbox"],
                "retained_components": keep,
                "removed_components": len(components) - keep,
                "removed_ink_pixels": int(mask.sum() - clean.sum()),
                "notes": "Private derived mask; visual review required.",
            }
        )
    (ROOT / "reports/seed-isolation.json").write_text(json.dumps(records, indent=2) + "\n")
    return isolated


def main():
    recipes = yaml.safe_load((ROOT / "vectors/seed-recipes.yaml").read_text())
    observations = json.loads(Path("data/observations/sas-1968/manifest.json").read_text())
    alphabet = {item["label"]: item for item in observations["items"]}
    model = build_seed(recipes)
    references = {
        character: ROOT / "observations" / alphabet[character]["crop_locator"].split(":", 1)[1]
        for character in model.glyphs
    }
    references = isolate_references(references, recipes, alphabet)
    reports = {}
    for component_id, family in recipes["families"].items():
        members = family["training"] + family["holdout"]
        result, history = fit_family(
            model, component_id, family, {char: references[char] for char in members}
        )
        model = result.model
        reports[component_id] = {
            "training": result.training_scores,
            "holdout": result.holdout_scores,
            "history": history,
        }
        print(component_id, json.dumps(reports[component_id]), flush=True)
    (ROOT / "vectors/seed.yaml").write_text(
        yaml.safe_dump(
            model.model_dump(mode="json", by_alias=True),
            allow_unicode=True,
            sort_keys=False,
        )
    )
    write_review(model, recipes, alphabet, references, reports)


if __name__ == "__main__":
    main()
