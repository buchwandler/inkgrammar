from __future__ import annotations

from ..models import RuntimeGlyphVariant, WritingModelV3
from .models import FontPlan


def _variant_name(plan: FontPlan, character: str, variant_id: str) -> str:
    try:
        return plan.variant_names[(character, variant_id)]
    except KeyError as error:
        raise ValueError(f"font feature references unknown variant {variant_id!r}") from error


def _condition_glyphs(
    model: WritingModelV3,
    plan: FontPlan,
    characters: list[str],
    classes: list[str],
    anchor_name: str,
) -> set[str] | None:
    candidates: set[str] | None = None
    if characters:
        candidates = {
            _variant_name(plan, character, model.glyphs[character].default_variant)
            for character in characters
            if character in model.glyphs
        }
    if classes:
        class_candidates = {
            plan.variant_names[(character, variant.id)]
            for character, glyph in model.glyphs.items()
            for variant in glyph.variants.values()
            if (anchor := getattr(variant, anchor_name)) is not None and anchor.class_ in classes
        }
        candidates = class_candidates if candidates is None else candidates & class_candidates
    return candidates


def _context_substitution(
    model: WritingModelV3,
    plan: FontPlan,
    rule,
    class_definitions: list[str],
) -> tuple[str | None, str | None]:
    if any(position in {"isolated", "initial", "medial", "final"} for position in rule.positions):
        return None, (
            f"context rule {rule.id!r} uses a glyph position that GSUB context cannot represent"
        )
    context_characters = (*rule.prev_characters, *rule.next_characters)
    unknown_characters = sorted(set(context_characters) - model.glyphs.keys())
    if unknown_characters:
        return None, f"context rule {rule.id!r} references unknown context characters"
    previous = _condition_glyphs(model, plan, rule.prev_characters, rule.prev_exit_classes, "exit")
    following = _condition_glyphs(
        model, plan, rule.next_characters, rule.next_entry_classes, "entry"
    )
    if (rule.prev_characters or rule.prev_exit_classes) and not previous:
        return None, f"context rule {rule.id!r} has no previous glyph candidates"
    if (rule.next_characters or rule.next_entry_classes) and not following:
        return None, f"context rule {rule.id!r} has no next glyph candidates"

    current = _variant_name(plan, rule.character, model.glyphs[rule.character].default_variant)
    output = _variant_name(plan, rule.character, rule.variant)
    if current == output:
        return None, None

    def token(candidates: set[str] | None, label: str) -> str | None:
        if not candidates:
            return None
        ordered = sorted(candidates)
        if len(ordered) == 1:
            return ordered[0]
        name = f"@IG{label}{len(class_definitions)}"
        class_definitions.append(f"{name} = [{' '.join(ordered)}];")
        return name

    previous_token = token(previous, "Prev")
    following_token = token(following, "Next")
    if previous_token is None and following_token is None:
        return f"sub {current} by {output};", None
    inputs = [part for part in (previous_token, f"{current}'", following_token) if part]
    return f"sub {' '.join(inputs)} by {output};", None


def _feature_block(tag: str, statements: list[str]) -> str:
    if not statements:
        return ""
    return f"feature {tag} {{\n" + "\n".join(f"    {line}" for line in statements) + f"\n}} {tag};"


def _anchor(anchor) -> str:
    if anchor is None:
        return "NULL"
    return f"<anchor {round(anchor.x)} {round(anchor.y)}>"


def generate_feature_source(model: WritingModelV3, plan: FontPlan) -> tuple[str, tuple[str, ...]]:
    required_rules: list[str] = []
    optional_rules: list[str] = []
    ligature_rules: list[str] = []
    issues: list[str] = []
    required_classes: list[str] = []
    optional_classes: list[str] = []

    for rule in sorted(model.context_rules, key=lambda item: (-item.priority, item.id)):
        class_definitions = required_classes if rule.required else optional_classes
        statement, issue = _context_substitution(model, plan, rule, class_definitions)
        if issue:
            issues.append(issue)
        elif statement:
            (required_rules if rule.required else optional_rules).append(statement)

    variants_by_id: dict[str, list[tuple[str, RuntimeGlyphVariant]]] = {}
    for character, glyph in model.glyphs.items():
        for variant in glyph.variants.values():
            variants_by_id.setdefault(variant.id, []).append((character, variant))

    for rule in sorted(model.connection_rules, key=lambda item: item.id):
        if rule.mode == "synthetic-bridge":
            issues.append(f"connection rule {rule.id!r} requires a synthetic bridge")
        if rule.mode != "ligature":
            continue
        left_matches = variants_by_id.get(rule.required_variant_left or "", [])
        right_matches = variants_by_id.get(rule.required_variant_right or "", [])
        ligature_matches = variants_by_id.get(rule.ligature_variant or "", [])
        if not (len(left_matches) == len(right_matches) == len(ligature_matches) == 1):
            issues.append(
                f"ligature rule {rule.id!r} does not identify unique input and output variants"
            )
            continue
        left_character, left = left_matches[0]
        right_character, right = right_matches[0]
        ligature_character, ligature = ligature_matches[0]
        left_name = _variant_name(plan, left_character, left.id)
        right_name = _variant_name(plan, right_character, right.id)
        ligature_name = _variant_name(plan, ligature_character, ligature.id)
        ligature_rules.append(f"sub {left_name} {right_name} by {ligature_name};")

    anchored_variants = [
        (character, variant)
        for character, glyph in sorted(model.glyphs.items())
        for variant in sorted(glyph.variants.values(), key=lambda item: item.id)
        if variant.entry is not None or variant.exit is not None
    ]

    cursive_lookup_rules = [
        f"pos cursive {_variant_name(plan, character, variant.id)} "
        f"{_anchor(variant.entry)} {_anchor(variant.exit)};"
        for character, variant in anchored_variants
    ]
    exit_variants = [
        (character, variant)
        for character, glyph in sorted(model.glyphs.items())
        for variant in sorted(glyph.variants.values(), key=lambda item: item.id)
        if variant.exit is not None
    ]
    entry_variants = [
        (character, variant)
        for character, glyph in sorted(model.glyphs.items())
        for variant in sorted(glyph.variants.values(), key=lambda item: item.id)
        if variant.entry is not None
    ]
    possible_pairs = {
        (
            _variant_name(plan, left_character, left.id),
            _variant_name(plan, right_character, right.id),
        )
        for left_character, left in exit_variants
        for right_character, right in entry_variants
    }
    allowed_pairs = set()
    for rule in sorted(model.connection_rules, key=lambda item: item.id):
        if rule.mode == "synthetic-bridge":
            issues.append(f"connection rule {rule.id!r} requires a synthetic bridge")
        if rule.mode not in {"cursive-anchor", "direct"}:
            continue
        left_matches = [
            (character, variant)
            for character, variant in exit_variants
            if variant.exit.class_ == rule.from_class
            and (rule.required_variant_left is None or variant.id == rule.required_variant_left)
        ]
        right_matches = [
            (character, variant)
            for character, variant in entry_variants
            if variant.entry.class_ == rule.to_class
            and (rule.required_variant_right is None or variant.id == rule.required_variant_right)
        ]
        if not left_matches or not right_matches:
            issues.append(f"connection rule {rule.id!r} has no matching cursive anchors")
            continue
        allowed_pairs.update(
            (
                _variant_name(plan, left_character, left.id),
                _variant_name(plan, right_character, right.id),
            )
            for left_character, left in left_matches
            for right_character, right in right_matches
        )

    source = ["languagesystem DFLT dflt;", "languagesystem latn dflt;"]
    if required_rules:
        source.extend(("", *required_classes, _feature_block("rclt", required_rules)))
    if optional_rules:
        source.extend(("", *optional_classes, _feature_block("calt", optional_rules)))
    if ligature_rules:
        source.extend(("", _feature_block("rlig", ligature_rules)))

    if allowed_pairs:
        if possible_pairs <= allowed_pairs:
            lookup = (
                "lookup IG_Cursive {\n"
                + "\n".join(f"    {line}" for line in cursive_lookup_rules)
                + "\n} IG_Cursive;"
            )
            source.extend(("", lookup, _feature_block("curs", ["lookup IG_Cursive;"])))
        else:
            issues.append(
                "cursive positioning cannot represent pair-specific connection rules "
                "for all anchored glyph variants"
            )
    return "\n".join(source).rstrip() + "\n", tuple(sorted(set(issues)))
