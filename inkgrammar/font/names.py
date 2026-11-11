from __future__ import annotations

import re

from fontTools.agl import UV2AGL

from ..models import WritingModelV3


def glyph_base_name(character: str) -> str:
    if character == " ":
        return "space"
    if len(character) != 1:
        raise ValueError(f"font glyphs must represent one Unicode character: {character!r}")
    codepoint = ord(character)
    return UV2AGL.get(codepoint) or (
        f"uni{codepoint:04X}" if codepoint <= 0xFFFF else f"u{codepoint:05X}"
    )


def _variant_suffix(character: str, variant_id: str) -> str:
    suffix = variant_id
    if suffix.startswith(character):
        suffix = suffix[len(character) :].lstrip("._-")
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", suffix) if part]
    if not parts:
        return "alt"
    first, *rest = parts
    return first.lower() + "".join(part[:1].upper() + part[1:] for part in rest)


def variant_glyph_names(model: WritingModelV3) -> dict[tuple[str, str], str]:
    names: dict[tuple[str, str], str] = {}
    used: set[str] = {".notdef"}
    for character, glyph in sorted(model.glyphs.items(), key=lambda item: ord(item[0][0])):
        base = glyph_base_name(character)
        for variant_id in sorted(glyph.variants):
            name = (
                base
                if variant_id == glyph.default_variant
                else f"{base}.{_variant_suffix(character, variant_id)}"
            )
            if name in used:
                raise ValueError(f"generated glyph name is not unique: {name}")
            used.add(name)
            names[(character, variant_id)] = name
    return names


def glyph_unicode_values(character: str, values: list[str]) -> tuple[int, ...]:
    if not values:
        return (ord(character),)
    codepoints: list[int] = []
    for value in values:
        match = re.fullmatch(r"U\+([0-9A-Fa-f]{4,6})", value)
        if match is None:
            raise ValueError(f"invalid Unicode value: {value}")
        codepoint = int(match.group(1), 16)
        if codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError(f"invalid Unicode scalar value: {value}")
        codepoints.append(codepoint)
    return tuple(codepoints)
