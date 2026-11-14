from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PlotSeries = tuple[str, np.ndarray, np.ndarray, tuple[int, int, int]]


def write_plot(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[PlotSeries],
) -> None:
    width, height = 900, 440
    left, top, right, bottom = 100, 70, 32, 76
    plot_right, plot_bottom = width - right, height - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 15), title, fill=(20, 20, 20), font=font)

    all_x = np.concatenate([np.asarray(item[1], dtype=float) for item in series])
    all_y = np.concatenate([np.asarray(item[2], dtype=float) for item in series])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_min == y_max:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    y_padding = (y_max - y_min) * 0.05
    y_min, y_max = y_min - y_padding, y_max + y_padding

    for fraction in np.linspace(0.0, 1.0, 5):
        x = round(left + fraction * (plot_right - left))
        y = round(plot_bottom - fraction * (plot_bottom - top))
        draw.line((x, top, x, plot_bottom), fill=(232, 232, 232), width=1)
        draw.line((left, y, plot_right, y), fill=(232, 232, 232), width=1)
        draw.text(
            (x - 8, plot_bottom + 8),
            f"{x_min + fraction * (x_max - x_min):.2g}",
            fill=(55, 55, 55),
            font=font,
        )
        draw.text(
            (left - 52, y - 4),
            f"{y_min + fraction * (y_max - y_min):.2g}",
            fill=(55, 55, 55),
            font=font,
        )
    draw.line((left, top, left, plot_bottom), fill=(40, 40, 40), width=2)
    draw.line((left, plot_bottom, plot_right, plot_bottom), fill=(40, 40, 40), width=2)
    draw.text((left, height - 24), x_label, fill=(30, 30, 30), font=font)
    draw.text((left, 42), y_label, fill=(30, 30, 30), font=font)

    for index, (label, x_values, y_values, color) in enumerate(series):
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        x_pixels = left + (x_values - x_min) / (x_max - x_min) * (plot_right - left)
        y_pixels = plot_bottom - (y_values - y_min) / (y_max - y_min) * (plot_bottom - top)
        points = list(
            zip(
                np.rint(x_pixels).astype(int),
                np.rint(y_pixels).astype(int),
                strict=True,
            )
        )
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        legend_y = top + 10 + index * 16
        draw.line(
            (plot_right - 160, legend_y + 5, plot_right - 145, legend_y + 5),
            fill=color,
            width=3,
        )
        draw.text((plot_right - 140, legend_y), label, fill=(30, 30, 30), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
