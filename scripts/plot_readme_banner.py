#!/usr/bin/env python3
"""Build the GitHub README banner with matplotlib; no external fonts required."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main() -> None:
    assets = Path(__file__).resolve().parents[1] / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "path",
        "svg.hashsalt": "adaptevo-readme-banner",
    })
    fig = plt.figure(figsize=(12, 1.65), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1800), ylim=(247.5, 0))
    ax.axis("off")
    ink, blue = "#172B46", "#3866FF"
    ax.add_patch(FancyBboxPatch(
        (8, 8), 1784, 231.5, boxstyle="round,pad=0,rounding_size=24",
        facecolor="#F0F4FB", edgecolor="none",
    ))
    first = ax.text(0, 124, "Adapt", fontsize=65, fontweight="bold",
                    color=ink, va="center")
    second = ax.text(0, 124, "Evo", fontsize=65, fontweight="bold",
                     color=blue, va="center")
    fig.canvas.draw()
    units_per_pixel = 1800 / ax.get_window_extent().width
    first_width = first.get_window_extent().width * units_per_pixel
    second_width = second.get_window_extent().width * units_per_pixel
    start = (1800 - first_width - second_width) / 2
    first.set_x(start)
    second.set_x(start + first_width)
    for ext in ("png", "svg"):
        options = {"metadata": {"Date": None}} if ext == "svg" else {"dpi": 180}
        fig.savefig(assets / f"banner.{ext}", facecolor="white", **options)
        if ext == "svg":
            svg = assets / "banner.svg"
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print("Rendered docs/assets/banner.png and banner.svg")


if __name__ == "__main__":
    main()
