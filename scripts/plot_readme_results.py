#!/usr/bin/env python3
"""Render the README result figure from the paper's public Table 1 values.

Install matplotlib, then run from any directory:
    python scripts/plot_readme_results.py

The JSON input contains all 13 Table 1 rows. The figure displays six
shared-backbone configurations and Gemini as a separate reference model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
INK = "#18283E"
MUTED = "#64748B"
BLUE = "#3866FF"
TEAL = "#208A8C"
GRID = "#E7ECF3"
COLORS = {
    "Qwen3.6-35B-A3B (base)": "#8795A9",
    "GRPO": "#BE8A4D",
    "GRPO (High-Conf.)": "#A896C8",
    "CA-GRPO": TEAL,
    "CA-GRPO + knowledge": "#6052A3",
    "AdaptEvo": BLUE,
    "Gemini3.8-Flash": "#45586E",
}
DISPLAY = {
    "Qwen3.6-35B-A3B (base)": "Base model",
    "Gemini3.8-Flash": "Gemini3.8-Flash",
}


def make_figure(data: dict, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "font.weight": "medium",
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "svg.fonttype": "path",
        "svg.hashsalt": "adaptevo-readme-main-results",
    })
    selected = [r for r in data["rows"] if r["group"] == "shared_backbone"]
    selected.append(next(r for r in data["rows"] if r["method"] == "Gemini3.8-Flash"))
    by_name = {r["method"]: r for r in data["rows"]}
    ours, grpo = by_name["AdaptEvo"], by_name["GRPO"]

    fig = plt.figure(figsize=(15.2, 11.2), facecolor="white")
    fig.text(.035, .952, "AdaptEvo", fontsize=28, fontweight="bold", color=BLUE)
    fig.text(.22, .953, "Main results", fontsize=25, fontweight="bold")
    fig.text(.035, .918, "Tool-using multimodal content moderation", fontsize=15, color=MUTED)

    for x, period, label in [(0.035, "in_period", "IN-PERIOD"),
                              (0.515, "out_of_period", "OUT-OF-PERIOD")]:
        fig.add_artist(FancyBboxPatch(
            (x, .795), .45, .10, boxstyle="round,pad=0.009,rounding_size=0.012",
            transform=fig.transFigure, facecolor="#F0F4FF", edgecolor="none",
        ))
        ela, bda = ours[f"{period}_ela"], ours[f"{period}_bda"]
        fig.text(x+.015, .868, label, fontsize=11.5, fontweight="bold", color=MUTED)
        fig.text(x+.015, .833, f"{ela:.1f}% ELA   /   {bda:.1f}% BDA",
                 fontsize=21, fontweight="bold", color=BLUE)
        fig.text(x+.015, .805,
                 f"+{ela-grpo[f'{period}_ela']:.1f} ELA / +{bda-grpo[f'{period}_bda']:.1f} BDA points vs GRPO",
                 fontsize=12, color=INK)

    # Each panel uses a zoomed dot axis, so distances do not imply zero-based bars.
    panels = [
        (.19, .47, "in_period_ela", "In-Period", "Exact-label accuracy (ELA)", (50, 65), [50, 55, 60, 65]),
        (.665, .47, "out_of_period_ela", "Out-of-Period", "Exact-label accuracy (ELA)", (50, 65), [50, 55, 60, 65]),
        (.19, .125, "in_period_bda", "In-Period", "Binary decision accuracy (BDA)", (64, 74.5), [64, 67, 70, 73]),
        (.665, .125, "out_of_period_bda", "Out-of-Period", "Binary decision accuracy (BDA)", (64, 71.8), [64, 66, 68, 70]),
    ]
    y_positions = [0, 1, 2, 3, 4, 5, 6.65]
    for x, y, key, period, metric, limits, ticks in panels:
        ax = fig.add_axes([x, y, .275, .245], facecolor="white")
        ax.set_title(f"{period}\n{metric}", fontsize=13.5, fontweight="bold",
                     loc="left", pad=14, linespacing=1.5)
        ax.set_xlim(*limits)
        ax.set_ylim(7.3, -.7)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t}%" for t in ticks], fontsize=12)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([DISPLAY.get(r["method"], r["method"]) for r in selected], fontsize=12)
        ax.tick_params(axis="both", length=0)
        ax.tick_params(axis="y", pad=12)
        ax.tick_params(axis="x", pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.axhspan(4.56, 5.44, color="#EEF3FF", zorder=0)
        ax.axhline(5.86, color="#CCD6E3", linestyle=(0, (3, 3)), linewidth=1.1)
        for label, row, ypos in zip(ax.get_yticklabels(), selected, y_positions):
            method = row["method"]
            if method == "AdaptEvo":
                label.set_color(BLUE)
                label.set_fontweight("bold")
            elif method == "Gemini3.8-Flash":
                label.set_color(MUTED)
            color, value = COLORS[method], row[key]
            marker = "D" if row["group"] == "reference_model" else "o"
            ax.scatter(value, ypos, s=112 if method == "AdaptEvo" else 74,
                       marker=marker, color=color, edgecolors="white", linewidths=1.3, zorder=3)
            ax.annotate(f"{value:.1f}", (value, ypos), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=12.5,
                        fontweight="bold" if method == "AdaptEvo" else "medium", color=color)

    fig.text(.035, .045, "Shared backbone: Qwen3.6-35B-A3B. Gemini is a separate reference model (diamond).",
             fontsize=11.5, color=MUTED)
    fig.text(.035, .022, "Source: paper Table 1. Accuracy (%), higher is better. Axes are zoomed; exact values are labeled.",
             fontsize=11, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=170, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "docs/assets/main-results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/main-results")
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    make_figure(data, args.output)
    print(f"Rendered {args.output.with_suffix('.png')} and .svg")


if __name__ == "__main__":
    main()
