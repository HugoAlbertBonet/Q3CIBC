#!/usr/bin/env python3
"""Table II of the paper as a success-vs-latency figure.

One panel per task: per-step inference time (x, log scale) against the task's own
score (y). Up and to the left is better. Colour is the method (the paper palette in
utils/plot_style.py), marker shape the variant; vertical bars are the std across
training seeds. Push-T gets two panels: absolute IoU (no std in the table) and the
Delta-vs-BC metric with the +/- reported in Table II. The Pareto front (methods no
other method beats on both speed and score) is a light gray step line, with a
light gray disk behind every point on it. No grid, no in-plot text; Palatino Linotype.

Values are copied from Table II in
6a8ca3b24200ec26f0fd2d36/sections/6.results.tex. Update both together.

Usage:
    uv run python scripts/plot_table2.py --out results/reviewer/table2_tradeoff.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.plot_style import METHOD_COLORS, TEXT, TEXT2, apply_house_style

FAMILY = dict(METHOD_COLORS)
BG = "#ffffff"  # pure white page and panels (not the house off-white)
FRONT_LINE, FRONT_DISK = "#e0e0e0", "#ebebeb"  # light gray

# key -> (legend label, family, marker)
METHODS = {
    "wifi_argmax": ("WiFI-BC-argmax", "wifi", "o"),
    "wifi_refined": ("WiFI-BC + DFO / Langevin", "wifi", "*"),
    "ibc": ("IBC (DFO / Langevin)", "ibc", "o"),
    "ddpm": ("DDPM-100", "dp", "o"),
    "ddim": ("DDIM (5-25 steps)", "dp", "D"),
    "cp": ("CP 1-step", "cp", "o"),
    "bc_mse": ("BC (MSE)", "bc", "o"),
    "bc_mdn": ("BC (MDN)", "bc", "^"),
}

WIDE_X = ((0.07, 900), [0.1, 1, 10, 100])

# (title, y label, y limits, (x limits, x ticks), [(method key, score, std or None, ms)])
TASKS = [
    ("Particle (16-D)", "Success (%)", (-5, 105), WIDE_X, [
        ("bc_mse", 3, 12.6, 0.11), ("ibc", 99, 4.3, 476.45),
        ("ddpm", 67.0, 7.2, 41.95), ("ddim", 71.0, 7.1, 6.45),
        ("cp", 0.0, 0.0, 0.90), ("wifi_argmax", 82.7, 3.1, 0.38),
        ("wifi_refined", 84.2, 7.2, 204.24),
    ]),
    ("Adroit Pen (human)", "Return", (1900, 3300), WIDE_X, [
        ("bc_mse", 2141, 109, 0.70), ("ibc", 2586, 65, 195.87),
        ("ddpm", 3050, 111, 58.39), ("ddim", 3077, 67, 4.32),
        ("cp", 2287, 143, 1.13), ("wifi_argmax", 2631, 110, 1.55),
    ]),
    ("Franka Kitchen (complete)", "Subtasks Completed", (0, 4), WIDE_X, [
        ("bc_mse", 1.76, 0.07, 0.69), ("ibc", 3.37, 0.01, 432.95),
        ("ddpm", 2.45, 0.41, 67.34), ("ddim", 2.60, 0.51, 4.49),
        ("cp", 1.12, 0.88, 1.20), ("wifi_refined", 3.41, 0.19, 172.67),
        ("wifi_argmax", 2.28, 0.35, 1.67),
    ]),
    # Zoomed: every method but CP 1-step sits at 98-100 %; CP is drawn as an off-scale marker.
    ("Pushing (states)", "Success (%)", (94, 101), WIDE_X, [
        ("bc_mse", 98.3, 0.5, 0.48), ("bc_mdn", 100, 0, 0.57),
        ("ibc", 100, 0, 5.12), ("ddpm", 99.3, 0.6, 57.92),
        ("ddim", 99.0, 1.7, 4.17), ("cp", 4.5, 3.5, 1.12),
        ("wifi_argmax", 99.0, 1.0, 0.79), ("wifi_refined", 100, 0, 3.40),
    ]),
    ("Pushing (pixels)", "Success (%)", (-5, 105), WIDE_X, [
        ("bc_mse", 87.0, 4.1, 0.87), ("bc_mdn", 10.0, 4.3, 0.87),
        ("ibc", 100, 0, 16.06), ("ddpm", 94.0, 2.2, 43.01),
        ("ddim", 92.7, 1.7, 3.69), ("cp", 75.0, 11.3, 1.57),
        ("wifi_argmax", 94.0, 3.0, 1.46), ("wifi_refined", 95.7, 1.7, 5.87),
    ]),
    ("LIBERO-Goal", "Success (%)", (35, 100), ((3.5, 180), [5, 10, 20, 50, 100]), [
        ("bc_mse", 94.1, 1.9, 5.39), ("ibc", 42.0, 8.7, 111.05),
        ("ddpm", 84.5, 3.4, 58.77), ("ddim", 81.2, 4.1, 14.92),
        ("cp", 91.1, 1.4, 5.81), ("wifi_argmax", 93.9, 0.7, 10.61),
    ]),
    ("Push-T (real robot)", "IoU (%)", (55, 90), ((6.5, 24), [7, 10, 15, 20]), [
        ("bc_mse", 61.0, None, 7.8), ("ibc", 73.41, None, 11.1),
        ("ddpm", 65.0, None, 17.6), ("ddim", 65.6, None, 19.4),
        ("wifi_argmax", 79.9, None, 13.9), ("wifi_refined", 83.6, None, 10.8),
    ]),
    # Mean per-position IoU gain over BC; BC is the zero reference.
    ("Push-T (real robot)", "ΔIoU vs. BC (pp)", (-20, 36), ((6.5, 24), [7, 10, 15, 20]), [
        ("bc_mse", 0.0, None, 7.8), ("ibc", 8.5, 26.3, 11.1),
        ("ddpm", 5.0, 3.9, 17.6), ("ddim", 5.6, 4.4, 19.4),
        ("wifi_argmax", 20.2, 6.2, 13.9), ("wifi_refined", 23.6, 6.6, 10.8),
    ]),
]


def pareto_front(points):
    """Points not beaten by any faster-or-equal point, sorted by latency. Ties in latency
    go best score first, so a point matched in speed and beaten in score stays off."""
    front, best = [], float("-inf")
    for ms, score in sorted(points, key=lambda p: (p[0], -p[1])):
        if score > best:
            front.append((ms, score))
            best = score
    return front


def marker_size(marker):
    return 14 if marker == "*" else 7.5


def style_panel(ax, title, ylabel, ylim, xspec):
    (xlim, xticks) = xspec
    ax.set_facecolor(BG)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:g}" for t in xticks])
    ax.minorticks_off()
    ax.set_ylim(*ylim)
    if ylabel == "Subtasks Completed":  # a count: whole-number ticks only
        ax.set_yticks(range(int(ylim[0]), int(ylim[1]) + 1))
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11.5, color=TEXT, loc="left", pad=5)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, labelsize=9)


def draw_panel(ax, task):
    title, ylabel, ylim, xspec, rows = task
    style_panel(ax, title, ylabel, ylim, xspec)
    xlim = xspec[0]

    # Front: drop from the fastest front point to the axis floor, step up through the
    # front, then run flat to the right edge.
    front = pareto_front([(ms, score) for _, score, _, ms in rows])
    xs = [front[0][0]] + [p[0] for p in front] + [xlim[1]]
    ys = [ylim[0]] + [p[1] for p in front] + [front[-1][1]]
    ax.step(xs, ys, where="post", color=FRONT_LINE, linewidth=2.0, zorder=1, solid_capstyle="butt")
    on_front = set(front)

    for key, score, std, ms in rows:
        name, family, marker = METHODS[key]
        color = FAMILY[family]
        if score < ylim[0]:
            # Off-scale below a zoomed axis: pin a down-pointing marker near the bottom edge (no text).
            y = ylim[0] + 0.08 * (ylim[1] - ylim[0])
            ax.plot(ms, y, linestyle="none", marker="v", markersize=7.5, markerfacecolor=color,
                    markeredgecolor=BG, markeredgewidth=1.2, zorder=3, clip_on=False)
            continue
        if (ms, score) in on_front:
            ax.plot(ms, score, linestyle="none", marker="o", markersize=19, markerfacecolor=FRONT_DISK,
                    markeredgecolor="none", zorder=1.5)
        if std:
            ax.errorbar(ms, score, yerr=std, fmt="none", ecolor=color, elinewidth=1.1, alpha=0.6, zorder=2)
        ax.plot(ms, score, linestyle="none", marker=marker, markersize=marker_size(marker),
                markerfacecolor=color, markeredgecolor=BG, markeredgewidth=1.2,
                zorder=4 if marker == "*" else 3)


def draw_legend(fig):
    handles = [Line2D([], [], linestyle="none", marker=marker, markersize=marker_size(marker),
                      markerfacecolor=FAMILY[family], markeredgecolor=BG, markeredgewidth=1.2, label=label)
               for label, family, marker in METHODS.values()]
    handles.append(Line2D([], [], color=FRONT_LINE, linewidth=2.0, marker="o", markersize=13,
                          markerfacecolor=FRONT_DISK, markeredgecolor="none", label="Pareto Front"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5,
                        frameon=False, fontsize=9.5, labelcolor=TEXT, handletextpad=0.5,
                        columnspacing=1.8, labelspacing=0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/reviewer/table2_tradeoff.png")
    args = parser.parse_args()

    family = apply_house_style()  # Palatino Linotype; raises if it is not installed
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.0), facecolor=BG)
    for ax, task in zip(axes.flat, TASKS):
        draw_panel(ax, task)
    # Reserve a bottom band for the shared x label and the horizontal legend.
    fig.tight_layout(rect=(0, 0.125, 1, 1), h_pad=0.8, w_pad=0.4)
    fig.text(0.5, 0.105, "Inference Time Per Step (ms)", ha="center", va="center", fontsize=10.5, color=TEXT2)
    draw_legend(fig)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} and {out.with_suffix('.pdf')} (font: {family})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
