#!/usr/bin/env python3
"""Table II of the paper as a success-vs-latency figure.

One panel per task: per-step inference time (x, log scale, shared across panels)
against the task's own score (y). Up and to the left is better. Colour is the
method family, marker shape the variant; vertical bars are the std across
training seeds. Push-T gets two panels: absolute IoU (no std in the table) and
the Delta-vs-BC metric with the +/- reported in Table II. A light step line
marks the Pareto front (no other method is both faster and better).

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
from utils.plot_style import GRID, METHOD_COLORS, SURFACE, TEXT, TEXT2, apply_house_style

# Family colours: the paper method palette, identical in every figure. Orange/green are close
# under deuteranopia, so each family also keeps its own marker shape (stars, circles, diamonds).
FAMILY = dict(METHOD_COLORS)
FRONT = "#dddcd7"

# key -> (legend label, family, marker, filled)
METHODS = {
    "wifi_argmax": ("WiFI-BC-argmax", "wifi", "*", True),
    "wifi_refined": ("WiFI-BC + DFO / Langevin", "wifi", "*", False),
    "ibc": ("IBC (DFO / Langevin)", "ibc", "o", True),
    "ddpm": ("DDPM-100", "dp", "o", True),
    "ddim": ("DDIM (5-25 steps)", "dp", "D", True),
    "cp": ("CP 1-step", "cp", "o", True),
    "bc_mse": ("BC (MSE)", "bc", "o", True),
    "bc_mdn": ("BC (MDN)", "bc", "^", True),
}

# task -> (y label, y limits, [(method key, score, std or None, ms, direct label or None)])
TASKS = [
    ("Particle 16-D", "Success (%)", (-5, 105), [
        ("bc_mse", 3, 12.6, 0.11, None), ("ibc", 99, 4.3, 476.45, None),
        ("ddpm", 67.0, 7.2, 41.95, None), ("ddim", 71.0, 7.1, 6.45, None),
        ("cp", 0.0, 0.0, 0.90, None), ("wifi_argmax", 82.7, 3.1, 0.38, "argmax"),
        ("wifi_refined", 84.2, 7.2, 204.24, "Langevin"),
    ]),
    ("Adroit pen-human", "Return", (1900, 3300), [
        ("bc_mse", 2141, 109, 0.70, None), ("ibc", 2586, 65, 195.87, None),
        ("ddpm", 3050, 111, 58.39, None), ("ddim", 3077, 67, 4.32, None),
        ("cp", 2287, 143, 1.13, None), ("wifi_argmax", 2631, 110, 1.55, "argmax"),
    ]),
    ("Franka kitchen-complete", "Subtasks (0-4)", (0, 4), [
        ("bc_mse", 1.76, 0.07, 0.69, None), ("ibc", 3.37, 0.01, 432.95, None),
        ("ddpm", 2.45, 0.41, 67.34, None), ("ddim", 2.60, 0.51, 4.49, None),
        ("cp", 1.12, 0.88, 1.20, None), ("wifi_refined", 3.41, 0.19, 172.67, "Langevin"),
        ("wifi_argmax", 2.28, 0.35, 1.67, "argmax"),
    ]),
    # Zoomed: every method but CP 1-step sits at 98-100 %; CP is drawn as an off-scale marker.
    ("Pushing, states", "Success (%)", (94, 101), [
        ("bc_mse", 98.3, 0.5, 0.48, None), ("bc_mdn", 100, 0, 0.57, None),
        ("ibc", 100, 0, 5.12, None), ("ddpm", 99.3, 0.6, 57.92, None),
        ("ddim", 99.0, 1.7, 4.17, None), ("cp", 4.5, 3.5, 1.12, None),
        ("wifi_argmax", 99.0, 1.0, 0.79, None), ("wifi_refined", 100, 0, 3.40, None),
    ]),
    ("Pushing, pixels", "Success (%)", (-5, 105), [
        ("bc_mse", 87.0, 4.1, 0.87, None), ("bc_mdn", 10.0, 4.3, 0.87, None),
        ("ibc", 100, 0, 16.06, None), ("ddpm", 94.0, 2.2, 43.01, None),
        ("ddim", 92.7, 1.7, 3.69, None), ("cp", 75.0, 11.3, 1.57, None),
        ("wifi_argmax", 94.0, 3.0, 1.46, None), ("wifi_refined", 95.7, 1.7, 5.87, None),
    ]),
    ("LIBERO-Goal, pixels", "Success (%)", (35, 100), [
        ("bc_mse", 94.1, 1.9, 111.05, None), ("ibc", 42.0, 8.7, 111.05, None),
        ("ddpm", 84.5, 3.4, 58.77, None), ("ddim", 81.2, 4.1, 14.92, None),
        ("cp", 91.1, 1.4, 5.81, None), ("wifi_argmax", 93.9, 0.7, 10.61, "argmax"),
    ]),
    ("Push-T, real WidowX (IoU)", "IoU (%)", (55, 90), [
        ("bc_mse", 61.0, None, 7.8, None), ("ibc", 73.41, None, 11.1, None),
        ("ddpm", 65.0, None, 17.6, None), ("ddim", 65.6, None, 19.4, None),
        ("wifi_argmax", 79.9, None, 13.9, "argmax"), ("wifi_refined", 83.6, None, 10.8, "DFO-5"),
    ]),
    # Mean per-position IoU gain over BC; BC is the zero reference.
    ("Push-T, real WidowX ($\\Delta$ vs. BC)", "$\\Delta$ IoU vs. BC (pp)", (-20, 36), [
        ("bc_mse", 0.0, None, 7.8, None), ("ibc", 8.5, 26.3, 11.1, None),
        ("ddpm", 5.0, 3.9, 17.6, None), ("ddim", 5.6, 4.4, 19.4, None),
        ("wifi_argmax", 20.2, 6.2, 13.9, "argmax"), ("wifi_refined", 23.6, 6.6, 10.8, ("DFO-5", (-9, 12), "right")),
    ]),
]

X_LIM = (0.07, 900)


def pareto_front(points):
    """Points not beaten by any faster point, sorted by latency."""
    front, best = [], float("-inf")
    for ms, score in sorted(points):
        if score > best:
            front.append((ms, score))
            best = score
    return front


def style_panel(ax, title, ylabel, ylim):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11.5, color=TEXT, loc="left", pad=6)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, labelsize=9)


def draw_panel(ax, task):
    title, ylabel, ylim, rows = task
    style_panel(ax, title, ylabel, ylim)

    front = pareto_front([(ms, score) for _, score, _, ms, _ in rows])
    xs = [p[0] for p in front] + [X_LIM[1]]
    ys = [p[1] for p in front] + [front[-1][1]]
    ax.step(xs, ys, where="post", color=FRONT, linewidth=1.1, zorder=1, solid_capstyle="round")

    for key, score, std, ms, label in rows:
        name, family, marker, filled = METHODS[key]
        color = FAMILY[family]
        if score < ylim[0]:
            # Off-scale below a zoomed axis: pin a down-pointing marker to the bottom edge.
            span = ylim[1] - ylim[0]
            ax.plot(ms, ylim[0] + 0.08 * span, linestyle="none", marker="v", markersize=7.5,
                    markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, clip_on=False)
            ax.annotate(f"{name}: {score:g}%", (ms, ylim[0] + 0.08 * span), xytext=(7, 0),
                        textcoords="offset points", fontsize=8.5, color=TEXT2, va="center", zorder=5)
            continue
        if std:
            ax.errorbar(ms, score, yerr=std, fmt="none", ecolor=color, elinewidth=1.1, alpha=0.55, zorder=2)
        star = marker == "*"
        ax.plot(ms, score, linestyle="none", marker=marker, markersize=13 if star else 7.5,
                markerfacecolor=color if filled else SURFACE, markeredgecolor=color if not filled else SURFACE,
                markeredgewidth=1.6 if not filled else 1.2, zorder=4 if star else 3)
        if label:
            # A label is text, or (text, (dx, dy) in points, horizontal alignment) to dodge a neighbour.
            text, offset, ha = (label, (7, -3), "left") if isinstance(label, str) else label
            ax.annotate(text, (ms, score), xytext=offset, textcoords="offset points",
                        fontsize=8.5, color=TEXT2, va="top", ha=ha, zorder=5)


def draw_legend(fig):
    handles = []
    for label, family, marker, filled in METHODS.values():
        color = FAMILY[family]
        star = marker == "*"
        handles.append(Line2D([], [], linestyle="none", marker=marker, markersize=13 if star else 7.5,
                              markerfacecolor=color if filled else SURFACE,
                              markeredgecolor=color if not filled else SURFACE,
                              markeredgewidth=1.6 if not filled else 1.2, label=label))
    handles.append(Line2D([], [], color=FRONT, linewidth=1.6, label="Pareto front"))
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5,
                        frameon=False, fontsize=9.5, labelcolor=TEXT, handletextpad=0.5,
                        columnspacing=1.8, labelspacing=0.5,
                        title="Up and to the left is better (higher score, faster inference)",
                        title_fontsize=9)
    legend.get_title().set_color(TEXT2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/reviewer/table2_tradeoff.png")
    args = parser.parse_args()

    apply_house_style()
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.3), facecolor=SURFACE)
    for ax, task in zip(axes.flat, TASKS):
        draw_panel(ax, task)
    # Reserve a bottom band for the shared x label and the horizontal legend.
    fig.tight_layout(rect=(0, 0.17, 1, 1), h_pad=1.4, w_pad=1.2)
    fig.text(0.5, 0.145, "Inference time per step (ms, log scale)", ha="center", va="center",
             fontsize=10.5, color=TEXT2)
    draw_legend(fig)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
