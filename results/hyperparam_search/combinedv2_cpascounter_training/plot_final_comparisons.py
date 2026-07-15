#!/usr/bin/env python3
"""Build the final cross-environment inference comparison figure.

The figure reads final comparison CSVs only. It does not inspect trial logs.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
OUT = ROOT / "final_comparisons_summary.png"
OUT_PDF = ROOT / "final_comparisons_summary.pdf"

# Restrained, color-blind-safe palette for print and screen.
BG = "#FFFFFF"
CARD = "#FFFFFF"
GRID = "#D9DEE7"
TEXT = "#171A1F"
MUTED = "#505761"
Q3C = "#0072B2"
IBC = "#D55E00"
DP = "#009E73"
EXPLICIT = "#CC79A7"
OTHER = "#7A8491"
GOOD = "#2E7D32"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "savefig.facecolor": BG,
    "savefig.edgecolor": BG,
})

FAMILY_COLORS = {
    "Q3CIBC": Q3C,
    "IBC EBM": IBC,
    "Explicit BC": EXPLICIT,
    "Diffusion Policy": DP,
    "Other": OTHER,
}


def read_csv(path: Path, comments: bool = False) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        lines = (line for line in handle if not comments or not line.startswith("#"))
        return list(csv.DictReader(lines))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def family(method: str) -> str:
    low = method.lower()
    if "q3c" in low:
        return "Q3CIBC"
    if low in ("ibc_mdn", "ibc_mse") or low.startswith("explicit bc"):
        return "Explicit BC"
    if low.startswith("ibc"):
        return "IBC EBM"
    if "diffusion" in low or low.startswith("dp_") or low.startswith("dp "):
        return "Diffusion Policy"
    return "Other"


def short_method(method: str) -> str:
    """Compact labels that remain readable when every point is annotated."""
    aliases = {
        "IBC + DFO": "IBC DFO",
        "q3c_0iter": "Q3C 0-it",
        "q3c_3iter": "Q3C 3-it",
        "q3c_5iter": "Q3C 5-it",
        "q3c_10iter": "Q3C 10-it",
        "ibc": "IBC EBM",
        "ibc_mdn": "MDN explicit",
        "ibc_mse": "MSE explicit",
        "dp_ddpm100": "DP DDPM-100",
        "dp_ddim5": "DP DDIM-5",
        "dp_ddim10": "DP DDIM-10",
        "dp_ddim25": "DP DDIM-25",
        "Explicit BC MSE (paper arch 2048x8-dense; quality: paper-reported)": "Explicit MSE",
        "Q3C CP-argmax (cp=200, resnet gen, no refinement)": "Q3C argmax",
        "Q3C+Langevin 30+20 (cp=200, lr 0.1, faithful chain)": "Q3C Lv 30+20",
        "Q3C+Langevin 50+30 (cp=200, lr 0.1, faithful chain)": "Q3C Lv 50+30",
        "Q3C+Langevin 50+30 (cp=200, lr 0.05, faithful chain)": "Q3C Lv 50+30 (0.05)",
        "Q3C+CP-DFO (cp=200, it5 std0.05 dec0.5)": "Q3C CP-DFO",
        "IBC full-faithful Langevin (100+100 iters x 512 samples)": "our IBC EBM",
        "IBC paper-reported EBM (official quality; faithful timing)": "IBC official",
        "Q3C-B4 CP-argmax (cp=100, no refinement)": "Q3C argmax",
        "Q3C+CP-DFO (cp=100, 10 iters, +32 uniform safety)": "Q3C CP-DFO",
        "Q3C+gentle Langevin (cp=80, 25 iters, very-gentle)": "Q3C gentle Lv",
        "IBC paper-reported EBM (official quality; paper-exact timing)": "IBC official",
        "Q3C CP-argmax (SN-off, cp=20, no refinement)": "Q3C argmax",
        "Q3C + Langevin-100 (SN-off, cp=20)": "Q3C Langevin-100",
        "DP + DDPM (100 steps, eps, resnet 512x4)": "DP DDPM-100",
        "DP + DDIM (5 steps, eps, resnet 512x4)": "DP DDIM-5",
        "DP + DDIM (10 steps, eps, resnet 512x4)": "DP DDIM-10",
        "DP + DDIM (25 steps, eps, resnet 512x4)": "DP DDIM-25",
    }
    if method in aliases:
        return aliases[method]
    if method.startswith("Q3CIBC + CP-DFO"):
        return method.replace("Q3CIBC + ", "Q3C ").split(" —")[0]
    if method.startswith("Diffusion Policy + "):
        return method.replace("Diffusion Policy + ", "DP ")
    return method.split(" (")[0]


def style_card(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(CARD)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#9AA1AA")
        ax.spines[name].set_linewidth(.8)
    ax.tick_params(colors=MUTED, labelsize=8.2, length=3, width=.7)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=.65, alpha=.75, linestyle=(0, (2, 2)))
    ax.set_title(title, loc="left", color=TEXT, fontsize=12.5,
                 fontweight="semibold", pad=17)
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, color=MUTED,
            fontsize=8.1, fontstyle="italic", va="bottom")


def annotate(ax, x: float, y: float, text: str, color: str, dx=7, dy=7) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        fontsize=7.2,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=.16", facecolor=BG, edgecolor=GRID,
                  linewidth=.35, alpha=.93),
        zorder=7,
    )


def connected_scatter(
    ax,
    rows: list[dict[str, str]],
    x_key: str,
    y_key: str,
    labels: dict[str, tuple[str, int, int]] | None = None,
    *,
    line_alpha: float = .7,
    yerr_key: str | None = None,
) -> None:
    """Connect variants from the same method family, then draw their points."""
    labels = labels or {}
    groups: dict[str, list[tuple[float, float, dict[str, str]]]] = defaultdict(list)
    for row in rows:
        groups[family(row["method"])].append((number(row, x_key), number(row, y_key), row))

    for fam, points in groups.items():
        color = FAMILY_COLORS[fam]
        points.sort(key=lambda item: item[0])
        if len(points) > 1:
            ax.plot([p[0] for p in points], [p[1] for p in points], color=color,
                    lw=1.55, alpha=line_alpha, zorder=2)
        for index, (x, y, row) in enumerate(points):
            if yerr_key and row.get(yerr_key, "").strip():
                yerr = float(row[yerr_key])
                if yerr > 0:
                    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color,
                                elinewidth=.75, capsize=2.2, capthick=.75,
                                alpha=.55, zorder=3)
            ax.scatter(x, y, s=48 if fam == "Q3CIBC" else 40, c=color,
                       edgecolor="white", linewidth=.65, zorder=4)
            if row["method"] in labels:
                text, dx, dy = labels[row["method"]]
            else:
                text = short_method(row["method"])
                dx = 7
                dy = 7 if index % 2 == 0 else -16
            annotate(ax, x, y, text, color, dx, dy)


def draw_particle_dimension(ax, particle: list[dict[str, str]]) -> None:
    style_card(ax, "(a) Particle: success vs. dimension",
               "Q3C best final experiments vs official IBC Figure 6 curves")
    particle = sorted(particle, key=lambda r: int(r["n_dim"]))
    q3_points = [(int(r["n_dim"]), 100 * number(r, "success_rate_q3cibc"))
                 for r in particle if r["success_rate_q3cibc"].strip()]
    ibc_points = [(int(r["n_dim"]), 100 * number(r, "success_rate_ibc_paper"))
                  for r in particle if r["success_rate_ibc_paper"].strip()]
    mse_points = [(int(r["n_dim"]), 100 * number(r, "success_rate_mse_paper"))
                  for r in particle if r["success_rate_mse_paper"].strip()]
    ax.plot([p[0] for p in q3_points], [p[1] for p in q3_points], "o-",
            color=Q3C, lw=1.65, ms=5.2, label="Q3CIBC", zorder=4)
    ax.plot([p[0] for p in ibc_points], [p[1] for p in ibc_points], "o--",
            color=IBC, lw=1.65, ms=5.2, label="IBC official (Langevin)", zorder=4)
    ax.plot([p[0] for p in mse_points], [p[1] for p in mse_points], marker="x",
            linestyle=":", color=EXPLICIT, lw=1.65, ms=5.2,
            label="Explicit MSE-BC official", zorder=4)
    for x, y in q3_points:
        ax.text(x, y + 3.0, f"{y:.0f}", color=Q3C, ha="center", fontsize=7.1,
                fontweight="semibold")
    for x, y in ibc_points:
        offset = -5.5 if y > 15 else 3.0
        ax.text(x, y + offset, f"{y:.0f}", color=IBC, ha="center", fontsize=7.1,
                fontweight="semibold")
    for x, y in mse_points:
        offset = 6.5 if y < 90 else 7.0
        ax.text(x, y + offset, f"{y:.0f}", color=EXPLICIT, ha="center", fontsize=7.1,
                fontweight="semibold")
    ax.set_xticks([int(r["n_dim"]) for r in particle])
    ax.set_ylim(-4, 112)
    ax.set_xlabel("particle dimension", color=MUTED)
    ax.set_ylabel("success rate (%)", color=MUTED)
    ax.legend(frameon=False, labelcolor=MUTED, loc="lower left", fontsize=9)


def draw_particle_latency(
    ax,
    particle: list[dict[str, str]],
) -> None:
    style_card(ax, "(b) Particle: success vs. inference time",
               "Official success rates + our locally calculated per-action timings")
    q3_points = []
    ibc_points = []
    for row in particle:
        dim = int(row["n_dim"])
        q_time = row["inference_time_q3cibc_local_ms_per_action"].strip()
        ibc_time = row["inference_time_ibc_local_ms_per_action"].strip()
        if q_time and row["success_rate_q3cibc"].strip():
            q3_points.append((float(q_time), 100 * number(row, "success_rate_q3cibc"), dim))
        if ibc_time and row["success_rate_ibc_paper"].strip():
            ibc_points.append((float(ibc_time), 100 * number(row, "success_rate_ibc_paper"), dim))

    q3_label_offsets = {
        2: (7, 8),
        3: (7, -18),
        4: (7, 8),
        5: (7, 8),
        6: (52, -10),
        8: (6, 7),
        16: (6, 7),
    }
    ibc_label_offsets = {
        2: (-68, -18),
        3: (8, -6),
        4: (-68, -18),
        5: (-68, 8),
        6: (8, 8),
        8: (8, -18),
        16: (6, -17),
    }
    for points, color, label in [(q3_points, Q3C, "Q3CIBC"),
                                 (ibc_points, IBC, "IBC official (Langevin)")]:
        points.sort()
        linestyle = "--" if label.startswith("IBC") else "-"
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                linestyle=linestyle,
                color=color, lw=1.65, ms=5.2, label=label, zorder=4)
        for x, y, dim in points:
            dx, dy = q3_label_offsets.get(dim, (6, 7))
            if label.startswith("IBC"):
                dx, dy = ibc_label_offsets.get(dim, (6, -17))
            time_text = f"{x:.2f}" if x < 10 else f"{x:.1f}"
            annotate(ax, x, y, f"{dim}D · {time_text} ms", color, dx, dy)
    ax.set_xscale("log")
    ax.set_xlim(.42, 720)
    ax.set_ylim(84, 102)
    ax.set_xlabel("inference time (ms, log scale)", color=MUTED)
    ax.set_ylabel("success rate (%)", color=MUTED)
    ax.legend(frameon=False, labelcolor=MUTED, loc="lower left", fontsize=9)
    ax.text(.98, .025,
            "2–8D: wall time / 2,500 actions\n16D: direct microbenchmark; MSE timing unavailable",
            transform=ax.transAxes, color=MUTED, fontsize=6.6,
            ha="right", va="bottom")


def main() -> None:
    particle = read_csv(PROJECT / "results" / "particle" / "success_rates.csv")
    pushing = read_csv(ROOT / "pushing" / "single_target_states.csv")
    pixels = read_csv(ROOT / "pushing_pixels" / "single_target_pixels.csv")
    kitchen = read_csv(ROOT / "d4rl" / "kitchen" / "kitchen_inference_results.csv")
    pen = read_csv(ROOT / "d4rl" / "pen" / "pen_inference_results.csv")
    # Retain the local IBC row in the final CSV for provenance, but honor the
    # requested paper comparison by plotting only its official reported quality.
    pen = [r for r in pen if not r["method"].startswith("IBC ")
           or r["method"].startswith("IBC paper-reported")]
    libero = read_csv(ROOT / "libero_goal" / "standard_results.csv", comments=True)

    fig = plt.figure(figsize=(15.5, 19.5), facecolor=BG)
    gs = fig.add_gridspec(4, 2, left=.075, right=.975, bottom=.105, top=.905,
                          hspace=.50, wspace=.25, height_ratios=[1, 1, 1, 1.18])

    fig.text(.075, .962, "Quality–latency trade-offs across environments", color=TEXT,
             fontsize=21, fontweight="semibold")
    fig.text(.075, .938,
             "Final comparison results; connected lines denote variants of the same method (door excluded).",
             color=MUTED, fontsize=9.5)
    fig.lines.append(Line2D([.075, .975], [.925, .925], transform=fig.transFigure,
                            color="#9AA1AA", linewidth=.8))

    # Row 1: the two requested particle views.
    draw_particle_dimension(fig.add_subplot(gs[0, 0]), particle)
    draw_particle_latency(fig.add_subplot(gs[0, 1]), particle)

    # Row 2 left: pushing states.
    ax = fig.add_subplot(gs[1, 0])
    style_card(ax, "(c) Pushing (states)", "Success vs. inference time; paper-faithful architecture")
    push = [r for r in pushing if r["comparison"] == "paper_faithful_both"]
    connected_scatter(ax, push, "inference_mean_ms", "success_pct", {
        "IBC + DFO": ("IBC DFO\n100% · 9.50 ms", 8, -25),
        "Q3CIBC + CP-DFO (3 iters)": ("Q3C 3-it CP-DFO\n100% · 4.64 ms", -20, 10),
        "Q3CIBC + CP-DFO (0 iters — pure CP-argmax)": ("Q3C argmax\n99% · 0.89 ms", 8, -5),
        "Diffusion Policy + DDPM (100 steps)": ("DP DDPM\n99.3% · 72.1 ms", -72, 8),
    }, yerr_key="success_stdev_pp")
    ax.set_xscale("log")
    ax.set_xlim(.65, 100)
    ax.set_ylim(97.0, 101.2)
    ax.set_xlabel("inference time (ms, log scale)", color=MUTED)
    ax.set_ylabel("success rate (%)", color=MUTED)

    # Row 2 right: pushing pixels with a broken y axis.
    pixel_cell = gs[1, 1].subgridspec(2, 1, height_ratios=[3.2, 1], hspace=.07)
    ax_hi = fig.add_subplot(pixel_cell[0])
    ax_lo = fig.add_subplot(pixel_cell[1], sharex=ax_hi)
    style_card(ax_hi, "(d) Pushing (pixels)", "Broken success axis resolves the 87–100% cluster")
    ax_lo.set_facecolor(CARD)
    for spine in ax_lo.spines.values():
        spine.set_visible(False)
    ax_lo.spines["left"].set_visible(True)
    ax_lo.spines["bottom"].set_visible(True)
    ax_lo.spines["left"].set_color("#9AA1AA")
    ax_lo.spines["bottom"].set_color("#9AA1AA")
    ax_lo.tick_params(colors=MUTED, labelsize=8.2, length=3, width=.7)
    ax_lo.set_axisbelow(True)
    ax_lo.grid(True, color=GRID, linewidth=.65, alpha=.75, linestyle=(0, (2, 2)))
    for ax_part in (ax_hi, ax_lo):
        connected_scatter(ax_part, pixels, "inference_time_ms", "success_rate_pct", {
            "q3c_0iter": ("Q3C argmax", 7, 5),
            "q3c_5iter": ("Q3C 5-it · 95.7%", 7, 6),
            "ibc": ("IBC EBM · 100%", -58, 7),
            "ibc_mdn": ("MDN explicit · 10%", 7, -2),
        }, yerr_key="success_rate_std_pct")
        ax_part.set_xscale("log")
        ax_part.set_xlim(.85, 80)
    ax_hi.set_ylim(82, 102)
    ax_lo.set_ylim(5, 15)
    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(labelbottom=False)
    ax_hi.set_ylabel("success rate (%)", color=MUTED)
    ax_lo.set_xlabel("inference time (ms, log scale)", color=MUTED)
    d = .012
    kwargs = dict(color=MUTED, clip_on=False, lw=1.2)
    ax_hi.plot((-d, +d), (-d, +d), transform=ax_hi.transAxes, **kwargs)
    ax_hi.plot((1-d, 1+d), (-d, +d), transform=ax_hi.transAxes, **kwargs)
    ax_lo.plot((-d, +d), (1-d, 1+d), transform=ax_lo.transAxes, **kwargs)
    ax_lo.plot((1-d, 1+d), (1-d, 1+d), transform=ax_lo.transAxes, **kwargs)

    # Row 3: D4RL.
    ax = fig.add_subplot(gs[2, 0])
    style_card(ax, "(e) D4RL Kitchen", "Tasks completed vs. latency; official IBC result included")
    connected_scatter(ax, kitchen, "inference_time_mean_ms", "avg_tasks_completed", {
        "Explicit BC MSE (paper arch 2048x8-dense; quality: paper-reported)":
            ("MSE-BC\n1.76 · 0.98 ms", 7, 7),
        "Q3C CP-argmax (cp=200, resnet gen, no refinement)":
            ("Q3C argmax\n2.28 · 2.86 ms", 7, 7),
        "Q3C+CP-DFO (cp=200, it5 std0.05 dec0.5)":
            ("Q3C CP-DFO\n2.29 · 14.1 ms", 7, -24),
        "Q3C+Langevin 30+20 (cp=200, lr 0.1, faithful chain)":
            ("Q3C Lv 30+20\n2.95 · 164.5 ms", -90, -23),
        "Q3C+Langevin 50+30 (cp=200, lr 0.1, faithful chain)":
            ("Q3C Lv 50+30 (0.10)\n3.25 · 249.4 ms", -105, -26),
        "Q3C+Langevin 50+30 (cp=200, lr 0.05, faithful chain)":
            ("Q3C Lv 50+30 (0.05)\n3.41 · 254.5 ms", -105, 9),
        "IBC full-faithful Langevin (100+100 iters x 512 samples)":
            ("our IBC\n3.05 · 1026 ms", -88, -25),
        "IBC paper-reported EBM (official quality; faithful timing)":
            ("IBC official\n3.37 · 1026 ms", -88, 8),
        "DP + DDPM (100 steps, eps, resnet 512x4)":
            ("DP DDPM-100\n2.45 · 110.8 ms", -35, -25),
        "DP + DDIM (5 steps, eps, resnet 512x4)":
            ("DP DDIM-5\n2.60 · 7.20 ms", -18, 9),
        "DP + DDIM (10 steps, eps, resnet 512x4)":
            ("DP DDIM-10\n2.63 · 12.2 ms", 7, -25),
        "DP + DDIM (25 steps, eps, resnet 512x4)":
            ("DP DDIM-25\n2.59 · 34.7 ms", 7, 8),
    }, yerr_key="SEM")
    ax.set_xscale("log")
    ax.set_xlim(.55, 1600)
    ax.set_ylim(1.5, 3.62)
    ax.set_xlabel("inference time (ms, log scale)", color=MUTED)
    ax.set_ylabel("average tasks completed", color=MUTED)
    ax.text(.98, .06, "Official quality paired with faithful-architecture timing.",
            transform=ax.transAxes, color=MUTED, fontsize=8.5, ha="right")

    ax = fig.add_subplot(gs[2, 1])
    style_card(ax, "(f) D4RL Pen", "Average reward vs. latency; only official IBC quality shown")
    connected_scatter(ax, pen, "inference_time_mean_ms", "avg_reward", {
        "Q3C CP-argmax (SN-off, cp=20, no refinement)":
            ("Q3C argmax\n2631 · 1.73 ms", 7, -24),
        "Q3C+CP-DFO (cp=100, 10 iters, +32 uniform safety)":
            ("Q3C CP-DFO\n2482 · 18.6 ms", 7, -24),
        "Q3C + Langevin-100 (SN-off, cp=20)":
            ("Q3C Lv-100\n2536 · 52.8 ms", 7, 8),
        "IBC paper-reported EBM (official quality; paper-exact timing)":
            ("IBC official\n2586 · 215.1 ms", -88, 8),
        "dp_ddpm100": ("DP DDPM-100\n3050 · 66.0 ms", 7, -24),
        "dp_ddim5": ("DP DDIM-5\n3077 · 4.88 ms", 7, 8),
        "dp_ddim10": ("DP DDIM-10\n3001 · 8.59 ms", 7, -25),
        "dp_ddim25": ("DP DDIM-25\n3008 · 21.3 ms", 7, 8),
    }, yerr_key="SEM")
    ax.set_xscale("log")
    ax.set_xlim(1, 320)
    ax.set_ylim(2200, 3250)
    ax.set_xlabel("inference time (ms, log scale)", color=MUTED)
    ax.set_ylabel("average reward", color=MUTED)
    ax.text(.98, .06, "Official quality paired with paper-exact architecture timing.",
            transform=ax.transAxes, color=MUTED, fontsize=8.5, ha="right")

    # Row 4: LIBERO bars, grouped visually by pretraining hatch.
    ax = fig.add_subplot(gs[3, :])
    style_card(ax, "(g) LIBERO-Goal: standard protocol",
               "Best Q3C rows with ≥3 seeds (50 evals/seed) · solid = pretrained · hatched = scratch")
    libero = sorted(libero, key=lambda r: number(r, "libero_goal_success_rate"))
    for i, row in enumerate(libero):
        value = number(row, "libero_goal_success_rate")
        fam = family(row["method"])
        color = FAMILY_COLORS[fam]
        scratch = row["pretraining"].strip().lower() == "scratch"
        ax.barh(i, value, height=.58, color=color, alpha=.90 if not scratch else .62,
                      hatch="///" if scratch else None, edgecolor=TEXT if scratch else color,
                      linewidth=.6, zorder=3)
        ax.text(value - .5, i, f"{value:.1f}%", va="center", ha="right",
                color="white", fontsize=8.2, fontweight="semibold", zorder=5)
        if "Q3CIBC" in row["method"]:
            suffix = "PRETRAINED" if not scratch else "SCRATCH"
            seeds = f" · {row['n_seeds']} seed" + ("s" if row["n_seeds"] != "1" else "")
            ax.text(value + .7, i, suffix + seeds, va="center", color=color,
                    fontsize=7.8, fontweight="semibold")
    ax.set_yticks(range(len(libero)), [r["method"] for r in libero])
    ax.set_xlim(60, 102)
    ax.set_xlabel("success rate (%)", color=MUTED)
    ax.set_ylabel("")
    ax.legend(handles=[
        Patch(facecolor=OTHER, edgecolor=OTHER, label="pretrained weights"),
        Patch(facecolor=OTHER, edgecolor=TEXT, hatch="///", alpha=.55, label="from scratch"),
    ], frameon=False, labelcolor=TEXT, loc="lower right", fontsize=8.2, ncol=2)

    method_legend = [
        Line2D([0], [0], marker="o", color=FAMILY_COLORS[name], lw=2,
               markerfacecolor=FAMILY_COLORS[name], markersize=7, label=name)
        for name in ["Q3CIBC", "IBC EBM", "Explicit BC", "Diffusion Policy", "Other"]
    ]
    fig.legend(handles=method_legend, loc="lower center", bbox_to_anchor=(.5, .061),
               ncol=5, frameon=False, labelcolor=TEXT, fontsize=8.7,
               handlelength=2.1, columnspacing=1.7)
    fig.text(.075, .036,
             "Summary. Q3C traces strong low-latency frontiers; refinement yields its largest gain on Kitchen, "
             "while pretrained visual representations improve LIBERO-Goal performance.",
             ha="left", color=TEXT, fontsize=9.1)
    fig.text(.075, .018,
             "Error bars show CSV-reported variability (SD for Pushing; SEM for D4RL). "
             "Source: final comparison CSVs in results/particle and results/hyperparam_search/combinedv2_cpascounter_training.",
             ha="left", color=MUTED, fontsize=7.6)

    fig.savefig(OUT, dpi=220, facecolor=BG, bbox_inches="tight")
    fig.savefig(OUT_PDF, facecolor=BG, bbox_inches="tight")
    print(OUT)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
