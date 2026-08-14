"""Plot Push-T real-robot coverage results from results/pusht/experiments.csv.

Produces seven figures under results/pusht/plots/ plus the matching aggregate
tables as CSV (the tables are the relief for the low-contrast series colors,
and let the numbers be read exactly).

Per start position, per algorithm x inference:
  1. min(cam0, cam1) coverage, pooled over refine_iters
  2. mean(cam0, cam1) coverage, pooled over refine_iters
  3. cam1 coverage, pooled over refine_iters
  4. cam1 coverage at the best refine_iters
  5. dist_centroid at the best refine_iters (best = lowest, it is an error)

Per algorithm x inference, pooling positions:
  6. mean(cam0, cam1) coverage at the best refine_iters
  7. cam1 coverage at the best refine_iters

"Pooled" means trials and refine_iters go into one sample; error bars are 1 std
over that sample. "Best refine_iters" means the cell keeps only the refine_iters
with the best mean, so the std is over trials (and positions, in 6-7) alone.
That selection is optimistic: with 1-4 trials per cell the winning iteration is
partly noise. The CSVs carry the full sweep with an is_best flag.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "pusht" / "experiments.csv"
OUT_DIR = ROOT / "results" / "pusht" / "plots"

# Composite encoding: hue = algorithm, lightness step = inference method within
# that algorithm. So the three Q3C variants read as one family at a glance, and
# DP and IBC read as separate algorithms rather than as two more variants.
#
# The three family hues validate as a categorical set on all pairs (worst CVD
# dE 9.2, aqua vs orange). Each family's steps validate as an ordinal ramp
# (monotone L, adjacent dL >= 0.06, light end clears 2:1 on the light surface).
# The blue steps are 250/450/650 of the documented blue ramp; the orange and
# aqua steps are that ramp's lightness levels re-hued onto each family anchor.
# Categorical checks flag the ramp ends for lightness/chroma by design — those
# bounds govern standalone slots, not steps inside an ordinal ramp.
SERIES_COLORS = {
    ("dp", "ddim"): "#eb6834",  # orange, mid
    ("dp", "ddpm"): "#c94908",  # orange, dark
    ("q3c", "argmax"): "#86b6ef",  # blue 250
    ("q3c", "dfo"): "#2a78d6",  # blue 450
    ("q3c", "langevin"): "#104281",  # blue 650
    ("ibc", "dfo"): "#1baf7a",  # aqua, mid
    ("ibc", "langevin"): "#00582b",  # aqua, dark
}

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dedcd6"

# Fixed series order so a series keeps its hue across every figure. Slots are
# indexed by position in this list, so excluding a series never repaints the
# survivors.
SERIES_ORDER = [
    ("dp", "ddim"),
    ("dp", "ddpm"),
    ("q3c", "argmax"),
    ("q3c", "dfo"),
    ("q3c", "langevin"),
    ("ibc", "dfo"),
    ("ibc", "langevin"),
]

# Series dropped from the figures (still present in experiments.csv).
EXCLUDED_INFERENCE = {"ddpm"}

# IBC has two checkpoints in the CSV and they are far apart (cam1 0.75 for
# Ibc2c_c256_imnet over all 9 positions vs 0.43 for Ibc2c_c256_conv over 5), so
# the figures keep only its best one. dp and q3c also have a second checkpoint
# each, but those are 2 and 14 rows against 52 and 178 and score within 0.06
# cam1 of the main one, so they stay pooled.
KEEP_CHECKPOINT = {"ibc": "Ibc2c_c256_imnet"}


def series_color(series: tuple[str, str]) -> str:
    return SERIES_COLORS[series]

POSITION_ORDER = [
    "top",
    "bottom",
    "left",
    "right",
    "top_right",
    "bottom_right",
    "turned_left",
    "turned_right",
    "upside_down",
]


def series_label(alg: str, inf: str) -> str:
    return f"{alg.upper()} / {inf}"


def load_rows() -> list[dict]:
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["cam0"] = float(r["coverage_cam0"])
        r["cam1"] = float(r["coverage_cam1"])
        r["min_cov"] = min(r["cam0"], r["cam1"])
        r["avg_cov"] = 0.5 * (r["cam0"] + r["cam1"])
        r["dist"] = float(r["dist_centroid"])
    return rows


def style_axes(ax, top: float | None = None) -> None:
    """Coverage axes (top=None) are fixed 0-1; other metrics scale to `top`."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    if top is None:
        ax.set_ylim(0, 1.25)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
    else:
        ax.set_ylim(0, top)


def grouped_plot(
    rows,
    metric: str,
    title: str,
    subtitle: str,
    stem: str,
    best_iters: bool = False,
    ylabel: str = "coverage",
    value_fmt: str = "{:.2f}",
    higher_is_better: bool = True,
) -> None:
    """Grouped bars: x = start position, one bar per algorithm x inference.

    With best_iters=False every refine_iters value of a cell is pooled. With
    best_iters=True the cell keeps only its best refine_iters (highest mean, or
    lowest when higher_is_better=False), so mean/std are over trials alone; the
    full sweep still goes to the table.
    """
    positions = [p for p in POSITION_ORDER if any(r["start_position"] == p for r in rows)]
    series = [s for s in SERIES_ORDER if any((r["algorithm"], r["inference"]) == s for r in rows)]

    table: list[dict] = []
    means = np.full((len(series), len(positions)), np.nan)
    stds = np.zeros_like(means)
    chosen_iters: dict[tuple[int, int], int] = {}

    for i, (alg, inf) in enumerate(series):
        for j, pos in enumerate(positions):
            cell = [
                r
                for r in rows
                if r["algorithm"] == alg and r["inference"] == inf and r["start_position"] == pos
            ]
            if not cell:
                continue

            per_iter: dict[int, list[float]] = {}
            for r in cell:
                per_iter.setdefault(int(r["refine_iters"]), []).append(r[metric])

            if best_iters:
                pick = max if higher_is_better else min
                best_it = pick(per_iter, key=lambda it: float(np.mean(per_iter[it])))
                groups = {best_it: per_iter[best_it]}
                chosen_iters[(i, j)] = best_it
                for it, vals in sorted(per_iter.items()):
                    table.append(
                        {
                            "algorithm": alg,
                            "inference": inf,
                            "start_position": pos,
                            "refine_iters": it,
                            "metric": metric,
                            "mean": round(float(np.mean(vals)), 4),
                            "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                            "n_trials": len(vals),
                            "is_best": "yes" if it == best_it else "",
                        }
                    )
            else:
                groups = {None: [r[metric] for r in cell]}

            vals = next(iter(groups.values()))
            means[i, j] = float(np.mean(vals))
            stds[i, j] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if not best_iters:
                table.append(
                    {
                        "algorithm": alg,
                        "inference": inf,
                        "start_position": pos,
                        "metric": metric,
                        "mean": round(means[i, j], 4),
                        "std": round(stds[i, j], 4),
                        "n_runs": len(vals),
                    }
                )

    is_coverage = ylabel == "coverage"
    # Cap labels/bars just above the tallest error bar so nothing clips.
    cap = float(np.nanmax(means + stds))
    label_cap = 1.0 if is_coverage else cap
    label_pad = 0.02 if is_coverage else 0.02 * cap

    fig, ax = plt.subplots(figsize=(13, 5.6), facecolor=SURFACE)
    style_axes(ax, top=None if is_coverage else cap * 1.28)

    n = len(series)
    group_w = 0.82
    bar_w = group_w / n
    x = np.arange(len(positions))

    for i, (alg, inf) in enumerate(series):
        offset = -group_w / 2 + bar_w * (i + 0.5)
        vals = np.nan_to_num(means[i], nan=0.0)
        present = ~np.isnan(means[i])
        ax.bar(
            x + offset,
            vals,
            width=bar_w * 0.88,  # 2px-equivalent surface gap between adjacent bars
            color=series_color((alg, inf)),
            label=series_label(alg, inf),
            zorder=2,
        )
        ax.errorbar(
            x[present] + offset,
            means[i][present],
            yerr=stds[i][present],
            fmt="none",
            ecolor=TEXT_SECONDARY,
            elinewidth=1.2,
            capsize=3,
            zorder=3,
        )
        for j in np.flatnonzero(present):
            ax.text(
                x[j] + offset,
                min(means[i, j] + stds[i, j], label_cap) + label_pad,
                (
                    value_fmt.format(means[i, j]) + f" · it {chosen_iters[(i, j)]}"
                    if best_iters
                    else value_fmt.format(means[i, j])
                ),
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=TEXT_SECONDARY,
                rotation=90,
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in positions], color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, loc="left", pad=18, weight="bold")
    ax.text(
        0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        color=TEXT_SECONDARY,
        fontsize=9,
        va="bottom",
    )
    leg = ax.legend(
        frameon=False,
        ncols=len(series),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        fontsize=9,
    )
    for txt in leg.get_texts():
        txt.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    write_table(OUT_DIR / f"{stem}.csv", table)


def best_iters_plot(rows, stem: str, metric: str, title: str, ylabel: str) -> None:
    """Bars: x = algorithm x inference, best refine_iters by mean of `metric`.

    Trials and start positions are pooled, so the std mixes trial noise with
    per-position difficulty (the position sets are not matched across series).
    """
    series = [s for s in SERIES_ORDER if any((r["algorithm"], r["inference"]) == s for r in rows)]

    labels, means, stds, best_iters, ns = [], [], [], [], []
    table: list[dict] = []

    for alg, inf in series:
        sub = [r for r in rows if r["algorithm"] == alg and r["inference"] == inf]
        per_iter = {}
        for r in sub:
            per_iter.setdefault(int(r["refine_iters"]), []).append(r[metric])
        # Full sweep goes to the table; the plot shows the argmax setting.
        for it, vals in sorted(per_iter.items()):
            table.append(
                {
                    "algorithm": alg,
                    "inference": inf,
                    "refine_iters": it,
                    "metric": metric,
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                    "n_runs": len(vals),
                    "is_best": "",
                }
            )
        best_it = max(per_iter, key=lambda it: float(np.mean(per_iter[it])))
        vals = per_iter[best_it]
        for row in table:
            if (
                row["algorithm"] == alg
                and row["inference"] == inf
                and row["refine_iters"] == best_it
            ):
                row["is_best"] = "yes"
        labels.append(series_label(alg, inf))
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        best_iters.append(best_it)
        ns.append(len(vals))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(labels))

    ax.bar(x, means, width=0.56, color=[series_color(s) for s in series], zorder=2)
    ax.errorbar(
        x, means, yerr=stds, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1.4, capsize=4, zorder=3
    )
    for xi, m, s, it, n in zip(x, means, stds, best_iters, ns):
        ax.text(
            xi,
            min(m + s, 1.0) + 0.03,
            f"{m:.2f}\niters {it} (n={n})",
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT_SECONDARY,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_title(
        title,
        color=TEXT_PRIMARY,
        fontsize=13,
        loc="left",
        pad=18,
        weight="bold",
    )
    ax.text(
        0,
        1.02,
        f"argmax over the refine_iters sweep of mean {ylabel}; "
        "error bars 1 std over trials and positions",
        transform=ax.transAxes,
        color=TEXT_SECONDARY,
        fontsize=9,
        va="bottom",
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    write_table(OUT_DIR / f"{stem}.csv", table)


def write_table(path: Path, table: list[dict]) -> None:
    if not table:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    def keep(r: dict) -> bool:
        if r["inference"] in EXCLUDED_INFERENCE:
            return False
        wanted = KEEP_CHECKPOINT.get(r["algorithm"])
        return wanted is None or r["seed_dir"].rsplit("/", 1)[-1] == wanted

    rows = [r for r in load_rows() if keep(r)]

    pooled = "mean over trials and refinement iterations; error bars 1 std over that pool"
    grouped_plot(
        rows,
        "min_cov",
        "Worst-camera coverage by start position",
        "min(cam0, cam1) — " + pooled,
        "coverage_min_by_position",
    )
    grouped_plot(
        rows,
        "avg_cov",
        "Mean-camera coverage by start position",
        "mean(cam0, cam1) — " + pooled,
        "coverage_avg_by_position",
    )
    grouped_plot(
        rows,
        "cam1",
        "Cam1 coverage by start position",
        "coverage_cam1 — " + pooled,
        "coverage_cam1_by_position",
    )
    grouped_plot(
        rows,
        "cam1",
        "Cam1 coverage by start position, best refinement setting",
        "coverage_cam1 — per cell, refine_iters with the highest mean; error bars 1 std over trials",
        "coverage_cam1_by_position_best_iters",
        best_iters=True,
    )
    grouped_plot(
        rows,
        "dist",
        "Centroid distance by start position, best refinement setting",
        "dist_centroid, lower is better — per cell, refine_iters with the lowest mean; "
        "error bars 1 std over trials",
        "dist_centroid_by_position_best_iters",
        best_iters=True,
        ylabel="centroid distance",
        value_fmt="{:.1f}",
        higher_is_better=False,
    )
    best_iters_plot(
        rows,
        "coverage_avg_best_iters",
        metric="avg_cov",
        title="Best refinement setting per method",
        ylabel="mean(cam0, cam1) coverage",
    )
    best_iters_plot(
        rows,
        "coverage_cam1_best_iters",
        metric="cam1",
        title="Best refinement setting per method, cam1",
        ylabel="cam1 coverage",
    )

    print(f"wrote 7 figures + tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
