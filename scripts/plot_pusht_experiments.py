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

matplotlib.rcParams["hatch.linewidth"] = 0.9

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "pusht" / "experiments.csv"
SPEED_PATH = ROOT / "results" / "pusht" / "inference_speed.csv"
DEVICE = "cuda"
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
    ("q3c", "argmax_fallback"): "#86b6ef",  # blue 250 + hatch
    ("q3c", "dfo"): "#2a78d6",  # blue 450
    ("q3c", "langevin"): "#104281",  # blue 650
    ("ibc", "dfo"): "#1baf7a",  # aqua, mid
    ("ibc", "langevin"): "#00582b",  # aqua, dark
}

# argmax and its DFO-fallback variant share a step and separate by texture. A
# fourth blue step is not available: the ramp's usable range cannot hold four
# steps that all clear the normal-vision floor while the light end still clears
# 2:1 on the surface (widest 4-step spread measures dE 9.8, against a floor of
# 15). Texture is the documented channel for exactly this case, and it carries
# the meaning better anyway - fallback is a variant of argmax, not a peer.
SERIES_HATCH = {("q3c", "argmax_fallback"): "///"}
HATCH_EDGE = "#104281"

# Inference labels that differ from the raw CSV value.
INFERENCE_LABEL = {"argmax_fallback": "argmax/fallback"}

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
    ("q3c", "argmax_fallback"),
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


def checkpoint_kept(r: dict) -> bool:
    wanted = KEEP_CHECKPOINT.get(r["algorithm"])
    return wanted is None or r["seed_dir"].rsplit("/", 1)[-1] == wanted

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
    return f"{alg.upper()} · {INFERENCE_LABEL.get(inf, inf)}"


def load_rows() -> list[dict]:
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["cam0"] = float(r["coverage_cam0"])
        r["cam1"] = float(r["coverage_cam1"])
        r["min_cov"] = min(r["cam0"], r["cam1"])
        r["avg_cov"] = 0.5 * (r["cam0"] + r["cam1"])
        r["dist"] = float(r["dist_centroid"])
        # Q3C argmax runs with refinement iterations are the DFO-fallback
        # variant (argmax, with DFO taking over when the argmax action stalls),
        # so they get their own series rather than being averaged into argmax.
        if (r["algorithm"], r["inference"]) == ("q3c", "argmax") and int(r["refine_iters"]) > 0:
            r["inference"] = "argmax_fallback"
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
            hatch=SERIES_HATCH.get((alg, inf)),
            edgecolor=HATCH_EDGE if (alg, inf) in SERIES_HATCH else "none",
            linewidth=0,
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

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(labels))

    bars = ax.bar(x, means, width=0.56, color=[series_color(s) for s in series], zorder=2)
    for bar, s in zip(bars, series):
        if s in SERIES_HATCH:
            bar.set_hatch(SERIES_HATCH[s])
            bar.set_edgecolor(HATCH_EDGE)
            bar.set_linewidth(0)
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
    ax.set_xticklabels([lb.replace(" · ", "\n") for lb in labels], color=TEXT_SECONDARY)
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


def speed_tradeoff_plot(rows, stem: str, metric: str = "cam1", ylabel: str = "cam1 coverage") -> None:
    """Scatter: x = mean inference time per step, y = task performance.

    One point per (algorithm, inference, refine_iters); points of the same
    series are joined in iteration order, so each line is that method's
    cost/quality sweep. Up and to the left is better.

    Timings come from inference_speed.csv and are restricted to CUDA — the CPU
    rows are ~10x slower for the same config, and two devices cannot share one
    axis. Configs without a timing (or without a rollout) are dropped.
    """
    with SPEED_PATH.open() as f:
        speed = [r for r in csv.DictReader(f) if r["device"] == DEVICE]
    speed = [r for r in speed if checkpoint_kept(r)]

    def cell(r: dict) -> tuple[str, str, int]:
        return (r["algorithm"], r["inference"], int(r["refine_iters"]))

    times: dict[tuple[str, str, int], list[float]] = {}
    for r in speed:
        # Same argmax/fallback split the rollout CSV gets, applied to timings.
        if (r["algorithm"], r["inference"]) == ("q3c", "argmax") and int(r["refine_iters"]) > 0:
            r["inference"] = "argmax_fallback"
        times.setdefault(cell(r), []).append(float(r["ms_per_step"]))

    scores: dict[tuple[str, str, int], list[float]] = {}
    for r in rows:
        scores.setdefault(cell(r), []).append(r[metric])

    table: list[dict] = []
    points: dict[tuple[str, str], list[tuple]] = {}
    for k in sorted(set(times) & set(scores)):
        alg, inf, iters = k
        t, s = times[k], scores[k]
        sem = float(np.std(s, ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0
        points.setdefault((alg, inf), []).append((float(np.mean(t)), float(np.mean(s)), sem, iters))
        table.append(
            {
                "algorithm": alg,
                "inference": inf,
                "refine_iters": iters,
                "ms_per_step": round(float(np.mean(t)), 2),
                "n_timing_runs": len(t),
                "metric": metric,
                "mean": round(float(np.mean(s)), 4),
                "sem": round(sem, 4),
                "n_rollouts": len(s),
            }
        )

    fig, ax = plt.subplots(figsize=(9.5, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)

    # Most configs land in a 8-12 ms cluster with DP's sweep out at 50 ms, so a
    # log x spreads the cluster instead of pinning it against the axis.
    ax.set_xscale("log")
    ax.set_xticks([8, 10, 12, 15, 20, 30, 50])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    lo = min(p[0] for pts in points.values() for p in pts)
    hi = max(p[0] for pts in points.values() for p in pts)
    ax.set_xlim(lo * 0.88, hi * 1.15)

    # Alternate the iteration labels above/below so the cluster stays legible.
    label_offsets = [(0, 12), (0, -17), (-14, -4), (15, 3), (0, -17), (0, 12), (0, 12)]

    for series in SERIES_ORDER:
        if series not in points:
            continue
        offset = label_offsets[SERIES_ORDER.index(series) % len(label_offsets)]
        pts = sorted(points[series], key=lambda p: p[3])
        xs, ys, sems, its = zip(*pts)
        color = series_color(series)
        marker = "D" if series in SERIES_HATCH else "o"
        if len(xs) > 1:
            ax.plot(xs, ys, color=color, linewidth=2, zorder=2, alpha=0.9)
        ax.errorbar(
            xs, ys, yerr=sems, fmt="none", ecolor=color, elinewidth=1.2, capsize=3, zorder=3
        )
        ax.plot(
            xs,
            ys,
            marker,
            color=color,
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            linestyle="none",
            label=series_label(*series),
            zorder=4,
        )
        for x, y, _, it in pts:
            ax.annotate(
                f"{it}",
                (x, y),
                textcoords="offset points",
                xytext=offset,
                ha="center",
                fontsize=7.5,
                color=TEXT_SECONDARY,
                zorder=5,
            )

    ax.set_xlabel(f"mean inference time per step (ms, {DEVICE})", color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_title(
        "Cost vs performance",
        color=TEXT_PRIMARY,
        fontsize=13,
        loc="left",
        pad=18,
        weight="bold",
    )
    ax.text(
        0,
        1.02,
        "point label = refinement iterations; error bars 1 SEM over rollouts; up and left is better",
        transform=ax.transAxes,
        color=TEXT_SECONDARY,
        fontsize=9,
        va="bottom",
    )
    leg = ax.legend(
        frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, handletextpad=0.6
    )
    for txt in leg.get_texts():
        txt.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
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
    rows = [
        r
        for r in load_rows()
        if r["inference"] not in EXCLUDED_INFERENCE and checkpoint_kept(r)
    ]

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

    speed_tradeoff_plot(rows, "cam1_vs_inference_time")

    print(f"wrote 8 figures + tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
