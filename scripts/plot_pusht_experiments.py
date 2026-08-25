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

# A timing run whose whole record is a handful of inferences is dominated by the
# first one, which carries CUDA warmup (~220 ms against a ~11 ms steady state).
# Those runs are not an estimate of anything, so they are dropped rather than
# averaged in: with them, q3c/argmax at horizon 4 reads 10.29 ms/step against
# 5.61 at horizon 2; without them it reads 2.78 and the sweep is monotone.
MIN_INFER_RUNS = 5

# A cost/quality point is only comparable to its neighbours if it was rolled out
# on the same task distribution. Points covering fewer start positions than this
# are dropped rather than plotted: the IBC ch16 horizon-1 cell is 3 rollouts all
# at `top` (the easiest position), which read as cam1 0.96 against sweep-mates
# that each cover all 9 positions.
MIN_POSITIONS = 8

# The IBC ch16 checkpoint was never rolled out at horizon 1 on the full position
# set, so its horizon sweep has no anchor. This borrows that one point from the
# sibling c256 checkpoint (same algorithm, inference and refinement, 25 rollouts
# over all 9 positions) and plots it as an ordinary point of the series. The
# substitution is recorded in the CSV's checkpoint column, not in the mark.
HORIZON_ANCHOR = {("ibc", "dfo", 5, "Ibc2c16_c256_imnet"): {1: "Ibc2c_c256_imnet"}}
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
    ("dp", "ddpm"): "#983000",  # orange, dark
    ("q3c", "argmax"): "#86b6ef",  # blue 250
    ("q3c", "argmax_fallback"): "#86b6ef",  # blue 250 + hatch
    ("q3c", "dfo"): "#2a78d6",  # blue 450
    ("q3c", "langevin"): "#104281",  # blue 650
    ("ibc", "dfo"): "#1baf7a",  # aqua, mid
    ("ibc", "langevin"): "#00582b",  # aqua, dark
    ("bc", "deterministic"): "#d4319b",  # magenta
}

# BC's magenta was picked by sweeping OKLCH hue x lightness x chroma and keeping
# only steps that clear the all-pairs gates against the whole plotted set. It is
# never the limiting pair: worst all-pairs stays aqua<->orange at CVD dE 9.2 and
# blue650<->blue450 at normal-vision dE 19.5, both pre-existing. Violet was the
# first pick but read as another cool step beside the three Q3C blues; the
# documented palette's yellow, magenta and red steps all fail beside orange.

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
    ("bc", "deterministic"),
]

# Series dropped from the figures (still present in experiments.csv). DDPM was
# excluded while it was 3 rollouts at one position; the i02 checkpoint now
# carries a full 100-iteration sweep over all 4 horizons and all 9 positions, so
# it is back in.
EXCLUDED_INFERENCE: set[str] = set()

# IBC has two checkpoints in the CSV and they are far apart (cam1 0.75 for
# Ibc2c_c256_imnet over all 9 positions vs 0.43 for Ibc2c_c256_conv over 5), so
# the figures keep only its best one. dp and q3c also have a second checkpoint
# each, but those are 2 and 14 rows against 52 and 178 and score within 0.06
# cam1 of the main one, so they stay pooled.
# DP now spans five checkpoints, of which only g01 (1-camera, horizon 1 only)
# and i02 (2-camera, the full refine_iters x horizon grid) have complete
# position coverage. Pooling them would average a 1-cam and a 2-cam model into
# one "DP" bar - they differ by up to 0.11 cam1 at the same setting - so the
# figures keep i02, the one the new sweeps were collected on.
KEEP_CHECKPOINT = {
    "ibc": {"Ibc2c_c256_imnet", "Ibc2c16_c256_imnet"},
    "dp": {"i02_resnet18_2cam_k16_s29_175k"},
}


def series_color(series: tuple[str, str]) -> str:
    return SERIES_COLORS[series]


def checkpoint_kept(r: dict) -> bool:
    wanted = KEEP_CHECKPOINT.get(r["algorithm"])
    return wanted is None or r["seed_dir"].rsplit("/", 1)[-1] in wanted


def checkpoint_of(r: dict) -> str:
    return r["seed_dir"].rsplit("/", 1)[-1]


def config_of(r: dict) -> tuple[str, str, int, int]:
    return (r["algorithm"], r["inference"], int(r["refine_iters"]), r["horizon"])


def eligible_configs(rows) -> set[tuple[str, str, int, int]]:
    """Configs rolled out on enough start positions to be picked as a `best`.

    Without this the argmax reaches into partially-collected sweeps: the DP
    h01 checkpoint currently has cells of 2 rollouts at a single position, one
    of which reads cam1 0.97 and would outrank every fully-swept config.
    """
    seen: dict[tuple[str, str, int, int], set[str]] = {}
    for r in rows:
        seen.setdefault(config_of(r), set()).add(r["start_position"])
    return {c for c, pos in seen.items() if len(pos) >= MIN_POSITIONS}

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
        r["horizon"] = int(r["exec_horizon"])
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

    With best_iters=False every (refine_iters, exec_horizon) setting of a cell
    is pooled. With best_iters=True the cell keeps only its best setting - the
    joint argmax over both knobs (argmin when higher_is_better=False) - so
    mean/std are over trials alone; the full sweep still goes to the table.
    """
    positions = [p for p in POSITION_ORDER if any(r["start_position"] == p for r in rows)]
    series = [s for s in SERIES_ORDER if any((r["algorithm"], r["inference"]) == s for r in rows)]

    table: list[dict] = []
    means = np.full((len(series), len(positions)), np.nan)
    stds = np.zeros_like(means)
    chosen: dict[tuple[int, int], tuple[int, int]] = {}

    for i, (alg, inf) in enumerate(series):
        for j, pos in enumerate(positions):
            cell = [
                r
                for r in rows
                if r["algorithm"] == alg and r["inference"] == inf and r["start_position"] == pos
            ]
            if not cell:
                continue

            per_cfg: dict[tuple[int, int], list[float]] = {}
            ckpts: dict[tuple[int, int], set[str]] = {}
            for r in cell:
                cfg = (int(r["refine_iters"]), r["horizon"])
                per_cfg.setdefault(cfg, []).append(r[metric])
                ckpts.setdefault(cfg, set()).add(checkpoint_of(r))

            if best_iters:
                pick = max if higher_is_better else min
                # Every config competes here: a bar is one start position, so a
                # config that only ran on a few positions is still a valid
                # measurement at the positions it did run. The position-coverage
                # guard belongs to the figures that pool positions.
                best_cfg = pick(per_cfg, key=lambda c: float(np.mean(per_cfg[c])))
                groups = {best_cfg: per_cfg[best_cfg]}
                chosen[(i, j)] = best_cfg
                for cfg, vals in sorted(per_cfg.items()):
                    table.append(
                        {
                            "algorithm": alg,
                            "inference": inf,
                            "start_position": pos,
                            "refine_iters": cfg[0],
                            "exec_horizon": cfg[1],
                            "checkpoint": "|".join(sorted(ckpts[cfg])),
                            "metric": metric,
                            "mean": round(float(np.mean(vals)), 4),
                            "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                            "n_trials": len(vals),
                            "is_best": "yes" if cfg == best_cfg else "",
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
                    value_fmt.format(means[i, j])
                    + f" · it{chosen[(i, j)][0]} h{chosen[(i, j)][1]}"
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


def best_iters_plot(
    rows,
    stem: str,
    metric: str,
    title: str,
    ylabel: str,
    eligible: set | None = None,
    exclude_positions: set[str] | None = None,
    position_balanced: bool = False,
) -> None:
    """Bars: x = algorithm x inference, best (refine_iters, exec_horizon) setting.

    With position_balanced=False trials and start positions are pooled, so the
    std mixes trial noise with per-position difficulty. With it True each
    position is averaged first and the bar is the mean of those per-position
    means, so a cell with extra rollouts on an easy position cannot outweigh the
    rest; the std is then taken across positions.
    """
    drop = exclude_positions or set()
    if drop:
        # Dropping a position changes which settings count as fully swept, so
        # eligibility is recomputed against what is left.
        rows = [r for r in rows if r["start_position"] not in drop]
        eligible = eligible_configs(rows)
    series = [s for s in SERIES_ORDER if any((r["algorithm"], r["inference"]) == s for r in rows)]

    labels, means, stds, best_cfgs, ns = [], [], [], [], []
    kept: list[tuple[str, str]] = []  # series that actually get a bar
    table: list[dict] = []

    for alg, inf in series:
        sub = [r for r in rows if r["algorithm"] == alg and r["inference"] == inf]
        per_cfg: dict[tuple[int, int], dict[str, list[float]]] = {}
        ckpts = {}
        for r in sub:
            cfg = (int(r["refine_iters"]), r["horizon"])
            per_cfg.setdefault(cfg, {}).setdefault(r["start_position"], []).append(r[metric])
            ckpts.setdefault(cfg, set()).add(checkpoint_of(r))

        def score(by_pos):
            if position_balanced:
                return float(np.mean([np.mean(v) for v in by_pos.values()]))
            return float(np.mean([x for v in by_pos.values() for x in v]))

        def spread(by_pos):
            sample = (
                [float(np.mean(v)) for v in by_pos.values()]
                if position_balanced
                else [x for v in by_pos.values() for x in v]
            )
            return float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0

        pool = [c for c in per_cfg if eligible is None or (alg, inf) + c in eligible]
        if not pool:
            continue
        best_cfg = max(pool, key=lambda c: score(per_cfg[c]))
        # Full sweep goes to the table; the plot shows the argmax setting.
        for cfg, by_pos in sorted(per_cfg.items()):
            sd = spread(by_pos)
            table.append(
                {
                    "algorithm": alg,
                    "inference": inf,
                    "refine_iters": cfg[0],
                    "exec_horizon": cfg[1],
                    "checkpoint": "|".join(sorted(ckpts[cfg])),
                    "metric": metric,
                    "mean": round(score(by_pos), 4),
                    "std": round(sd, 4),
                    "sem": round(sd / np.sqrt(len(by_pos)), 4) if position_balanced else "",
                    "n_positions": len(by_pos),
                    "n_runs": sum(len(v) for v in by_pos.values()),
                    "eligible": ""
                    if eligible is not None and (alg, inf) + cfg not in eligible
                    else "yes",
                    "is_best": "yes" if cfg == best_cfg else "",
                }
            )
        by_pos = per_cfg[best_cfg]
        kept.append((alg, inf))
        labels.append(series_label(alg, inf))
        means.append(score(by_pos))
        stds.append(spread(by_pos))
        best_cfgs.append(best_cfg)
        ns.append(sum(len(v) for v in by_pos.values()))

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(labels))

    bars = ax.bar(x, means, width=0.56, color=[series_color(s) for s in kept], zorder=2)
    for bar, s in zip(bars, kept):
        if s in SERIES_HATCH:
            bar.set_hatch(SERIES_HATCH[s])
            bar.set_edgecolor(HATCH_EDGE)
            bar.set_linewidth(0)
    ax.errorbar(
        x, means, yerr=stds, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1.4, capsize=4, zorder=3
    )
    for xi, m, s, cfg, n in zip(x, means, stds, best_cfgs, ns):
        ax.text(
            xi,
            min(m + s, 1.0) + 0.03,
            f"{m:.2f}\nit {cfg[0]} · h {cfg[1]}",
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
        pad=30,
        weight="bold",
    )
    ax.text(
        0,
        1.02,
        f"argmax over the refine_iters x exec_horizon sweep of {ylabel}\n"
        + (
            "bar = mean of the per-position means; error bars 1 std across positions"
            if position_balanced
            else "error bars 1 std over trials and positions"
        )
        + (f"\nexcluded: {', '.join(sorted(drop)).replace('_', ' ')}" if drop else ""),
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

    One point per (algorithm, inference, refine_iters) at exec_horizon 1;
    points of the same series are joined in iteration order, so each line is
    that method's cost/quality sweep. Up and to the left is better. The horizon
    sweeps get their own figure - mixing them in here would put two different
    knobs on one line.

    Timings come from inference_speed.csv and are restricted to CUDA — the CPU
    rows are ~10x slower for the same config, and two devices cannot share one
    axis. Configs without a timing (or without a rollout) are dropped.
    """
    with SPEED_PATH.open() as f:
        speed = [r for r in csv.DictReader(f) if r["device"] == DEVICE]
    speed = [
        r
        for r in speed
        if checkpoint_kept(r) and int(r["exec_horizon"]) == 1 and int(r["n_infer"]) >= MIN_INFER_RUNS
    ]
    rows = [r for r in rows if r["horizon"] == 1]

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
    label_offsets = [(0, 12), (0, -17), (-14, -4), (15, 3), (0, -17), (0, 12), (0, 12), (-14, -4)]

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
        "Cost vs performance — refinement sweep",
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


# No checkpoint currently needs its own step: IBC contributes a single horizon
# line, and its dark-aqua override collided with the DP ddpm step under protan
# simulation (dE 4.5). The checkpoint is still named in the legend via
# CHECKPOINT_TAG.
CHECKPOINT_COLOR: dict[str, str] = {}
CHECKPOINT_TAG = {"Ibc2c16_c256_imnet": "ch16"}


def horizon_tradeoff_plot(
    rows,
    stem: str,
    metric: str = "cam1",
    ylabel: str = "cam1 coverage",
    title: str = "Cost vs performance — execution-horizon sweep",
    show_error: bool = True,
) -> None:
    """Scatter: x = mean inference time per step, y = performance, over horizon.

    A series is one (algorithm, inference, refine_iters, checkpoint) combination
    that was swept over more than one exec_horizon; it is drawn as a line with a
    dot per horizon, joined shortest horizon to longest. Configs that exist at
    only one horizon are not plotted, nor are points covering fewer than
    MIN_POSITIONS start positions. Each point is position-balanced (mean of the
    per-position means) and its error bar is 1 SEM across positions. CUDA
    timings only.
    """
    with SPEED_PATH.open() as f:
        speed = [r for r in csv.DictReader(f) if r["device"] == DEVICE]
    speed = [r for r in speed if checkpoint_kept(r) and int(r["n_infer"]) >= MIN_INFER_RUNS]

    def cfg(r: dict) -> tuple:
        inf = r["inference"]
        if (r["algorithm"], inf) == ("q3c", "argmax") and int(r["refine_iters"]) > 0:
            inf = "argmax_fallback"
        return (r["algorithm"], inf, int(r["refine_iters"]), checkpoint_of(r), int(r["exec_horizon"]))

    times: dict[tuple, list[float]] = {}
    for r in speed:
        times.setdefault(cfg(r), []).append(float(r["ms_per_step"]))
    # Keep the rollouts per position so each point can be position-balanced -
    # an unweighted mean would let a cell with extra easy-position rollouts
    # outscore one with the same policy and a harder mix.
    scores: dict[tuple, dict[str, list[float]]] = {}
    for r in rows:
        scores.setdefault(cfg(r), {}).setdefault(r["start_position"], []).append(r[metric])

    series: dict[tuple, list[tuple]] = {}
    table: list[dict] = []
    joint = set(times) & set(scores)
    covered = {k for k in joint if len(scores[k]) >= MIN_POSITIONS}
    swept_cfgs = {c for c in {k[:4] for k in covered} if len({k[4] for k in covered if k[:4] == c}) > 1}
    for k in sorted(joint):
        alg, inf, iters, ckpt, horizon = k
        if k[:4] not in swept_cfgs:
            continue
        by_pos = scores[k]
        per_pos = [float(np.mean(v)) for _, v in sorted(by_pos.items())]
        n_roll = sum(len(v) for v in by_pos.values())
        mean = float(np.mean(per_pos))
        sem = float(np.std(per_pos, ddof=1) / np.sqrt(len(per_pos))) if len(per_pos) > 1 else 0.0
        row = {
            "algorithm": alg,
            "inference": inf,
            "refine_iters": iters,
            "exec_horizon": horizon,
            "checkpoint": ckpt,
            "ms_per_step": round(float(np.mean(times[k])), 2),
            "n_timing_runs": len(times[k]),
            "metric": metric,
            "mean": round(mean, 4),
            "sem": round(sem, 4),
            "n_rollouts": n_roll,
            "n_positions": len(per_pos),
            "plotted": "yes" if k in covered else "",
        }
        table.append(row)
        if k in covered:
            series.setdefault((alg, inf, iters, ckpt), []).append(
                (float(np.mean(times[k])), mean, sem, horizon, False)
            )

    # Splice in the borrowed anchor points.
    for cfg_key, anchors in HORIZON_ANCHOR.items():
        if cfg_key not in series:
            continue
        alg, inf, iters, _ = cfg_key
        for horizon, src_ckpt in anchors.items():
            if any(pt[3] == horizon for pt in series[cfg_key]):
                continue
            src = (alg, inf, iters, src_ckpt, horizon)
            if src not in covered:
                continue
            per_pos = [float(np.mean(v)) for _, v in sorted(scores[src].items())]
            mean = float(np.mean(per_pos))
            sem = float(np.std(per_pos, ddof=1) / np.sqrt(len(per_pos)))
            series[cfg_key].append((float(np.mean(times[src])), mean, sem, horizon, True))
            table.append(
                {
                    "algorithm": alg,
                    "inference": inf,
                    "refine_iters": iters,
                    "exec_horizon": horizon,
                    "checkpoint": f"{src_ckpt} (borrowed by {cfg_key[3]})",
                    "ms_per_step": round(float(np.mean(times[src])), 2),
                    "n_timing_runs": len(times[src]),
                    "metric": metric,
                    "mean": round(mean, 4),
                    "sem": round(sem, 4),
                    "n_rollouts": sum(len(v) for v in scores[src].values()),
                    "n_positions": len(per_pos),
                    "plotted": "yes (spliced)",
                }
            )

    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 10, 20, 50])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    allpts = [p for pts in series.values() for p in pts]  # noqa: F841
    ax.set_xlim(min(p[0] for p in allpts) * 0.8, max(p[0] for p in allpts) * 1.25)
    pad = (lambda p: p[2]) if show_error else (lambda p: 0.0)
    lo = min(p[1] - pad(p) for p in allpts)
    hi = max(p[1] + pad(p) for p in allpts)
    ax.set_ylim(lo - 0.06, hi + 0.06)

    # Greedy label placement: try right, left, above, below, take the first slot
    # that clears everything already placed.
    placed: list[tuple[float, float]] = []

    def place(ax, text, xy):
        for dx, dy in ((10, -3), (-10, -3), (0, 11), (0, -17)):
            px, py = ax.transData.transform(xy)
            cand = (px + dx * 2.2, py + dy * 2.2)
            if all(abs(cand[0] - q[0]) > 34 or abs(cand[1] - q[1]) > 26 for q in placed):
                placed.append(cand)
                ha = "left" if dx > 0 else "right" if dx < 0 else "center"
                ax.annotate(
                    text,
                    xy,
                    textcoords="offset points",
                    xytext=(dx, dy),
                    ha=ha,
                    fontsize=8,
                    color=TEXT_SECONDARY,
                    zorder=6,
                )
                return
        placed.append(ax.transData.transform(xy))
        ax.annotate(
            text,
            xy,
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=8,
            color=TEXT_SECONDARY,
            zorder=6,
        )

    def sort_key(item):
        (alg, inf, iters, ckpt), pts = item
        return (SERIES_ORDER.index((alg, inf)), iters)

    # A family can now contribute several lines (DP sweeps horizons at 5, 10 and
    # 25 refinement iterations), which would all wear the same hue. Dash pattern
    # separates them inside the family; the legend names the iteration count.
    dashes = ["-", "--", ":", "-."]
    family_rank: dict[tuple[str, str], int] = {}

    for (alg, inf, iters, ckpt), pts in sorted(series.items(), key=sort_key):
        rank = family_rank.get((alg, inf), 0)
        family_rank[(alg, inf)] = rank + 1
        pts = sorted(pts, key=lambda p: p[3])
        xs, ys, sems, hs, spliced = zip(*pts)
        color = CHECKPOINT_COLOR.get(ckpt, series_color((alg, inf)))
        marker = "D" if (alg, inf) in SERIES_HATCH else "o"
        tag = f" [{CHECKPOINT_TAG[ckpt]}]" if ckpt in CHECKPOINT_TAG else ""
        label = f"{series_label(alg, inf)} · {iters} it{tag}"
        if show_error:
            ax.errorbar(
                xs, ys, yerr=sems, fmt="none", ecolor=color, elinewidth=1.2, capsize=3, zorder=3
            )
        # Line and markers in one call so the legend handle carries the dash
        # pattern - a family can contribute several lines in the same hue.
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            alpha=0.9,
            linestyle=dashes[rank % len(dashes)],
            marker=marker,
            markersize=10,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=label,
            zorder=5,
        )
        for x, y, _, h, _sp in sorted(pts, key=lambda q: -q[1]):
            place(ax, f"h{h}", (x, y))

    ax.set_xlabel(f"mean inference time per step (ms, {DEVICE})", color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_title(
        title,
        color=TEXT_PRIMARY,
        fontsize=13,
        loc="left",
        pad=32,
        weight="bold",
    )
    ax.text(
        0,
        1.03,
        "lines join a config's horizons shortest to longest (point label = horizon)\n"
        "position-balanced over all 9 start positions"
        + (
            "; error bars 1 SEM across positions"
            if show_error
            else "; uncertainty omitted — see the error-bar version before comparing points"
        ),
        transform=ax.transAxes,
        color=TEXT_SECONDARY,
        fontsize=9,
        va="bottom",
    )
    leg = ax.legend(
        frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, handletextpad=0.6
    )
    for txt in leg.get_texts():
        txt.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    write_table(OUT_DIR / f"{stem}.csv", table)


def baseline_delta_plot(
    rows,
    stem: str,
    metric: str = "cam1",
    xlabel: str = "Δ cam1 coverage vs BC baseline",
    baseline_alg: str = "bc",
) -> None:
    """Forest plot of per-position paired differences against a BC baseline.

    Start position explains more of the spread than the algorithm does, so the
    absolute bars carry error bars too wide to separate anything. This blocks on
    position: every config is compared to the baseline *within* each position,
    and the 9 differences are then averaged.

    The interval is 1 SEM over positions, which answers "is this config better
    than the baseline". The per-position differences are drawn as dots behind
    it, because the spread across positions is real signal (methods fail on
    different positions), not just noise to be averaged away.

    Two limits worth remembering when reading it. Differencing cancels the
    position effect but adds the trial noise of both arms - with a median of 2
    trials per cell the SE of one per-position mean is ~0.12, so the difference
    carries a noise floor of ~0.17 no matter how well the blocking works. And it
    only helps where the two position-profiles correlate: that runs from r=0.94
    (DP) down to r=0.24 (IBC), and below about r=0.5 the pairing adds variance
    rather than removing it. The CSV carries r per config so this is checkable.
    """
    per_pos: dict[tuple, dict[str, list[float]]] = {}
    for r in rows:
        cfg = config_of(r)
        per_pos.setdefault(cfg, {}).setdefault(r["start_position"], []).append(r[metric])

    means = {
        cfg: {p: float(np.mean(v)) for p, v in d.items()}
        for cfg, d in per_pos.items()
        if len(d) >= MIN_POSITIONS
    }
    if not means:
        return

    # Baseline = the baseline algorithm's best fully-swept combination.
    cands = {c: d for c, d in means.items() if c[0] == baseline_alg}
    if not cands:
        return
    base = max(cands, key=lambda c: float(np.mean(list(cands[c].values()))))
    positions = sorted(means[base])

    table, points = [], []
    for cfg, d in means.items():
        shared = [p for p in positions if p in d]
        if len(shared) < MIN_POSITIONS or cfg == base:
            continue
        a = np.array([d[p] for p in shared])
        b = np.array([means[base][p] for p in shared])
        diff = a - b
        sem = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
        corr = float(np.corrcoef(a, b)[0, 1])
        points.append((cfg, float(diff.mean()), sem, list(zip(shared, diff))))
        row = {
            "algorithm": cfg[0],
            "inference": cfg[1],
            "refine_iters": cfg[2],
            "exec_horizon": cfg[3],
            "metric": metric,
            "abs_mean": round(float(a.mean()), 4),
            "delta_mean": round(float(diff.mean()), 4),
            "delta_std": round(float(np.std(diff, ddof=1)), 4),
            "delta_sem": round(sem, 4),
            "corr_with_baseline": round(corr, 3),
            "n_positions": len(shared),
            "n_rollouts": sum(len(v) for v in per_pos[cfg].values()),
        }
        row.update({f"delta_{p}": round(float(x), 4) for p, x in zip(shared, diff)})
        table.append(row)

    points.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(11, 0.52 * len(points) + 2.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1.4, zorder=1)

    labels = []
    for i, (cfg, mean, sem, pos_diffs) in enumerate(points):
        alg, inf, iters, horizon = cfg
        color = series_color((alg, inf))
        marker = "D" if (alg, inf) in SERIES_HATCH else "o"
        # Per-position differences: the spread the mean is hiding.
        ax.plot(
            [x for _, x in pos_diffs],
            [i] * len(pos_diffs),
            "o",
            color=color,
            markersize=5,
            alpha=0.35,
            markeredgecolor="none",
            zorder=2,
        )
        ax.errorbar(
            [mean], [i], xerr=[sem], fmt="none", ecolor=color, elinewidth=2.2, capsize=4, zorder=3
        )
        ax.plot(
            [mean],
            [i],
            marker,
            color=color,
            markersize=10,
            markeredgecolor=SURFACE,
            markeredgewidth=1.6,
            zorder=4,
        )
        it = f"{iters} it · " if iters else ""
        labels.append(f"{series_label(alg, inf)} · {it}h{horizon}")

    ax.set_yticks(range(len(points)))
    ax.set_yticklabels(labels, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylim(-0.8, len(points) - 0.2)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    bl = f"{series_label(base[0], base[1])} · h{base[3]}"
    ax.set_title(
        "Paired difference from the BC baseline",
        color=TEXT_PRIMARY,
        fontsize=13,
        loc="left",
        pad=62,
        weight="bold",
    )
    ax.text(
        0,
        1.008,
        f"each config minus {bl} within every start position, then averaged\n"
        "solid mark = mean over 9 positions, bar = 1 SEM; faint dots = the 9 per-position differences\n"
        "configs not run on all 9 positions are excluded",
        transform=ax.transAxes,
        color=TEXT_SECONDARY,
        fontsize=9,
        va="bottom",
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    write_table(OUT_DIR / f"{stem}.csv", table)


def baseline_delta_bars(
    rows,
    stem: str,
    metric: str = "cam1",
    ylabel: str = "Δ cam1 coverage vs BC baseline",
    baseline_alg: str = "bc",
    exclude_positions: set[str] | None = None,
) -> None:
    """Bars: one per algorithm x inference, at its best (iters, horizon) setting.

    Same paired statistic as `baseline_delta_plot` - each config differenced
    against the baseline within every start position - collapsed to one bar per
    method so it reads like the best-setting bar charts. Only settings rolled
    out on all positions can be picked, and the error bar is 1 SEM over
    positions rather than 1 std over trials, which is what makes these bars
    narrow enough to rank.

    The baseline series itself is not drawn: its best setting *is* the baseline,
    so its bar would be zero by construction. The zero line is that baseline.
    """
    drop = exclude_positions or set()
    rows = [r for r in rows if r["start_position"] not in drop]
    per_pos: dict[tuple, dict[str, list[float]]] = {}
    for r in rows:
        per_pos.setdefault(config_of(r), {}).setdefault(r["start_position"], []).append(r[metric])
    # With a position removed the coverage bar moves with it: a setting still has
    # to have been run on every remaining position to qualify.
    n_needed = min(MIN_POSITIONS, len({r["start_position"] for r in rows}))
    means = {
        cfg: {p: float(np.mean(v)) for p, v in d.items()}
        for cfg, d in per_pos.items()
        if len(d) >= n_needed
    }
    cands = {c: d for c, d in means.items() if c[0] == baseline_alg}
    if not cands:
        return
    base = max(cands, key=lambda c: float(np.mean(list(cands[c].values()))))
    positions = sorted(means[base])

    best: dict[tuple[str, str], tuple] = {}
    table = []
    for cfg, d in means.items():
        shared = [p for p in positions if p in d]
        if len(shared) < n_needed:
            continue
        diff = np.array([d[p] for p in shared]) - np.array([means[base][p] for p in shared])
        sem = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
        n_roll = sum(len(v) for v in per_pos[cfg].values())
        entry = (float(diff.mean()), sem, cfg[2], cfg[3], n_roll)
        table.append(
            {
                "algorithm": cfg[0],
                "inference": cfg[1],
                "refine_iters": cfg[2],
                "exec_horizon": cfg[3],
                "metric": metric,
                "delta_mean": round(entry[0], 4),
                "delta_sem": round(sem, 4),
                "delta_std": round(float(np.std(diff, ddof=1)), 4),
                "n_positions": len(shared),
                "n_rollouts": n_roll,
                "is_best": "",
            }
        )
        key = (cfg[0], cfg[1])
        if key not in best or entry[0] > best[key][0]:
            best[key] = entry
    for row in table:
        key = (row["algorithm"], row["inference"])
        if key in best and (row["refine_iters"], row["exec_horizon"]) == best[key][2:4]:
            row["is_best"] = "yes"

    series = [s for s in SERIES_ORDER if s in best and s[0] != baseline_alg]
    if not series:
        return
    vals = [best[s] for s in series]

    fig, ax = plt.subplots(figsize=(1.6 * len(series) + 3.5, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1.4, zorder=3)

    x = np.arange(len(series))
    bars = ax.bar(
        x, [v[0] for v in vals], width=0.56, color=[series_color(s) for s in series], zorder=2
    )
    for bar, s in zip(bars, series):
        if s in SERIES_HATCH:
            bar.set_hatch(SERIES_HATCH[s])
            bar.set_edgecolor(HATCH_EDGE)
            bar.set_linewidth(0)
    ax.errorbar(
        x,
        [v[0] for v in vals],
        yerr=[v[1] for v in vals],
        fmt="none",
        ecolor=TEXT_SECONDARY,
        elinewidth=1.4,
        capsize=4,
        zorder=4,
    )

    lo = min(v[0] - v[1] for v in vals)
    hi = max(v[0] + v[1] for v in vals)
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = hi - lo
    for xi, (m, sem, iters, horizon, n) in zip(x, vals):
        up = m >= 0
        it = f"{iters} it · " if iters else ""
        ax.text(
            xi,
            m + (sem + 0.035 * span) * (1 if up else -1),
            f"{m:+.3f}\n{it}h{horizon}",
            ha="center",
            va="bottom" if up else "top",
            fontsize=8,
            color=TEXT_SECONDARY,
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [series_label(*s).replace(" · ", "\n") for s in series], color=TEXT_SECONDARY
    )
    ax.set_ylim(lo - 0.3 * span, hi + 0.28 * span)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    bl = f"{series_label(base[0], base[1])} · h{base[3]}"
    ax.set_title(
        "Best setting per method, relative to the BC baseline",
        color=TEXT_PRIMARY,
        fontsize=13,
        loc="left",
        pad=48,
        weight="bold",
    )
    ax.text(
        0,
        1.008,
        f"paired against {bl} within every start position; bar = mean over {len(positions)} "
        f"positions, error bar = 1 SEM\nbest (refine_iters, exec_horizon) per method among "
        f"settings run on all {len(positions)} positions"
        + (f"\nexcluded: {', '.join(sorted(drop)).replace('_', ' ')}" if drop else ""),
        transform=ax.transAxes,
        color=TEXT_SECONDARY,
        fontsize=9,
        va="bottom",
    )

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
    eligible = eligible_configs(rows)

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
        "coverage_cam1 — per cell, the refine_iters x exec_horizon setting with the highest mean; "
        "error bars 1 std over trials",
        "coverage_cam1_by_position_best_iters",
        best_iters=True,
    )
    grouped_plot(
        rows,
        "dist",
        "Centroid distance by start position, best refinement setting",
        "dist_centroid, lower is better — per cell, the refine_iters x exec_horizon setting with "
        "the lowest mean; "
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
        eligible=eligible,
    )
    best_iters_plot(
        rows,
        "coverage_cam1_best_iters",
        metric="cam1",
        title="Best refinement setting per method, cam1",
        ylabel="cam1 coverage",
        eligible=eligible,
    )

    speed_tradeoff_plot(rows, "cam1_vs_inference_time")
    horizon_tradeoff_plot(
        rows,
        "cam1_vs_inference_time_horizon",
        title="Cost vs performance — execution-horizon sweep, cam1",
    )
    horizon_tradeoff_plot(
        rows,
        "cam1_vs_inference_time_horizon_no_error",
        title="Cost vs performance — execution-horizon sweep, cam1",
        show_error=False,
    )
    horizon_tradeoff_plot(
        rows,
        "coverage_avg_vs_inference_time_horizon",
        metric="avg_cov",
        ylabel="mean(cam0, cam1) coverage",
        title="Cost vs performance — execution-horizon sweep, both cameras",
    )

    best_iters_plot(
        rows,
        "coverage_avg_best_iters_no_upside_down",
        metric="avg_cov",
        title="Best refinement setting per method",
        ylabel="mean(cam0, cam1) coverage",
        exclude_positions={"upside_down"},
    )

    best_iters_plot(
        rows,
        "coverage_avg_best_iters_position_balanced",
        metric="avg_cov",
        title="Best refinement setting per method, position-balanced",
        ylabel="mean(cam0, cam1) coverage",
        eligible=eligible,
        position_balanced=True,
    )

    baseline_delta_plot(rows, "cam1_delta_vs_bc")
    baseline_delta_plot(
        rows,
        "coverage_avg_delta_vs_bc",
        metric="avg_cov",
        xlabel="Δ mean(cam0, cam1) coverage vs BC baseline",
    )

    baseline_delta_bars(rows, "cam1_delta_vs_bc_best")
    baseline_delta_bars(
        rows,
        "coverage_avg_delta_vs_bc_best",
        metric="avg_cov",
        ylabel="Δ mean(cam0, cam1) coverage vs BC baseline",
    )

    baseline_delta_bars(
        rows,
        "coverage_avg_delta_vs_bc_best_no_upside_down",
        metric="avg_cov",
        ylabel="Δ mean(cam0, cam1) coverage vs BC baseline",
        exclude_positions={"upside_down"},
    )

    print(f"wrote 18 figures + tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
