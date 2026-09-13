#!/usr/bin/env python3
"""Separation loss vs entropy loss in the q3c one-at-a-time ablations.

Rebuilds the data from trials.jsonl (never hand-typed): for each ablation batch it
matches every command line to its scored training record by exact fixed params,
takes the reference arm (which trains with separation_loss="entropy") and the
"separation_loss=separation" arm, 3 seeds each, and plots every seed plus the
mean +/- std, with the paired separation - entropy difference per environment.

Usage:
    uv run python scripts/plot_ablation_separation.py --out results/ablation/separation_vs_entropy.png
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics as st
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/hyperparam_search/combinedv2_cpascounter_training"
ENVS = [("Pushing (states)", "ablPushingStates.txt", "pushing"),
        ("Pushing (pixels)", "ablPushingPixels.txt", "pushing_pixels"),
        ("LIBERO-Goal (pixels)", "ablLibero.txt", "libero_goal_pixels")]

# Reference palette (dataviz skill, light mode): categorical slots 1-2, text & surface tokens.
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_ENT, C_SEP = "#2a78d6", "#eb6834"


def load_arm_values(batch: str, env: str) -> tuple[dict, dict, int]:
    recs = [json.loads(l) for l in open(RES / env / "trials.jsonl") if l.strip()]
    trained = [r for r in recs if not (r.get("note") or "").startswith("reeval")
               and r.get("success_rate") is not None and not r.get("error")]
    arms: dict[str, dict[int, dict]] = {}
    name = None
    for line in open(ROOT / "batches" / batch):
        m = re.match(r"#\s*──\s*(.+?)\s+seed\s+(\d+)", line)
        if m:
            name = m.group(1).strip()
            continue
        if line.startswith("uv run") and "--fixed-params" in line:
            tok = shlex.split(line)
            fp = json.loads(tok[tok.index("--fixed-params") + 1])
            hit = sorted((r for r in trained if all(str(r["params"].get(k)) == str(v) for k, v in fp.items())),
                         key=lambda r: r["run_id"])
            if not hit:
                raise SystemExit(f"{batch}: no scored record for arm {name!r} seed {fp['trial_seed']}")
            arms.setdefault(name, {})[fp["trial_seed"]] = hit[-1]
    ref = arms["ref"]
    sep = arms["separation_loss=separation"]
    for r in ref.values():
        assert r["params"].get("separation_loss") == "entropy", f"{env}: reference arm is not entropy"
    for r in sep.values():
        assert r["params"].get("separation_loss") == "separation", f"{env}: separation arm mislabeled"
    eps = next(iter(ref.values())).get("num_seeds")
    return ({s: r["success_rate"] * 100 for s, r in ref.items()},
            {s: r["success_rate"] * 100 for s, r in sep.items()}, eps)


def paired_p(d: list[float]) -> float | None:
    if len(d) < 2 or st.stdev(d) == 0:
        return None
    try:
        from scipy import stats
    except ImportError:
        return None
    t = st.mean(d) / (st.stdev(d) / sqrt(len(d)))
    return float(2 * (1 - stats.t.cdf(abs(t), len(d) - 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results/ablation/separation_vs_entropy.png")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2})
    fig, ax = plt.subplots(figsize=(9.2, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    offset, jitter = 0.17, [-0.035, 0.0, 0.035]
    ticklabels, ymin = [], 100.0
    for gi, (label, batch, env) in enumerate(ENVS):
        ent, sep, eps = load_arm_values(batch, env)
        seeds = sorted(set(ent) & set(sep))
        diffs = [sep[s] - ent[s] for s in seeds]
        for vals, color, dx in ((ent, C_ENT, -offset), (sep, C_SEP, offset)):
            v = [vals[s] for s in seeds]
            ymin = min(ymin, *v)
            xs = [gi + dx + jitter[i % 3] for i in range(len(v))]
            ax.scatter(xs, v, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, alpha=0.9)
            m, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
            xm = gi + dx + (0.12 if dx > 0 else -0.12)
            ax.plot([xm, xm], [m - sd, m + sd], color=color, linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot([xm - 0.035, xm + 0.035], [m, m], color=color, linewidth=2.6, solid_capstyle="round", zorder=2)
            ax.text(xm + (0.05 if dx > 0 else -0.05), m, f"{m:.1f}", color=TEXT2, fontsize=9,
                    ha="left" if dx > 0 else "right", va="center")
        p = paired_p(diffs)
        dm = st.mean(diffs)
        ptxt = f"p = {p:.2f}" if p is not None else "p n/a"
        # Short enough to fit inside its own group's slot (the long form collided).
        ax.text(gi, 101.4, f"Δ {dm:+.1f} pts · {ptxt}", ha="center", va="bottom", color=TEXT, fontsize=9.5)
        ticklabels.append(f"{label}\n{len(seeds)} seeds · {eps} eval episodes")

    lo = max(0, 5 * int((ymin - 6) // 5))
    ax.set_ylim(lo, 106)
    ax.set_yticks(range(lo, 101, 5))
    ax.set_xlim(-0.6, len(ENVS) - 0.4)
    ax.set_xticks(range(len(ENVS)))
    ax.set_xticklabels(ticklabels)
    ax.set_ylabel("Success rate (%)")
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0)
    fig.suptitle("Separation loss vs entropy loss (q3c ablation)", x=0.06, ha="left", y=0.975,
                 color=TEXT, fontsize=13, fontweight="semibold")
    ax.set_title("Dots: individual training seeds. Line: mean ± std. Δ = paired separation − entropy. DFO 0 iterations.",
                 loc="left", color=TEXT2, fontsize=9.5, pad=18)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=C_ENT, markeredgecolor=SURFACE,
                      markeredgewidth=2, label="Entropy loss (reference)"),
               Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=C_SEP, markeredgecolor=SURFACE,
                      markeredgewidth=2, label="Separation loss")]
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9.5, labelcolor=TEXT)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
