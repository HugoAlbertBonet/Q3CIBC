#!/usr/bin/env python3
"""Separation loss vs entropy loss in the q3c one-at-a-time ablations.

Rebuilds the data from trials.jsonl (never hand-typed): for each ablation batch it
matches every command line to its scored training record by exact fixed params,
takes the reference arm (which trains with separation_loss="entropy") and the
"separation_loss=separation" arm, and plots mean +/- std per environment in a
compact, untitled paper figure set in Palatino Linotype (PNG + PDF).

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


def use_palatino() -> str:
    """Register Palatino Linotype and return its family name; raise if unavailable."""
    from matplotlib import font_manager as fm
    name = "Palatino Linotype"
    if not any(f.name == name for f in fm.fontManager.ttflist):
        for path in sorted(Path("/mnt/c/Windows/Fonts").glob("pala*.ttf")) + sorted(Path.home().glob(".fonts/pala*.ttf")):
            fm.fontManager.addfont(str(path))
    if not any(f.name == name for f in fm.fontManager.ttflist):
        raise SystemExit("Palatino Linotype not found (looked in matplotlib, /mnt/c/Windows/Fonts, ~/.fonts)")
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results/ablation/separation_vs_entropy.png")
    args = ap.parse_args()

    family = use_palatino()
    plt.rcParams.update({"font.family": family, "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT, "ytick.color": TEXT2, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.6, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    offset, lows = 0.16, []
    for gi, (label, batch, env) in enumerate(ENVS):
        ent, sep, _ = load_arm_values(batch, env)
        for vals, color, dx in ((ent, C_ENT, -offset), (sep, C_SEP, offset)):
            v = list(vals.values())
            m, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
            lows.append(m - sd)
            x = gi + dx
            ax.plot([x, x], [m - sd, m + sd], color=color, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([x], [m], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(x + (0.07 if dx > 0 else -0.07), m, f"{m:.1f}", color=TEXT2, fontsize=9.5,
                    ha="left" if dx > 0 else "right", va="center")

    # Round ticks that always include 100, so points near the ceiling have a gridline to read against.
    lo = max(0, 10 * int((min(lows) - 1) // 10))
    ax.set_ylim(lo - 1, 101.5)
    ax.set_yticks(range(lo, 101, 10))
    ax.set_xlim(-0.55, len(ENVS) - 0.45)
    ax.set_xticks(range(len(ENVS)))
    ax.set_xticklabels([e[0] for e in ENVS])
    ax.set_ylabel("Success rate (%)")
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):  # the lowest gridline is the baseline
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=c, markeredgecolor=SURFACE,
                      markeredgewidth=2, label=l) for c, l in ((C_ENT, "Entropy loss"), (C_SEP, "Separation loss"))]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=10, labelcolor=TEXT, handletextpad=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor=SURFACE)
    fig.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"-> {args.out} (+ .pdf), font: {family}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
