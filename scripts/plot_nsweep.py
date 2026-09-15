#!/usr/bin/env python3
"""Success rate and inference latency vs the number of control points N.

Reviewer question (1). Rebuilds the data from trials.jsonl (never hand-typed): every
entropy-loss command line of the N-sweep batch is matched to its scored training
record by exact fixed params; latency per env step comes from the argmax rows of
results/reviewer/latency_vs_N_nsweep.csv (bench_inference_rv_n.py, batch size 1,
random weights, same networks as the batch).

Left panel: success rate, mean +/- std over seeds (line and band). Right panel:
latency per step, mean +/- std over timed steps. Both share a log-scaled N axis.
Compact, untitled, Palatino Linotype (PNG + PDF), same palette as the ablation
figure.

Usage:
    uv run python scripts/plot_nsweep.py --out results/reviewer/nsweep_particle16.png
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_ablation_separation import (C_ENT, GRID, RES, ROOT, SURFACE, TEXT, TEXT2, match,
                                      scored_records, use_palatino)

FIXED_RE = re.compile(r"--fixed-params '(\{.*?\})'")
# (label, batch, results subdir, latency-CSV env). Pen joins once its sweep is scored.
SWEEPS = [("Particle (16D)", "nsweepParticle16.txt", "particle/16", "particle")]


def load_success(batch: str, env: str) -> dict[int, dict[int, float]]:
    trained = scored_records(env)
    out: dict[int, dict[int, float]] = defaultdict(dict)
    for line in open(ROOT / "batches" / batch):
        if line.startswith("#"):
            continue
        for m in FIXED_RE.finditer(line):
            fp = json.loads(m.group(1))
            if fp.get("separation_loss") != "entropy":
                continue
            r = match(trained, fp, f"{batch} N={fp['control_points']} seed {fp['trial_seed']}")
            out[fp["control_points"]][fp["trial_seed"]] = r["success_rate"] * 100
    return dict(sorted(out.items()))


def load_latency(csv_env: str) -> dict[int, tuple[float, float]]:
    rows = [r for r in csv.DictReader(open(ROOT / "results/reviewer/latency_vs_N_nsweep.csv"))
            if r["env"] == csv_env and r["eval"] == "argmax"]
    return {int(r["N"]): (float(r["ms_mean"]), float(r["ms_std"])) for r in rows}


def style(ax, ns: list[int]) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlim(ns[0] / 1.35, ns[-1] * 1.35)
    ax.set_xlabel("Control points $N$")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):  # axis lines
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(TEXT2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results/reviewer/nsweep_particle16.png")
    args = ap.parse_args()

    family = use_palatino()
    plt.rcParams.update({"font.family": family, "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT, "ytick.color": TEXT2, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(7.0, 3.0), facecolor=SURFACE)

    for label, batch, env, csv_env in SWEEPS:
        succ = load_success(batch, env)
        lat = load_latency(csv_env)
        ns = list(succ)
        missing = [n for n in ns if n not in lat]
        if missing:
            raise SystemExit(f"latency CSV has no argmax rows for {csv_env} N={missing}")
        means = [st.mean(succ[n].values()) for n in ns]
        sds = [st.stdev(succ[n].values()) if len(succ[n]) > 1 else 0.0 for n in ns]
        ax_s.fill_between(ns, [max(0, m - s) for m, s in zip(means, sds)], [min(100, m + s) for m, s in zip(means, sds)],
                          color=C_ENT, alpha=0.15, linewidth=0, zorder=1)
        ax_s.plot(ns, means, color=C_ENT, linewidth=2, zorder=2, label="WiFI-BC-argmax")
        ax_s.scatter(ns, means, s=46, color=C_ENT, edgecolor=SURFACE, linewidth=1.5, zorder=4, clip_on=False)

        lm = [lat[n][0] for n in ns]
        ls = [lat[n][1] for n in ns]
        ax_l.fill_between(ns, [max(0, m - s) for m, s in zip(lm, ls)], [m + s for m, s in zip(lm, ls)],
                          color=C_ENT, alpha=0.15, linewidth=0, zorder=1)
        ax_l.plot(ns, lm, color=C_ENT, linewidth=2, zorder=2)
        ax_l.scatter(ns, lm, s=46, color=C_ENT, edgecolor=SURFACE, linewidth=1.5, zorder=4, clip_on=False)

        for n, m, s, l_m in zip(ns, means, sds, lm):
            print(f"{label} N={n:<4} seeds={sorted(succ[n])} success={[round(succ[n][k], 1) for k in sorted(succ[n])]} "
                  f"mean={m:.1f} std={s:.1f} latency={l_m:.3f} ms")
        style(ax_s, ns)
        style(ax_l, ns)
        top_ms = max(m + s for m, s in zip(lm, ls))

    ax_s.set_ylim(0, 103)  # axis starts on the 0 gridline so the x-axis line does not double it
    ax_s.set_yticks(range(0, 101, 20))
    ax_s.set_ylabel("Success rate (%)")
    ax_s.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=TEXT, handlelength=1.8)
    ax_l.set_ylim(0, max(1.0, round(top_ms * 1.25, 1)))
    ax_l.set_ylabel("Latency per step (ms)")
    fig.tight_layout(w_pad=2.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor=SURFACE)
    fig.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"-> {args.out} (+ .pdf), font: {family}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
