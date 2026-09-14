#!/usr/bin/env python3
"""Separation loss vs entropy loss in the q3c ablations.

Rebuilds the data from trials.jsonl (never hand-typed): for each batch it matches
every command line to its scored training record by exact fixed params, takes the
entropy arm and the separation arm, and plots mean +/- std per environment in a
compact, untitled paper figure set in Palatino Linotype (PNG + PDF).

  Pushing / LIBERO: one-at-a-time ablation batches; the reference arm trains with
                    separation_loss="entropy", the "separation_loss=separation" arm
                    changes only that key.
  Particle (16D):   the N=5 runs of the N-sweep batch plus nsweepParticle16SepSeeds
                    (argmax recipe), entropy vs
                    separation at identical fixed params, every scored seed of each arm.

Whiskers are clipped to the 0-100 % range success rates can take.

Usage:
    uv run python scripts/plot_ablation_separation.py --out results/ablation/separation_vs_entropy.png
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/hyperparam_search/combinedv2_cpascounter_training"
FIXED_RE = re.compile(r"--fixed-params '(\{.*?\})'")

# Reference palette (dataviz skill, light mode): categorical slots 1-2, text & surface tokens.
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_ENT, C_SEP = "#2a78d6", "#eb6834"


def scored_records(env: str) -> list[dict]:
    recs = [json.loads(l) for l in open(RES / env / "trials.jsonl") if l.strip()]
    return [r for r in recs if not (r.get("note") or "").startswith("reeval")
            and r.get("success_rate") is not None and not r.get("error")]


def match(trained: list[dict], fp: dict, what: str) -> dict:
    hit = sorted((r for r in trained if all(str(r["params"].get(k)) == str(v) for k, v in fp.items())),
                 key=lambda r: r["run_id"])
    if not hit:
        raise SystemExit(f"no scored record for {what}")
    return hit[-1]


def as_percent(arm: dict[int, dict]) -> dict[int, float]:
    return {s: r["success_rate"] * 100 for s, r in arm.items()}


def load_arm_values(batch: str, env: str) -> tuple[dict, dict]:
    trained = scored_records(env)
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
            arms.setdefault(name, {})[fp["trial_seed"]] = match(trained, fp, f"{batch} arm {name!r} seed {fp['trial_seed']}")
    ref, sep = arms["ref"], arms["separation_loss=separation"]
    for r in ref.values():
        assert r["params"].get("separation_loss") == "entropy", f"{env}: reference arm is not entropy"
    for r in sep.values():
        assert r["params"].get("separation_loss") == "separation", f"{env}: separation arm mislabeled"
    return as_percent(ref), as_percent(sep)


def load_nsweep_arms(batches: tuple[str, ...], env: str, n_cp: int) -> tuple[dict, dict]:
    """Entropy vs separation at N=n_cp from N-sweep batches (packed job lines)."""
    trained = scored_records(env)
    arms: dict[str, dict[int, dict]] = {"entropy": {}, "separation": {}}
    for batch in batches:
        for line in open(ROOT / "batches" / batch):
            if line.startswith("#"):
                continue
            for m in FIXED_RE.finditer(line):
                fp = json.loads(m.group(1))
                if fp["control_points"] != n_cp:
                    continue
                loss = fp["separation_loss"]
                arms[loss][fp["trial_seed"]] = match(trained, fp, f"{batch} N={n_cp} {loss} seed {fp['trial_seed']}")
    if len(arms["entropy"]) < 2 or len(arms["separation"]) < 2:
        raise SystemExit(f"{batches}: need >= 2 scored seeds per arm at N={n_cp}: "
                         f"entropy {sorted(arms['entropy'])}, separation {sorted(arms['separation'])}")
    return as_percent(arms["entropy"]), as_percent(arms["separation"])


ENVS = [("Particle (16D)", lambda: load_nsweep_arms(("nsweepParticle16.txt", "nsweepParticle16SepSeeds.txt"), "particle/16", n_cp=5)),
        ("Pushing (states)", lambda: load_arm_values("ablPushingStates.txt", "pushing")),
        ("Pushing (pixels)", lambda: load_arm_values("ablPushingPixels.txt", "pushing_pixels")),
        ("LIBERO-Goal (pixels)", lambda: load_arm_values("ablLibero.txt", "libero_goal_pixels"))]


def use_palatino() -> str:
    """Register Palatino Linotype and return its family name; raise if unavailable."""
    from matplotlib import font_manager as fm
    name = "Palatino Linotype"
    if not any(f.name == name for f in fm.fontManager.ttflist):
        for path in sorted(Path("/mnt/c/Windows/Fonts").glob("pala*.ttf")) + sorted(Path.home().glob(".fonts/pala*.ttf")):
            fm.fontManager.addfont(str(path))
    if not any(f.name == name for f in fm.fontManager.ttflist):
        raise SystemExit("Palatino Linotype not found (looked in matplotlib, /mnt/c/Windows/Fonts, ~/.fonts)")
    styles = {f.style for f in fm.fontManager.ttflist if f.name == name}
    if "italic" not in styles:
        raise SystemExit(f"Palatino Linotype italic not found (styles registered: {sorted(styles)})")
    # Math text ($N$) otherwise falls back to DejaVu; typeset it in Palatino too.
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": name, "mathtext.it": f"{name}:italic",
                         "mathtext.bf": f"{name}:bold", "mathtext.default": "it"})
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results/ablation/separation_vs_entropy.png")
    args = ap.parse_args()

    family = use_palatino()
    plt.rcParams.update({"font.family": family, "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT, "ytick.color": TEXT2, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.0, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    offset, lows = 0.16, []
    for gi, (label, load) in enumerate(ENVS):
        ent, sep = load()
        for vals, color, dx in ((ent, C_ENT, -offset), (sep, C_SEP, offset)):
            v = list(vals.values())
            m, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
            lo_w, hi_w = max(0.0, m - sd), min(100.0, m + sd)
            lows.append(lo_w)
            x = gi + dx
            ax.plot([x, x], [lo_w, hi_w], color=color, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([x], [m], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(x + (0.07 if dx > 0 else -0.07), m, f"{m:.1f}", color=TEXT2, fontsize=9.5,
                    ha="left" if dx > 0 else "right", va="center")
            print(f"{label:22} {'entropy' if color == C_ENT else 'separation':10} seeds={sorted(vals)} "
                  f"values={[round(vals[s], 1) for s in sorted(vals)]} mean={m:.1f} std={sd:.1f}")

    # Round ticks that always include 100, so points near the ceiling have a gridline to read against.
    lo = max(0, 10 * int((min(lows) - 1) // 10))
    step = 20 if 100 - lo > 60 else 10
    lo = step * (lo // step)
    ax.set_ylim(lo - 1.5, 101.5)
    ax.set_yticks(range(lo, 101, step))
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
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10, labelcolor=TEXT, handletextpad=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor=SURFACE)
    fig.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"-> {args.out} (+ .pdf), font: {family}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
