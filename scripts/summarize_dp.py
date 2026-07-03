"""Summarize Diffusion Policy trials (DDPM vs DDIM) as markdown tables.

Reads results/hyperparam_search/diffusion_policy_training/<env>/trials.jsonl and
prints:
  1. Per-trial x per-sampler table (success_rate, avg_reward, ms/step).
  2. Cross-seed aggregate (mean +/- std) per sampler for the BASE recipe
     (the trials whose only varied param is trial_seed).

Usage:
  uv run python scripts/summarize_dp.py [--env pushing] [--min-trial-id N]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent
SAMPLER_RE = re.compile(r"^(ddpm|ddim\d+)_success_rate$")


def load(env: str, min_trial_id: int) -> list[dict]:
    p = ROOT / "results" / "hyperparam_search" / "diffusion_policy_training" / env / "trials.jsonl"
    if not p.exists():
        raise SystemExit(f"No trials file at {p}")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in rows if int(r.get("trial_id", 0)) >= min_trial_id]


def samplers(rows: list[dict]) -> list[str]:
    names: set[str] = set()
    for r in rows:
        for k in r:
            m = SAMPLER_RE.match(k)
            if m:
                names.add(m.group(1))
    # ddpm first, then ddim by step count.
    def key(n):
        return (0, 0) if n == "ddpm" else (1, int(n[4:]))
    return sorted(names, key=key)


def pget(r: dict, k: str, default=None):
    return r.get("params", {}).get(k, default)


def fmt_pct(x):
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="pushing")
    ap.add_argument("--min-trial-id", type=int, default=0)
    args = ap.parse_args()

    rows = load(args.env, args.min_trial_id)
    rows = [r for r in rows if not r.get("training_failed")]
    if not rows:
        raise SystemExit("No successful trials.")
    smp = samplers(rows)

    # ── Table 1: per trial ────────────────────────────────────────────────
    print(f"\n## DP trials — {args.env}  ({len(rows)} trials)\n")
    hdr = ["trial", "seed", "T", "sched", "depth"]
    for s in smp:
        hdr += [f"{s} SR", f"{s} R", f"{s} ms"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in sorted(rows, key=lambda r: int(r.get("trial_id", 0))):
        line = [
            str(r.get("trial_id")),
            str(pget(r, "trial_seed", "-")),
            str(pget(r, "num_train_timesteps", "-")),
            str(pget(r, "beta_schedule", "-")),
            str(pget(r, "denoiser_depth", "-")),
        ]
        for s in smp:
            line += [
                fmt_pct(r.get(f"{s}_success_rate")),
                f"{r.get(f'{s}_avg_reward', float('nan')):.2f}",
                f"{r.get(f'{s}_ms_per_step', float('nan')):.1f}",
            ]
        print("| " + " | ".join(line) + " |")

    # ── Table 2: base-recipe cross-seed aggregate ─────────────────────────
    # Base = trials sharing the modal (T, sched, depth); vary only seed.
    def sig(r):
        return (pget(r, "num_train_timesteps"), pget(r, "beta_schedule"), pget(r, "denoiser_depth"))
    from collections import Counter
    modal = Counter(sig(r) for r in rows).most_common(1)[0][0]
    base = [r for r in rows if sig(r) == modal]
    if len(base) > 1:
        print(f"\n## Base recipe cross-seed (T={modal[0]}, {modal[1]}, depth={modal[2]}; n={len(base)} seeds)\n")
        print("| sampler | success_rate mean±std | avg_reward mean±std | ms/step |")
        print("|---|---|---|---|")
        for s in smp:
            srs = [r[f"{s}_success_rate"] for r in base if f"{s}_success_rate" in r]
            rws = [r[f"{s}_avg_reward"] for r in base if f"{s}_avg_reward" in r]
            mss = [r[f"{s}_ms_per_step"] for r in base if f"{s}_ms_per_step" in r]
            if not srs:
                continue
            sd_sr = pstdev(srs) if len(srs) > 1 else 0.0
            sd_rw = pstdev(rws) if len(rws) > 1 else 0.0
            print(f"| {s} | {100*mean(srs):.1f}% ± {100*sd_sr:.1f} | "
                  f"{mean(rws):.2f} ± {sd_rw:.2f} | {mean(mss):.1f} |")
    print()


if __name__ == "__main__":
    main()
