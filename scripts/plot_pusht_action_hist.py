#!/usr/bin/env python3
"""Histogram expert (GT) actions vs each checkpoint's predictions.

Reads the per-sample arrays dumped by diagnose_pusht_actions.py --dump-arrays
(<dir>/<tag>_seed<seed>.npz, keys pred/gt). GT is identical across checkpoints
(same seed-0 sample), so it is drawn once as the reference on every row.

Usage:
    python scripts/plot_pusht_action_hist.py \
        --arrays-dir results/diag_arrays_raw \
        --out results/pusht_action_hist_raw.png
"""
import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arrays-dir", type=Path, required=True,
                   help="dir with <tag>_seed*.npz from --dump-arrays")
    p.add_argument("--out", type=Path, default=Path("results/pusht_action_hist.png"))
    p.add_argument("--bins", type=int, default=60)
    args = p.parse_args()

    files = sorted(args.arrays_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz in {args.arrays_dir}")

    # GT is the same sample for every checkpoint; take it from the first file.
    gt = np.load(files[0])["gt"]
    lo = float(min(gt.min(), min(np.load(f)["pred"].min() for f in files)))
    hi = float(max(gt.max(), max(np.load(f)["pred"].max() for f in files)))
    bins = np.linspace(lo, hi, args.bins + 1)

    nrows = len(files)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 2.4 * nrows), squeeze=False)
    axis_name = ("dx", "dy")
    for r, f in enumerate(files):
        d = np.load(f)
        pred = d["pred"]
        tag = f.stem
        for c in range(2):
            ax = axes[r][c]
            ax.hist(gt[:, c], bins=bins, alpha=0.5, label="expert (GT)",
                    color="#888888", density=True)
            ax.hist(pred[:, c], bins=bins, alpha=0.5, label="pred",
                    color="#d1495b", density=True)
            mae = float(np.abs(pred[:, c] - gt[:, c]).mean())
            ax.set_title(f"{tag}  {axis_name[c]}  MAE={mae:.3f}", fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}  ({nrows} checkpoints)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
