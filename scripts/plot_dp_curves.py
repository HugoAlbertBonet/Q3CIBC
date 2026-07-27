#!/usr/bin/env python3
"""Parse DP training .out logs and plot train-loss + held-out val-MAE curves.

The DP trainer (train_pusht_real_dp.py) prints, per step interval:
    Step 12000/350000 | Loss: 0.041234 | LR: 2.87e-04 | 512.3s
    [val] step 25000/350000 | held-out sampled-action MAE (ddim5): 0.07321

This reads any number of such logs, overlays their train-loss and val-MAE
curves in two stacked panels, and writes a PNG (and prints the last value per
run so you get the numbers even without opening the image).

Usage:
    python scripts/plot_dp_curves.py results/curves/*.out --out results/curves/dp_curves.png
    # label runs explicitly (else the filename stem is used):
    python scripts/plot_dp_curves.py a.out=conv-750k b.out=resnet-s11 c.out=resnet-s29
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LOSS = re.compile(r"Step\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([0-9.eE+-]+)")
_VAL = re.compile(r"\[val\]\s+step\s+(\d+)/\d+\s+\|.*MAE\s+\(ddim\d+\):\s+([0-9.eE+-]+)")


def parse(path: Path):
    loss_x, loss_y, val_x, val_y = [], [], [], []
    total = None
    for line in path.read_text(errors="replace").splitlines():
        m = _LOSS.search(line)
        if m:
            loss_x.append(int(m.group(1)))
            total = int(m.group(2))
            loss_y.append(float(m.group(3)))
            continue
        m = _VAL.search(line)
        if m:
            val_x.append(int(m.group(1)))
            val_y.append(float(m.group(2)))
    return loss_x, loss_y, val_x, val_y, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="paths, optionally path=label")
    ap.add_argument("--out", type=Path, default=Path("results/curves/dp_curves.png"))
    args = ap.parse_args()

    fig, (ax_loss, ax_val) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    print(f"{'run':28s} {'steps':>8s} {'final loss':>12s} {'final valMAE':>13s} {'best valMAE':>12s}")
    for spec in args.logs:
        path_s, _, label = spec.partition("=")
        path = Path(path_s)
        label = label or path.stem
        lx, ly, vx, vy, total = parse(path)
        if lx:
            ax_loss.plot(lx, ly, label=label, alpha=0.85)
        if vx:
            ax_val.plot(vx, vy, marker="o", ms=3, label=label, alpha=0.85)
        fl = f"{ly[-1]:.5f}" if ly else "-"
        fv = f"{vy[-1]:.5f}" if vy else "-"
        bv = f"{min(vy):.5f}" if vy else "-"
        last = lx[-1] if lx else (vx[-1] if vx else 0)
        print(f"{label:28s} {last:>8d} {fl:>12s} {fv:>13s} {bv:>12s}")

    ax_loss.set_ylabel("train loss")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8)
    ax_val.set_ylabel("held-out sampled-action MAE")
    ax_val.set_xlabel("step")
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
