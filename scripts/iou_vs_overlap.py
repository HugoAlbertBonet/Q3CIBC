#!/usr/bin/env python3
"""IoU as a function of intersection, for two shapes of identical size and shape.

Both shapes have area A and overlap by I. The union is then A + A - I, so with
the overlap expressed as a fraction of one shape, i = I / A in [0, 1]:

    IoU(i) = i / (2 - i)          inverse:  i = 2 * IoU / (1 + IoU)

The point of plotting it: IoU is much harsher than raw overlap. Covering half
the target scores 0.33, not 0.5, and you need 2/3 overlap just to reach 0.5 IoU.

    uv run python scripts/iou_vs_overlap.py --n 11
    uv run python scripts/iou_vs_overlap.py --area 4500 --plot iou.png
"""

from __future__ import annotations

import argparse

import numpy as np


def iou_from_overlap(i: np.ndarray) -> np.ndarray:
    """i = I/A in [0, 1] -> IoU. Union is 2A - I, so IoU = i / (2 - i)."""
    return i / (2.0 - i)


def overlap_from_iou(iou: np.ndarray) -> np.ndarray:
    """Inverse of the above."""
    return 2.0 * iou / (1.0 + iou)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=21, help="points in the linspace")
    ap.add_argument("--area", type=float, default=None,
                    help="shape area A. If given, I is printed in the same "
                         "units as well as as a fraction.")
    ap.add_argument("--plot", default=None, help="write a PNG here instead of only printing")
    args = ap.parse_args()

    i = np.linspace(0.0, 1.0, args.n)
    iou = iou_from_overlap(i)

    head = f"{'I/A':>8} {'IoU':>8}" + (f" {'I':>12} {'union':>12}" if args.area else "")
    print(head); print("-" * len(head))
    for f, v in zip(i, iou):
        line = f"{f:8.3f} {v:8.4f}"
        if args.area:
            line += f" {f*args.area:12.1f} {(2-f)*args.area:12.1f}"
        print(line)

    print("\nreference points")
    for target in (0.25, 0.5, 0.75, 0.9):
        print(f"  IoU {target:.2f} needs {100*overlap_from_iou(np.array(target)):.1f}% overlap")
    print(f"  50% overlap gives IoU {iou_from_overlap(np.array(0.5)):.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fine = np.linspace(0, 1, 400)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(fine, iou_from_overlap(fine), lw=2, label="IoU = i / (2 - i)")
        ax.plot(fine, fine, ls="--", lw=1, color="0.6", label="y = i (reference)")
        ax.scatter(i, iou, s=14, zorder=3)
        ax.set_xlabel("intersection as a fraction of one shape, I/A")
        ax.set_ylabel("IoU")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(args.plot, dpi=150)
        print(f"\n-> {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
