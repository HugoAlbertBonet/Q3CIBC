#!/usr/bin/env python3
"""Is the live camera's red T as bright as the training demos?

The deploy policy drifts mid-rollout partly because the live red T is dimmer
than the training frames (measured: peak red-score 83 live vs 120 in demo
im_0.jpg), a photometric covariate shift. Use this to tune the rig's lighting /
camera exposure until the live frame matches the demo target, WITHOUT a full
policy run.

Capture frames first:
    python scripts/deploy_pusht_real.py --seed-dir <seed> --device cpu \
        --dry-run --dry-run-steps 5
then:
    python scripts/check_brightness_parity.py            # reads deploy_dryrun/raw_*.npy

"Redness" of a pixel = channel0 - mean(channel1, channel2) (red is in ch0 on
this rig). The T is the high-redness region.
"""

from __future__ import annotations

import argparse
import glob

import numpy as np

# Reference measured on the training data (data/03-23-pusht-data.zip,
# traj_group0/traj0/images1/im_0.jpg, 640x480):
DEMO_PEAK_REDNESS = 120      # max(ch0 - mean(ch1,ch2)) over the T
DEMO_RED_FRAC = 0.074        # fraction of pixels with redness > 30
PEAK_TOLERANCE = 15          # accept live peak within this of the demo peak


def redness(im: np.ndarray) -> np.ndarray:
    im = im.astype(np.int32)
    return im[:, :, 0] - (im[:, :, 1] + im[:, :, 2]) // 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-glob", default="deploy_dryrun/raw_*.npy")
    p.add_argument("--thresh", type=int, default=30,
                   help="redness threshold that counts as 'red T' pixel")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(glob.glob(args.raw_glob))
    if not paths:
        raise FileNotFoundError(f"no frames match {args.raw_glob} "
                                "(run deploy_pusht_real.py --dry-run first)")

    peaks, fracs, brights = [], [], []
    for pth in paths:
        im = np.load(pth)
        sc = redness(im)
        peaks.append(int(sc.max()))
        fracs.append(float((sc > args.thresh).mean()))
        brights.append(float(im.mean()))
    peak = float(np.mean(peaks))
    frac = float(np.mean(fracs))
    bright = float(np.mean(brights))

    print(f"{len(paths)} frames from {args.raw_glob}")
    print(f"  live  peak-redness {peak:6.1f}   red-frac {frac:.4f}   mean-brightness {bright:5.1f}")
    print(f"  demo  peak-redness {DEMO_PEAK_REDNESS:6.1f}   red-frac {DEMO_RED_FRAC:.4f}")

    ok = peak >= DEMO_PEAK_REDNESS - PEAK_TOLERANCE
    if ok:
        print(f"  => OK: live T within {PEAK_TOLERANCE} of demo peak. Lighting matched.")
    else:
        deficit = DEMO_PEAK_REDNESS - peak
        print(f"  => TOO DIM by {deficit:.0f}. Increase scene light or camera "
              f"exposure/gain until peak-redness >= {DEMO_PEAK_REDNESS - PEAK_TOLERANCE}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
