"""Analyze a deploy rollout for the dead-zone / z-droop stall (D4 + G4).

Reads a `--log-dir` produced by deploy_pusht_real.py (raw/*.npy + steps.jsonl)
and reports the stall signature: frame-pair motion vs noise floor, exact-zero
and sub-min-step commands, executed EEF motion, z-droop vs reach, and the
fixed-point onset. Use it to compare the ablation runs (baseline / min-step /
z-hold / both) on the same footing.

Usage:
    python scripts/analyze_rollout.py results/roll_c09_snap
    python scripts/analyze_rollout.py results/roll_c09_both --min-step 0.0015
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log_dir", type=Path)
    p.add_argument("--min-step", type=float, default=0.0015,
                   help="dead-zone edge (m) to count sub-min-step commands")
    p.add_argument("--z-target", type=float, default=0.0197)
    p.add_argument("--stall-run", type=int, default=10,
                   help="consecutive near-frozen steps that mark a stall")
    args = p.parse_args()

    S = [json.loads(l) for l in open(args.log_dir / "steps.jsonl")]
    N = len(S)
    act = np.array([s["action"][:2] for s in S])
    env = np.array([s.get("env_action", s["action"])[:2] for s in S])
    st = np.array([s["state"] for s in S])
    x, z = st[:, 0], st[:, 2]
    amag = np.linalg.norm(act, axis=1)
    eefmove = np.linalg.norm(np.diff(st[:, :2], axis=0), axis=1)

    # frame-pair motion vs noise floor (if raw frames were dumped)
    raws = sorted(glob.glob(str(args.log_dir / "raw" / "*.npy")))
    fdiff = None
    if len(raws) >= 2:
        R = np.stack([np.load(f).astype(np.int16) for f in raws])
        fdiff = np.array([np.abs(R[t] - R[t - 1]).mean() for t in range(1, len(R))])

    # stall onset: first step starting a run of >=stall_run near-frozen steps
    frozen = eefmove < 1e-3
    onset = None
    for t in range(len(frozen) - args.stall_run):
        if frozen[t:t + args.stall_run].all():
            onset = t
            break

    print(f"== {args.log_dir}  ({N} steps) ==")
    print(f"exact-zero commands: {(amag < 1e-4).mean():.1%}   "
          f"sub-min-step (0<|comp|<{args.min_step*1000:.1f}mm): "
          f"{np.mean((np.abs(act) > 1e-5) & (np.abs(act) < args.min_step)):.1%}")
    print(f"clipped (action != env_action): {(np.abs(act-env).sum(1) > 1e-9).mean():.1%}")
    if fdiff is not None:
        floor = np.percentile(fdiff, 5)
        print(f"frame-diff: move-median={np.median(fdiff):.2f}  noise-floor(p5)={floor:.2f}")
    print(f"EEF x: start={x[0]:.3f} max={x.max():.3f} final={x[-1]:.3f}  (demos push x→0.49)")
    print(f"z: corr(x,z)={np.corrcoef(x, z)[0,1]:+.2f}  z@xmax={z[x.argmax()]:.4f}  "
          f"min={z.min():.4f}  (target {args.z_target})")
    if onset is not None:
        print(f"STALL: fixed-point onset ~step {onset}; "
              f"after it |action|mean={amag[onset:].mean():.5f} "
              f"eefmove={eefmove[onset:].mean():.6f}")
    else:
        print("NO sustained stall detected — arm kept moving. ✅")
    # verdict for the ablation
    reached = x.max()
    print(f"\nVERDICT: x_reached={reached:.3f}  "
          f"{'PROGRESS past 0.30' if reached > 0.30 else 'stalled short'}  "
          f"z@reach {'OK' if abs(z[x.argmax()]-args.z_target) < 0.004 else 'LOW'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
