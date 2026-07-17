#!/usr/bin/env python3
"""Find why deploy actions land in the wrong quadrant vs the offline policy.

Offline (diagnose_pusht_actions.py) the checkpoints predict spread, ++-leaning
actions and match ground truth. On the robot they produced (-,-). Channel order
already matches training, so the remaining suspects are geometric: a vertical /
horizontal flip, a rotation, or a scale/view mismatch between the live camera
and the training frames.

This loads the RAW blue frames captured by `deploy_pusht_real.py --dry-run`
(deploy_dryrun/raw_*.npy) and runs the policy under a set of candidate image
transforms, printing the predicted action + quadrant for each. Whichever
transform moves the actions back toward the offline distribution (mostly ++,
matching diagnose_pusht_actions.py) is the deploy correction we need.

    .venv/bin/python scripts/check_preproc_parity.py \
        --seed-dir checkpoints/pusht_real_combinedv2/seed_0011 \
        --raw-glob 'deploy_dryrun/raw_*.npy'
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("deploy", ROOT / "scripts" / "deploy_pusht_real.py")
deploy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deploy)


# Candidate geometric transforms applied to the raw (H,W,3) frame.
TRANSFORMS = {
    "identity":     lambda a: a,
    "flip_vert":    lambda a: a[::-1, :, :],
    "flip_horiz":   lambda a: a[:, ::-1, :],
    "rot180":       lambda a: a[::-1, ::-1, :],
    "chan_reverse": lambda a: a[:, :, ::-1],          # RGB<->BGR
    "transpose":    lambda a: np.transpose(a, (1, 0, 2)),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-dir", type=Path, required=True)
    p.add_argument("--raw-glob", default="deploy_dryrun/raw_*.npy")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_run_config(seed_dir: Path) -> dict:
    with (seed_dir / "config.json").open() as fh:
        config = json.load(fh)
    return config["environments"][config["active_env"]]


def quad(a: np.ndarray) -> str:
    dx, dy = a[:, 0], a[:, 1]
    return (f"--:{int(np.sum((dx<0)&(dy<0)))} -+:{int(np.sum((dx<0)&(dy>=0)))} "
            f"+-:{int(np.sum((dx>=0)&(dy<0)))} ++:{int(np.sum((dx>=0)&(dy>=0)))}")


def main() -> int:
    import cv2  # noqa: F401  (parity uses deploy.preprocess which imports cv2)

    args = parse_args()
    seed_dir = args.seed_dir.resolve()
    env = load_run_config(seed_dir)
    fs = int(env.get("frame_stack", 2))
    hw = (int(env.get("image_height", 240)), int(env.get("image_width", 320)))
    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu", weights_only=False)
    act_min = np.asarray(norm_stats["act_min"], np.float32)
    act_max = np.asarray(norm_stats["act_max"], np.float32)
    norm_range = tuple(norm_stats.get("action_norm_range", (-1.0, 1.0)))
    cp_selection = str(norm_stats.get("cp_selection", "argmax"))
    cp_temp = float(norm_stats.get("cp_selection_temperature", 1.0))

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    in_channels = 3 * fs
    cp_gen, q_net = deploy.build_models(env, in_channels, device)
    suffix = "" if args.no_ema else "_ema"
    deploy.load_weights(cp_gen, seed_dir / f"control_point_generator{suffix}.pt", device)
    deploy.load_weights(q_net, seed_dir / f"q_estimator{suffix}.pt", device)

    raw_paths = sorted(glob.glob(args.raw_glob))
    if not raw_paths:
        raise FileNotFoundError(f"no raw frames match {args.raw_glob} "
                                "(run deploy_pusht_real.py --dry-run first)")
    frames = [np.load(p) for p in raw_paths]
    print(f"{len(frames)} raw frames, shape {frames[0].shape}, transforms={list(TRANSFORMS)}")
    print("Offline reference (diagnose): pred ~++ leaning, mean~[+0.03,-0.04].\n")

    for name, fn in TRANSFORMS.items():
        acts = []
        for f in frames:
            t = np.ascontiguousarray(fn(f))
            # feed as-is (channel handling is one of the transforms under test)
            proc = deploy.preprocess(t, hw, swap_rgb=False)
            buf = [proc] * fs                          # single-frame stack proxy
            obs_u8 = deploy.stack_to_tensor(buf, device)
            na = deploy.select_action(cp_gen, q_net, obs_u8, cp_selection, cp_temp)
            acts.append(na)
        acts = np.stack(acts)
        print(f"{name:12s} mean={np.round(acts.mean(0),3)} std={np.round(acts.std(0),3)}  "
              f"quad[{quad(acts)}]")

    print("\nPick the transform whose mean/quadrants best match the offline "
          "reference; that is the geometric correction the deploy client needs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
