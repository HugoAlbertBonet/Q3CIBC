#!/usr/bin/env python3
"""Launch combinedv2_cpascounter_training on the real-robot Push-T zip.

This writes an immutable per-run config beside the checkpoints and points the
existing trainer at it with Q3C_CONFIG_PATH.  The shared config_json/config.json
is never modified, so Slurm array tasks can train concurrently without races.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "03-23-pusht-data.zip",
        help="BridgeData-style Push-T demonstration zip",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-stack", type=int, default=2)
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["images0", "images1"],
        help="Ordered RGB streams; channel ordering is persisted in norm_stats.pt",
    )
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "checkpoints" / "pusht_real_combinedv2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and print the run config without starting training",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, run_dir: Path) -> dict:
    with (ROOT / "config_json" / "config.json").open() as handle:
        config = json.load(handle)

    # Start from the already exercised pixel-pushing setup, then make every
    # real-robot-specific choice explicit in the per-run config.
    env = copy.deepcopy(config["environments"]["pushing_pixels"])
    env.update(
        {
            "data_archive": str(args.dataset.resolve()),
            "env_id": "PushTRealRobot-v0",
            "state_dim": [
                3 * len(args.cameras) * args.frame_stack,
                args.image_height,
                args.image_width,
            ],
            "action_dim": 2,
            "frame_stack": args.frame_stack,
            "camera_streams": list(args.cameras),
            "image_height": args.image_height,
            "image_width": args.image_width,
            "dataloader_num_workers": args.workers,
            "action_bounds": [-1.0, 1.0],
            "encoder_target_height": 180,
            "encoder_target_width": 240,
        }
    )

    env["training"].update(
        {
            "training_steps": args.steps,
            "batch_size": args.batch_size,
            "trial_seed": args.seed,
            # There is deliberately no simulator-based model selection for
            # this dataset. Each array task is an independently testable seed.
            "best_ckpt": False,
            "ema_decay": 0.999,
            # Real-robot control should rank the CP cloud directly; iterative
            # inference adds latency and was not validated on this hardware.
            "langevin_num_iterations": 0,
            "inference_langevin_iterations": 0,
        }
    )
    env["model"].update(
        {
            "encoder_kind": "conv_maxpool",
            "control_points": 20,
            "num_hidden_layers": 2,
            "num_neurons": 256,
            "value_width": 1024,
            "value_num_blocks": 1,
        }
    )

    config["active_env"] = "pusht_real_pixels"
    config["environments"]["pusht_real_pixels"] = env
    config["training_shared"]["model_save_dir"] = str(run_dir)
    config["training_shared"]["save_interval"] = 10_000
    return config


def main() -> int:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if args.steps <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("steps/batch-size must be positive and workers non-negative")

    run_dir = args.output_root.resolve() / f"seed_{args.seed:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config = build_config(args, run_dir)
    with config_path.open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print(f"Dataset:    {args.dataset.resolve()}")
    print(f"Seed:       {args.seed}")
    print(f"Config:     {config_path}")
    print(f"Checkpoints:{run_dir}")
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["Q3C_CONFIG_PATH"] = str(config_path)
    command = [sys.executable, str(ROOT / "combinedv2_cpascounter_training.py")]
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
