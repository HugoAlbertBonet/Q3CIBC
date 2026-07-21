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
        default=ROOT / "data" / "pusht_widowx_data.zip",
        help="Push-T demonstration archive. Diffusion-Policy format "
             "(replay_buffer.zarr + videos/<ep>/<cam>.mp4) by default; pass "
             "--data-format bridge_zip for the older 03-23-pusht-data.zip.",
    )
    parser.add_argument(
        "--data-format",
        choices=["zarr_video", "bridge_zip"],
        default="zarr_video",
        help="On-disk layout of --dataset (default: zarr_video)",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-stack", type=int, default=2)
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["images1"],
        help="bridge_zip only: ordered RGB streams; channel ordering is "
             "persisted in norm_stats.pt",
    )
    parser.add_argument(
        "--video-camera",
        type=int,
        default=1,
        help="zarr_video only: which per-episode MP4 to train on. 1 is the "
             "fixed blue scene camera (== the old images1 and the stream the "
             "deploy client reads); camera 0 is a second viewpoint and is "
             "discarded.",
    )
    parser.add_argument(
        "--frame-cache-dir",
        type=Path,
        default=None,
        help="zarr_video only: where to build the decoded uint8 frame memmap "
             "(default: <dataset dir>/_frame_cache). ~17 GB for this "
             "collection at 240x320; built once, shared by concurrent runs.",
    )
    # ── Idle-transition handling (the stalling fix) ───────────────────────
    parser.add_argument(
        "--idle-filter",
        choices=["none", "drop_zero", "drop_static", "subsample"],
        default="drop_zero",
        help="How to treat transitions whose target action is ~0. 24%% of this "
             "dataset is the teleoperator pausing: a delta spike holding a "
             "quarter of the probability mass at a single point, while real "
             "pushes spread over a 2-D continuum — so an energy-argmax policy "
             "preferentially lands on the spike, and P(a_t=0 | a_t-1=0)=0.69 "
             "(vs 0.24 base rate) makes it absorbing. Measured effect on the "
             "full archive: none -> 24.1%% zeros kept; drop_static -> 21.5%% "
             "(removes only 3.3%% of data, spike survives); drop_zero -> 0%% "
             "(removes 24.1%%). Default 'drop_zero' is the one that actually "
             "eliminates the stalling mode.",
    )
    parser.add_argument("--idle-eps", type=float, default=0.0,
                        help="|action| <= this counts as idle (metres)")
    parser.add_argument("--idle-move-eps", type=float, default=1e-4,
                        help="drop_static: EEF displacement below this means "
                             "the frame pair carries no visible motion")
    parser.add_argument("--idle-keep-frac", type=float, default=0.25,
                        help="subsample: fraction of idle transitions to keep")
    parser.add_argument(
        "--cond-eef-xy",
        action="store_true",
        help="zarr_video only: condition the policy on the current end-effector "
             "(x, y), min-max normalized to [-1,1] over the training workspace. "
             "The policy is otherwise pixels-only and cannot distinguish 'at the "
             "demo start' from 'stalled in a corner'. x/y are the only proprio "
             "channels that match between this archive and the live server "
             "(rotations use a different convention, z has a ~1cm deploy droop). "
             "Requires a deploy client that feeds state[:2] — see "
             "deployment_forensics.md.",
    )
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument(
        "--output-root",
        type=Path,
        # v2: retuned to the best pushing_pixels hyperparam-search recipe +
        # appearance augmentation. Separate root so the deployed v1 seeds
        # (pusht_real_combinedv2) are never overwritten.
        default=ROOT / "checkpoints" / "pusht_real_combinedv2_v2",
    )
    parser.add_argument(
        "--no-aug",
        action="store_true",
        help="Disable train-time appearance augmentation (photometric + "
             "small view crop). Default ON: deploy forensics measured the "
             "robot's T ~33%% darker than training with identical mat "
             "exposure, so lighting/color robustness must come from data.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Run-directory name under --output-root (default: seed_XXXX). Use "
             "a distinct tag per hyperparameter config so batch runs don't "
             "overwrite each other.",
    )
    # ── Hyperparameters (defaults = best pushing_pixels search recipe) ────
    # Exposed so a batch can sweep them; every one lands in the per-run
    # config.json, which is what the trainer and later diagnose/deploy read.
    hp = parser.add_argument_group("hyperparameters")
    hp.add_argument("--lr", type=float, default=3e-4)
    hp.add_argument("--est-lr", type=float, default=3e-4)
    hp.add_argument("--control-points", type=int, default=20)
    hp.add_argument("--top-k-cps", type=int, default=8)
    hp.add_argument("--cp-width", type=int, default=256)
    hp.add_argument("--cp-depth", type=int, default=2)
    hp.add_argument("--value-width", type=int, default=1024)
    hp.add_argument("--value-num-blocks", type=int, default=1)
    hp.add_argument("--mse-weight", type=float, default=10.0)
    hp.add_argument("--info-nce-weight", type=float, default=0.5)
    hp.add_argument("--sep-weight", type=float, default=0.1)
    hp.add_argument("--entropy-bandwidth", type=float, default=0.01)
    hp.add_argument("--uniform-negatives", type=int, default=0)
    hp.add_argument("--langevin-negatives", type=int, default=0)
    hp.add_argument("--langevin-iters", type=int, default=0)
    hp.add_argument("--infonce-clamp", type=float, default=10.0)
    hp.add_argument("--scheduler", default="cosine_warm_restarts",
                    choices=["cosine", "cosine_warm_restarts"])
    hp.add_argument("--cosine-t0", type=int, default=50_000)
    hp.add_argument("--ema-decay", type=float, default=0.999)
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
    # zarr_video trains on ONE camera (the blue scene view); bridge_zip keeps
    # the explicit stream list. Both end up as a 1-element channel group so
    # in_channels arithmetic is identical.
    n_cams = 1 if args.data_format == "zarr_video" else len(args.cameras)
    env.update(
        {
            "data_archive": str(args.dataset.resolve()),
            "data_format": args.data_format,
            "env_id": "PushTRealRobot-v0",
            "state_dim": [
                3 * n_cams * args.frame_stack,
                args.image_height,
                args.image_width,
            ],
            "action_dim": 2,
            "frame_stack": args.frame_stack,
            "camera_streams": (
                [f"video{args.video_camera}"]
                if args.data_format == "zarr_video"
                else list(args.cameras)
            ),
            "video_camera": args.video_camera,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "dataloader_num_workers": args.workers,
            "action_bounds": [-1.0, 1.0],
            "encoder_target_height": 180,
            "encoder_target_width": 240,
            # Train-time appearance augmentation (see PushTRealPixelsDataset).
            # Ranges cover the measured train->deploy shift (T at ~0.67x red).
            "image_aug": not args.no_aug,
            # Idle-transition handling — see PushTWidowXVideoDataset docstring.
            "idle_filter": args.idle_filter,
            "idle_eps": args.idle_eps,
            "idle_move_eps": args.idle_move_eps,
            "idle_keep_frac": args.idle_keep_frac,
            "cond_eef_xy": args.cond_eef_xy,
        }
    )
    if args.frame_cache_dir is not None:
        env["frame_cache_dir"] = str(args.frame_cache_dir.resolve())

    env["training"].update(
        {
            "training_steps": args.steps,
            "batch_size": args.batch_size,
            "trial_seed": args.seed,
            # There is deliberately no simulator-based model selection for
            # this dataset. Each run is an independently testable config.
            "best_ckpt": False,
            "ema_decay": args.ema_decay,
            # Real-robot control ranks the CP cloud directly; iterative
            # inference adds latency and was not validated on this hardware.
            "inference_langevin_iterations": 0,
            # Defaults below are the best pushing_pixels recipe from the
            # combinedv2 hyperparam search (results/hyperparam_search/
            # combinedv2_cpascounter_training/pushing_pixels/trials.jsonl,
            # trial 95: success_rate 0.99). Every one is overridable so a batch
            # can sweep them.
            "learning_rate": args.lr,
            "estimator_learning_rate": args.est_lr,
            "infonce_logit_clamp": args.infonce_clamp,
            "scheduler_type": args.scheduler,
            "cosine_t0": args.cosine_t0,
            "cosine_t_max": args.steps,
            "num_uniform_negatives": args.uniform_negatives,
            "num_langevin_negatives": args.langevin_negatives,
            "langevin_num_iterations": args.langevin_iters,
            "noisy_expert_count": 0,
            "top_k_control_points": args.top_k_cps,
            "separation_epsilon": 0.02,
            "entropy_bandwidth": args.entropy_bandwidth,
        }
    )
    # These three live in training_shared in config_json, so the trainer reads
    # them from there rather than from the env block — set both to be safe.
    config["training_shared"].update(
        {
            "mse_weight": args.mse_weight,
            "info_nce_weight": args.info_nce_weight,
            "separation_weight": args.sep_weight,
        }
    )
    env["model"].update(
        {
            "encoder_kind": "conv_maxpool",
            "control_points": args.control_points,
            # cp_width/cp_depth are what the pixel CP generator actually reads
            # (num_hidden_layers/num_neurons are only their fallbacks), so set
            # both explicitly to keep the built network unambiguous.
            "num_hidden_layers": args.cp_depth,
            "num_neurons": args.cp_width,
            "cp_width": args.cp_width,
            "cp_depth": args.cp_depth,
            "value_width": args.value_width,
            "value_num_blocks": args.value_num_blocks,
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

    run_name = args.tag if args.tag else f"seed_{args.seed:04d}"
    run_dir = args.output_root.resolve() / run_name
    # Refuse to silently overwrite a finished run: checkpoints are expensive and
    # a repeated --tag in a batch file is an easy mistake to make.
    existing = sorted(run_dir.glob("*.pt")) if run_dir.exists() else []
    if existing and not args.dry_run:
        raise FileExistsError(
            f"{run_dir} already holds checkpoints ({[p.name for p in existing[:3]]}...). "
            f"Pass a different --tag, or delete the directory to retrain."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config = build_config(args, run_dir)
    with config_path.open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print(f"Dataset:    {args.dataset.resolve()}  (format={args.data_format})")
    print(f"Seed:       {args.seed}   tag={run_name}")
    print(f"Idle filter:{args.idle_filter} (eps={args.idle_eps})")
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
