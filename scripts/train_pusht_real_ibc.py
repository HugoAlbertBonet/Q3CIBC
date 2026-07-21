#!/usr/bin/env python3
"""Train a paper-faithful IBC (EBM + DFO) policy on the real-robot Push-T zip.

IBC baseline counterpart of scripts/train_pusht_real.py: same dataset,
preprocessing, and checkpoint layout, but the model and recipe follow the
official google-research/ibc optimal config for the Pushing-Pixels task
(ibc/configs/pushing_pixels/pixel_ebm_best.gin, the run behind Florence et
al. 2021 Table 3 "Block Pushing — Pixels, EBM (DFO): 100% ± 0%"):

    - PixelEBM = ConvMaxpoolEncoder (target 180x240) + DenseResnetValue
      (width 1024, 1 block)  == utils.models.PixelQEstimator.
    - Late fusion: encoder runs once per state; the value head scores
      [features, action] pairs.
    - InfoNCE over 256 uniform counter-examples (boundary buffer 0.05),
      NO gradient penalty, NO Langevin (best gin sets
      fraction_langevin_samples=0, add_grad_penalty=False).
    - Adam, lr 1e-3, exponential decay 0.99 every 100 steps (continuous,
      as keras ExponentialDecay staircase=False).
    - Batch 128, 100k steps, sequence_length 2 == frame_stack 2.

Inference (scripts/deploy_pusht_real_ibc.py) uses the matching best-gin
policy: DFO with 2048 action samples, 3 iterations, iteration_std 0.33.

Each run writes an immutable per-run config + norm_stats.pt beside the
checkpoints so Slurm array tasks can train concurrently without races, and
so the deploy client can rebuild the exact model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Official pixel_ebm_best.gin values (google-research/ibc, pushing_pixels).
IBC_BEST = {
    "training_steps": 100_000,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "lr_decay_rate": 0.99,
    "lr_decay_steps": 100,
    "num_counter_examples": 256,
    "uniform_boundary_buffer": 0.05,
    "softmax_temperature": 1.0,
    # Model (PixelEBM: ConvMaxpoolEncoder + DenseResnetValue).
    "encoder_target_height": 180,
    "encoder_target_width": 240,
    "value_width": 1024,
    "value_num_blocks": 1,
    # Inference (IbcPolicy + mcmc.iterative_dfo defaults).
    "inference_dfo_samples": 2048,
    "inference_dfo_iterations": 3,
    "inference_dfo_iteration_std": 0.33,
    "inference_dfo_std_decay": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--steps", type=int, default=IBC_BEST["training_steps"])
    parser.add_argument("--batch-size", type=int, default=IBC_BEST["batch_size"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-stack", type=int, default=2,
                        help="best gin sequence_length = 2")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["images1"],
        help="bridge_zip only: ordered RGB streams. Default images1: the deploy "
             "rig runs a single fixed scene camera (blue), see "
             "PUSHT_DEPLOY_HANDOFF.md",
    )
    parser.add_argument(
        "--video-camera",
        type=int,
        default=1,
        help="zarr_video only: which per-episode MP4 to train on. 1 is the "
             "fixed blue scene camera (== the old images1 and the stream the "
             "deploy client reads); camera 0 is a second viewpoint, discarded.",
    )
    parser.add_argument(
        "--frame-cache-dir",
        type=Path,
        default=None,
        help="zarr_video only: where to build the decoded uint8 frame memmap "
             "(default: <dataset dir>/_frame_cache). ~17 GB for this collection "
             "at 240x320; build it once up front with "
             "scripts/prepare_pusht_video_cache.py, or array tasks will idle on "
             "GPUs waiting for whichever one wins the build lock.",
    )
    # ── Idle-transition handling (the stalling fix) ───────────────────────
    parser.add_argument(
        "--idle-filter",
        choices=["none", "drop_zero", "drop_static", "subsample"],
        default="drop_zero",
        help="How to treat transitions whose target action is ~0. 24%% of this "
             "dataset is the teleoperator pausing: a delta spike holding a "
             "quarter of the probability mass at a single point, while real "
             "pushes spread over a 2-D continuum. IBC is exactly the kind of "
             "policy this breaks — InfoNCE fits the density and DFO returns its "
             "argmax, so the spike wins. Measured on the full archive: none -> "
             "24.1%% zeros kept; drop_static -> 21.5%% (removes only 3.3%%, the "
             "spike survives); drop_zero -> 0%% (removes 24.1%%). Default "
             "'drop_zero' is the one that eliminates the stalling mode. Matches "
             "scripts/train_pusht_real.py so IBC and Q3C stay comparable.",
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
        help="zarr_video only: condition the EBM on the current end-effector "
             "(x, y). NOTE this deviates from pixel_ebm_best.gin, which is "
             "pixels-only — enable it only to match a Q3C run trained the same "
             "way, and record that the baseline is no longer paper-faithful. "
             "Requires a deploy client that feeds state[:2].",
    )
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--counter-examples", type=int,
                        default=IBC_BEST["num_counter_examples"])
    parser.add_argument("--save-interval", type=int, default=10_000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "checkpoints" / "pusht_real_ibc",
    )
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Enable train-time appearance augmentation (photometric + small "
             "view crop). OFF by default, matching scripts/train_pusht_real.py: "
             "it targets the deploy lighting shift, which is a SEPARATE problem "
             "from the stalling, and leaving it on would confound the "
             "idle-filter comparison. Also off is the paper-faithful setting.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Run-directory name under --output-root (default: seed_XXXX). Use "
             "a distinct tag per config so batch runs don't overwrite each other.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and print the run config without starting training",
    )
    return parser.parse_args()


def build_run_config(args: argparse.Namespace, run_dir: Path) -> dict:
    """Immutable per-run config, shaped like the q3c seed dirs so the deploy
    client's load_run_config() pattern works unchanged."""
    # zarr_video trains on ONE camera (the blue scene view); bridge_zip keeps
    # the explicit stream list. Both end up as a 1-element channel group so the
    # in_channels arithmetic is identical.
    n_cams = 1 if args.data_format == "zarr_video" else len(args.cameras)
    env = {
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
        "action_bounds": [-1.0, 1.0],
        "encoder_target_height": IBC_BEST["encoder_target_height"],
        "encoder_target_width": IBC_BEST["encoder_target_width"],
        "image_aug": args.aug,
        # Idle-transition handling — see PushTWidowXVideoDataset.
        "idle_filter": args.idle_filter,
        "idle_eps": args.idle_eps,
        "idle_move_eps": args.idle_move_eps,
        "idle_keep_frac": args.idle_keep_frac,
        "cond_eef_xy": args.cond_eef_xy,
        "training": {
            "training_steps": args.steps,
            "batch_size": args.batch_size,
            "trial_seed": args.seed,
            "learning_rate": IBC_BEST["learning_rate"],
            "lr_decay_rate": IBC_BEST["lr_decay_rate"],
            "lr_decay_steps": IBC_BEST["lr_decay_steps"],
            "num_counter_examples": args.counter_examples,
            "uniform_boundary_buffer": IBC_BEST["uniform_boundary_buffer"],
            "softmax_temperature": IBC_BEST["softmax_temperature"],
        },
        "model": {
            "kind": "ibc_pixel_ebm",
            "encoder_kind": "conv_maxpool",
            "value_width": IBC_BEST["value_width"],
            "value_num_blocks": IBC_BEST["value_num_blocks"],
        },
        "inference": {
            "dfo_samples": IBC_BEST["inference_dfo_samples"],
            "dfo_iterations": IBC_BEST["inference_dfo_iterations"],
            "dfo_iteration_std": IBC_BEST["inference_dfo_iteration_std"],
            "dfo_std_decay": IBC_BEST["inference_dfo_std_decay"],
            "uniform_boundary_buffer": IBC_BEST["uniform_boundary_buffer"],
        },
    }
    if args.frame_cache_dir is not None:
        env["frame_cache_dir"] = str(args.frame_cache_dir.resolve())
    return {
        "active_env": "pusht_real_pixels_ibc",
        "environments": {"pusht_real_pixels_ibc": env},
        "training_shared": {
            "model_save_dir": str(run_dir),
            "save_interval": args.save_interval,
        },
    }


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
    config = build_run_config(args, run_dir)
    config_path = run_dir / "config.json"
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

    # Deterministic seeding — same seed => same training trajectory.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")

    from utils.models import PixelQEstimator

    if args.data_format == "zarr_video":
        from utils.datasets import PushTWidowXVideoDataset

        dataset = PushTWidowXVideoDataset(
            archive_path=str(args.dataset),
            frame_stack=args.frame_stack,
            camera=args.video_camera,
            resize_hw=(args.image_height, args.image_width),
            normalize_actions=True,
            action_norm_range=(-1.0, 1.0),
            augment=args.aug,
            idle_filter=args.idle_filter,
            idle_eps=args.idle_eps,
            idle_move_eps=args.idle_move_eps,
            idle_keep_frac=args.idle_keep_frac,
            idle_seed=args.seed,
            cache_dir=(
                str(args.frame_cache_dir.resolve())
                if args.frame_cache_dir is not None
                else None
            ),
            cond_eef_xy=args.cond_eef_xy,
        )
    else:
        from utils.datasets import PushTRealPixelsDataset

        if args.cond_eef_xy:
            raise ValueError("--cond-eef-xy requires --data-format zarr_video")
        dataset = PushTRealPixelsDataset(
            archive_path=str(args.dataset),
            frame_stack=args.frame_stack,
            camera_streams=tuple(args.cameras),
            resize_hw=(args.image_height, args.image_width),
            normalize_actions=True,
            action_norm_range=(-1.0, 1.0),
            augment=args.aug,
        )
    cond_dim = int(getattr(dataset, "cond_dim", 0))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )

    ebm = PixelQEstimator(
        action_dim=2,
        in_channels=dataset.in_channels,
        encoder_target_height=IBC_BEST["encoder_target_height"],
        encoder_target_width=IBC_BEST["encoder_target_width"],
        value_width=IBC_BEST["value_width"],
        value_num_blocks=IBC_BEST["value_num_blocks"],
        cond_dim=cond_dim,
    ).to(device)
    n_params = sum(p.numel() for p in ebm.parameters())
    print(f"PixelEBM (ConvMaxpool + DenseResnetValue 1024x1): {n_params:,} params")
    if cond_dim:
        print(
            f"  conditioned on {cond_dim}-D EEF (x, y) — this DEVIATES from "
            f"pixel_ebm_best.gin, which is pixels-only"
        )

    lr0 = IBC_BEST["learning_rate"]
    optimizer = torch.optim.Adam(ebm.parameters(), lr=lr0)

    # norm_stats.pt: everything the deploy client needs to rebuild the exact
    # preprocessing + denormalization (mirrors the q3c trainer's file).
    norm_stats = {
        "act_min": dataset.act_min,
        "act_max": dataset.act_max,
        "action_norm_range": (-1.0, 1.0),
        "frame_stack": args.frame_stack,
        # Take the stream names from the dataset so zarr_video records
        # ("video1",) rather than the unused bridge_zip --cameras default.
        "camera_streams": list(dataset.camera_streams),
        "in_channels": dataset.in_channels,
        "image_hw": [args.image_height, args.image_width],
        "encoder_target_height": IBC_BEST["encoder_target_height"],
        "encoder_target_width": IBC_BEST["encoder_target_width"],
        "value_width": IBC_BEST["value_width"],
        "value_num_blocks": IBC_BEST["value_num_blocks"],
        "encoder_kind": "conv_maxpool",
        "cond_dim": cond_dim,
        "data_format": args.data_format,
        "idle_filter": args.idle_filter,
        "action_semantics": dataset.action_semantics,
    }
    torch.save(norm_stats, run_dir / "norm_stats.pt")

    n_counter = args.counter_examples
    buffer = IBC_BEST["uniform_boundary_buffer"]
    a_lo, a_hi = -1.0, 1.0
    sample_lo = a_lo - (a_hi - a_lo) * buffer
    sample_hi = a_hi + (a_hi - a_lo) * buffer
    temperature = IBC_BEST["softmax_temperature"]
    decay_rate = IBC_BEST["lr_decay_rate"]
    decay_steps = IBC_BEST["lr_decay_steps"]

    def save_ckpt(tag: str, step: int) -> None:
        torch.save(ebm.state_dict(), run_dir / f"q_estimator{tag}.pt")
        (run_dir / "last_step.json").write_text(json.dumps({"step": step}) + "\n")

    step = 0
    start = time.time()
    log_interval = 500
    consecutive_nan = 0
    print(f"Training {args.steps} steps: batch={args.batch_size}, "
          f"lr={lr0} (x{decay_rate}/{decay_steps} steps), "
          f"{n_counter} uniform counter-examples in [{sample_lo}, {sample_hi}]")

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            states = batch["state"].to(device, non_blocking=True)
            expert = batch["action"].float().to(device, non_blocking=True)
            B = expert.shape[0]
            # PixelQEstimator late-fuses this alongside the image features.
            if cond_dim:
                ebm._cond = batch["cond"].float().to(device, non_blocking=True)

            # 256 uniform counter-examples with the 5% boundary buffer, expert
            # appended last (official ibc_agent: negatives + expert -> softmax
            # cross-entropy against the expert index).
            negatives = (
                torch.rand(B, n_counter, 2, device=device) * (sample_hi - sample_lo)
                + sample_lo
            )
            all_actions = torch.cat([negatives, expert.unsqueeze(1)], dim=1)

            # Late fusion: encoder once (with grad — it trains through the
            # InfoNCE), value head scores all candidates on cached features.
            features = ebm.encode(states)
            logits = ebm.score(features, all_actions).squeeze(-1) / temperature
            log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            loss = -log_probs[:, -1].mean()

            if torch.isnan(loss):
                consecutive_nan += 1
                optimizer.state.clear()
                if consecutive_nan >= 50:
                    raise RuntimeError("Training diverged: 50 consecutive NaN batches")
                continue
            consecutive_nan = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            # keras ExponentialDecay (staircase=False): lr = lr0 * rate^(t/T).
            lr = lr0 * decay_rate ** (step / decay_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if step % log_interval == 0:
                with torch.no_grad():
                    acc = (logits.argmax(dim=1) == n_counter).float().mean().item()
                elapsed = time.time() - start
                print(f"  Step {step}/{args.steps} | NCE: {loss.item():.4f} | "
                      f"Acc: {acc:.3f} | LR: {lr:.2e} | {elapsed:.1f}s", flush=True)
            if step % args.save_interval == 0:
                save_ckpt(f"_step{step:06d}", step)

    save_ckpt("", args.steps)
    total = time.time() - start
    print(f"Done in {total / 60:.1f} min. Final weights: {run_dir / 'q_estimator.pt'}")
    print(f"Deploy with: python scripts/deploy_pusht_real_ibc.py --seed-dir {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
