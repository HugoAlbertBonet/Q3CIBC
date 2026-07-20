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
        default=ROOT / "data" / "03-23-pusht-data.zip",
        help="BridgeData-style Push-T demonstration zip",
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
        help="Ordered RGB streams. Default images1 only: the deploy rig runs "
             "a single fixed scene camera (blue), see PUSHT_DEPLOY_HANDOFF.md",
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
        "--no-aug",
        action="store_true",
        help="Disable train-time appearance augmentation (photometric + small "
             "view crop). Default ON for parity with the q3c v2 runs: deploy "
             "forensics measured the robot's T ~33%% darker than training, so "
             "lighting/color robustness must come from data. Pass --no-aug for "
             "a strictly paper-faithful (no-augmentation) run.",
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
    env = {
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
        "action_bounds": [-1.0, 1.0],
        "encoder_target_height": IBC_BEST["encoder_target_height"],
        "encoder_target_width": IBC_BEST["encoder_target_width"],
        "image_aug": not args.no_aug,
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

    run_dir = args.output_root.resolve() / f"seed_{args.seed:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = build_run_config(args, run_dir)
    config_path = run_dir / "config.json"
    with config_path.open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print(f"Dataset:    {args.dataset.resolve()}")
    print(f"Seed:       {args.seed}")
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

    from utils.datasets import PushTRealPixelsDataset
    from utils.models import PixelQEstimator

    dataset = PushTRealPixelsDataset(
        archive_path=str(args.dataset),
        frame_stack=args.frame_stack,
        camera_streams=tuple(args.cameras),
        resize_hw=(args.image_height, args.image_width),
        normalize_actions=True,
        action_norm_range=(-1.0, 1.0),
        augment=not args.no_aug,
    )
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
        cond_dim=0,
    ).to(device)
    n_params = sum(p.numel() for p in ebm.parameters())
    print(f"PixelEBM (ConvMaxpool + DenseResnetValue 1024x1): {n_params:,} params")

    lr0 = IBC_BEST["learning_rate"]
    optimizer = torch.optim.Adam(ebm.parameters(), lr=lr0)

    # norm_stats.pt: everything the deploy client needs to rebuild the exact
    # preprocessing + denormalization (mirrors the q3c trainer's file).
    norm_stats = {
        "act_min": dataset.act_min,
        "act_max": dataset.act_max,
        "action_norm_range": (-1.0, 1.0),
        "frame_stack": args.frame_stack,
        "camera_streams": list(args.cameras),
        "in_channels": dataset.in_channels,
        "image_hw": [args.image_height, args.image_width],
        "encoder_target_height": IBC_BEST["encoder_target_height"],
        "encoder_target_width": IBC_BEST["encoder_target_width"],
        "value_width": IBC_BEST["value_width"],
        "value_num_blocks": IBC_BEST["value_num_blocks"],
        "encoder_kind": "conv_maxpool",
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
