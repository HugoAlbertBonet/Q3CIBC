#!/usr/bin/env python3
"""Train a Diffusion Policy denoiser on the real-robot Push-T zip.

Diffusion-Policy counterpart of scripts/train_pusht_real.py (Q3C) and
scripts/train_pusht_real_ibc.py (IBC): same dataset, preprocessing, and
checkpoint layout, but the model is a conditional denoiser sampled at deploy
time with DDPM or DDIM.

ONE checkpoint serves BOTH samplers. DDPM and DDIM are inference-time
schedules over the same trained denoiser (see utils.diffusion.GaussianDiffusion
.ddpm_sample / .ddim_sample), so there is no "DDPM run" and "DDIM run" to
train separately — the sampler is chosen by the deploy client.

Recipe = the best pushing_pixels DP configuration (batches/pushingPixelsDPD.txt
and pushingPixelsDPE.txt), pixels being the closest environment to this rig:

    - PixelDiffusionDenoiser = ConvMaxpoolEncoder (target 180x240) ->
      256-D feature, conditioning a DenseResnet denoiser head (width 1024,
      1 block, 128-D time embedding). Encoder trained jointly.
    - T = 100 train timesteps, cosine beta schedule.
    - AdamW lr 3e-4, cosine anneal to 1e-6, grad-norm clip 1.0, batch 128.
    - EMA 0.999 (deploy uses the EMA weights).
    - 750k steps: the convergence ladder in pushingPixelsDPD showed DP is
      still improving at 300k/500k and only flattens near 750k (~5x the
      budget Q3C needs). Under-training DP is not a fair baseline.

--prediction-type selects the training target: "epsilon" (Ho et al. 2020, the
original Diffusion Policy) or "v" (Salimans & Ho 2022). Both are capacity- and
budget-matched; v-pred was the stronger of the two on pushing_pixels.

Each run writes an immutable per-run config + norm_stats.pt beside the
checkpoints so Slurm array tasks can train concurrently without races, and so
the deploy client can rebuild the exact model.
"""

from __future__ import annotations

import argparse
import copy
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


# Best pushing_pixels DP recipe (batches/pushingPixelsDPD.txt, DPE.txt).
DP_BEST = {
    "training_steps": 750_000,
    "batch_size": 128,
    "learning_rate": 3e-4,
    "lr_eta_min": 1e-6,
    "grad_clip_norm": 1.0,
    # Diffusion process.
    "num_train_timesteps": 100,
    "beta_schedule": "cosine",
    # Denoiser head (matched to the Q3C/IBC pixel value head).
    "time_emb_dim": 128,
    "denoiser_network_kind": "dense_resnet",
    "denoiser_width": 1024,
    "denoiser_depth": 1,
    "denoiser_use_spectral_norm": False,
    "ema_decay": 0.999,
    # Encoder (same ConvMaxpoolEncoder geometry as Q3C/IBC).
    "encoder_target_height": 180,
    "encoder_target_width": 240,
    "encoder_feature_dim": 256,
    # Eval-only sampler knobs, persisted for the deploy client.
    "ddim_eval_steps": [5, 10, 25],
    "ddim_eta": 0.0,
    "eval_ddpm": True,
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
    parser.add_argument(
        "--prediction-type",
        choices=["epsilon", "v"],
        default="v",
        help="Denoiser training target. v-pred was the stronger recipe on "
             "pushing_pixels; epsilon is the original Diffusion Policy.",
    )
    parser.add_argument("--steps", type=int, default=DP_BEST["training_steps"])
    parser.add_argument("--batch-size", type=int, default=DP_BEST["batch_size"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-stack", type=int, default=2,
                        help="Matches the Q3C/IBC real-robot runs")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["images1"],
        help="Ordered RGB streams. Default images1 only: the deploy rig runs "
             "a single fixed scene camera (blue), see PUSHT_DEPLOY_HANDOFF.md",
    )
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--save-interval", type=int, default=25_000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "checkpoints" / "pusht_real_dp",
    )
    parser.add_argument(
        "--no-aug",
        action="store_true",
        help="Disable train-time appearance augmentation (photometric + small "
             "view crop). Default ON for parity with the q3c v2 and IBC runs: "
             "deploy forensics measured the robot's T ~33%% darker than "
             "training, so lighting/color robustness must come from data.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and print the run config without starting training",
    )
    return parser.parse_args()


def build_run_config(args: argparse.Namespace, run_dir: Path) -> dict:
    """Immutable per-run config, shaped like the q3c/IBC seed dirs so the
    deploy client's load_run_config() pattern works unchanged."""
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
        "encoder_target_height": DP_BEST["encoder_target_height"],
        "encoder_target_width": DP_BEST["encoder_target_width"],
        "image_aug": not args.no_aug,
        "training": {
            "training_steps": args.steps,
            "batch_size": args.batch_size,
            "trial_seed": args.seed,
            "learning_rate": DP_BEST["learning_rate"],
        },
        "model": {
            "kind": "dp_pixel_denoiser",
            "encoder_kind": "conv_maxpool",
            "diffusion": {
                "num_train_timesteps": DP_BEST["num_train_timesteps"],
                "beta_schedule": DP_BEST["beta_schedule"],
                "prediction_type": args.prediction_type,
                "time_emb_dim": DP_BEST["time_emb_dim"],
                "denoiser_network_kind": DP_BEST["denoiser_network_kind"],
                "denoiser_width": DP_BEST["denoiser_width"],
                "denoiser_depth": DP_BEST["denoiser_depth"],
                "denoiser_use_spectral_norm": DP_BEST["denoiser_use_spectral_norm"],
                "ema_decay": DP_BEST["ema_decay"],
                "ddim_eval_steps": DP_BEST["ddim_eval_steps"],
                "ddim_eta": DP_BEST["ddim_eta"],
                "eval_ddpm": DP_BEST["eval_ddpm"],
            },
        },
        # DDPM and DDIM both read the single trained denoiser; these are the
        # schedules the deploy client should sweep.
        "inference": {
            "samplers": ["ddpm", "ddim"],
            "ddim_eval_steps": DP_BEST["ddim_eval_steps"],
            "ddim_eta": DP_BEST["ddim_eta"],
            "num_train_timesteps": DP_BEST["num_train_timesteps"],
        },
    }
    return {
        "active_env": "pusht_real_pixels_dp",
        "environments": {"pusht_real_pixels_dp": env},
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

    run_dir = (args.output_root.resolve()
               / f"{args.prediction_type}pred_seed_{args.seed:04d}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = build_run_config(args, run_dir)
    config_path = run_dir / "config.json"
    with config_path.open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print(f"Dataset:    {args.dataset.resolve()}")
    print(f"Seed:       {args.seed}")
    print(f"Pred type:  {args.prediction_type}")
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
    from utils.diffusion import build_diffusion, build_pixel_denoiser

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

    dp = {
        "num_train_timesteps": DP_BEST["num_train_timesteps"],
        "beta_schedule": DP_BEST["beta_schedule"],
        "prediction_type": args.prediction_type,
        "time_emb_dim": DP_BEST["time_emb_dim"],
        "denoiser_network_kind": DP_BEST["denoiser_network_kind"],
        "denoiser_width": DP_BEST["denoiser_width"],
        "denoiser_depth": DP_BEST["denoiser_depth"],
        "denoiser_use_spectral_norm": DP_BEST["denoiser_use_spectral_norm"],
    }
    denoiser = build_pixel_denoiser(
        2, dataset.in_channels, dp,
        encoder_target_height=DP_BEST["encoder_target_height"],
        encoder_target_width=DP_BEST["encoder_target_width"],
        device=device,
    )
    diffusion = build_diffusion(dp, device, (-1.0, 1.0))
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"PixelDiffusionDenoiser (ConvMaxpool + DenseResnet "
          f"{DP_BEST['denoiser_width']}x{DP_BEST['denoiser_depth']}, "
          f"{args.prediction_type}-pred): {n_params:,} params")

    optimizer = torch.optim.AdamW(denoiser.parameters(),
                                  lr=DP_BEST["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=DP_BEST["lr_eta_min"],
    )

    ema_decay = DP_BEST["ema_decay"]
    ema_denoiser = copy.deepcopy(denoiser)
    for p in ema_denoiser.parameters():
        p.requires_grad_(False)

    # norm_stats.pt: everything the deploy client needs to rebuild the exact
    # preprocessing + denormalization (mirrors the q3c and IBC trainers).
    norm_stats = {
        "act_min": dataset.act_min,
        "act_max": dataset.act_max,
        "action_norm_range": (-1.0, 1.0),
        "frame_stack": args.frame_stack,
        "camera_streams": list(args.cameras),
        "in_channels": dataset.in_channels,
        "image_hw": [args.image_height, args.image_width],
        "encoder_target_height": DP_BEST["encoder_target_height"],
        "encoder_target_width": DP_BEST["encoder_target_width"],
        "encoder_feature_dim": DP_BEST["encoder_feature_dim"],
        "encoder_kind": "conv_maxpool",
        "action_semantics": dataset.action_semantics,
        # Sampler reconstruction — DDPM/DDIM both rebuild GaussianDiffusion
        # from these, so the deploy client never has to guess.
        "num_train_timesteps": DP_BEST["num_train_timesteps"],
        "beta_schedule": DP_BEST["beta_schedule"],
        "prediction_type": args.prediction_type,
        "time_emb_dim": DP_BEST["time_emb_dim"],
        "denoiser_network_kind": DP_BEST["denoiser_network_kind"],
        "denoiser_width": DP_BEST["denoiser_width"],
        "denoiser_depth": DP_BEST["denoiser_depth"],
        "ddim_eval_steps": DP_BEST["ddim_eval_steps"],
        "ddim_eta": DP_BEST["ddim_eta"],
    }
    torch.save(norm_stats, run_dir / "norm_stats.pt")

    def save_ckpt(tag: str, step: int) -> None:
        torch.save(denoiser.state_dict(), run_dir / f"denoiser{tag}.pt")
        torch.save(ema_denoiser.state_dict(), run_dir / f"denoiser_ema{tag}.pt")
        (run_dir / "last_step.json").write_text(json.dumps({"step": step}) + "\n")

    step = 0
    start = time.time()
    log_interval = 500
    running_loss = 0.0
    running_n = 0
    print(f"Training {args.steps} steps: batch={args.batch_size}, "
          f"lr={DP_BEST['learning_rate']} (cosine -> {DP_BEST['lr_eta_min']}), "
          f"T={DP_BEST['num_train_timesteps']} {DP_BEST['beta_schedule']}, "
          f"EMA {ema_decay}")

    denoiser.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            # Pixels: no obs standardizer — ConvMaxpoolEncoder does
            # uint8 -> /255 -> resize internally (same as the q3c/IBC path).
            states = batch["state"].float().to(device, non_blocking=True)
            actions = batch["action"].float().to(device, non_blocking=True)

            loss = diffusion.training_loss(denoiser, states, actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(),
                                           max_norm=DP_BEST["grad_clip_norm"])
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for ep, p in zip(ema_denoiser.parameters(), denoiser.parameters()):
                    ep.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)

            running_loss += float(loss.item())
            running_n += 1
            step += 1

            if step % log_interval == 0:
                avg = running_loss / max(running_n, 1)
                elapsed = time.time() - start
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {step}/{args.steps} | Loss: {avg:.6f} | "
                      f"LR: {lr:.2e} | {elapsed:.1f}s", flush=True)
                running_loss = 0.0
                running_n = 0
            if step % args.save_interval == 0:
                save_ckpt(f"_step{step:06d}", step)

    save_ckpt("", args.steps)
    total = time.time() - start
    print(f"Done in {total / 60:.1f} min. Final weights: {run_dir / 'denoiser.pt'} "
          f"(deploy with denoiser_ema.pt)")
    print("Same checkpoint serves both samplers: DDPM (T=100) and "
          f"DDIM ({DP_BEST['ddim_eval_steps']} steps, eta={DP_BEST['ddim_eta']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
