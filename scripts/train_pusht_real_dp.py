#!/usr/bin/env python3
"""Train a Diffusion Policy denoiser on the real-robot Push-T archive.

Diffusion-Policy counterpart of scripts/train_pusht_real.py (Q3C) and
scripts/train_pusht_real_ibc.py (IBC): same dataset, preprocessing, idle-filter
handling, and checkpoint layout, but the model is a conditional denoiser
sampled at deploy time with DDPM or DDIM.

ONE checkpoint serves BOTH samplers. DDPM and DDIM are inference-time schedules
over the same trained denoiser (utils.diffusion.GaussianDiffusion.ddpm_sample /
.ddim_sample), so there is no "DDPM run" and "DDIM run" to train separately —
the sampler is chosen by the deploy client.

Recipe = the best pushing_pixels DP configuration (batches/pushingPixelsDPD.txt
and pushingPixelsDPE.txt), pixels being the closest environment to this rig:

    - PixelDiffusionDenoiser = ConvMaxpoolEncoder (target 180x240) ->
      256-D feature, conditioning a DenseResnet denoiser head (width 1024,
      1 block, 128-D time embedding). Encoder trained jointly. The head is
      the same DenseResnetValue architecture as the Q3C/IBC value net, so
      the three baselines stay capacity-matched.
    - T = 100 train timesteps, cosine beta schedule.
    - AdamW lr 3e-4, cosine anneal to 1e-6, grad-norm clip 1.0, batch 128.
    - EMA 0.999 (deploy uses the EMA weights).
    - 750k steps: the convergence ladder in pushingPixelsDPD showed DP is
      still improving at 300k/500k and only flattens near 750k (~5x the
      budget Q3C needs). Under-training DP is not a fair baseline.

--prediction-type selects the training target: "epsilon" (Ho et al. 2020, the
original Diffusion Policy) or "v" (Salimans & Ho 2022). Both are capacity- and
budget-matched; v-pred was the stronger of the two on pushing_pixels.

Unlike Q3C and IBC, DP samples from the fitted distribution rather than taking
its argmax, so the idle spike (24% of this archive's actions are exactly (0,0))
does not automatically dominate — but it is still a quarter of the density and
still absorbing on the robot. --idle-filter defaults to drop_zero to match the
other two baselines; run --idle-filter none as the control.

Each run writes an immutable per-run config + norm_stats.pt beside the
checkpoints so concurrent jobs never race, and so the deploy client can rebuild
the exact model.
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
import torch.nn as nn

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


class CondPixelDiffusionDenoiser(nn.Module):
    """PixelDiffusionDenoiser + an extra conditioning vector on the feature.

    utils.diffusion.PixelDiffusionDenoiser is pixels-only. --cond-eef-xy needs
    the denoiser head to also see the end-effector (x, y), exactly as
    PixelQEstimator's cond_dim does for IBC. Rather than change the shared
    module (whose pushing_pixels results are already published), this local
    subclass widens the head's state input by cond_dim and concatenates.

    The per-batch conditioning vector is handed over via `self._cond`, mirroring
    the `ebm._cond` convention in scripts/train_pusht_real_ibc.py.
    """

    def __init__(self, action_dim: int, *, in_channels: int, cond_dim: int,
                 encoder_target_height: int, encoder_target_width: int,
                 encoder_feature_dim: int, dp: dict) -> None:
        super().__init__()
        from utils.diffusion import DiffusionDenoiser
        from utils.models import ConvMaxpoolEncoder

        self.cond_dim = int(cond_dim)
        self._cond: torch.Tensor | None = None
        self.encoder = ConvMaxpoolEncoder(
            in_channels=in_channels,
            target_height=encoder_target_height,
            target_width=encoder_target_width,
            feature_dim=encoder_feature_dim,
        )
        self.denoiser = DiffusionDenoiser(
            state_dim=encoder_feature_dim + self.cond_dim,
            action_dim=action_dim,
            time_emb_dim=dp["time_emb_dim"],
            network_kind=dp["denoiser_network_kind"],
            width=dp["denoiser_width"],
            depth=dp["denoiser_depth"],
            use_spectral_norm=dp["denoiser_use_spectral_norm"],
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(images)
        if self.cond_dim:
            if self._cond is None:
                raise RuntimeError("cond_dim > 0 but no conditioning vector set")
            feat = torch.cat([feat, self._cond], dim=-1)
        return feat

    def forward(self, images: torch.Tensor, noisy_action: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        return self.denoiser(self.encode(images), noisy_action, t)


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
        "--action-chunk",
        type=int,
        default=1,
        help="Predict K planar deltas per step (open-loop action chunking), so "
             "the denoiser's action_dim becomes 2*K. K=1 is single-step "
             "(default); K=16 matches the libero_goal_pixels recipe. "
             "deploy_pusht_real_dp.py already executes chunks -- it reads the "
             "chunk length off act_min and --exec-horizon picks how many of the "
             "K deltas run open-loop before re-predicting. zarr_video only.",
    )
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
             "deploy client reads); camera 0 is a second viewpoint, discarded. "
             "Ignored when --video-cameras is given.",
    )
    parser.add_argument(
        "--video-cameras",
        type=int,
        nargs="+",
        default=None,
        help="zarr_video only: ordered list of per-episode MP4 cameras to stack "
             "as input (e.g. `0 1` for both views). Overrides --video-camera. "
             "Each camera adds 3 channels per stacked frame; a per-camera frame "
             "cache is built once per camera.",
    )
    parser.add_argument(
        "--frame-cache-dir",
        type=Path,
        default=None,
        help="zarr_video only: where to build the decoded uint8 frame memmap "
             "(default: <dataset dir>/_frame_cache). ~17 GB for this collection "
             "at 240x320; build it once up front with "
             "scripts/prepare_pusht_video_cache.py, or concurrent jobs will "
             "idle on GPUs waiting for whichever one wins the build lock.",
    )
    # ── Idle-transition handling (the stalling fix) ───────────────────────
    parser.add_argument(
        "--idle-filter",
        choices=["none", "drop_zero", "drop_static", "subsample"],
        default="drop_zero",
        help="How to treat transitions whose target action is ~0. 24%% of this "
             "dataset is the teleoperator pausing. DP is less exposed to this "
             "than Q3C/IBC — it SAMPLES the fitted distribution instead of "
             "taking its argmax, so the spike does not automatically win — but "
             "a quarter of the density still sits on an absorbing action. "
             "Measured on the full archive: none -> 24.1%% zeros kept; "
             "drop_static -> 21.5%% (removes only 3.3%%, the spike survives); "
             "drop_zero -> 0%% (removes 24.1%%). Default 'drop_zero' matches "
             "scripts/train_pusht_real.py and train_pusht_real_ibc.py so all "
             "three baselines stay comparable.",
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
        help="zarr_video only: condition the denoiser on the current "
             "end-effector (x, y). NOTE this deviates from the pushing_pixels "
             "DP recipe, which is pixels-only — enable it only to match a Q3C "
             "or IBC run trained the same way. Requires a deploy client that "
             "feeds state[:2].",
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
        "--aug",
        action="store_true",
        help="Enable train-time appearance augmentation (photometric + small "
             "view crop). OFF by default, matching scripts/train_pusht_real.py "
             "and train_pusht_real_ibc.py: it targets the deploy lighting "
             "shift, which is a SEPARATE problem from the stalling, and "
             "leaving it on would confound the idle-filter comparison.",
    )
    # ── Held-out validation (episode-level split; live generalization MAE) ──
    parser.add_argument(
        "--val-frac", type=float, default=0.0,
        help="Hold out this fraction of EPISODES (no frame leakage) as a "
             "validation set. The trainer logs a live held-out sampled-action "
             "MAE. 0.0 (default) = no split, train on everything.",
    )
    parser.add_argument(
        "--val-seed", type=int, default=0,
        help="RNG seed choosing which episodes are held out (share it across a "
             "sweep so every run holds out the SAME episodes — comparable MAE).",
    )
    parser.add_argument(
        "--val-interval", type=int, default=None,
        help="Steps between held-out MAE evals (default: --save-interval).",
    )
    # ── Encoder architecture (conv_maxpool default, resnet18 optional) ──────
    parser.add_argument(
        "--encoder-kind", choices=["conv_maxpool", "resnet18"],
        default="conv_maxpool",
        help="conv_maxpool = IBC ConvMaxpool (default, the pushing_pixels DP "
             "recipe); resnet18 = torchvision ResNet-18 + SpatialSoftmax (the "
             "LIBERO-standard BC encoder).",
    )
    parser.add_argument(
        "--encoder-pretrained", choices=["none", "imagenet"], default="none",
        help="resnet18 only: 'imagenet' loads ImageNet-pretrained weights, "
             "'none' trains from scratch.",
    )
    parser.add_argument(
        "--encoder-norm-kind", choices=["bn", "gn", "bn_frozen"], default="bn",
        help="resnet18 only: normalization ('gn' = GroupNorm).",
    )
    parser.add_argument("--encoder-num-kp", type=int, default=64,
                        help="resnet18 only: SpatialSoftmax keypoints.")
    parser.add_argument(
        "--encoder-per-camera", action="store_true",
        help="Give each camera stream its own encoder instead of a shared one.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Run-directory name under --output-root (default: "
             "<pred>pred_seed_XXXX). Use a distinct tag per config so batch "
             "runs don't overwrite each other.",
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
    # zarr_video trains on ONE camera (the blue scene view); bridge_zip keeps
    # the explicit stream list. Both end up as a 1-element channel group so the
    # in_channels arithmetic is identical.
    # zarr_video can now stack MULTIPLE per-episode cameras (--video-cameras);
    # bridge_zip keeps its explicit stream list. Both end up as n_cams channel
    # groups so the in_channels arithmetic (3 * n_cams * frame_stack) is shared.
    video_cameras = (
        list(args.video_cameras) if args.video_cameras
        else [args.video_camera]
    )
    n_cams = len(video_cameras) if args.data_format == "zarr_video" else len(args.cameras)
    env = {
        "data_archive": str(args.dataset.resolve()),
        "data_format": args.data_format,
        "env_id": "PushTRealRobot-v0",
        "state_dim": [
            3 * n_cams * args.frame_stack,
            args.image_height,
            args.image_width,
        ],
        # 2 planar deltas per chunk entry; K=1 keeps the old 2-D action exactly.
        "action_dim": 2 * args.action_chunk,
        "action_chunk": args.action_chunk,
        "frame_stack": args.frame_stack,
        "camera_streams": (
            [f"video{c}" for c in video_cameras]
            if args.data_format == "zarr_video"
            else list(args.cameras)
        ),
        "video_camera": video_cameras[0] if args.data_format == "zarr_video" else args.video_camera,
        "video_cameras": video_cameras,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "action_bounds": [-1.0, 1.0],
        "encoder_target_height": DP_BEST["encoder_target_height"],
        "encoder_target_width": DP_BEST["encoder_target_width"],
        "image_aug": args.aug,
        # Idle-transition handling — see PushTWidowXVideoDataset.
        "idle_filter": args.idle_filter,
        "idle_eps": args.idle_eps,
        "idle_move_eps": args.idle_move_eps,
        "idle_keep_frac": args.idle_keep_frac,
        "cond_eef_xy": args.cond_eef_xy,
        # Episode-level held-out validation (0.0 = off).
        "val_frac": args.val_frac,
        "val_seed": args.val_seed,
        "training": {
            "training_steps": args.steps,
            "batch_size": args.batch_size,
            "trial_seed": args.seed,
            "learning_rate": DP_BEST["learning_rate"],
        },
        "model": {
            "kind": "dp_pixel_denoiser",
            "encoder_kind": args.encoder_kind,
            "encoder_pretrained": (args.encoder_pretrained == "imagenet"),
            "encoder_norm_kind": args.encoder_norm_kind,
            "encoder_num_kp": args.encoder_num_kp,
            "encoder_per_camera": args.encoder_per_camera,
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
    if args.frame_cache_dir is not None:
        env["frame_cache_dir"] = str(args.frame_cache_dir.resolve())
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

    run_name = args.tag if args.tag else f"{args.prediction_type}pred_seed_{args.seed:04d}"
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
    print(f"Pred type:  {args.prediction_type}")
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

    from utils.diffusion import build_diffusion, build_pixel_denoiser

    video_cameras = (
        list(args.video_cameras) if args.video_cameras
        else [args.video_camera]
    )
    # Episode-level held-out validation: train on the "train" split, eval MAE on
    # "val". val_frac=0 => split "all" (train on everything), val dataset None.
    use_val = args.data_format == "zarr_video" and args.val_frac > 0.0
    val_dataset = None
    if args.data_format == "zarr_video":
        from utils.datasets import PushTWidowXVideoDataset

        ds_common = dict(
            archive_path=str(args.dataset),
            frame_stack=args.frame_stack,
            cameras=video_cameras,
            resize_hw=(args.image_height, args.image_width),
            normalize_actions=True,
            action_norm_range=(-1.0, 1.0),
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
            val_frac=args.val_frac,
            val_seed=args.val_seed,
            action_chunk=args.action_chunk,
        )
        dataset = PushTWidowXVideoDataset(
            augment=args.aug,
            split=("train" if use_val else "all"),
            **ds_common,
        )
        if use_val:
            # Val mirrors deploy: no augmentation, same normalization stats.
            val_dataset = PushTWidowXVideoDataset(
                augment=False, split="val", **ds_common,
            )
    else:
        from utils.datasets import PushTRealPixelsDataset

        if args.cond_eef_xy:
            raise ValueError("--cond-eef-xy requires --data-format zarr_video")
        if args.action_chunk > 1:
            # PushTRealPixelsDataset pins self.action_chunk = 1, so asking for a
            # chunk here would silently train a single-step model under a
            # chunked config -- the deploy client would then read chunk_len from
            # act_min, see 1, and clamp --exec-horizon back to 1.
            raise ValueError("--action-chunk > 1 requires --data-format zarr_video")
        if args.val_frac > 0.0:
            raise ValueError("--val-frac is only wired for --data-format zarr_video")
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
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=max(1, args.workers // 2),
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
        )
        print(f"Held-out val: {len(val_dataset)} samples "
              f"({args.val_frac:.0%} of episodes, seed {args.val_seed})")

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
    # Width of one training target. Read off the dataset rather than recomputed
    # from --action-chunk so the model can never disagree with the chunked
    # vector the loader actually yields (or with act_min, which the deploy
    # client divides by 2 to recover the chunk length).
    action_dim = int(np.asarray(dataset.act_min).size)
    if action_dim != 2 * args.action_chunk:
        raise RuntimeError(
            f"dataset action width {action_dim} != 2 * --action-chunk "
            f"{args.action_chunk}; the chunking did not reach the dataset")
    if args.action_chunk > 1:
        print(f"Action chunking: K={args.action_chunk} -> action_dim={action_dim} "
              f"(deploy executes the first --exec-horizon of the K deltas)")

    if cond_dim:
        # Local subclass — pixels-only, conv_maxpool-only (the resnet18 encoder
        # is not wired through the cond head).
        if args.encoder_kind != "conv_maxpool":
            raise ValueError(
                "--cond-eef-xy is only supported with --encoder-kind conv_maxpool")
        denoiser = CondPixelDiffusionDenoiser(
            action_dim,
            in_channels=dataset.in_channels,
            cond_dim=cond_dim,
            encoder_target_height=DP_BEST["encoder_target_height"],
            encoder_target_width=DP_BEST["encoder_target_width"],
            encoder_feature_dim=DP_BEST["encoder_feature_dim"],
            dp=dp,
        ).to(device)
    else:
        denoiser = build_pixel_denoiser(
            action_dim, dataset.in_channels, dp,
            encoder_target_height=DP_BEST["encoder_target_height"],
            encoder_target_width=DP_BEST["encoder_target_width"],
            encoder_feature_dim=DP_BEST["encoder_feature_dim"],
            encoder_kind=args.encoder_kind,
            encoder_pretrained=(args.encoder_pretrained == "imagenet"),
            encoder_num_kp=args.encoder_num_kp,
            encoder_norm_kind=args.encoder_norm_kind,
            encoder_per_camera=args.encoder_per_camera,
            device=device,
        )
    diffusion = build_diffusion(dp, device, (-1.0, 1.0))
    n_params = sum(p.numel() for p in denoiser.parameters())
    enc_desc = (args.encoder_kind
                + ("-imagenet" if args.encoder_pretrained == "imagenet" else "")
                + (f" x{len(video_cameras)}cam" if args.data_format == "zarr_video" else ""))
    print(f"PixelDiffusionDenoiser ({enc_desc} + DenseResnet "
          f"{DP_BEST['denoiser_width']}x{DP_BEST['denoiser_depth']}, "
          f"{args.prediction_type}-pred): {n_params:,} params")
    if cond_dim:
        print(f"  conditioned on {cond_dim}-D EEF (x, y) — this DEVIATES from "
              f"the pixels-only pushing_pixels DP recipe")

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
        # Take the stream names from the dataset so zarr_video records
        # ("video1",) rather than the unused bridge_zip --cameras default.
        "camera_streams": list(dataset.camera_streams),
        "in_channels": dataset.in_channels,
        "image_hw": [args.image_height, args.image_width],
        "encoder_target_height": DP_BEST["encoder_target_height"],
        "encoder_target_width": DP_BEST["encoder_target_width"],
        "encoder_feature_dim": DP_BEST["encoder_feature_dim"],
        "encoder_kind": args.encoder_kind,
        # resnet18-specific knobs (ignored by conv_maxpool) — the deploy /
        # diagnose clients rebuild the SAME encoder from these.
        "encoder_pretrained": (args.encoder_pretrained == "imagenet"),
        "encoder_num_kp": args.encoder_num_kp,
        "encoder_norm_kind": args.encoder_norm_kind,
        "encoder_per_camera": args.encoder_per_camera,
        "cond_dim": cond_dim,
        "data_format": args.data_format,
        "idle_filter": args.idle_filter,
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

    # ── Held-out validation: sampled-action MAE on the EMA denoiser (the copy
    #    the deploy client uses). DDIM with a few steps keeps the eval cheap. ──
    val_interval = args.val_interval or args.save_interval
    val_ddim_steps = int(DP_BEST["ddim_eval_steps"][0])

    @torch.no_grad()
    def _val_action_mae() -> float:
        ema_denoiser.eval()
        tot, n = 0.0, 0
        for vb in val_loader:
            vs = vb["state"].float().to(device, non_blocking=True)
            va = vb["action"].float().to(device, non_blocking=True)
            if cond_dim:
                ema_denoiser._cond = vb["cond"].float().to(device, non_blocking=True)
            pred = diffusion.ddim_sample(ema_denoiser, vs, action_dim=action_dim,
                                         num_steps=val_ddim_steps, eta=0.0)
            tot += (pred - va).abs().mean().item() * vs.shape[0]
            n += vs.shape[0]
        return tot / max(n, 1)

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
            if cond_dim:
                denoiser._cond = batch["cond"].float().to(device, non_blocking=True)

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

            running_loss += float(loss.detach())
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
            if val_loader is not None and step % val_interval == 0:
                v = _val_action_mae()
                print(f"  [val] step {step}/{args.steps} | "
                      f"held-out sampled-action MAE (ddim{val_ddim_steps}): "
                      f"{v:.5f}", flush=True)

    save_ckpt("", args.steps)
    total = time.time() - start
    print(f"Done in {total / 60:.1f} min. Final weights: {run_dir / 'denoiser.pt'} "
          f"(deploy with denoiser_ema.pt)")
    print("Same checkpoint serves both samplers: DDPM (T=100) and "
          f"DDIM ({DP_BEST['ddim_eval_steps']} steps, eta={DP_BEST['ddim_eta']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
