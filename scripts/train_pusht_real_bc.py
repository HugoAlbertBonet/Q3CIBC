#!/usr/bin/env python3
"""Train an EXPLICIT behavior-cloning policy (MSE regression) on the real-robot
Push-T archive.

Explicit-BC counterpart of scripts/train_pusht_real.py (Q3C),
train_pusht_real_ibc.py (IBC / implicit BC) and train_pusht_real_dp.py
(Diffusion Policy): same dataset, preprocessing, idle-filter handling, camera
selection, action chunking, validation split and checkpoint layout — only the
policy class and the loss change.

    pi(o) -> a, trained with  L = mean((pi(o) - a_expert)^2)

There is no energy model, no counter-examples, no sampler and no iterative
inference: one forward pass returns the action. That is the point of the
baseline — it is the classical explicit-policy control the IBC paper argues
against (Florence et al. 2021, Sec. 4), so it bounds from below what the
implicit/generative methods have to beat, at ~1 net evaluation per action.

Model (deliberately capacity-matched to the Q3C CP generator so the comparison
is about the objective, not the parameter count):

    utils.models.PixelControlPointGenerator with control_points=1

i.e. the SAME image encoder (ConvMaxpool or ResNet-18 + SpatialSoftmax, all the
same knobs) feeding the SAME MLP/ResNet head, which here emits a single
tanh-bounded action instead of a cloud of candidates. Reusing that class rather
than writing a bespoke net is what lets scripts/deploy_pusht_real_bc.py rebuild
the policy with the exact code path the other deploy clients use.

Multimodality caveat, stated up front so the numbers are read correctly: MSE
regression fits the CONDITIONAL MEAN. Where the demonstrations are multimodal
(two equally good ways around the T), the mean of the modes is itself not a
valid action, and the policy averages them. Action chunking (--action-chunk)
mitigates this somewhat by committing to a trajectory segment rather than
re-averaging every step; nothing else here does.

--idle-filter defaults to drop_zero, matching the other three trainers: 24% of
this archive's actions are exactly (0,0) (the teleoperator pausing). For an MSE
policy those do not create an absorbing argmax the way they do for Q3C/IBC, but
they DO drag the regression target toward zero everywhere they are frequent,
which is the same freeze on the robot by a different route.

Each run writes an immutable per-run config + norm_stats.pt beside the
checkpoints so concurrent Slurm jobs never race, and so the deploy client can
rebuild the exact model.
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


# Defaults: the pixel encoder geometry shared by every Push-T baseline here, and
# a plain, well-behaved regression recipe (AdamW + cosine + grad clip + EMA).
BC_DEFAULTS = {
    "training_steps": 150_000,
    "batch_size": 128,
    "learning_rate": 3e-4,
    "lr_eta_min": 1e-6,
    "weight_decay": 1e-6,
    "grad_clip_norm": 1.0,
    "ema_decay": 0.999,
    "encoder_target_height": 180,
    "encoder_target_width": 240,
    "encoder_feature_dim": 256,
    "head_width": 512,
    "head_depth": 4,
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
    parser.add_argument("--steps", type=int, default=BC_DEFAULTS["training_steps"])
    parser.add_argument("--batch-size", type=int, default=BC_DEFAULTS["batch_size"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-stack", type=int, default=2,
                        help="Matches the Q3C/IBC/DP real-robot runs")
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
             "deploy client reads); camera 0 is a second viewpoint. Ignored "
             "when --video-cameras is given.",
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
        "--action-chunk",
        type=int,
        default=1,
        help="Predict K planar deltas per step (open-loop action chunking), so "
             "the head's output width becomes 2*K. K=1 is single-step "
             "(default); K=16 matches the libero_goal_pixels recipe. "
             "deploy_pusht_real_bc.py reads the chunk length back from "
             "norm_stats['act_min'].size // 2 and --exec-horizon picks how many "
             "of the K deltas run open-loop before re-predicting. zarr_video "
             "only.",
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
    # ── Idle-transition handling ──────────────────────────────────────────
    parser.add_argument(
        "--idle-filter",
        choices=["none", "drop_zero", "drop_static", "subsample"],
        default="drop_zero",
        help="How to treat transitions whose target action is ~0. 24%% of this "
             "dataset is the teleoperator pausing. An MSE policy does not take "
             "an argmax, so the (0,0) spike does not win outright the way it "
             "does for Q3C/IBC — but a quarter of the regression targets being "
             "exactly zero still shrinks the fitted conditional mean toward "
             "zero, which stalls the arm just the same. Measured on the full "
             "archive: none -> 24.1%% zeros kept; drop_static -> 21.5%% "
             "(removes only 3.3%%, the spike survives); drop_zero -> 0%% "
             "(removes 24.1%%). Default 'drop_zero' matches the other three "
             "trainers so the baselines stay comparable.",
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
             "Requires a deploy client that feeds state[:2].",
    )
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--save-interval", type=int, default=10_000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "checkpoints" / "pusht_real_bc",
    )
    # ── Appearance augmentation ───────────────────────────────────────────
    parser.add_argument(
        "--aug-photometric",
        action="store_true",
        help="PHOTOMETRIC-ONLY augmentation: brightness / contrast / saturation "
             "/ per-channel-gain jitter to harden against the deploy lighting "
             "shift, with NO crop-zoom (zoom fixed at 1.0). Preferred over "
             "--aug for this FIXED camera, where a view crop would translate "
             "the scene and break the pixel->world map.",
    )
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Train-time appearance augmentation (photometric + small view "
             "crop). OFF by default, matching the other trainers.",
    )
    # ── Held-out validation (episode-level split) ─────────────────────────
    parser.add_argument(
        "--val-frac", type=float, default=0.0,
        help="Hold out this fraction of EPISODES (no frame leakage). The "
             "trainer logs a live held-out action-MAE plus the train-batch MAE "
             "and their gap. 0.0 (default) = train on everything.",
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
    # ── Image encoder ─────────────────────────────────────────────────────
    enc = parser.add_argument_group("encoder")
    enc.add_argument(
        "--encoder-kind", choices=["conv_maxpool", "resnet18"],
        default="conv_maxpool",
        help="conv_maxpool = IBC ConvMaxpool (default); resnet18 = torchvision "
             "ResNet-18 + SpatialSoftmax (the LIBERO-standard BC encoder).",
    )
    enc.add_argument(
        "--encoder-pretrained", choices=["none", "imagenet"], default="none",
        help="resnet18 only: 'imagenet' loads ImageNet-pretrained weights.",
    )
    enc.add_argument(
        "--encoder-norm-kind", choices=["bn", "gn", "bn_frozen"], default="bn",
        help="resnet18 only: normalization ('gn' = GroupNorm, the "
             "libero_goal_pixels choice).",
    )
    enc.add_argument("--encoder-num-kp", type=int, default=64,
                     help="resnet18 only: SpatialSoftmax keypoints (libero=128).")
    enc.add_argument(
        "--encoder-per-camera", action="store_true",
        help="Give each camera stream its own encoder instead of a shared one.",
    )
    enc.add_argument(
        "--cond-fusion", choices=["concat", "film"], default="concat",
        help="How conditioning is fused into the pixel net. Only relevant with "
             "--cond-eef-xy.",
    )
    # ── Policy head / optimization ────────────────────────────────────────
    hp = parser.add_argument_group("hyperparameters")
    hp.add_argument("--head-width", type=int, default=BC_DEFAULTS["head_width"],
                    help="MLP width of the regression head (the Q3C runs' "
                         "--cp-width; keep them equal to capacity-match).")
    hp.add_argument("--head-depth", type=int, default=BC_DEFAULTS["head_depth"],
                    help="MLP depth of the regression head (the Q3C runs' "
                         "--cp-depth).")
    hp.add_argument("--head-network-kind", choices=["mlp", "resnet"],
                    default="mlp",
                    help="Head backbone: plain MLP (default) or "
                         "ResNetPreActivation blocks of --head-width.")
    hp.add_argument("--lr", type=float, default=BC_DEFAULTS["learning_rate"])
    hp.add_argument("--lr-eta-min", type=float, default=BC_DEFAULTS["lr_eta_min"],
                    help="Final LR of the cosine anneal.")
    hp.add_argument("--weight-decay", type=float,
                    default=BC_DEFAULTS["weight_decay"])
    hp.add_argument("--grad-clip-norm", type=float,
                    default=BC_DEFAULTS["grad_clip_norm"],
                    help="0 disables gradient clipping.")
    hp.add_argument("--ema-decay", type=float, default=BC_DEFAULTS["ema_decay"],
                    help="Deploy uses the EMA weights by default.")
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
    """Immutable per-run config, shaped like the q3c/IBC/DP seed dirs so the
    deploy client's load_run_config() pattern works unchanged."""
    # zarr_video can stack MULTIPLE per-episode cameras (--video-cameras);
    # bridge_zip keeps its explicit stream list. Both end up as n_cams channel
    # groups so the in_channels arithmetic (3 * n_cams * frame_stack) is shared.
    if args.data_format == "zarr_video":
        video_cameras = (list(args.video_cameras) if args.video_cameras
                         else [args.video_camera])
        n_cams = len(video_cameras)
    else:
        video_cameras = None
        n_cams = len(args.cameras)
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
        "video_camera": (video_cameras[0] if args.data_format == "zarr_video"
                         else args.video_camera),
        "video_cameras": video_cameras,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "action_bounds": [-1.0, 1.0],
        "encoder_target_height": BC_DEFAULTS["encoder_target_height"],
        "encoder_target_width": BC_DEFAULTS["encoder_target_width"],
        "image_aug": bool(args.aug or args.aug_photometric),
        "image_aug_params": (
            {"zoom_range": [1.0, 1.0]} if args.aug_photometric else None
        ),
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
            "learning_rate": args.lr,
            "lr_eta_min": args.lr_eta_min,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "ema_decay": args.ema_decay,
            "loss": "mse",
            "action_chunk": args.action_chunk,
            **({"val_interval": args.val_interval}
               if args.val_interval is not None else {}),
        },
        "model": {
            # Explicit policy: one deterministic action per observation. The
            # module is PixelControlPointGenerator with control_points=1, so the
            # deploy client rebuilds it through the same code path as Q3C.
            "kind": "bc_pixel_policy",
            "control_points": 1,
            "encoder_kind": args.encoder_kind,
            "encoder_pretrained": (args.encoder_pretrained == "imagenet"),
            "encoder_norm_kind": args.encoder_norm_kind,
            "encoder_num_kp": args.encoder_num_kp,
            "encoder_per_camera": args.encoder_per_camera,
            "encoder_feature_dim": BC_DEFAULTS["encoder_feature_dim"],
            "cond_fusion": args.cond_fusion,
            # Both spellings, as in train_pusht_real.py: cp_width/cp_depth are
            # what the pixel head actually reads, num_* are their fallbacks.
            "cp_width": args.head_width,
            "cp_depth": args.head_depth,
            "num_neurons": args.head_width,
            "num_hidden_layers": args.head_depth,
            "cp_network_kind": args.head_network_kind,
            "cp_use_spectral_norm": False,
        },
        # One forward pass, no search. Recorded so the deploy client and the
        # cost tables can state the inference contract without guessing.
        "inference": {"kind": "deterministic", "net_evals": 1},
    }
    if args.frame_cache_dir is not None:
        env["frame_cache_dir"] = str(args.frame_cache_dir.resolve())
    return {
        "active_env": "pusht_real_pixels_bc",
        "environments": {"pusht_real_pixels_bc": env},
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
    # a repeated --tag in a batch file is an easy mistake to make. norm_stats.pt
    # is written BEFORE the training loop, so it must not count as a "finished
    # run" — otherwise a crash during warmup blocks a clean re-run.
    existing = (
        [p for p in sorted(run_dir.glob("*.pt")) if p.name != "norm_stats.pt"]
        if run_dir.exists() else []
    )
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
    print("Policy:     explicit BC (MSE regression), 1 net eval per action")
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

    device = torch.device(args.device if (torch.cuda.is_available()
                                          or args.device == "cpu") else "cpu")

    from utils.models import PixelControlPointGenerator

    aug_on = bool(args.aug or args.aug_photometric)
    aug_params = {"zoom_range": [1.0, 1.0]} if args.aug_photometric else None
    video_cameras = (list(args.video_cameras) if args.video_cameras
                     else [args.video_camera])
    # Episode-level held-out validation: train on the "train" split, eval MAE on
    # "val". val_frac=0 => split "all" (train on everything), val dataset None.
    if args.data_format == "zarr_video":
        from utils.datasets import PushTWidowXVideoDataset

        def _make_video_ds(split: str):
            return PushTWidowXVideoDataset(
                archive_path=str(args.dataset),
                frame_stack=args.frame_stack,
                cameras=video_cameras,
                resize_hw=(args.image_height, args.image_width),
                normalize_actions=True,
                action_norm_range=(-1.0, 1.0),
                # Val mirrors deploy: no augmentation, same normalization stats.
                augment=aug_on if split != "val" else False,
                aug_params=aug_params,
                idle_filter=args.idle_filter,
                idle_eps=args.idle_eps,
                idle_move_eps=args.idle_move_eps,
                idle_keep_frac=args.idle_keep_frac,
                idle_seed=args.seed,
                cache_dir=(str(args.frame_cache_dir.resolve())
                           if args.frame_cache_dir is not None else None),
                cond_eef_xy=args.cond_eef_xy,
                action_chunk=args.action_chunk,
                split=split,
                val_frac=args.val_frac,
                val_seed=args.val_seed,
            )

        use_val = args.val_frac > 0.0
        dataset = _make_video_ds("train" if use_val else "all")
        val_dataset = _make_video_ds("val") if use_val else None
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
            augment=aug_on,
        )
        val_dataset = None

    cond_dim = int(getattr(dataset, "cond_dim", 0))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
        timeout=600 if args.workers > 0 else 0,
    )
    # Single-process val loader (avoids the second-persistent-loader hang).
    val_loader = None
    val_interval = args.val_interval if args.val_interval is not None else args.save_interval
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        print(f"Held-out val: {len(val_dataset)} transitions "
              f"({args.val_frac:.0%} of episodes, seed {args.val_seed}); "
              f"action-MAE logged every {val_interval} steps.")

    # Width of one training target. Read off the dataset rather than recomputed
    # from --action-chunk so the model can never disagree with the chunked
    # vector the loader actually yields (or with act_min, which the deploy
    # client divides by 2 to recover the chunk length).
    action_dim = int(dataset.action_shape)
    if args.data_format == "zarr_video" and action_dim != 2 * args.action_chunk:
        raise RuntimeError(
            f"dataset action width {action_dim} != 2 * --action-chunk "
            f"{args.action_chunk}; the chunking did not reach the dataset")

    policy = PixelControlPointGenerator(
        output_dim=action_dim,
        # THE difference from Q3C: one action, not a cloud to be ranked.
        control_points=1,
        hidden_dims=[args.head_width for _ in range(args.head_depth)],
        action_bounds=(-1.0, 1.0),
        network_kind=args.head_network_kind,
        width=args.head_width,
        depth=args.head_depth,
        use_spectral_norm=False,
        in_channels=dataset.in_channels,
        encoder_target_height=BC_DEFAULTS["encoder_target_height"],
        encoder_target_width=BC_DEFAULTS["encoder_target_width"],
        encoder_feature_dim=BC_DEFAULTS["encoder_feature_dim"],
        cond_dim=cond_dim,
        encoder_kind=args.encoder_kind,
        encoder_pretrained=(args.encoder_pretrained == "imagenet"),
        encoder_num_kp=args.encoder_num_kp,
        encoder_norm_kind=args.encoder_norm_kind,
        encoder_per_camera=args.encoder_per_camera,
        cond_fusion=args.cond_fusion,
        goal_dim=0,
    ).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    enc_desc = (args.encoder_kind
                + ("-imagenet" if args.encoder_pretrained == "imagenet" else "")
                + (f" x{len(video_cameras)}cam" if args.data_format == "zarr_video"
                   else ""))
    print(f"Explicit BC policy ({enc_desc} + {args.head_network_kind} "
          f"{args.head_width}x{args.head_depth}, action_dim={action_dim}): "
          f"{n_params:,} params")
    if cond_dim:
        print(f"  conditioned on {cond_dim}-D EEF (x, y)")
    if args.action_chunk > 1:
        print(f"Action chunking: K={args.action_chunk} -> action_dim={action_dim} "
              f"(deploy executes the first --exec-horizon of the K deltas)")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr_eta_min)
    criterion = nn.MSELoss()

    ema_decay = float(args.ema_decay)
    ema_policy = copy.deepcopy(policy)
    for p in ema_policy.parameters():
        p.requires_grad_(False)

    # norm_stats.pt: everything the deploy client needs to rebuild the exact
    # preprocessing + denormalization (mirrors the q3c / IBC / DP trainers).
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
        "state_shape": list(dataset.state_shape),
        "encoder_target_height": BC_DEFAULTS["encoder_target_height"],
        "encoder_target_width": BC_DEFAULTS["encoder_target_width"],
        "encoder_feature_dim": BC_DEFAULTS["encoder_feature_dim"],
        "encoder_kind": args.encoder_kind,
        # resnet18-specific knobs (ignored by conv_maxpool) — the deploy client
        # rebuilds the SAME encoder from these.
        "encoder_pretrained": (args.encoder_pretrained == "imagenet"),
        "encoder_num_kp": args.encoder_num_kp,
        "encoder_norm_kind": args.encoder_norm_kind,
        "encoder_per_camera": args.encoder_per_camera,
        "cond_fusion": args.cond_fusion,
        # Head geometry, so the deploy client never has to read config.json's
        # model block to size the regression head.
        "policy_kind": "bc_pixel_policy",
        "control_points": 1,
        "cp_width": args.head_width,
        "cp_depth": args.head_depth,
        "cp_network_kind": args.head_network_kind,
        "action_chunk": args.action_chunk,
        "cond_dim": cond_dim,
        "data_format": args.data_format,
        "idle_filter": args.idle_filter,
        "action_semantics": dataset.action_semantics,
    }
    if cond_dim > 0:
        # Deploy must normalize the live proprio with these exact bounds or the
        # conditioning vector is silently on a different scale.
        norm_stats["cond_kind"] = "eef_xy"
        norm_stats["cond_min"] = dataset.cond_min
        norm_stats["cond_max"] = dataset.cond_max
    torch.save(norm_stats, run_dir / "norm_stats.pt")

    def save_ckpt(tag: str, step: int) -> None:
        torch.save(policy.state_dict(), run_dir / f"bc_policy{tag}.pt")
        torch.save(ema_policy.state_dict(), run_dir / f"bc_policy_ema{tag}.pt")
        (run_dir / "last_step.json").write_text(json.dumps({"step": step}) + "\n")

    @torch.no_grad()
    def val_action_mae() -> float:
        """Held-out action-MAE of the EMA policy — the weights deploy uses, and
        the exact same forward pass (no sampling, no search)."""
        ema_policy.eval()
        tot, n = 0.0, 0
        for vb in val_loader:
            vs = vb["state"].float().to(device, non_blocking=True)
            va = vb["action"].float().to(device, non_blocking=True)
            if cond_dim:
                ema_policy._cond = vb["cond"].float().to(device, non_blocking=True)
            pred = ema_policy(vs)[:, 0, :]
            tot += (pred - va).abs().sum().item()
            n += va.numel()
        return tot / max(n, 1)

    step = 0
    start = time.time()
    log_interval = 500
    running_loss = 0.0
    running_mae = 0.0
    running_n = 0
    # Train-MAE accumulated since the LAST val eval, kept separate from the
    # log-line window: val_interval and log_interval are independent, and the
    # train/val gap is only meaningful against a train number that exists.
    val_window_mae = 0.0
    val_window_n = 0
    print(f"Training {args.steps} steps: batch={args.batch_size}, "
          f"lr={args.lr} (cosine -> {args.lr_eta_min}), wd={args.weight_decay}, "
          f"clip={args.grad_clip_norm}, EMA {ema_decay}, loss=MSE")

    policy.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            # Pixels: no obs standardizer — the encoder does uint8 -> /255 ->
            # resize internally (same as the q3c/IBC/DP paths).
            states = batch["state"].float().to(device, non_blocking=True)
            actions = batch["action"].float().to(device, non_blocking=True)
            if cond_dim:
                policy._cond = batch["cond"].float().to(device, non_blocking=True)

            pred = policy(states)[:, 0, :]              # (B, action_dim)
            loss = criterion(pred, actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                               max_norm=args.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for ep, p in zip(ema_policy.parameters(), policy.parameters()):
                    ep.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)
                # Buffers (BatchNorm running stats) are copied, not averaged —
                # they are not gradient-trained parameters.
                for eb, b in zip(ema_policy.buffers(), policy.buffers()):
                    eb.copy_(b)
                batch_mae = (pred - actions).abs().mean().item()
            running_mae += batch_mae
            val_window_mae += batch_mae
            val_window_n += 1
            running_loss += float(loss.detach())
            running_n += 1
            step += 1

            if step % log_interval == 0:
                avg = running_loss / max(running_n, 1)
                mae = running_mae / max(running_n, 1)
                elapsed = time.time() - start
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {step}/{args.steps} | MSE: {avg:.6f} | "
                      f"train MAE: {mae:.5f} | LR: {lr:.2e} | {elapsed:.1f}s",
                      flush=True)
                running_loss = 0.0
                running_mae = 0.0
                running_n = 0
            if step % args.save_interval == 0:
                save_ckpt(f"_step{step:06d}", step)
            if val_loader is not None and step % val_interval == 0:
                v = val_action_mae()
                policy.train()
                train_mae = val_window_mae / max(val_window_n, 1)
                val_window_mae = 0.0
                val_window_n = 0
                print(f"[val] step {step}/{args.steps}: action_MAE "
                      f"train={train_mae:.5f} val={v:.5f} "
                      f"gap={v - train_mae:+.5f}", flush=True)

    save_ckpt("", args.steps)
    total = time.time() - start
    print(f"Done in {total / 60:.1f} min. Final weights: "
          f"{run_dir / 'bc_policy.pt'} (deploy with bc_policy_ema.pt)")
    print(f"Deploy with: python scripts/deploy_pusht_real_bc.py --seed-dir {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
