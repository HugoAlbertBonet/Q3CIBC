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


# libero_goal_pixels IBC recipe (results/hyperparam_search/
# ibc_dfo_libero_goal_pixels/trials.jsonl, best sr 0.46): Langevin-refined
# counter-examples during training + Langevin inference, on the same
# ConvMaxpool/DenseResnet PixelEBM.
IBC_BEST = {
    "training_steps": 100_000,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "lr_decay_rate": 0.99,
    "lr_decay_steps": 100,
    "num_counter_examples": 8,
    "uniform_boundary_buffer": 0.05,
    "softmax_temperature": 1.0,
    # Model (PixelEBM: ConvMaxpoolEncoder + DenseResnetValue).
    "encoder_target_height": 180,
    "encoder_target_width": 240,
    "value_width": 1024,
    "value_num_blocks": 1,
    # Langevin MCMC that generates the training counter-examples (Florence
    # et al. App. D.1) — schedule shared with inference.
    "langevin_train_iterations": 100,
    "langevin_stepsize_init": 0.1,
    "langevin_stepsize_final": 1e-5,
    "langevin_stepsize_power": 2.0,
    "langevin_delta_clip": 0.1,
    "langevin_noise_scale": 0.1,
    # Inference (Langevin, matching the libero trial).
    "inference_num_samples": 1024,
    "inference_num_iterations": 100,
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
        "--video-cameras", type=int, nargs="+", default=None,
        help="zarr_video only: ordered list of per-episode MP4 cameras to stack "
             "(e.g. `0 1`). Overrides --video-camera. Each adds 3 channels per "
             "stacked frame; a per-camera frame cache is built once per camera.",
    )
    parser.add_argument(
        "--action-chunk", type=int, default=1,
        help="Predict K planar deltas per step (open-loop action chunking). The "
             "EBM's action_dim becomes K*2 and Langevin sampling runs in that "
             "space; deploy_pusht_real_ibc.py reads the chunk length back from "
             "norm_stats['act_min'].size // 2 and honours --exec-horizon. "
             "Default 1 is the paper-faithful setting AND the safe one: the "
             "sample budget (counter-examples at train, 1024x100 Langevin at "
             "inference) is fixed, so per-dimension coverage degrades as K "
             "grows. K=16 matches the libero_goal_pixels recipe and is swept in "
             "batches/pushtWidowXlibIbc2camChunk16.txt.",
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
    # ── Image encoder (libero_goal_pixels IBC uses conv_maxpool; resnet18 is a
    #    sweep arm) ─────────────────────────────────────────────────────────
    parser.add_argument("--encoder-kind", choices=["conv_maxpool", "resnet18"],
                        default="conv_maxpool")
    parser.add_argument("--encoder-pretrained", choices=["none", "imagenet"],
                        default="none", help="resnet18 only")
    parser.add_argument("--encoder-norm-kind", choices=["bn", "gn", "bn_frozen"],
                        default="bn", help="resnet18 only")
    parser.add_argument("--encoder-num-kp", type=int, default=64,
                        help="resnet18 only: SpatialSoftmax keypoints")
    # ── Held-out validation (episode-level split) ─────────────────────────
    parser.add_argument("--val-frac", type=float, default=0.0,
                        help="Fraction of EPISODES held out; the trainer logs a "
                             "live held-out action-MAE. 0 = train on everything.")
    parser.add_argument("--val-seed", type=int, default=0,
                        help="Seed choosing held-out episodes (share across a "
                             "sweep for comparable MAE).")
    parser.add_argument("--val-interval", type=int, default=None,
                        help="Steps between held-out MAE evals (default: "
                             "save_interval).")
    parser.add_argument(
        "--aug-photometric", action="store_true",
        help="Photometric-only augmentation (brightness/contrast/saturation/"
             "channel-gain, NO crop) to harden against the deploy lighting shift.",
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
        "action_dim": 2,
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
        "encoder_target_height": IBC_BEST["encoder_target_height"],
        "encoder_target_width": IBC_BEST["encoder_target_width"],
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
        "val_frac": args.val_frac,
        "val_seed": args.val_seed,
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
            "action_chunk": args.action_chunk,
            "langevin_train_iterations": IBC_BEST["langevin_train_iterations"],
            **({"val_interval": args.val_interval}
               if args.val_interval is not None else {}),
        },
        "model": {
            "kind": "ibc_pixel_ebm",
            "encoder_kind": args.encoder_kind,
            "encoder_pretrained": (args.encoder_pretrained == "imagenet"),
            "encoder_norm_kind": args.encoder_norm_kind,
            "encoder_num_kp": args.encoder_num_kp,
            "value_width": IBC_BEST["value_width"],
            "value_num_blocks": IBC_BEST["value_num_blocks"],
        },
        "inference": {
            "num_samples": IBC_BEST["inference_num_samples"],
            "num_iterations": IBC_BEST["inference_num_iterations"],
            "langevin_stepsize_init": IBC_BEST["langevin_stepsize_init"],
            "langevin_stepsize_final": IBC_BEST["langevin_stepsize_final"],
            "langevin_stepsize_power": IBC_BEST["langevin_stepsize_power"],
            "langevin_delta_clip": IBC_BEST["langevin_delta_clip"],
            "langevin_noise_scale": IBC_BEST["langevin_noise_scale"],
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
    # norm_stats.pt is written before the training loop, so it must not count as
    # a finished run (a crash during warmup would otherwise block a clean re-run).
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
    from utils.sampling import sample_langevin

    aug_on = bool(args.aug or args.aug_photometric)
    aug_params = {"zoom_range": [1.0, 1.0]} if args.aug_photometric else None
    cameras = (list(args.video_cameras) if args.video_cameras
               else [args.video_camera])
    if args.data_format == "zarr_video":
        from utils.datasets import PushTWidowXVideoDataset

        def _make_video_ds(split):
            return PushTWidowXVideoDataset(
                archive_path=str(args.dataset),
                frame_stack=args.frame_stack,
                cameras=cameras,
                resize_hw=(args.image_height, args.image_width),
                normalize_actions=True,
                action_norm_range=(-1.0, 1.0),
                augment=aug_on if split == "train" else False,
                aug_params=aug_params,
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
                action_chunk=args.action_chunk,
                split=split,
                val_frac=args.val_frac,
                val_seed=args.val_seed,
            )
        dataset = _make_video_ds("train")
        val_dataset = _make_video_ds("val") if args.val_frac > 0.0 else None
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
        print(f"Validation: {len(val_dataset)} held-out transitions; val "
              f"action-MAE logged every {val_interval} steps.")

    ebm = PixelQEstimator(
        action_dim=dataset.action_shape,
        in_channels=dataset.in_channels,
        encoder_target_height=IBC_BEST["encoder_target_height"],
        encoder_target_width=IBC_BEST["encoder_target_width"],
        value_width=IBC_BEST["value_width"],
        value_num_blocks=IBC_BEST["value_num_blocks"],
        cond_dim=cond_dim,
        encoder_kind=args.encoder_kind,
        encoder_pretrained=(args.encoder_pretrained == "imagenet"),
        encoder_num_kp=args.encoder_num_kp,
        encoder_norm_kind=args.encoder_norm_kind,
    ).to(device)
    n_params = sum(p.numel() for p in ebm.parameters())
    print(f"PixelEBM ({args.encoder_kind} + DenseResnetValue "
          f"{IBC_BEST['value_width']}x{IBC_BEST['value_num_blocks']}, "
          f"action_dim={dataset.action_shape}): {n_params:,} params")
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
        "encoder_kind": args.encoder_kind,
        "encoder_pretrained": (args.encoder_pretrained == "imagenet"),
        "encoder_num_kp": args.encoder_num_kp,
        "encoder_norm_kind": args.encoder_norm_kind,
        "action_chunk": args.action_chunk,
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
    # Langevin MCMC config for the training counter-examples.
    A_dim = dataset.action_shape
    lv_min = torch.full((A_dim,), sample_lo, device=device)
    lv_max = torch.full((A_dim,), sample_hi, device=device)
    lv_iters = IBC_BEST["langevin_train_iterations"]
    lv = dict(
        lr_init=IBC_BEST["langevin_stepsize_init"],
        lr_final=IBC_BEST["langevin_stepsize_final"],
        polynomial_decay_power=IBC_BEST["langevin_stepsize_power"],
        delta_action_clip=IBC_BEST["langevin_delta_clip"],
        noise_scale=IBC_BEST["langevin_noise_scale"],
    )
    inf_samples = IBC_BEST["inference_num_samples"]

    def save_ckpt(tag: str, step: int) -> None:
        torch.save(ebm.state_dict(), run_dir / f"q_estimator{tag}.pt")
        (run_dir / "last_step.json").write_text(json.dumps({"step": step}) + "\n")

    @torch.no_grad()
    def val_action_mae() -> float:
        """Held-out action-MAE via IBC inference (sample candidates, argmax
        score). Uniform-sample proxy of the Langevin inference — cheap, but the
        same selection rule, so it tracks generalization."""
        was_training = ebm.training
        ebm.eval()
        tot, n = 0.0, 0
        for vb in val_loader:
            vs = vb["state"].to(device)
            va = vb["action"].float().to(device)
            if cond_dim:
                ebm._cond = vb["cond"].float().to(device)
            Bv = vs.shape[0]
            feats = ebm.encode(vs)
            cand = (torch.rand(Bv, inf_samples, A_dim, device=device)
                    * (sample_hi - sample_lo) + sample_lo)
            logits = ebm.score(feats, cand).squeeze(-1)          # (Bv, S)
            best = cand[torch.arange(Bv, device=device), logits.argmax(dim=1)]
            tot += (best - va).abs().sum().item()
            n += va.numel()
        if was_training:
            ebm.train()
        return tot / max(n, 1)

    step = 0
    start = time.time()
    log_interval = 500
    consecutive_nan = 0
    print(f"Training {args.steps} steps: batch={args.batch_size}, "
          f"lr={lr0} (x{decay_rate}/{decay_steps} steps), "
          f"{n_counter} Langevin counter-examples ({lv_iters} iters) in "
          f"[{sample_lo}, {sample_hi}]")

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

            # Late fusion: encode the image ONCE (with grad — the encoder trains
            # through the InfoNCE), value head scores all candidates on the
            # cached features.
            features = ebm.encode(states)

            # Langevin-refined counter-examples (Florence et al. App. D.1):
            # descend -score (ascend the model's preferred actions) from uniform
            # starts, using DETACHED features so the sampler does not backprop
            # into the encoder. Expert appended last -> softmax CE at that index.
            feats_det = features.detach()

            def _neg_energy(_obs, acts):
                return -ebm.score(feats_det, acts).squeeze(-1)

            negatives = sample_langevin(
                energy_function=_neg_energy,
                observations=states,
                num_samples=n_counter,
                action_min=lv_min,
                action_max=lv_max,
                num_iterations=lv_iters,
                **lv,
            ).detach()
            all_actions = torch.cat([negatives, expert.unsqueeze(1)], dim=1)

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
            if val_loader is not None and step % val_interval == 0:
                print(f"[val] step {step}: action_MAE val={val_action_mae():.4f}",
                      flush=True)

    save_ckpt("", args.steps)
    total = time.time() - start
    print(f"Done in {total / 60:.1f} min. Final weights: {run_dir / 'q_estimator.pt'}")
    print(f"Deploy with: python scripts/deploy_pusht_real_ibc.py --seed-dir {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
