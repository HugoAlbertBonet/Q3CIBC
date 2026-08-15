#!/usr/bin/env python3
"""Launch dpq3c_training on the real-robot Push-T archive.

Same mechanism as scripts/train_pusht_real.py: write an immutable per-run config
beside the checkpoints and point the trainer at it with Q3C_CONFIG_PATH, so the
shared config_json/config.json is never modified and Slurm jobs can train
concurrently without racing.

Difference from train_pusht_real.py: the proposal distribution is a diffusion
policy instead of the Q3C control-point generator, so the CP-generator
hyperparameters (--cp-width/--cp-depth/--control-points/--mse-weight/
--sep-weight/--entropy-bandwidth) are replaced by the diffusion knobs and by the
critic-side knobs that decide what the Q estimator is actually trained to be.

One run produces BOTH halves of the policy, so deploy it with the same directory
on both flags::

    python scripts/deploy_pusht_real_dpq3c.py \
        --dp-dir  checkpoints/pusht_real_dpq3c/D2c_ch16_p01_n64 \
        --q-dir   checkpoints/pusht_real_dpq3c/D2c_ch16_p01_n64 \
        --cp 64 --dp-method ddim --dp-iters 10 --steps 700 --measure

The three axes worth understanding before sweeping anything else:

  --dp-negatives N     The critic's negatives are drawn from the DIFFUSION
                       POLICY, so it learns to rank the exact distribution it
                       will face at deploy. This is the thing a separately
                       trained DP + Q3C pair cannot have. N=0 turns it off and
                       gives the "trained apart" control.
  --progress-weight W  W>0 anchors Q to the Monte-Carlo time-to-go return, which
                       needs no reward labels and is what makes Q a value
                       function rather than a density model. W=0 leaves it a
                       pure ranker — the Q3C-equivalent critic, and the honest
                       baseline to beat.
  --action-chunk K     Open-loop chunking, the established anti-stall fix on
                       this rig.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data" / "pusht_widowx_data.zip",
        help="Push-T demonstration archive. Diffusion-Policy format "
             "(replay_buffer.zarr + videos/<ep>/<cam>.mp4) by default.",
    )
    parser.add_argument("--data-format", choices=["zarr_video", "bridge_zip"],
                        default="zarr_video",
                        help="On-disk layout of --dataset (default: zarr_video)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-stack", type=int, default=2)
    parser.add_argument("--cameras", nargs="+", default=["images1"],
                        help="bridge_zip only: ordered RGB streams.")
    parser.add_argument("--video-camera", type=int, default=1,
                        help="zarr_video only: which per-episode MP4 to train on. "
                             "Ignored when --video-cameras is given.")
    parser.add_argument("--video-cameras", type=int, nargs="+", default=None,
                        help="zarr_video only: ordered per-episode MP4 cameras to "
                             "stack as input (e.g. `0 1` for both views).")
    parser.add_argument(
        "--action-chunk", type=int, default=1,
        help="Predict K planar deltas per step (open-loop action chunking). "
             "K=1 is single-step; K=16 matches the libero recipe. The deploy "
             "client executes --exec-horizon of them before re-predicting.",
    )
    parser.add_argument("--val-frac", type=float, default=0.0,
                        help="Hold out this fraction of EPISODES as validation. "
                             "The trainer logs a live held-out action-MAE using "
                             "the SAME DP-cloud + argmax-Q selection deploy uses.")
    parser.add_argument("--val-seed", type=int, default=0,
                        help="RNG seed choosing which episodes are held out "
                             "(share it across a sweep for comparable MAE).")
    parser.add_argument("--val-interval", type=int, default=None,
                        help="Steps between held-out MAE evals (default: the "
                             "checkpoint save_interval).")
    parser.add_argument("--frame-cache-dir", type=Path, default=None,
                        help="zarr_video only: where to build the decoded uint8 "
                             "frame memmap (default: <dataset dir>/_frame_cache).")

    # ── Idle-transition handling ─────────────────────────────────────────────
    parser.add_argument(
        "--idle-filter", choices=["none", "drop_zero", "drop_static", "subsample"],
        default="drop_zero",
        help="How to treat transitions whose target action is ~0. 24%% of this "
             "dataset is the teleoperator pausing. 'drop_zero' (default) is the "
             "one that eliminates the stalling mode. NOTE this filters the "
             "sample INDEX list, not the raw arrays, so --progress-weight "
             "computes its time-to-go against the raw contiguous timeline.",
    )
    parser.add_argument("--idle-eps", type=float, default=0.0)
    parser.add_argument("--idle-move-eps", type=float, default=1e-4)
    parser.add_argument("--idle-keep-frac", type=float, default=0.25)
    parser.add_argument(
        "--cond-eef-xy", action="store_true",
        help="Condition the policy on the current end-effector (x, y). NOTE: "
             "deploy_pusht_real_dpq3c.py cannot yet rebuild a CONDITIONED "
             "denoiser, so a checkpoint trained this way is not deployable "
             "until that is wired up.",
    )
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "checkpoints" / "pusht_real_dpq3c")
    parser.add_argument("--aug-photometric", action="store_true",
                        help="PHOTOMETRIC-ONLY augmentation (no crop-zoom), to "
                             "harden against the deploy lighting shift.")
    parser.add_argument("--aug", action="store_true",
                        help="Photometric + small view crop. The crop is "
                             "questionable for this FIXED camera.")
    parser.add_argument("--tag", default=None,
                        help="Run-directory name under --output-root "
                             "(default: seed_XXXX).")

    # ── Image encoder (shared by the denoiser and the Q estimator) ───────────
    enc = parser.add_argument_group("encoder")
    enc.add_argument("--encoder-kind", choices=["conv_maxpool", "resnet18"],
                     default="conv_maxpool")
    enc.add_argument("--encoder-pretrained", choices=["none", "imagenet"],
                     default="none")
    enc.add_argument("--encoder-norm-kind", choices=["bn", "gn", "bn_frozen"],
                     default="bn",
                     help="resnet18 only. 'gn' is the libero choice — BN's "
                          "train/eval stat mismatch is hostile to EBM training.")
    enc.add_argument("--encoder-num-kp", type=int, default=64)
    enc.add_argument("--encoder-feature-dim", type=int, default=256)
    enc.add_argument("--encoder-per-camera", action="store_true")
    enc.add_argument("--cond-fusion", choices=["concat", "film"], default="concat")

    # ── Diffusion policy (the actor) ────────────────────────────────────────
    dpg = parser.add_argument_group("diffusion policy (actor)")
    dpg.add_argument("--num-train-timesteps", type=int, default=100)
    dpg.add_argument("--beta-schedule", choices=["cosine", "linear"], default="cosine")
    dpg.add_argument("--prediction-type", choices=["epsilon", "v"], default="epsilon")
    dpg.add_argument("--denoiser-network-kind", default="dense_resnet",
                     help="Head architecture. dense_resnet matches the Q3C/IBC "
                          "pixel value head, so actor and critic are "
                          "capacity-matched.")
    dpg.add_argument("--denoiser-width", type=int, default=1024)
    dpg.add_argument("--denoiser-depth", type=int, default=1)
    dpg.add_argument("--time-emb-dim", type=int, default=128)
    dpg.add_argument("--ddim-eval-steps", type=int, nargs="+", default=[5, 10, 25],
                     help="Recorded in norm_stats; the deploy client's --dp-iters "
                          "defaults to the first entry.")
    dpg.add_argument("--ddim-eta", type=float, default=0.0)

    # ── Critic: what the Q estimator is trained to be ───────────────────────
    cr = parser.add_argument_group("critic")
    cr.add_argument(
        "--dp-negatives", type=int, default=16,
        help="Diffusion samples per state used as InfoNCE negatives. This is "
             "the alignment that a separately-trained DP + Q3C pair cannot "
             "have: the critic learns to rank the distribution it will actually "
             "face. Match it to the deploy --cp — the score gaps inside a cloud "
             "grow with N, so a critic (and a selection temperature) calibrated "
             "at one N is not the same at another. 0 = off (the control).",
    )
    cr.add_argument("--dp-negative-iters", type=int, default=4,
                    help="Denoising steps used to DRAW those negatives. They "
                         "need to be the kind of thing that gets proposed, not "
                         "perfect samples, so a short chain is the right trade. "
                         "This is the dominant per-step cost knob.")
    cr.add_argument("--dp-negative-method", choices=["ddim", "ddpm"], default="ddim")
    cr.add_argument("--dp-negative-warmup", type=int, default=5000,
                    help="Steps before DP negatives kick in. Until the denoiser "
                         "has learned anything its samples are indistinguishable "
                         "from noise, so uniform draws stand in and the sampler "
                         "cost is not paid for nothing.")
    cr.add_argument(
        "--progress-weight", type=float, default=0.0,
        help="Weight on the REWARD-FREE value anchor: regress Q at the expert "
             "action toward the Monte-Carlo time-to-go return "
             "(-remaining_steps / mean_episode_length). Every demo dataset "
             "already contains this label. Without it, InfoNCE and the margin "
             "only constrain score DIFFERENCES, so the critic is identified "
             "only up to a per-state shift and stays a ranker. 0 = off.",
    )
    cr.add_argument("--margin-weight", type=float, default=0.0,
                    help="DQfD-style large-margin hinge: the expert chunk must "
                         "outrank the best DP proposal by --margin. Same "
                         "information as InfoNCE, one-sided. 0 = off.")
    cr.add_argument("--margin", type=float, default=0.1)
    cr.add_argument("--q-actor-weight", type=float, default=0.0,
                    help="Training-time analogue of the deploy client's "
                         "--q-guidance: pull the denoiser's predicted CLEAN "
                         "sample toward high Q (estimator frozen for the term). "
                         "0 = off; leave it off until the critic has passed the "
                         "rank-correlation check.")
    cr.add_argument("--value-width", type=int, default=1024)
    cr.add_argument("--value-num-blocks", type=int, default=1)
    cr.add_argument("--info-nce-weight", type=float, default=0.5)
    cr.add_argument("--infonce-clamp", type=float, default=10.0)
    cr.add_argument("--uniform-negatives", type=int, default=0)
    cr.add_argument("--langevin-negatives", type=int, default=0)
    cr.add_argument("--langevin-iters", type=int, default=0)
    cr.add_argument("--noisy-expert-count", type=int, default=0)
    cr.add_argument("--noisy-expert-sigma-start", type=float, default=0.1)
    cr.add_argument("--noisy-expert-sigma-final", type=float, default=0.02)
    cr.add_argument("--gradient-penalty-weight", type=float, default=0.0)
    cr.add_argument("--gradient-penalty-margin", type=float, default=1.0)
    cr.add_argument(
        "--deploy-control-points", type=int, default=None,
        help="Cloud size recorded in the run config as model.control_points, "
             "which is what the deploy client's --cp defaults to. Defaults to "
             "--dp-negatives so deploy inherits the N the critic was calibrated "
             "on.",
    )

    # ── Optimization ────────────────────────────────────────────────────────
    hp = parser.add_argument_group("optimization")
    hp.add_argument("--lr", type=float, default=3e-4, help="denoiser LR")
    hp.add_argument("--est-lr", type=float, default=3e-4, help="Q estimator LR")
    hp.add_argument("--scheduler", default="cosine",
                    choices=["cosine", "cosine_warm_restarts"])
    hp.add_argument("--cosine-t0", type=int, default=50_000)
    hp.add_argument("--ema-decay", type=float, default=0.999)
    hp.add_argument("--encoder-lr-scale", type=float, default=1.0,
                    help="Multiply the encoder's LR by this (pretrained trunks "
                         "usually want < 1). 1.0 = off.")

    parser.add_argument("--dry-run", action="store_true",
                        help="Write and print the run config without training")
    return parser.parse_args()


def build_config(args: argparse.Namespace, run_dir: Path) -> dict:
    with (ROOT / "config_json" / "config.json").open() as handle:
        config = json.load(handle)

    # Same starting point as train_pusht_real.py so every non-algorithmic choice
    # (dataset plumbing, encoder geometry, action bounds) is identical and a
    # dpq3c-vs-q3c comparison is not confounded by the data pipeline.
    env = copy.deepcopy(config["environments"]["pushing_pixels"])
    if args.data_format == "zarr_video":
        video_cameras = list(args.video_cameras) if args.video_cameras else [args.video_camera]
        n_cams = len(video_cameras)
    else:
        video_cameras = None
        n_cams = len(args.cameras)

    env.update({
        "data_archive": str(args.dataset.resolve()),
        "data_format": args.data_format,
        "env_id": "PushTRealRobot-v0",
        "state_dim": [3 * n_cams * args.frame_stack, args.image_height, args.image_width],
        "action_dim": 2,
        "frame_stack": args.frame_stack,
        "camera_streams": ([f"video{c}" for c in video_cameras]
                           if args.data_format == "zarr_video" else list(args.cameras)),
        "video_camera": (video_cameras[0] if args.data_format == "zarr_video"
                         else args.video_camera),
        "video_cameras": video_cameras,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "dataloader_num_workers": args.workers,
        "action_bounds": [-1.0, 1.0],
        "encoder_target_height": 180,
        "encoder_target_width": 240,
        "image_aug": bool(args.aug or args.aug_photometric),
        "image_aug_params": ({"zoom_range": [1.0, 1.0]} if args.aug_photometric else None),
        "val_frac": args.val_frac,
        "val_seed": args.val_seed,
        "idle_filter": args.idle_filter,
        "idle_eps": args.idle_eps,
        "idle_move_eps": args.idle_move_eps,
        "idle_keep_frac": args.idle_keep_frac,
        "cond_eef_xy": args.cond_eef_xy,
    })
    if args.frame_cache_dir is not None:
        env["frame_cache_dir"] = str(args.frame_cache_dir.resolve())

    env["training"].update({
        "training_steps": args.steps,
        "batch_size": args.batch_size,
        "trial_seed": args.seed,
        "action_chunk": args.action_chunk,
        **({"val_interval": args.val_interval} if args.val_interval is not None else {}),
        "best_ckpt": False,
        "ema_decay": args.ema_decay,
        "learning_rate": args.lr,
        "estimator_learning_rate": args.est_lr,
        "encoder_lr_scale": args.encoder_lr_scale,
        "infonce_logit_clamp": args.infonce_clamp,
        "scheduler_type": args.scheduler,
        "cosine_t0": args.cosine_t0,
        "cosine_t_max": args.steps,
        # Critic negatives.
        "dp_negatives": args.dp_negatives,
        "dp_negative_iters": args.dp_negative_iters,
        "dp_negative_method": args.dp_negative_method,
        "dp_negative_warmup_steps": args.dp_negative_warmup,
        "num_uniform_negatives": args.uniform_negatives,
        "num_langevin_negatives": args.langevin_negatives,
        "langevin_num_iterations": args.langevin_iters,
        "noisy_expert_count": args.noisy_expert_count,
        "noisy_expert_sigma_start": args.noisy_expert_sigma_start,
        "noisy_expert_sigma_final": args.noisy_expert_sigma_final,
        # Critic objective terms.
        "margin_weight": args.margin_weight,
        "margin": args.margin,
        "progress_weight": args.progress_weight,
        "gradient_penalty_weight": args.gradient_penalty_weight,
        "gradient_penalty_margin": args.gradient_penalty_margin,
        # Actor <- critic feedback.
        "q_actor_weight": args.q_actor_weight,
        # Diffusion process. resolve_dp_params reads env["training"] FIRST, so
        # these are the authoritative values for both the trainer and the deploy
        # client's build_dp_denoiser.
        "num_train_timesteps": args.num_train_timesteps,
        "beta_schedule": args.beta_schedule,
        "prediction_type": args.prediction_type,
        "time_emb_dim": args.time_emb_dim,
        "denoiser_network_kind": args.denoiser_network_kind,
        "denoiser_width": args.denoiser_width,
        "denoiser_depth": args.denoiser_depth,
        "denoiser_use_spectral_norm": False,
        "ddim_eval_steps": list(args.ddim_eval_steps),
        "ddim_eta": args.ddim_eta,
    })

    config["training_shared"].update({"info_nce_weight": args.info_nce_weight})

    env["model"].update({
        "encoder_kind": args.encoder_kind,
        "encoder_pretrained": (args.encoder_pretrained == "imagenet"),
        "encoder_norm_kind": args.encoder_norm_kind,
        "encoder_num_kp": args.encoder_num_kp,
        "encoder_feature_dim": args.encoder_feature_dim,
        "encoder_per_camera": args.encoder_per_camera,
        "cond_fusion": args.cond_fusion,
        "value_width": args.value_width,
        "value_num_blocks": args.value_num_blocks,
        # There is no control-point generator any more, but the deploy client
        # reads model.control_points as the default cloud size for --cp (and
        # rebuilds — then discards — a CP generator from the same block, so the
        # cp_* keys must stay present and well-formed).
        "control_points": (args.deploy_control_points
                           if args.deploy_control_points is not None
                           else max(1, args.dp_negatives)),
    })

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
    if args.dp_negatives < 0:
        raise ValueError("--dp-negatives must be >= 0")
    if args.dp_negatives == 0 and args.margin_weight > 0:
        raise SystemExit(
            "--margin-weight > 0 needs DP candidates to form the hinge against; "
            "--dp-negatives 0 leaves it with nothing to compare. Either raise "
            "--dp-negatives or set --margin-weight 0.")
    if args.dp_negative_iters < 1:
        raise ValueError("--dp-negative-iters must be >= 1")

    run_name = args.tag if args.tag else f"seed_{args.seed:04d}"
    run_dir = args.output_root.resolve() / run_name
    # Refuse to silently overwrite a finished run. norm_stats.pt is written
    # BEFORE the training loop, so it must not count as "finished".
    existing = ([p for p in sorted(run_dir.glob("*.pt")) if p.name != "norm_stats.pt"]
                if run_dir.exists() else [])
    if existing and not args.dry_run:
        raise FileExistsError(
            f"{run_dir} already holds checkpoints ({[p.name for p in existing[:3]]}...). "
            f"Pass a different --tag, or delete the directory to retrain.")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config = build_config(args, run_dir)
    with config_path.open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print(f"Dataset:     {args.dataset.resolve()}  (format={args.data_format})")
    print(f"Seed:        {args.seed}   tag={run_name}")
    print(f"Idle filter: {args.idle_filter} (eps={args.idle_eps})")
    print(f"Actor:       diffusion T={args.num_train_timesteps} "
          f"{args.beta_schedule}/{args.prediction_type} "
          f"head={args.denoiser_network_kind}({args.denoiser_width}x{args.denoiser_depth})")
    print(f"Critic:      dp_neg={args.dp_negatives}x{args.dp_negative_iters} "
          f"(warmup {args.dp_negative_warmup})  info_nce={args.info_nce_weight} "
          f"margin={args.margin_weight}  progress={args.progress_weight}")
    print(f"             {'VALUE-anchored' if args.progress_weight > 0 else 'RANKER only (no absolute scale)'}")
    print(f"Config:      {config_path}")
    print(f"Checkpoints: {run_dir}")
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["Q3C_CONFIG_PATH"] = str(config_path)
    command = [sys.executable, str(ROOT / "dpq3c_training.py")]
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
