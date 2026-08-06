#!/usr/bin/env python3
"""Deploy a trained Diffusion-Policy Push-T checkpoint on the real WidowX arm.

DP counterpart of deploy_pusht_real.py. The robot-facing half is IDENTICAL —
every server call, safety clip, z-hold, approach floor, cond, dry-run,
calibration, scoring and forensic-log path is imported from deploy_pusht_real.py
so the two clients agree bit-for-bit. Only the policy differs:

  * Model: a PixelDiffusionDenoiser (ConvMaxpool encoder + DenseResnet head)
    plus a GaussianDiffusion sampler, rebuilt from the checkpoint's
    config.json / norm_stats exactly as train_pusht_real_dp.py wrote them.
  * Action: instead of the Q3C CP-cloud argmax, we SAMPLE the fitted action
    distribution once per step via DDPM (or DDIM with --inference ddim). Both
    are inference schedules over the same denoiser.

Weights: denoiser_ema.pt (default) or denoiser.pt (--no-ema).

Cameras: driven by the checkpoint. norm_stats["camera_streams"] is ("video1",)
for the blue-only lines (g01/g02/g03) and ("video0", "video1") for the bothcam
line (g04); the live stack is built in that same camera order, so both work
without a flag. The topics are registered in the collection's order (D435 then
blue) and each camera is read from its POSITION in that list -- see
deploy_pusht_real.frame_for_camera. On a rig without the D435:
`--camera-topics /blue/image_raw --topic-camera-ids 1`.

Usage (server already up):

    python scripts/deploy_pusht_real_dp.py \
        --seed-dir checkpoints/pusht_real_dp_2026_07/g01_resnet18_s11_350k \
        --device cpu --dry-run
    python scripts/deploy_pusht_real_dp.py \
        --seed-dir checkpoints/pusht_real_dp_2026_07/g01_resnet18_s11_350k \
        --device cpu --steps 700 --log-dir results/run_dp

--steps: demos in the 2026-07 collection run 183-998 steps (mean 608) at the
default 0.05 s period, so 200 is a third of a demo -- raise it for a real trial.

Scoring: ``--measure`` reads the final frame of every registered camera through
measure_target_coverage.py and appends one row per episode to
``results/pusht/experiments.csv`` (``--results-csv``) -- the SAME table the Q3C
and IBC clients write, which is what the ``algorithm`` column is for. This
script always records ``dp``; the ``inference`` column records the sampler
schedule (``ddpm`` or ``ddim``). Repeating a parameter combination adds a row
with the next ``trial`` number instead of overwriting the previous one. Scoring
runs even if the episode is interrupted, and a failure there never fails the
run::

    python scripts/deploy_pusht_real_dp.py \
        --seed-dir checkpoints/pusht_real_dp_2026_07/g01_resnet18_s11_350k \
        --steps 700 --inference ddim --ddim-steps 10 \
        --measure --start-position top
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse every WidowX/server/safety/preprocess helper from the Q3C deploy client
# so the DP client and the Q3C client behave identically off-policy.
_spec = importlib.util.spec_from_file_location(
    "deploy", ROOT / "scripts" / "deploy_pusht_real.py")
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)

from utils.diffusion import build_diffusion, build_pixel_denoiser, resolve_dp_params

# This script deploys Diffusion Policy. Q3C lives in deploy_pusht_real.py and
# IBC in deploy_pusht_real_ibc.py; all three append to the same results table,
# so the label is fixed here rather than exposed as a flag -- a row written by
# this file is a DP row by construction.
ALGORITHM = "dp"


def parse_args() -> argparse.Namespace:
    # Full copy of deploy_pusht_real.parse_args, with the Q3C CP-selection knobs
    # replaced by the DP sampler knobs. Everything else is deliberately identical.
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-ema", action="store_true",
                   help="use raw denoiser instead of the EMA copy")
    p.add_argument("--ip", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--widowx-envs-path", type=Path, default=None,
                   help="OPTIONAL path prepended to sys.path before importing "
                        "widowx_envs (see deploy_pusht_real.py).")
    p.add_argument("--camera-topics", nargs="+", default=d.CAMERA_TOPICS,
                   help="ROS topics, registered in THIS order. Default = the "
                        "order the training data was collected in "
                        f"({d.DATASET_CAMERA_TOPICS}).")
    p.add_argument("--topic-camera-ids", nargs="+", type=int, default=None,
                   help="dataset camera id of each --camera-topics entry "
                        "(default 0,1,...). The checkpoint's camera_streams "
                        "(video1 -> id 1) are resolved through this map, so a "
                        "blue-only rig needs `--camera-topics /blue/image_raw "
                        "--topic-camera-ids 1`.")

    # --- service image geometry (confirmed-working values) ------------------
    p.add_argument("--im-size", type=int, default=480, help="service image height")
    p.add_argument("--im-width", type=int, default=640, help="service image width")

    # --- control -------------------------------------------------------------
    p.add_argument("--steps", type=int, default=200, help="max control steps")
    p.add_argument("--exec-horizon", type=int, default=1,
                   help="receding horizon: how many sub-actions of a predicted "
                        "action chunk to execute open-loop before re-predicting. "
                        "1 (default) = re-predict every control step. Clipped to "
                        "the checkpoint's chunk length (act_min size / 2), so it "
                        "is a no-op for unchunked checkpoints.")
    p.add_argument("--step-duration", type=float, default=d.STEP_DURATION,
                   help="control period; also used as env move_duration. Default "
                        "is the collection's move_duration (20 Hz).")
    p.add_argument("--non-blocking", action="store_true",
                   help="the working reference uses blocking=True; this opts out")
    p.add_argument("--action-mode", default="2trans",
                   choices=["2trans", "3trans", "3trans1rot", "3trans3rot"])
    p.add_argument("--safety-max-xy-delta", type=float, default=d.SAFETY_MAX_XY_DELTA)
    p.add_argument("--workspace-xyz", type=float, nargs=6, default=None,
                   metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                   help="override the server's workspace box (metres); see "
                        "deploy_pusht_real.py. Only applied on init().")
    p.add_argument("--min-step-xy", type=float, default=0.0,
                   help="metres. If >0, any nonzero |dx|/|dy| below this is "
                        "snapped UP to it (sign kept); exact 0 stays 0. The "
                        "expert teleop is bang-bang (0 or >=1.5mm; measured "
                        "dead zone in (0,1.5mm)), so a policy can emit "
                        "sub-min-step OOD actions that the arm can't execute and "
                        "it locks. Suggested 0.0015. Default 0 = off.")
    p.add_argument("--lock-z", dest="lock_z", action="store_true", default=True)
    p.add_argument("--no-lock-z", dest="lock_z", action="store_false")
    p.add_argument("--fixed-z-height", type=float, default=d.FIXED_Z_HEIGHT)
    p.add_argument("--neutral-z-height", type=float, default=d.NEUTRAL_Z_HEIGHT)
    p.add_argument("--z-hold", type=float, default=0.0,
                   help="metres. If >0, inject a per-step dz to hold EEF z "
                        "(needs a z-carrying action_mode). See deploy_pusht_real.py.")
    p.add_argument("--z-hold-gain", type=float, default=1.0,
                   help="proportional gain on (z_target - cur_z) for --z-hold.")
    p.add_argument("--z-hold-max", type=float, default=0.01,
                   help="metres, per-step |dz| clip for --z-hold.")
    p.add_argument("--fixed-gripper", type=float, default=d.FIXED_GRIPPER,
                   help="gripper target (0.0 = CLOSED, 1.0 = OPEN).")
    p.add_argument("--gripper-command", type=float, default=0.0,
                   help="explicitly actuate gripper after reset (0.0 = closed). "
                        "Negative to skip.")
    p.add_argument("--skip-move-to-neutral", action="store_true")
    p.add_argument("--i-traj", type=int, default=0,
                   help="trajectory index passed to reset(itraj=N).")

    # --- initial pose (matches deploy_pusht_real.py) ------------------------
    p.add_argument("--move-to-demo-start", dest="move_to_demo_start",
                   action="store_true", default=True,
                   help="after reset, move the EEF to the demo start pose in "
                        "--start-eep-npy.")
    p.add_argument("--no-move-to-demo-start", dest="move_to_demo_start",
                   action="store_false")
    p.add_argument("--start-eep-npy", type=Path, default=d.START_EEP_NPY,
                   help="4x4 EEF start transform (mean of demo starts, x~0.117).")
    p.add_argument("--demo-start-state", dest="demo_start_state",
                   action="store_true", default=True,
                   help="derive the env's start_state from --start-eep-npy so "
                        "reset() itself lands on the demo start pose. Off means "
                        "reset() uses the WidowXConfigs default (0.3, 0.0) and "
                        "the arm crosses the board on every reset.")
    p.add_argument("--no-demo-start-state", dest="demo_start_state",
                   action="store_false")
    p.add_argument("--start-move-duration", type=float, default=1.5)
    p.add_argument("--max-initial-move-retries", type=int, default=5)

    # --- HARD approach guard: never move CLOSER to the robot than the start -
    p.add_argument("--approach-floor", dest="approach_floor",
                   action="store_true", default=True,
                   help="HARD SAFETY: never let the EEF move closer to the robot "
                        "base than the start pose. Any commanded step that would "
                        "take x below the floor is clipped so x stops AT the floor.")
    p.add_argument("--no-approach-floor", dest="approach_floor",
                   action="store_false",
                   help="disable the approach guard (NOT recommended).")
    p.add_argument("--approach-floor-x", type=float, default=None,
                   help="override the x floor (metres). Default: the start pose's "
                        "x (from --start-eep-npy, or the post-reset EEF x).")

    # --- init / reset robustness (confirmed-working values) -----------------
    p.add_argument("--init-timeout-ms", type=int, default=180_000)
    p.add_argument("--init-retries", type=int, default=8)
    p.add_argument("--init-retry-sleep", type=float, default=2.0)
    p.add_argument("--reset-timeout-ms", type=int, default=60_000)
    p.add_argument("--reset-retries", type=int, default=3)
    p.add_argument("--reset-retry-sleep", type=float, default=1.0)
    p.add_argument("--rpc-timeout-ms", type=int, default=5_000)
    p.add_argument("--force-fresh-init", action="store_true")
    p.add_argument("--no-reuse-existing-env", dest="reuse_existing_env",
                   action="store_false", default=True)

    # --- policy (DP sampler) -------------------------------------------------
    # Named --inference to line up with deploy_pusht_real.py, whose value lands
    # in the same results-CSV column. --sampler is kept as an alias so the
    # existing command lines keep working.
    p.add_argument("--inference", "--sampler", dest="inference", default="ddpm",
                   choices=["ddpm", "ddim"],
                   help="action sampling schedule over the trained denoiser. "
                        "ddpm (default) runs the full reverse chain; ddim runs "
                        "--ddim-steps sub-sampled steps. Recorded in the results "
                        "CSV's `inference` column.")
    p.add_argument("--ddim-steps", type=int, default=None,
                   help="DDIM sub-sampled steps (default: ddim_eval_steps[0] "
                        "from norm_stats).")
    p.add_argument("--ddim-eta", type=float, default=None,
                   help="DDIM stochasticity (default: ddim_eta from norm_stats).")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="if set, torch.manual_seed before sampling for a "
                        "reproducible dry run.")

    # --- exposure matching (OFF by default) ---------------------------------
    # The 2026-07 collection was captured in the CURRENT scene, so the live
    # frames already sit at the training white point and no gain is wanted. The
    # flag stays available for the older checkpoints, whose training scene was
    # measured ~16% brighter than the deploy one.
    p.add_argument("--match-exposure", action="store_true",
                   help="OFF by default. Lift the live frame to the training "
                        "white point with per-channel gains (see "
                        "--exposure-gains). Only meaningful for checkpoints "
                        "trained on a differently-lit collection.")
    p.add_argument("--exposure-gains", type=float, nargs=3, default=[1.22, 1.18, 1.17],
                   metavar=("R", "G", "B"),
                   help="per-channel gains applied only with --match-exposure.")
    p.add_argument("--calibrate", action="store_true",
                   help="ignore the policy; command scripted OPEN-LOOP moves "
                        "(+dx,-dx,+dy,-dy) and log raw frames + state to "
                        "--log-dir. Lets you verify (a) the action->image "
                        "direction matches training and (b) the arm actually "
                        "pushes the T. Analyze with check_action_image_frame.py.")
    p.add_argument("--calibrate-step", type=float, default=0.006,
                   help="metres per calibration step (default 6mm, a clear move).")
    p.add_argument("--calibrate-reps", type=int, default=8,
                   help="steps per direction (out then back each axis).")

    # --- diagnostics ---------------------------------------------------------
    p.add_argument("--dry-run", action="store_true",
                   help="no motion: dump fed frames + print sampled actions")
    p.add_argument("--dry-run-steps", type=int, default=20)
    p.add_argument("--dump-dir", type=Path, default=ROOT / "deploy_dryrun_dp")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="per-step forensic log: raw/*.npy, fed/*.png, steps.jsonl")
    p.add_argument("--measure", action="store_true",
                   help="after the episode, score the final frame with "
                        "measure_target_coverage.py and append a row to "
                        "--results-csv")
    p.add_argument("--start-position", default="top",
                   help="where the block started; recorded in the results CSV")
    p.add_argument("--results-csv", type=Path, default=d.RESULTS_CSV,
                   help="results table appended to by --measure (created, with "
                        "its parent directories, if missing)")
    return p.parse_args()


def build_dp_policy(env_cfg: dict, norm_stats: dict, in_channels: int, device):
    """Rebuild the denoiser + diffusion sampler exactly as the trainer did."""
    dp = resolve_dp_params(env_cfg)
    # norm_stats is the authority on what was actually trained; let it win.
    for k in ("num_train_timesteps", "beta_schedule", "prediction_type",
              "time_emb_dim", "denoiser_network_kind", "denoiser_width",
              "denoiser_depth"):
        if k in norm_stats:
            dp[k] = norm_stats[k]

    cond_dim = int(norm_stats.get("cond_dim", 0))
    if cond_dim:
        # The pushtWidowXdp batch (d01..d06) is pixels-only; conditioned DP
        # would need CondPixelDiffusionDenoiser from train_pusht_real_dp.py.
        raise NotImplementedError(
            f"cond_dim={cond_dim}: conditioned DP deploy not wired up "
            "(the pushtWidowXdp batch is pixels-only).")

    enc_h = int(norm_stats.get("encoder_target_height",
                               env_cfg.get("encoder_target_height", 180)))
    enc_w = int(norm_stats.get("encoder_target_width",
                               env_cfg.get("encoder_target_width", 240)))
    denoiser = build_pixel_denoiser(
        2, in_channels, dp,
        encoder_target_height=enc_h, encoder_target_width=enc_w,
        encoder_feature_dim=int(norm_stats.get("encoder_feature_dim", 256)),
        encoder_kind=str(norm_stats.get("encoder_kind", "conv_maxpool")),
        # Weights come from the checkpoint's state_dict; pretrained only affects
        # train-time init, so force False here to skip a needless ImageNet fetch.
        encoder_pretrained=False,
        encoder_num_kp=int(norm_stats.get("encoder_num_kp", 64)),
        encoder_norm_kind=str(norm_stats.get("encoder_norm_kind", "bn")),
        encoder_per_camera=bool(norm_stats.get("encoder_per_camera", False)),
        device=device)
    diffusion = build_diffusion(dp, device, (-1.0, 1.0))
    return denoiser, diffusion, dp


@torch.no_grad()
def dp_sample_action(diffusion, denoiser, obs_u8, inference, ddim_steps, ddim_eta,
                     cond=None, action_dim: int = 2):
    """One draw of the (normalized) action. obs_u8: (1,C,H,W) uint8 -> (A,).

    `action_dim` is the checkpoint's action width (2 * chunk length), so a
    chunked checkpoint returns the whole flat chunk exactly as the Q3C client's
    select_action does.
    """
    if cond is not None:
        denoiser._cond = cond
    state = obs_u8.float()
    if inference == "ddim":
        a = diffusion.ddim_sample(denoiser, state, action_dim=action_dim,
                                  num_steps=ddim_steps, eta=ddim_eta)
    else:
        a = diffusion.ddpm_sample(denoiser, state, action_dim=action_dim)
    return a[0].detach().cpu().numpy()


def main() -> int:
    args = parse_args()
    if args.z_hold > 0 and args.action_mode == "2trans":
        raise SystemExit(
            "--z-hold needs an action_mode that sends z "
            "(3trans/3trans1rot/3trans3rot); got 2trans. The injected dz would "
            "be dropped. Re-run with e.g. --action-mode 3trans.")
    seed_dir = args.seed_dir.resolve()

    # --- checkpoint metadata -------------------------------------------------
    env_cfg = d.load_run_config(seed_dir)
    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu",
                            weights_only=False)
    act_min = np.asarray(norm_stats["act_min"], np.float32)
    act_max = np.asarray(norm_stats["act_max"], np.float32)
    norm_range = tuple(norm_stats.get("action_norm_range", (-1.0, 1.0)))

    frame_stack = int(norm_stats.get("frame_stack", env_cfg.get("frame_stack", 2)))
    cams = tuple(norm_stats.get("camera_streams",
                                env_cfg.get("camera_streams", ["video1"])))
    image_hw = norm_stats.get("image_hw",
                              (int(env_cfg.get("image_height", 240)),
                               int(env_cfg.get("image_width", 320))))
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    in_channels = int(norm_stats.get("in_channels", 3 * len(cams) * frame_stack))
    # One camera (`--video-camera 1`, the blue-only lines) or two
    # (`--video-cameras 0 1`, the bothcam line) -- the checkpoint decides, and
    # the live stack is built in the same camera order the dataset used.
    cam_ids = d.camera_ids_from_streams(cams)
    topic_camera_ids = d.resolve_topic_camera_ids(args.camera_topics,
                                                  args.topic_camera_ids)
    expected_channels = 3 * len(cam_ids) * frame_stack
    if expected_channels != in_channels:
        raise SystemExit(
            f"checkpoint says in_channels={in_channels} but its camera_streams "
            f"{cams} x frame_stack {frame_stack} imply {expected_channels}.")

    # EEF (x,y) conditioning mirror (pixels-only for the DP batch -> cond None).
    cond_dim = int(norm_stats.get("cond_dim", 0))
    cond_min = cond_max = None
    if cond_dim:
        if str(norm_stats.get("cond_kind", "")) != "eef_xy":
            raise ValueError(
                f"checkpoint cond_dim={cond_dim} cond_kind="
                f"{norm_stats.get('cond_kind')!r}; only eef_xy is known")
        cond_min = np.asarray(norm_stats["cond_min"], np.float32)
        cond_max = np.asarray(norm_stats["cond_max"], np.float32)

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    denoiser, diffusion, dp = build_dp_policy(env_cfg, norm_stats, in_channels, device)
    suffix = "" if args.no_ema else "_ema"
    weights = torch.load(seed_dir / f"denoiser{suffix}.pt", map_location=device,
                         weights_only=True)
    denoiser.load_state_dict(weights)
    denoiser.eval()

    action_dim = int(act_min.size)
    ddim_steps = args.ddim_steps
    if ddim_steps is None:
        ev = norm_stats.get("ddim_eval_steps", dp.get("ddim_eval_steps", [10]))
        ddim_steps = int(ev[0]) if ev else 10
    ddim_eta = args.ddim_eta
    if ddim_eta is None:
        ddim_eta = float(norm_stats.get("ddim_eta", dp.get("ddim_eta", 0.0)))
    if args.sample_seed is not None:
        torch.manual_seed(args.sample_seed)

    # The results table's `refine_iters` column counts the inference iterations
    # the run actually paid for. Q3C puts its langevin/DFO iteration count there;
    # the DP analogue is the number of denoising steps -- the sub-sampled
    # --ddim-steps for DDIM, the full training chain for DDPM. Recording it keeps
    # two DDIM trials at different step counts as distinct parameter combos.
    refine_iters = (int(ddim_steps) if args.inference == "ddim"
                    else int(dp.get("num_train_timesteps", 0)))

    print(f"Loaded denoiser ({'raw' if args.no_ema else 'EMA'}) from {seed_dir}")
    print(f"  frame_stack={frame_stack} cameras={cams} (ids {cam_ids}) "
          f"model_hw=({image_h},{image_w}) in_channels={in_channels}")
    print(f"  inference={args.inference}"
          + (f" ({ddim_steps} steps, eta={ddim_eta})" if args.inference == "ddim" else "")
          + f"  pred={dp.get('prediction_type')}  T={dp.get('num_train_timesteps')}  device={device}")
    print(f"  act_min={act_min} act_max={act_max} norm_range={norm_range}")
    print(f"  cond_dim={cond_dim} (pixels only)")

    def make_cond(raw_obs):
        if not cond_dim:
            return None
        st = None if raw_obs is None else raw_obs.get("state")
        if st is None:
            raise RuntimeError("checkpoint needs EEF conditioning but obs has no 'state'")
        xy = np.asarray(st, np.float32).reshape(-1)[:2]
        span = np.where(cond_max == cond_min, np.ones_like(cond_max), cond_max - cond_min)
        z = np.clip(-1.0 + 2.0 * (xy - cond_min) / span, -1.0, 1.0)
        return torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)

    def sample(obs_u8, raw_obs):
        return dp_sample_action(diffusion, denoiser, obs_u8, args.inference,
                                ddim_steps, ddim_eta, cond=make_cond(raw_obs),
                                action_dim=action_dim)

    # --- connect -------------------------------------------------------------
    WidowXClient, WidowXConfigs, WidowXStatus = d.load_widowx_dependencies(
        args.widowx_envs_path)
    print(f"WidowX SDK: {WidowXClient.__module__} "
          f"({getattr(sys.modules.get(WidowXClient.__module__), '__file__', '?')})")

    env_params = d.build_env_params(args, WidowXConfigs)
    print(f"Camera topics: {args.camera_topics} -> dataset camera ids "
          f"{topic_camera_ids}; policy reads {cam_ids}")
    print(f"action_mode={args.action_mode} lock_z={args.lock_z} "
          f"fixed_z_height={args.fixed_z_height} move_duration={args.step_duration}")
    _ss = env_params["start_state"]
    print(f"reset start_state=({_ss[0]:.4f}, {_ss[1]:.4f}, {_ss[2]:.4f})"
          + ("  [demo start pose]" if args.demo_start_state else
             "  [WidowXConfigs default -- NOT where the demos start]"))

    client = WidowXClient(host=args.ip, port=args.port)

    reuse_existing_env = False
    if args.reuse_existing_env and not args.force_fresh_init:
        reuse_existing_env = d.widowx_server_has_live_env(client, max_wait_sec=1.0)
        if reuse_existing_env:
            print("[INFO] Server already has a live env; reusing it (skipping init()).")
            print("[WARN] The live env keeps its FIRST env_params -- INCLUDING its "
                  "camera_topics. If it was started with a different topic list "
                  "than --camera-topics, the camera ids above are wrong and the "
                  "policy is fed the wrong view: restart the server or pass "
                  "--no-reuse-existing-env.")

    if reuse_existing_env:
        d.set_reqrep_timeout_ms(client, max(1, args.rpc_timeout_ms))
    else:
        init_status = d.init_widowx_with_retry(
            client, env_params, args.im_size, WidowXStatus, args)
        if init_status != WidowXStatus.SUCCESS:
            raise RuntimeError(
                f"WidowX init failed after {args.init_retries} attempts with "
                f"status={d.status_name(init_status, WidowXStatus)}. If this is a "
                f"config-hash error, restart widowx_env_service --server.")
    print("WidowX connection established.")

    reset_status = d.reset_widowx_with_retry(client, WidowXStatus, args, args.i_traj)
    if reset_status != WidowXStatus.SUCCESS:
        raise RuntimeError(
            f"WidowX reset failed with status={d.status_name(reset_status, WidowXStatus)}")
    print(f"Reset done (itraj={args.i_traj}).")

    if args.gripper_command >= 0.0:
        if hasattr(client, "move_gripper"):
            try:
                gstatus = client.move_gripper(float(args.gripper_command))
                print(f"Gripper commanded to {args.gripper_command} "
                      f"(0=closed,1=open); status={d.status_name(gstatus, WidowXStatus)}")
                time.sleep(1.0)
            except Exception as exc:
                print(f"[WARN] move_gripper({args.gripper_command}) failed: {exc}")
        else:
            print("[WARN] WidowXClient has no move_gripper(); cannot actuate the clamp.")

    # --- move to the demo start pose ----------------------------------------
    start_T = None
    if args.move_to_demo_start:
        start_path = Path(args.start_eep_npy).expanduser()
        if not start_path.is_file():
            raise FileNotFoundError(
                f"--start-eep-npy not found: {start_path}. Pass "
                "--no-move-to-demo-start to skip (arm then starts ~17cm OOD).")
        start_T = np.load(start_path).astype(np.float32)
        print(f"[INFO] Moving EEF to demo start pose (x={start_T[0,3]:.3f}, "
              f"y={start_T[1,3]:.3f}, z={start_T[2,3]:.3f})...")
        move_status, tries = None, 0
        while move_status != WidowXStatus.SUCCESS and tries < args.max_initial_move_retries:
            move_status = client.move(start_T, duration=args.start_move_duration)
            tries += 1
        if move_status != WidowXStatus.SUCCESS:
            print(f"[WARN] initial move not SUCCESS after {tries} tries "
                  f"(status={d.status_name(move_status, WidowXStatus)}); continuing.")

    # --- resolve the HARD approach floor ------------------------------------
    approach_floor_x = None
    if args.approach_floor:
        if args.approach_floor_x is not None:
            approach_floor_x = float(args.approach_floor_x)
        elif start_T is not None:
            approach_floor_x = float(start_T[0, 3])
        else:
            try:
                approach_floor_x = d.eef_x_from_obs(client.get_observation())
            except Exception:
                approach_floor_x = None
        if approach_floor_x is None:
            raise RuntimeError(
                "Approach guard ON but x floor undeterminable. Pass "
                "--approach-floor-x <metres> or --no-approach-floor.")
        print(f"[SAFETY] Approach floor ARMED: EEF x never below "
              f"{approach_floor_x:.4f} m.")

    # --- warm up the frame buffer -------------------------------------------
    frame_buf = collections.deque(maxlen=frame_stack)

    def grab_obs(retries: int = 50):
        for _ in range(retries):
            obs = client.get_observation()
            if obs is not None:
                return obs
            time.sleep(0.2)
        raise RuntimeError("no observation from server after retries")

    exposure_gains = args.exposure_gains if args.match_exposure else None
    if exposure_gains is not None:
        print(f"[match-exposure] per-channel gains RGB={exposure_gains}")

    def policy_frames(raw_obs):
        return d.build_stack_frame(raw_obs, cam_ids, topic_camera_ids,
                                   (image_h, image_w), gains=exposure_gains)

    def raw_frame(raw_obs):
        return d.frame_for_camera(raw_obs, cam_ids[0], topic_camera_ids)

    first_obs = grab_obs()
    first = policy_frames(first_obs)
    print(f"Stacked frame per timestep: {first.shape} (cameras {cam_ids})")
    for _ in range(frame_stack):
        frame_buf.append(first)

    # --- dry run -------------------------------------------------------------
    if args.dry_run:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"DRY RUN: dumping {args.dry_run_steps} frames to {args.dump_dir} "
              f"(no step_action). Confirm the T renders RED.")
        for i in range(args.dry_run_steps):
            raw_obs = grab_obs()
            np.save(args.dump_dir / f"raw_{i:03d}.npy",
                    np.ascontiguousarray(raw_frame(raw_obs)))
            frame_buf.append(policy_frames(raw_obs))
            obs_u8 = d.stack_to_tensor(frame_buf, device)
            na = sample(obs_u8, raw_obs)
            act = d.unnormalize(na, act_min, act_max, norm_range)
            d.save_fed_png(args.dump_dir / f"fed_{i:03d}", list(frame_buf)[-1], cam_ids)
            print(f"[{i:03d}] norm={np.round(na, 3)} -> action(dx,dy)={np.round(act, 4)}")
            time.sleep(args.step_duration)
        client.stop()
        print("Dry run done. Inspect the fed_000.png before live control.")
        return 0

    # --- calibration: scripted open-loop moves (no policy) -------------------
    if args.calibrate:
        if args.log_dir is None:
            raise SystemExit("--calibrate needs --log-dir to write raw/ + steps.jsonl")
        (args.log_dir / "raw").mkdir(parents=True, exist_ok=True)
        log_fh = (args.log_dir / "steps.jsonl").open("w")
        phases = [("+dx", (1.0, 0.0)), ("-dx", (-1.0, 0.0)),
                  ("+dy", (0.0, 1.0)), ("-dy", (0.0, -1.0))]
        print(f"CALIBRATE: {args.calibrate_reps} steps/dir @ {args.calibrate_step*1000:.0f}mm "
              f"in +dx,-dx,+dy,-dy. Watch the image + whether the T moves.")
        input("Press [Enter] to start calibration.")
        step = 0
        for name, (ux, uy) in phases:
            for _ in range(args.calibrate_reps):
                raw_obs = grab_obs()
                np.save(args.log_dir / f"raw/{step:04d}.npy",
                        np.ascontiguousarray(raw_frame(raw_obs)))
                act_xy = np.array([ux, uy], np.float64) * args.calibrate_step
                cur_x = d.eef_x_from_obs(raw_obs)
                act_xy2, floored = d.apply_approach_floor(act_xy, cur_x, approach_floor_x)
                a7 = d.safety_clip_action(d.to_action_7d(act_xy2, args.fixed_gripper),
                                          args.action_mode, args.safety_max_xy_delta)
                env_action = d.project_action_to_env_mode(a7, args.action_mode)
                st = client.step_action(env_action, blocking=not args.non_blocking)
                if st != WidowXStatus.SUCCESS:
                    raise RuntimeError(f"step_action failed: status={st}")
                log_fh.write(json.dumps({
                    "step": step, "phase": name, "t": time.time(),
                    "action": act_xy.tolist(), "env_action": np.asarray(env_action).tolist(),
                    "floored": bool(floored),
                    "state": (None if raw_obs is None else
                              np.asarray(raw_obs.get("state")).tolist()),
                }) + "\n")
                log_fh.flush()
                print(f"[{step:03d}] {name} cmd={np.round(act_xy,4)} floored={floored}")
                step += 1
                time.sleep(args.step_duration)
        log_fh.close(); client.stop()
        print(f"Calibration done -> {args.log_dir}. Analyze with "
              f"check_action_image_frame.py (or eyeball raw/ frames per phase).")
        return 0

    # --- forensic logging ----------------------------------------------------
    log_fh = None
    if args.log_dir is not None:
        (args.log_dir / "raw").mkdir(parents=True, exist_ok=True)
        (args.log_dir / "fed").mkdir(parents=True, exist_ok=True)
        log_fh = (args.log_dir / "steps.jsonl").open("w")
        print(f"Forensic log -> {args.log_dir}")

    blocking = not args.non_blocking

    # Receding horizon. A chunked checkpoint predicts `chunk_len` consecutive
    # (dx,dy) pairs in one flat vector (act_min has size 2 * chunk_len); we
    # execute the first `exec_horizon` of them open-loop, then re-predict.
    # Observations are still appended to frame_buf on EVERY control step, so
    # the frame stack stays a run of adjacent env steps as in training.
    chunk_len = max(1, int(act_min.size) // 2)
    if args.exec_horizon < 1:
        raise SystemExit("--exec-horizon must be >= 1")
    exec_horizon = min(args.exec_horizon, chunk_len)
    if exec_horizon < args.exec_horizon:
        print(f"[WARN] --exec-horizon {args.exec_horizon} > chunk length "
              f"{chunk_len}; clipped to {exec_horizon}.")

    print(f"Closed-loop control up to {args.steps} steps, blocking={blocking}, "
          f"step_duration={args.step_duration}s. Keep a hand on the E-stop.")
    print(f"  chunk_len={chunk_len} exec_horizon={exec_horizon} "
          f"(re-predict every {exec_horizon} step(s))")
    input("Press [Enter] to start.")

    step = 0
    last_exec = time.time()
    pending: list[np.ndarray] = []   # unexecuted (dx,dy) tail of the chunk
    pending_norm: list[np.ndarray] = []
    chunk_idx = 0
    try:
        for step in range(args.steps):
            raw_obs = grab_obs()
            raw = raw_frame(raw_obs)
            frame_buf.append(policy_frames(raw_obs))
            obs_u8 = d.stack_to_tensor(frame_buf, device)

            if not pending:
                na_full = sample(obs_u8, raw_obs)
                act_full = d.unnormalize(na_full, act_min, act_max, norm_range)
                pending = list(np.asarray(act_full).reshape(-1, 2)[:exec_horizon])
                pending_norm = list(np.asarray(na_full).reshape(-1, 2)[:exec_horizon])
                chunk_idx = 0
            else:
                chunk_idx += 1

            na = pending_norm.pop(0)
            act_xy = pending.pop(0)

            # Snap sub-min-step OOD dead-zone actions onto the supported grid
            # (see apply_min_step) so tiny nonzero commands actually execute
            # instead of freezing the arm at a fixed point.
            act_xy, snapped = d.apply_min_step(act_xy, args.min_step_xy)
            if snapped:
                print(f"[min-step] snapped {np.round(na, 3)} -> dx,dy={np.round(act_xy, 4)}")

            # HARD SAFETY: never move closer to the robot than the start pose.
            cur_x = d.eef_x_from_obs(raw_obs)
            act_xy, floored = d.apply_approach_floor(act_xy, cur_x, approach_floor_x)
            if floored:
                print(f"[SAFETY] approach floor: clipped dx at x={cur_x:.4f} "
                      f"(floor={approach_floor_x:.4f})")

            action_7d = d.to_action_7d(act_xy, args.fixed_gripper)
            action_7d = d.safety_clip_action(action_7d, args.action_mode,
                                             args.safety_max_xy_delta)
            # G4 z-droop compensation: inject dz AFTER safety_clip (which zeros
            # dims 2-6 in 2trans) so it survives, and only when the mode carries
            # z. Startup guard already rejects --z-hold with action_mode=2trans.
            if args.z_hold > 0:
                dz = d.z_hold_dz(d.z_from_obs(raw_obs), args.z_hold,
                                 args.z_hold_gain, args.z_hold_max)
                action_7d[2] = dz
            env_action = d.project_action_to_env_mode(action_7d, args.action_mode)

            if not blocking:
                wait_s = (last_exec + args.step_duration) - time.time()
                if wait_s > 0:
                    time.sleep(wait_s)

            step_status = client.step_action(env_action, blocking=blocking)
            last_exec = time.time()
            if step_status != WidowXStatus.SUCCESS:
                raise RuntimeError(
                    "WidowX step_action failed: status="
                    f"{d.status_name(step_status, WidowXStatus)}, "
                    f"env_action={np.asarray(env_action).tolist()}")

            print(f"[{step:03d}] chunk[{chunk_idx}/{exec_horizon - 1}] "
                  f"norm={np.round(na, 3)} -> "
                  f"env_action={np.round(env_action, 5)}")

            if log_fh is not None:
                np.save(args.log_dir / "raw" / f"{step:04d}.npy",
                        np.ascontiguousarray(raw))
                d.save_fed_png(args.log_dir / "fed" / f"{step:04d}",
                               list(frame_buf)[-1], cam_ids)
                st = raw_obs.get("state")
                log_fh.write(json.dumps({
                    "step": step,
                    "t": time.time(),
                    "chunk_idx": chunk_idx,
                    "exec_horizon": exec_horizon,
                    "norm": [float(x) for x in np.ravel(na)],
                    "action": [float(x) for x in np.ravel(act_xy)],
                    "env_action": [float(x) for x in np.ravel(env_action)],
                    "state": (np.ravel(np.asarray(st, dtype=np.float64)).tolist()
                              if st is not None else None),
                }) + "\n")
                log_fh.flush()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if log_fh is not None:
            log_fh.close()
        # Score before stopping the client: the observation stream is what the
        # measurement reads, and it dies with the connection. An interrupted
        # episode is still worth scoring, so this sits in the finally block --
        # and it must never be the reason a run ends badly, hence the catch-all.
        if args.measure:
            try:
                final_obs = grab_obs()
                frames = {cam: d.frame_for_camera(final_obs, cam, topic_camera_ids)
                          for cam in topic_camera_ids}
                scores = d.score_final_frames(frames)
                row = {
                    # Always dp: this file only ever deploys a diffusion policy.
                    "algorithm": ALGORITHM,
                    "seed_dir": str(Path(args.seed_dir).expanduser().resolve()),
                    "inference": args.inference,     # ddpm | ddim
                    "refine_iters": refine_iters,
                    "start_position": args.start_position,
                    **scores,
                }
                trial = d.append_result_row(args.results_csv, row)
                print(f"[measure] trial {trial}: "
                      f"coverage cam0={scores['coverage_cam0']} "
                      f"cam1={scores['coverage_cam1']} "
                      f"centroid={scores['dist_centroid']} px "
                      f"-> {args.results_csv}")
            except Exception as exc:
                print(f"[measure] FAILED, no row written: {exc!r}")
        try:
            client.stop()
        except Exception:
            pass
        print(f"Stopped after {step + 1} steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
