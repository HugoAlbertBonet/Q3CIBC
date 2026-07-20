#!/usr/bin/env python3
"""Deploy a trained Q3C(IBC) Push-T policy on the real WidowX arm.

Server/client split (bridge_data_robot): the robot runs
``widowx_env_service --server``; this script is the *client*. It pulls the
fixed scene camera, runs the checkpoint, and streams 2-D translation deltas
back via ``step_action`` (action_mode ``2trans``).

Camera identity (must match training):
    seed_00XX trained on ``images1`` == ``/blue/image_raw`` == the fixed
    Logitech scene camera. The D435 (``images0``) is no longer on the rig, so
    the server runs a single camera: blue is ``full_image[0]`` and arrives as
    ``external_img``. This client auto-picks the blue frame: ``over_shoulder_img``
    if a second camera is present (legacy dual-cam), else ``external_img``.

Preprocessing reproduced from utils.datasets.PushTRealPixelsDataset:
    - decode -> RGB, resize to (image_height, image_width) with INTER_AREA,
      keep uint8 [0, 255] (the conv encoder does /255 + resize internally),
    - channel-stack frame_stack frames oldest->newest: [t-1 RGB, t RGB] = 6ch,
    - action = model output in the normalized range, unnormalized with the
      dataset's act_min/act_max (persisted in norm_stats.pt), then executed.

Channel order: on this rig the server frame already has red in channel 0 (same
as the tf-decoded training JPEGs), so we feed it as-is (no swap). Verified via
`--dry-run`, which dumps the fed frames: the T renders red. `--swap-rgb` is
available for a rig whose camera pipeline differs.

Run on the Alienware (localhost) with the server already up:

    python scripts/deploy_pusht_real.py \
        --seed-dir checkpoints/pusht_real_combinedv2/seed_0011 --dry-run
    # then, once the dry-run frames look right and the arm is clear:
    python scripts/deploy_pusht_real.py \
        --seed-dir checkpoints/pusht_real_combinedv2/seed_0011
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- Deploy-time env params for the server's robot env init. ---------------
# Mirrors experiments/bridge_data_v2/conf_clam_pusht.py so the arm behaves as
# during data collection: planar 2trans control, z locked at the table height.
# camera_topics order fixes the observation indexing: index 0 -> external_img
# (D435), index 1 -> over_shoulder_img (blue) == training images1.
FIXED_Z_HEIGHT = 0.02
# Same bounds the reference WidowX eval scripts use (bridge / condBFNPol).
# Format per row: [x, y, z, yaw, ?] as lower, upper. The training demos live
# well inside these (x 0.117->0.30, y -0.019->+0.219, z 0.02), and without them
# the policy was free to drift to y=-0.223 and park in a workspace corner
# (results/run01, results/run02).
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
# One control step's worth of motion; matches the reference STEP_DURATION and
# the default --hz 5 (0.2 s period). Was 0.08, i.e. the move completed well
# before the next observation.
STEP_DURATION = 0.2
DEPLOY_ENV_PARAMS = {
    # Single camera on the current rig: blue (Logitech). It is full_image[0],
    # so the server returns it as external_img.
    "camera_topics": [
        {"name": "/blue/image_raw"},
    ],
    "gripper_attached": "custom",
    "skip_move_to_neutral": False,
    "move_to_rand_start_freq": -1,
    "fix_zangle": 0.1,
    "action_mode": "2trans",
    "move_duration": STEP_DURATION,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "adaptive_wait": True,
    "fixed_z_height": FIXED_Z_HEIGHT,
    "neutral_z_height": FIXED_Z_HEIGHT,
    "fixed_gripper": 0.0,
    "lock_z": True,
    "action_clipping": None,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-dir", type=Path, required=True,
                   help="checkpoint dir with *_ema.pt, norm_stats.pt, config.json")
    p.add_argument("--ip", default="localhost", help="robot server host")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--steps", type=int, default=985,
                   help="max control steps (default = longest training episode)")
    p.add_argument("--hz", type=float, default=5.0, help="control loop rate")
    p.add_argument("--non-blocking", action="store_true",
                   help="send actions non-blocking (legacy). Default is BLOCKING: "
                        "wait for the arm to finish each delta before grabbing the "
                        "next frame so the 2-frame stack carries real inter-frame "
                        "motion, as in training. Non-blocking captured frames before "
                        "the move landed -> ~zero-motion obs -> 10x MAE -> (-,-) "
                        "collapse (see diagnose --zero-motion).")
    p.add_argument("--cp-selection", choices=["argmax", "sample"], default=None,
                   help="override how a control point is picked from the CP cloud "
                        "(default: whatever norm_stats recorded, usually argmax). "
                        "argmax on the robot under-steps badly: measured mean |a| "
                        "is 0.325x the demos' and it emits exactly-zero actions "
                        "0.4%% of the time vs 42.5%% in the demos, i.e. it mode-"
                        "averages a bimodal hold/push distribution. 'sample' may "
                        "restore commitment.")
    p.add_argument("--cp-temperature", type=float, default=None,
                   help="softmax temperature for --cp-selection sample "
                        "(default: from norm_stats). Lower = closer to argmax.")
    p.add_argument("--action-dim", type=int, default=7,
                   help="dimensionality of the action sent to the server. Default 7 "
                        "matches the reference WidowX evals AND the demo archive, "
                        "whose policy_out actions are (7,) with dims 2-6 exactly "
                        "zero across all 49463 transitions. The policy predicts "
                        "(dx,dy); the rest are padded with zeros + --gripper-value. "
                        "Use 2 to send a bare (dx,dy) with action_mode 2trans "
                        "(the previous behaviour).")
    p.add_argument("--gripper-value", type=float, default=0.0,
                   help="value written to the gripper dim when padding to 7. The "
                        "demos record 0.0 for every transition (the reference's "
                        "generic default is 1.0=open, but our data is explicit).")
    p.add_argument("--action-mode", default=None,
                   help="override the server env action_mode. Default: unset when "
                        "--action-dim 7 (server default, as the reference does), "
                        "'2trans' when --action-dim 2.")
    p.add_argument("--initial-eep", type=float, nargs=3, default=None,
                   help="override the EEF start position (x y z). Default is the "
                        "measured mean demo start (0.117,-0.019,0.02) from "
                        "--start-eep-npy. The reference evals default to "
                        "0.3 0.0 0.15, but that is the generic bridge default and "
                        "sits 13cm above the table, out of contact with the T.")
    p.add_argument("--max-delta-translation", type=float, default=0.03,
                   help="safety clip on each commanded translation delta, metres "
                        "(reference eval uses 0.03). Trained actions are ~+/-0.008 "
                        "so this only ever catches a blow-up. 0 disables.")
    p.add_argument("--settle", type=float, default=0.0,
                   help="extra seconds to wait after each (blocking) step for the "
                        "arm/scene to settle before capturing the next frame")
    p.add_argument("--no-require-fresh", action="store_true",
                   help="disable the duplicate-frame guard. By default the client "
                        "rejects a get_observation() frame byte-identical to the "
                        "previous one (the server occasionally repeats images) and "
                        "re-fetches until a fresh frame arrives or --fresh-timeout, "
                        "so the 2-frame stack never has a stale zero-motion slot.")
    p.add_argument("--fresh-timeout", type=float, default=0.5,
                   help="max seconds to wait for a non-duplicate frame before "
                        "proceeding with the stale one (logs a warning)")
    p.add_argument("--no-initial-move", action="store_true",
                   help="skip moving the EEF to the demo start pose before the "
                        "rollout. By default the arm is moved to the fixed pose all "
                        "training demos start from (x~0.117, y~-0.019, z~0.02). "
                        "Without it the arm starts near neutral (~x=0.29) which is "
                        "~17cm out of the training distribution -> the policy drifts "
                        "to a workspace corner and stalls (verified via --log-dir).")
    p.add_argument("--start-eep-npy", type=Path,
                   default=ROOT / "scripts" / "assets" / "pusht_start_eep.npy",
                   help="4x4 EEF start transform to move to (mean of demo starts)")
    p.add_argument("--start-move-duration", type=float, default=1.5)
    p.add_argument("--log-dir", type=Path, default=None,
                   help="if set, forensic-log every closed-loop step here: raw + "
                        "fed frames (PNG) and a steps.jsonl row with timestamp, "
                        "normalized + metric action, and the EEF proprio state. "
                        "For offline review of what the arm saw and did.")
    p.add_argument("--no-ema", action="store_true",
                   help="use raw weights instead of the EMA copy")
    p.add_argument("--swap-rgb", action="store_true",
                   help="swap channels before feeding the model. Default OFF: this "
                        "rig's server frame already has red in channel 0, matching "
                        "training (verified via dry-run; imshow shows the T blue).")
    p.add_argument("--obs-key", default="auto",
                   choices=["auto", "external_img", "over_shoulder_img"],
                   help="which get_observation() field holds the blue frame "
                        "(auto: over_shoulder_img if present else external_img)")
    p.add_argument("--dry-run", action="store_true",
                   help="no motion: dump fed frames + print predicted actions")
    p.add_argument("--dry-run-steps", type=int, default=20)
    p.add_argument("--dump-dir", type=Path, default=ROOT / "deploy_dryrun",
                   help="where --dry-run writes the fed RGB frames")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_run_config(seed_dir: Path) -> dict:
    cfg_path = seed_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing per-run config: {cfg_path}")
    with cfg_path.open() as fh:
        config = json.load(fh)
    active = config["active_env"]
    env = config["environments"][active]
    return env


def build_models(env: dict, in_channels: int, device):
    """Reconstruct CP generator + Q estimator exactly as the trainer did.

    Reads the model block from the per-run config so any hyperparameter change
    is picked up rather than hardcoded. pusht_real_pixels is unconditioned
    (cond_dim = goal_dim = 0).
    """
    from utils.models import PixelControlPointGenerator, PixelQEstimator

    m = env["model"]
    enc_h = int(env.get("encoder_target_height", 180))
    enc_w = int(env.get("encoder_target_width", 240))
    action_dim = int(env.get("action_dim", 2))
    a_lo, a_hi = env.get("action_bounds", [-1.0, 1.0])

    control_points = int(m.get("control_points", 50))
    num_neurons = int(m.get("num_neurons", 512))
    num_hidden_layers = int(m.get("num_hidden_layers", 8))
    cp_width = int(m.get("cp_width", num_neurons))
    cp_depth = int(m.get("cp_depth", num_hidden_layers))
    cp_network_kind = m.get("cp_network_kind", "mlp")
    value_width = int(m.get("value_width", 1024))
    value_num_blocks = int(m.get("value_num_blocks", 1))
    encoder_kind = m.get("encoder_kind", "conv_maxpool")

    cp_gen = PixelControlPointGenerator(
        output_dim=action_dim,
        control_points=control_points,
        hidden_dims=[cp_width for _ in range(cp_depth)],
        action_bounds=(float(a_lo), float(a_hi)),
        network_kind=cp_network_kind,
        width=cp_width,
        depth=cp_depth,
        in_channels=in_channels,
        encoder_target_height=enc_h,
        encoder_target_width=enc_w,
        cond_dim=0,
        encoder_kind=encoder_kind,
        goal_dim=0,
    ).to(device).eval()

    q_net = PixelQEstimator(
        action_dim=action_dim,
        in_channels=in_channels,
        encoder_target_height=enc_h,
        encoder_target_width=enc_w,
        value_width=value_width,
        value_num_blocks=value_num_blocks,
        cond_dim=0,
        encoder_kind=encoder_kind,
        goal_dim=0,
    ).to(device).eval()
    return cp_gen, q_net


def load_weights(model, path: Path, device):
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint weights: {path}")
    # weights_only=False: our own trusted checkpoints (state dicts + numpy).
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    return model


def pick_blue_frame(obs: dict, obs_key: str) -> np.ndarray:
    """Return the blue-camera BGR frame from a get_observation() dict.

    auto: prefer over_shoulder_img (legacy dual-cam: blue = full_image[1]),
    else external_img (single-cam: blue = full_image[0]).
    """
    if obs_key == "auto":
        if obs.get("over_shoulder_img") is not None:
            return obs["over_shoulder_img"]
        if obs.get("external_img") is not None:
            return obs["external_img"]
        raise RuntimeError("no blue frame: obs has neither over_shoulder_img nor external_img")
    if obs.get(obs_key) is None:
        raise RuntimeError(f"requested --obs-key {obs_key} missing from observation")
    return obs[obs_key]


def preprocess(frame: np.ndarray, out_hw, swap_rgb: bool) -> np.ndarray:
    """Live blue frame (H,W,3 uint8) -> (H',W',3 uint8), channels as training.

    On this rig the server frame already has red in channel 0 (same as the
    tf-decoded training JPEGs), so no swap by default. `--swap-rgb` flips it for
    a rig whose pipeline differs.
    """
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if swap_rgb else frame
    H, W = out_hw
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def stack_to_tensor(frame_buf, device) -> "torch.Tensor":
    """frame_buf oldest->newest, each (H,W,3) -> (1, 3*fs, H, W) uint8 tensor."""
    stacked = np.concatenate(list(frame_buf), axis=-1)     # (H, W, 3*fs)
    stacked = np.transpose(stacked, (2, 0, 1))             # (3*fs, H, W)
    return torch.from_numpy(np.ascontiguousarray(stacked)).unsqueeze(0).to(device)


@torch.no_grad()
def select_action(cp_gen, q_net, obs_u8, cp_selection: str, temperature: float):
    """Pure CP-cloud ranking (langevin/DFO disabled for this hardware)."""
    features = q_net.encode(obs_u8)          # (1, feat)
    cps = cp_gen(obs_u8)                      # (1, P, action_dim)
    logits = q_net.score(features, cps).squeeze(-1)   # (1, P)
    if cp_selection == "sample":
        probs = torch.softmax(logits.squeeze(0) / max(temperature, 1e-6), dim=-1)
        idx = int(torch.multinomial(probs, 1).item())
    else:
        idx = int(logits.squeeze(0).argmax().item())
    return cps[0, idx].detach().cpu().numpy()   # normalized action


def adapt_action_dim(act, action_dim: int, gripper_value: float) -> np.ndarray:
    """(dx,dy) -> the vector the server expects.

    The demo archive records 7-D actions whose dims 2-6 (dz, droll, dpitch,
    dyaw, gripper) are exactly zero for all 49463 transitions, so padding with
    zeros reproduces the commands the data was collected with.
    """
    act = np.asarray(act, np.float32).ravel()
    if action_dim <= act.size:
        return act[:action_dim]
    out = np.zeros(action_dim, np.float32)
    out[:act.size] = act
    if action_dim >= 7:
        out[6] = gripper_value
    return out


def unnormalize(norm_action, act_min, act_max, norm_range):
    lo, hi = norm_range
    scale = (act_max - act_min) / (hi - lo)
    return (act_min + (np.asarray(norm_action, np.float32) - lo) * scale).astype(np.float32)


def main() -> int:
    args = parse_args()
    seed_dir = args.seed_dir.resolve()

    # --- checkpoint metadata ------------------------------------------------
    env = load_run_config(seed_dir)
    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu",
                            weights_only=False)
    act_min = np.asarray(norm_stats["act_min"], np.float32)
    act_max = np.asarray(norm_stats["act_max"], np.float32)
    norm_range = tuple(norm_stats.get("action_norm_range", (-1.0, 1.0)))
    frame_stack = int(norm_stats.get("frame_stack", env.get("frame_stack", 2)))
    cp_selection = args.cp_selection or str(norm_stats.get("cp_selection", "argmax"))
    cp_temp = (args.cp_temperature if args.cp_temperature is not None
               else float(norm_stats.get("cp_selection_temperature", 1.0)))

    cams = list(env.get("camera_streams", ["images1"]))
    if cams != ["images1"]:
        print(f"WARNING: checkpoint camera_streams={cams}; this client only "
              f"feeds the single fixed scene camera (images1/blue). Verify.")
    image_h = int(env.get("image_height", 240))
    image_w = int(env.get("image_width", 320))
    in_channels = 3 * len(cams) * frame_stack

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    print(f"Seed dir:     {seed_dir}")
    print(f"Cameras:      {cams} (deploy stream = over_shoulder_img / blue)")
    print(f"Frame stack:  {frame_stack}  in_channels={in_channels}  "
          f"input={image_h}x{image_w}")
    print(f"Action range: {act_min} -> {act_max}  norm={norm_range}")
    print(f"CP selection: {cp_selection} (temp={cp_temp})  device={device}")

    # --- models -------------------------------------------------------------
    cp_gen, q_net = build_models(env, in_channels, device)
    suffix = "" if args.no_ema else "_ema"
    load_weights(cp_gen, seed_dir / f"control_point_generator{suffix}.pt", device)
    load_weights(q_net, seed_dir / f"q_estimator{suffix}.pt", device)
    print(f"Loaded weights ({'raw' if args.no_ema else 'EMA'}).")

    # --- connect to robot ---------------------------------------------------
    from widowx_envs.widowx_env_service import WidowXClient, WidowXStatus

    # The demo archive's actions are (7,) with dims 2-6 identically zero, and the
    # reference WidowX evals drive this same robot/server with 7-D actions and no
    # action_mode override. Match that by default; --action-dim 2 restores the
    # previous 2trans interface.
    env_params = dict(DEPLOY_ENV_PARAMS)
    if args.action_mode is not None:
        env_params["action_mode"] = args.action_mode
    elif args.action_dim == 7:
        env_params.pop("action_mode", None)      # server default, as reference
    else:
        env_params["action_mode"] = "2trans"
    print(f"Env action_mode={env_params.get('action_mode', '<server default>')} "
          f"action_dim={args.action_dim}")

    client = WidowXClient(host=args.ip, port=args.port)
    client.init(env_params, image_size=256)
    # init only constructs the env; reset() homes the arm AND starts the
    # control loop that step_action depends on. Without it the first
    # step_action throws server-side and drops the connection.
    print("Resetting robot (home + start control loop)...")
    client.reset()

    # Move to the pose all demos start from. Skipping this leaves the arm at
    # neutral (~x=0.29), ~17cm off the demo start (x~0.117); from there the
    # policy is out-of-distribution and drifts to a workspace corner and stalls.
    if not args.no_initial_move:
        start_T = np.load(args.start_eep_npy).astype(np.float32)
        if args.initial_eep is not None:
            start_T = start_T.copy()
            start_T[:3, 3] = np.asarray(args.initial_eep, np.float32)
        print(f"Moving EEF to demo start pose (x={start_T[0,3]:.3f}, "
              f"y={start_T[1,3]:.3f}, z={start_T[2,3]:.3f})...")
        move_status, tries = None, 0
        while move_status != WidowXStatus.SUCCESS and tries < 5:
            move_status = client.move(start_T, duration=args.start_move_duration)
            tries += 1
        if move_status != WidowXStatus.SUCCESS:
            print(f"[warn] initial move did not report SUCCESS after {tries} tries "
                  f"(status={move_status}); continuing anyway.")

    obs = None
    while obs is None:
        obs = client.get_observation()
        if obs is None:
            print("Waiting for robot/cameras...")
            time.sleep(1.0)
    blue0 = pick_blue_frame(obs, args.obs_key)   # raises with a clear msg if absent
    resolved_key = ("over_shoulder_img"
                    if (args.obs_key == "auto" and obs.get("over_shoulder_img") is not None)
                    else ("external_img" if args.obs_key == "auto" else args.obs_key))
    print(f"Blue frame source: {resolved_key} (raw {blue0.shape})")

    frame_buf = collections.deque(maxlen=frame_stack)
    period = 1.0 / max(args.hz, 1e-3)

    require_fresh = not args.no_require_fresh
    last_raw = {"v": None}   # newest RAW blue frame, for duplicate detection
    last_obs = {"v": None}   # newest full observation dict, for proprio logging

    def grab_raw(retries: int = 25):
        # get_observation() can transiently return None (server busy/restarting).
        # Retry instead of crashing on None.get in pick_blue_frame.
        for _ in range(retries):
            o = client.get_observation()
            frame = None if o is None else pick_blue_frame(o, args.obs_key)
            if frame is not None:
                last_obs["v"] = o
                return frame
            time.sleep(0.2)
        raise RuntimeError("no observation from server after retries (server down?)")

    def refresh_frame():
        # The server occasionally returns a repeated (stale) image. A duplicate
        # in the frame stack means zero inter-frame motion, which the policy maps
        # to garbage / hold actions (see diagnose --zero-motion: MAE 0.02 -> 0.22).
        # Re-fetch until the raw frame differs from the previous one, bounded by
        # --fresh-timeout so a genuinely static scene can't hang the loop.
        raw = grab_raw()
        if require_fresh and last_raw["v"] is not None:
            t0 = time.time()
            while np.array_equal(raw, last_raw["v"]) and (time.time() - t0) < args.fresh_timeout:
                time.sleep(0.05)
                raw = grab_raw()
            if np.array_equal(raw, last_raw["v"]):
                print(f"[warn] stale frame: server repeated image "
                      f"(no fresh frame within {args.fresh_timeout}s)")
        last_raw["v"] = raw
        return preprocess(raw, (image_h, image_w), args.swap_rgb)

    first = refresh_frame()
    for _ in range(frame_stack):     # pad episode start with the first frame
        frame_buf.append(first)

    # --- dry run: no motion, dump frames + print actions --------------------
    if args.dry_run:
        import cv2
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"DRY RUN: dumping {args.dry_run_steps} frames to {args.dump_dir} "
              f"(no step_action). Confirm the T renders RED.")
        for i in range(args.dry_run_steps):
            # Also persist the RAW blue frame (exactly as the server returns it)
            # for offline preprocessing-parity analysis vs the training pipeline.
            raw = None
            while raw is None:
                o = client.get_observation()
                raw = None if o is None else pick_blue_frame(o, args.obs_key)
                if raw is None:
                    time.sleep(0.2)
            np.save(args.dump_dir / f"raw_{i:03d}.npy", np.ascontiguousarray(raw))
            frame_buf.append(preprocess(raw, (image_h, image_w), args.swap_rgb))
            obs_u8 = stack_to_tensor(frame_buf, device)
            na = select_action(cp_gen, q_net, obs_u8, cp_selection, cp_temp)
            act = unnormalize(na, act_min, act_max, norm_range)
            newest_rgb = list(frame_buf)[-1]
            cv2.imwrite(str(args.dump_dir / f"fed_{i:03d}.png"),
                        cv2.cvtColor(newest_rgb, cv2.COLOR_RGB2BGR))
            print(f"[{i:03d}] norm={np.round(na,3)} -> action(dx,dy)={np.round(act,4)}")
            time.sleep(period)
        client.stop()
        print("Dry run done. Inspect deploy_dryrun/fed_000.png before live control.")
        return 0

    # --- closed loop --------------------------------------------------------
    mode = "non-blocking (legacy)" if args.non_blocking else "blocking"
    print(f"Closed-loop control up to {args.steps} steps @ {args.hz}Hz, "
          f"step={mode}, settle={args.settle}s. "
          f"Keep a hand on the E-stop. Ctrl-C to stop.")
    # --- forensic logging setup ---------------------------------------------
    log_fh = None
    if args.log_dir is not None:
        import cv2
        args.log_dir.mkdir(parents=True, exist_ok=True)
        (args.log_dir / "raw").mkdir(exist_ok=True)
        (args.log_dir / "fed").mkdir(exist_ok=True)
        log_fh = (args.log_dir / "steps.jsonl").open("w")
        print(f"Forensic log -> {args.log_dir} (raw/*.npy, fed/*.png, steps.jsonl)")

    def log_step(step, na, act, fed_rgb):
        if log_fh is None:
            return
        np.save(args.log_dir / "raw" / f"{step:04d}.npy",
                np.ascontiguousarray(last_raw["v"]))
        cv2.imwrite(str(args.log_dir / "fed" / f"{step:04d}.png"),
                    cv2.cvtColor(fed_rgb, cv2.COLOR_RGB2BGR))
        o = last_obs["v"] or {}
        state = o.get("state")
        row = {
            "step": step,
            "t": time.time(),
            "norm": [float(x) for x in np.ravel(na)],
            "action": [float(x) for x in np.ravel(act)],
            "state": (np.ravel(state).astype(float).tolist()
                      if state is not None else None),
        }
        log_fh.write(json.dumps(row) + "\n")
        log_fh.flush()

    step = 0
    try:
        for step in range(args.steps):
            t0 = time.time()
            frame_buf.append(refresh_frame())
            obs_u8 = stack_to_tensor(frame_buf, device)
            na = select_action(cp_gen, q_net, obs_u8, cp_selection, cp_temp)
            act = unnormalize(na, act_min, act_max, norm_range)
            if args.max_delta_translation > 0:
                act = np.clip(act, -args.max_delta_translation,
                              args.max_delta_translation)
            cmd = adapt_action_dim(act, args.action_dim, args.gripper_value)
            # Server env needs action.shape (numpy array). Pickle compatibility
            # with the server's numpy 1.x requires the CLIENT env to also run
            # numpy<2 (else numpy._core is unresolvable server-side).
            res = client.step_action(cmd, blocking=not args.non_blocking)
            if args.settle > 0:
                time.sleep(args.settle)
            log_step(step, na, act, list(frame_buf)[-1])
            print(f"[{step:03d}] norm={np.round(na, 3)} -> action(dx,dy)={np.round(act, 4)}")
            if res == WidowXStatus.NO_CONNECTION:
                print("Lost connection to server. Stopping.")
                break
            # Rate-limit only in non-blocking mode: a blocking step_action
            # already paces the loop by waiting for the arm (matches the
            # reference eval, which skips its STEP_DURATION wait when blocking).
            if args.non_blocking:
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if log_fh is not None:
            log_fh.close()
        client.stop()
        print(f"Stopped after {step + 1} steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
