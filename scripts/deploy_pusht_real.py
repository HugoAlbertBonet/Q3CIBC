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

The live camera comes back BGR (edgeml jpeg_to_mat = cv2.imdecode); training
decoded JPEGs as RGB (tf.io.decode_jpeg). We convert BGR->RGB by default so
channel-0 is red, exactly as in training. `--dry-run` dumps the fed frames so
you can eyeball that the T renders red before any motion.

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
    "move_duration": 0.08,
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
    p.add_argument("--no-ema", action="store_true",
                   help="use raw weights instead of the EMA copy")
    p.add_argument("--keep-bgr", action="store_true",
                   help="do NOT convert live BGR->RGB (debug only; breaks color match)")
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
    state = torch.load(path, map_location=device)
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


def preprocess(frame_bgr: np.ndarray, out_hw, keep_bgr: bool) -> np.ndarray:
    """Live over_shoulder_img (H,W,3 BGR uint8) -> (H',W',3 RGB uint8)."""
    import cv2

    rgb = frame_bgr if keep_bgr else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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


def unnormalize(norm_action, act_min, act_max, norm_range):
    lo, hi = norm_range
    scale = (act_max - act_min) / (hi - lo)
    return (act_min + (np.asarray(norm_action, np.float32) - lo) * scale).astype(np.float32)


def main() -> int:
    args = parse_args()
    seed_dir = args.seed_dir.resolve()

    # --- checkpoint metadata ------------------------------------------------
    env = load_run_config(seed_dir)
    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu")
    act_min = np.asarray(norm_stats["act_min"], np.float32)
    act_max = np.asarray(norm_stats["act_max"], np.float32)
    norm_range = tuple(norm_stats.get("action_norm_range", (-1.0, 1.0)))
    frame_stack = int(norm_stats.get("frame_stack", env.get("frame_stack", 2)))
    cp_selection = str(norm_stats.get("cp_selection", "argmax"))
    cp_temp = float(norm_stats.get("cp_selection_temperature", 1.0))

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

    client = WidowXClient(host=args.ip, port=args.port)
    client.init(DEPLOY_ENV_PARAMS, image_size=256)
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

    def refresh_frame():
        o = client.get_observation()
        f = preprocess(pick_blue_frame(o, args.obs_key), (image_h, image_w), args.keep_bgr)
        return f

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
            frame_buf.append(refresh_frame())
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
    print(f"Closed-loop control up to {args.steps} steps @ {args.hz}Hz. "
          f"Keep a hand on the E-stop. Ctrl-C to stop.")
    step = 0
    try:
        for step in range(args.steps):
            t0 = time.time()
            frame_buf.append(refresh_frame())
            obs_u8 = stack_to_tensor(frame_buf, device)
            na = select_action(cp_gen, q_net, obs_u8, cp_selection, cp_temp)
            act = unnormalize(na, act_min, act_max, norm_range)
            res = client.step_action(np.asarray(act, np.float32), blocking=False)
            if res == WidowXStatus.NO_CONNECTION:
                print("Lost connection to server. Stopping.")
                break
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        client.stop()
        print(f"Stopped after {step + 1} steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
