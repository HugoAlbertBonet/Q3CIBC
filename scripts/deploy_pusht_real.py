#!/usr/bin/env python3
"""Deploy a trained Q3C(IBC) Push-T policy on the real WidowX arm.

The robot-facing half of this script is modelled directly on
``data/eval_widowx_bfn.py`` -- a script CONFIRMED WORKING on this exact rig for
the same Push-T task with a different algorithm. Everything that touches the
WidowX server (env params, init/reset retry policy, action projection, safety
clipping, control loop) mirrors that script. Only the policy is ours.

Key facts taken from the confirmed-working script (do not "fix" these):
  * ``action_mode="2trans"``: the client sends a **2-element** (dx, dy) action.
    ``_project_action_to_env_mode`` slices ``action_7d[:2]``.
  * ``im_size=480, im_width=640`` -- the service is told the native camera
    geometry, not a square 256.
  * ``lock_z=True, fixed_z_height=0.02, neutral_z_height=0.02,
    fixed_gripper=0.0`` plus the z-lock / deadband / vr_* tuning keys.
  * ``env_params`` is built on top of ``WidowXConfigs.DefaultEnvParams.copy()``.
  * Init needs a LONG rpc timeout (180 s) and several retries; the short default
    is what produces spurious init failures.
  * The start pose comes from ``reset(itraj=N)`` (the collect-style reset), NOT
    from ``start_state`` and NOT from an explicit absolute ``move``. The optional
    absolute move is documented there as "recommended false".
  * ``blocking=True`` by default.

Q3C-specific (differs from the BFN reference by necessity):
  * Observation: uint8 [0,255], resized to (image_height, image_width) with
    INTER_AREA and channel-CONCATENATED oldest->newest into (3*frame_stack,H,W),
    reproducing utils.datasets.PushTRealPixelsDataset. Not float[0,1], not
    stacked on a new axis.
  * The policy emits a normalized (dx, dy); it is min-max denormalized with
    norm_stats (act_min/act_max) before being sent.
  * Camera: seed_00XX trained on ``images1`` == ``/blue/image_raw``. With the
    D435 removed the blue camera is the only one, so it arrives as
    ``external_img``.

Usage (server already up):

    python scripts/deploy_pusht_real.py \
        --seed-dir checkpoints/pusht_real_combinedv2/seed_0011 \
        --device cpu --dry-run
    python scripts/deploy_pusht_real.py \
        --seed-dir checkpoints/pusht_real_combinedv2/seed_0011 \
        --device cpu --steps 200 --log-dir results/run_new
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- constants copied from the confirmed-working eval_widowx_bfn.py ---------
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = ["/blue/image_raw"]
FIXED_Z_HEIGHT = 0.02
NEUTRAL_Z_HEIGHT = 0.02
FIXED_GRIPPER = 0.0
# The demo archive's actions are ±0.008 in x/y; the working script clips at the
# same magnitude via vr_xy_step_clip.
SAFETY_MAX_XY_DELTA = 0.008


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-ema", action="store_true",
                   help="use raw weights instead of the EMA copy")
    p.add_argument("--ip", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--widowx-envs-path", type=Path,
                   default=Path.home() / "bridge_data_robot" / "widowx_envs",
                   help="path prepended to sys.path before importing widowx_envs. "
                        "MUST match the widowx_envs the server runs, or the edgeml "
                        "handshake fails with 'Incompatible config with hash'.")
    p.add_argument("--camera-topics", nargs="+", default=CAMERA_TOPICS)

    # --- service image geometry (confirmed-working values) ------------------
    p.add_argument("--im-size", type=int, default=480, help="service image height")
    p.add_argument("--im-width", type=int, default=640, help="service image width")

    # --- control -------------------------------------------------------------
    p.add_argument("--steps", type=int, default=200, help="max control steps")
    p.add_argument("--step-duration", type=float, default=0.1,
                   help="control period; also used as env move_duration")
    p.add_argument("--non-blocking", action="store_true",
                   help="the working reference uses blocking=True; this opts out")
    p.add_argument("--action-mode", default="2trans",
                   choices=["2trans", "3trans", "3trans1rot", "3trans3rot"])
    p.add_argument("--safety-max-xy-delta", type=float, default=SAFETY_MAX_XY_DELTA)
    p.add_argument("--lock-z", dest="lock_z", action="store_true", default=True)
    p.add_argument("--no-lock-z", dest="lock_z", action="store_false")
    p.add_argument("--fixed-z-height", type=float, default=FIXED_Z_HEIGHT)
    p.add_argument("--neutral-z-height", type=float, default=NEUTRAL_Z_HEIGHT)
    p.add_argument("--fixed-gripper", type=float, default=FIXED_GRIPPER)
    p.add_argument("--skip-move-to-neutral", action="store_true")
    p.add_argument("--i-traj", type=int, default=0,
                   help="trajectory index passed to reset(itraj=N). This is what "
                        "puts the arm at the collect-time start pose.")

    # --- init / reset robustness (confirmed-working values) -----------------
    p.add_argument("--init-timeout-ms", type=int, default=180_000)
    p.add_argument("--init-retries", type=int, default=8)
    p.add_argument("--init-retry-sleep", type=float, default=2.0)
    p.add_argument("--reset-timeout-ms", type=int, default=60_000)
    p.add_argument("--reset-retries", type=int, default=3)
    p.add_argument("--reset-retry-sleep", type=float, default=1.0)
    p.add_argument("--rpc-timeout-ms", type=int, default=5_000)

    # --- policy --------------------------------------------------------------
    p.add_argument("--cp-selection", choices=["argmax", "sample"], default=None,
                   help="override CP-cloud selection (default: from norm_stats)")
    p.add_argument("--cp-temperature", type=float, default=None)

    # --- diagnostics ---------------------------------------------------------
    p.add_argument("--dry-run", action="store_true",
                   help="no motion: dump fed frames + print predicted actions")
    p.add_argument("--dry-run-steps", type=int, default=20)
    p.add_argument("--dump-dir", type=Path, default=ROOT / "deploy_dryrun")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="per-step forensic log: raw/*.npy, fed/*.png, steps.jsonl")
    return p.parse_args()


# ---------------------------------------------------------------------------
# WidowX plumbing (mirrors data/eval_widowx_bfn.py)
# ---------------------------------------------------------------------------

def load_widowx_dependencies(widowx_envs_path: Path):
    path = Path(widowx_envs_path).expanduser()
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        from widowx_envs.widowx_env_service import (  # type: ignore
            WidowXClient, WidowXConfigs, WidowXStatus,
        )
    except Exception as exc:
        raise ImportError(
            f"Failed to import widowx_envs from {path}. "
            "Set --widowx-envs-path correctly."
        ) from exc
    return WidowXClient, WidowXConfigs, WidowXStatus


def status_name(status: Any, WidowXStatus: Any) -> str:
    for name in ("SUCCESS", "NO_CONNECTION", "EXECUTION_FAILURE", "NOT_INITIALIZED"):
        if hasattr(WidowXStatus, name) and status == getattr(WidowXStatus, name):
            return name
    return str(status)


def set_reqrep_timeout_ms(client: Any, timeout_ms: int) -> None:
    """Best-effort update of the underlying req/rep timeout used by widowx_envs."""
    try:
        action_client = getattr(client, "_WidowXClient__client", None)
        if action_client is None:
            return
        reqrep_client = getattr(action_client, "client", None)
        if reqrep_client is None:
            return
        reqrep_client.timeout_ms = int(timeout_ms)
        reqrep_client.reset_socket()
    except Exception:
        pass


def build_env_params(args, WidowXConfigs) -> Dict[str, Any]:
    """Exactly the dict the confirmed-working BFN eval sends."""
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update({
        "camera_topics": [{"name": t} for t in args.camera_topics],
        "override_workspace_boundaries": WORKSPACE_BOUNDS,
        "move_duration": args.step_duration,
        "action_mode": args.action_mode,
        "skip_move_to_neutral": bool(args.skip_move_to_neutral),
        "move_to_rand_start_freq": -1,
        "fix_zangle": 0.1,
        "adaptive_wait": True,
        "fixed_z_height": float(args.fixed_z_height),
        "neutral_z_height": float(args.neutral_z_height),
        "z_lock_feedback_gain": 0.2,
        "z_lock_max_delta": 0.0015,
        "z_lock_deadband": 0.002,
        "xy_action_deadband": 0.0015,
        "vr_vertical_reject_ratio": 0.6,
        "vr_xy_step_deadband": 0.0015,
        "vr_xy_step_clip": 0.008,
        "vr_xy_scale": 0.9,
        "fixed_gripper": float(args.fixed_gripper),
        "lock_z": bool(args.lock_z),
        "action_clipping": None,
    })
    return env_params


def init_widowx_with_retry(client, env_params, image_size, WidowXStatus, args):
    set_reqrep_timeout_ms(client, max(1, args.init_timeout_ms))
    last_status = None
    for attempt in range(1, max(1, args.init_retries) + 1):
        print(f"[INFO] WidowX init attempt {attempt}/{args.init_retries} "
              f"(timeout={args.init_timeout_ms} ms, server={args.ip}:{args.port})")
        t0 = time.time()
        last_status = client.init(env_params, image_size=image_size)
        elapsed = time.time() - t0
        if last_status == WidowXStatus.SUCCESS:
            set_reqrep_timeout_ms(client, max(1, args.rpc_timeout_ms))
            return last_status
        print(f"[WARN] init attempt {attempt} failed with "
              f"status={status_name(last_status, WidowXStatus)} after {elapsed:.2f}s.")
        if last_status == getattr(WidowXStatus, "NO_CONNECTION", None):
            print("[HINT] No response from the WidowX action server. Make sure "
                  "`widowx_env_service --server` is running and reachable at "
                  f"{args.ip}:{args.port}.")
        if attempt < args.init_retries and args.init_retry_sleep > 0:
            time.sleep(args.init_retry_sleep)
    set_reqrep_timeout_ms(client, max(1, args.rpc_timeout_ms))
    return last_status


def reset_widowx_with_retry(client, WidowXStatus, args, i_traj: int | None):
    set_reqrep_timeout_ms(client, max(args.reset_timeout_ms, args.rpc_timeout_ms))
    last_status = None
    warned = False
    for attempt in range(1, max(1, args.reset_retries) + 1):
        if i_traj is None:
            last_status = client.reset()
        else:
            try:
                last_status = client.reset(itraj=int(i_traj))
            except TypeError:
                if not warned:
                    print("[WARN] reset(itraj=...) unsupported by this widowx_envs "
                          "version; falling back to reset().")
                    warned = True
                last_status = client.reset()
        if last_status == WidowXStatus.SUCCESS:
            break
        print(f"[WARN] reset attempt {attempt} failed with "
              f"status={status_name(last_status, WidowXStatus)}.")
        if attempt < args.reset_retries and args.reset_retry_sleep > 0:
            time.sleep(args.reset_retry_sleep)
    set_reqrep_timeout_ms(client, max(1, args.rpc_timeout_ms))
    return last_status


# ---------------------------------------------------------------------------
# Observation handling (mirrors eval_widowx_bfn.py, then q3c preprocessing)
# ---------------------------------------------------------------------------

def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Image channel mismatch, expected 3 channels, got {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def extract_blue_frame(raw_obs: Dict[str, Any]) -> np.ndarray:
    """Return the fixed scene (blue) camera frame as (H,W,3) uint8 RGB.

    Preference order matches eval_widowx_bfn._extract_widowx_rgb_obs: the
    single-camera rig delivers blue as external_img.
    """
    for key in ("external_img", "over_shoulder_img"):
        if raw_obs.get(key) is not None:
            return to_uint8_rgb(np.asarray(raw_obs[key]))

    full_image = raw_obs.get("full_image")
    if full_image is not None:
        arr = np.asarray(full_image)
        if arr.ndim == 4:
            return to_uint8_rgb(arr[0])
        if arr.ndim == 3:
            return to_uint8_rgb(arr)

    raise RuntimeError(
        "WidowX observation has no usable camera frame "
        f"(keys={sorted(raw_obs.keys())})"
    )


def preprocess(frame: np.ndarray, out_hw) -> np.ndarray:
    """(H,W,3) uint8 RGB -> (H',W',3) uint8, as PushTRealPixelsDataset does.

    The training pipeline decodes to RGB and resizes with AREA, keeping uint8
    (the conv encoder does the /255 itself).
    """
    import cv2

    H, W = out_hw
    if frame.shape[:2] != (H, W):
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)


def stack_to_tensor(frame_buf, device) -> torch.Tensor:
    """oldest->newest (H,W,3) frames -> (1, 3*fs, H, W) uint8 tensor."""
    stacked = np.concatenate(list(frame_buf), axis=-1)     # (H, W, 3*fs)
    stacked = np.transpose(stacked, (2, 0, 1))             # (3*fs, H, W)
    return torch.from_numpy(np.ascontiguousarray(stacked)).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Q3C policy (unchanged -- deploy_pusht_real_v2.py imports these)
# ---------------------------------------------------------------------------

def load_run_config(seed_dir: Path) -> dict:
    with (seed_dir / "config.json").open() as fh:
        config = json.load(fh)
    return config["environments"][config["active_env"]]


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
        width=value_width,
        num_blocks=value_num_blocks,
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


@torch.no_grad()
def select_action(cp_gen, q_net, obs_u8, cp_selection: str, temperature: float):
    """Pure CP-cloud ranking (langevin/DFO disabled for this hardware)."""
    features = q_net.encode(obs_u8)                   # (1, feat)
    cps = cp_gen(obs_u8)                              # (1, P, action_dim)
    logits = q_net.score(features, cps).squeeze(-1)   # (1, P)
    if cp_selection == "sample":
        probs = torch.softmax(logits.squeeze(0) / max(temperature, 1e-6), dim=-1)
        idx = int(torch.multinomial(probs, 1).item())
    else:
        idx = int(logits.squeeze(0).argmax().item())
    return cps[0, idx].detach().cpu().numpy()         # normalized action


def unnormalize(norm_action, act_min, act_max, norm_range):
    lo, hi = norm_range
    scale = (act_max - act_min) / (hi - lo)
    return (act_min + (np.asarray(norm_action, np.float32) - lo) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Action shaping (mirrors eval_widowx_bfn.py)
# ---------------------------------------------------------------------------

def to_action_7d(act_xy: np.ndarray, gripper_value: float) -> np.ndarray:
    """(dx,dy) -> 7-D [dx,dy,dz,droll,dpitch,dyaw,grip].

    All 49463 demo transitions have dims 2-6 exactly zero, so zeros reproduce
    the commands the data was collected with.
    """
    out = np.zeros(7, dtype=np.float64)
    out[:2] = np.asarray(act_xy, np.float64).ravel()[:2]
    out[6] = float(gripper_value)
    return out


def safety_clip_action(action_7d: np.ndarray, action_mode: str,
                       max_xy_delta: float) -> np.ndarray:
    action = np.asarray(action_7d, dtype=np.float64).copy()
    if action_mode == "2trans":
        if max_xy_delta > 0:
            action[:2] = np.clip(action[:2], -max_xy_delta, max_xy_delta)
        # 2trans should only use planar translation deltas.
        action[2:6] = 0.0
    return action


def project_action_to_env_mode(action_7d: np.ndarray, action_mode: str) -> np.ndarray:
    if action_mode == "2trans":
        return action_7d[:2]
    if action_mode == "3trans":
        return np.array([action_7d[0], action_7d[1], action_7d[2], action_7d[6]],
                        dtype=np.float64)
    if action_mode == "3trans1rot":
        return np.array([action_7d[0], action_7d[1], action_7d[2],
                         action_7d[5], action_7d[6]], dtype=np.float64)
    if action_mode == "3trans3rot":
        return action_7d
    raise ValueError(f"Unsupported action_mode: {action_mode}")


def main() -> int:
    args = parse_args()
    seed_dir = args.seed_dir.resolve()

    # --- checkpoint metadata -------------------------------------------------
    env_cfg = load_run_config(seed_dir)
    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu",
                            weights_only=False)
    act_min = np.asarray(norm_stats["act_min"], np.float32)
    act_max = np.asarray(norm_stats["act_max"], np.float32)
    norm_range = tuple(norm_stats.get("action_norm_range", (-1.0, 1.0)))
    cp_selection = args.cp_selection or str(norm_stats.get("cp_selection", "argmax"))
    cp_temp = (args.cp_temperature if args.cp_temperature is not None
               else float(norm_stats.get("cp_selection_temperature", 1.0)))

    frame_stack = int(env_cfg.get("frame_stack", 2))
    cams = tuple(env_cfg.get("camera_streams", ["images1"]))
    image_h = int(env_cfg.get("image_height", 240))
    image_w = int(env_cfg.get("image_width", 320))
    in_channels = 3 * len(cams) * frame_stack

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    cp_gen, q_net = build_models(env_cfg, in_channels, device)
    suffix = "" if args.no_ema else "_ema"
    load_weights(cp_gen, seed_dir / f"control_point_generator{suffix}.pt", device)
    load_weights(q_net, seed_dir / f"q_estimator{suffix}.pt", device)
    print(f"Loaded weights ({'raw' if args.no_ema else 'EMA'}) from {seed_dir}")
    print(f"  frame_stack={frame_stack} cameras={cams} model_hw=({image_h},{image_w}) "
          f"in_channels={in_channels}")
    print(f"  cp_selection={cp_selection} (temp={cp_temp})  device={device}")
    print(f"  act_min={act_min} act_max={act_max} norm_range={norm_range}")

    # --- connect -------------------------------------------------------------
    WidowXClient, WidowXConfigs, WidowXStatus = load_widowx_dependencies(
        args.widowx_envs_path)
    print(f"WidowX SDK: {WidowXClient.__module__} "
          f"({getattr(sys.modules.get(WidowXClient.__module__), '__file__', '?')})")

    env_params = build_env_params(args, WidowXConfigs)
    print(f"Camera topics: {args.camera_topics}")
    print(f"action_mode={args.action_mode} lock_z={args.lock_z} "
          f"fixed_z_height={args.fixed_z_height} move_duration={args.step_duration}")

    client = WidowXClient(host=args.ip, port=args.port)
    init_status = init_widowx_with_retry(
        client, env_params, args.im_size, WidowXStatus, args)
    if init_status != WidowXStatus.SUCCESS:
        raise RuntimeError(
            f"WidowX init failed after {args.init_retries} attempts with "
            f"status={status_name(init_status, WidowXStatus)}. "
            f"Check server reachability at {args.ip}:{args.port}, and that "
            f"--widowx-envs-path ({args.widowx_envs_path}) matches the server's.")
    print("WidowX connection established.")

    # Collect-style reset: this is what puts the arm at the data-collection
    # start pose. No explicit absolute move (the working reference calls that
    # path "recommended false").
    reset_status = reset_widowx_with_retry(client, WidowXStatus, args, args.i_traj)
    if reset_status != WidowXStatus.SUCCESS:
        raise RuntimeError(
            f"WidowX reset failed with "
            f"status={status_name(reset_status, WidowXStatus)}")
    print(f"Reset done (itraj={args.i_traj}).")

    # --- warm up the frame buffer -------------------------------------------
    frame_buf = collections.deque(maxlen=frame_stack)

    def grab_frame(retries: int = 50) -> np.ndarray:
        for _ in range(retries):
            obs = client.get_observation()
            if obs is not None:
                try:
                    return extract_blue_frame(obs)
                except RuntimeError:
                    pass
            time.sleep(0.2)
        raise RuntimeError("no observation from server after retries (server down?)")

    def grab_obs(retries: int = 50):
        for _ in range(retries):
            obs = client.get_observation()
            if obs is not None:
                return obs
            time.sleep(0.2)
        raise RuntimeError("no observation from server after retries")

    first_obs = grab_obs()
    first = extract_blue_frame(first_obs)
    print(f"Blue frame: raw {first.shape}")
    for _ in range(frame_stack):
        frame_buf.append(preprocess(first, (image_h, image_w)))

    # --- dry run -------------------------------------------------------------
    if args.dry_run:
        import cv2
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"DRY RUN: dumping {args.dry_run_steps} frames to {args.dump_dir} "
              f"(no step_action). Confirm the T renders RED.")
        for i in range(args.dry_run_steps):
            raw = grab_frame()
            np.save(args.dump_dir / f"raw_{i:03d}.npy", np.ascontiguousarray(raw))
            frame_buf.append(preprocess(raw, (image_h, image_w)))
            obs_u8 = stack_to_tensor(frame_buf, device)
            na = select_action(cp_gen, q_net, obs_u8, cp_selection, cp_temp)
            act = unnormalize(na, act_min, act_max, norm_range)
            cv2.imwrite(str(args.dump_dir / f"fed_{i:03d}.png"),
                        cv2.cvtColor(list(frame_buf)[-1], cv2.COLOR_RGB2BGR))
            print(f"[{i:03d}] norm={np.round(na, 3)} -> action(dx,dy)={np.round(act, 4)}")
            time.sleep(args.step_duration)
        client.stop()
        print("Dry run done. Inspect deploy_dryrun/fed_000.png before live control.")
        return 0

    # --- forensic logging ----------------------------------------------------
    log_fh = None
    if args.log_dir is not None:
        import cv2
        (args.log_dir / "raw").mkdir(parents=True, exist_ok=True)
        (args.log_dir / "fed").mkdir(parents=True, exist_ok=True)
        log_fh = (args.log_dir / "steps.jsonl").open("w")
        print(f"Forensic log -> {args.log_dir}")

    blocking = not args.non_blocking
    print(f"Closed-loop control up to {args.steps} steps, blocking={blocking}, "
          f"step_duration={args.step_duration}s. Keep a hand on the E-stop.")
    input("Press [Enter] to start.")

    step = 0
    last_exec = time.time()
    try:
        for step in range(args.steps):
            raw_obs = grab_obs()
            raw = extract_blue_frame(raw_obs)
            frame_buf.append(preprocess(raw, (image_h, image_w)))
            obs_u8 = stack_to_tensor(frame_buf, device)

            na = select_action(cp_gen, q_net, obs_u8, cp_selection, cp_temp)
            act_xy = unnormalize(na, act_min, act_max, norm_range)

            action_7d = to_action_7d(act_xy, args.fixed_gripper)
            action_7d = safety_clip_action(action_7d, args.action_mode,
                                           args.safety_max_xy_delta)
            env_action = project_action_to_env_mode(action_7d, args.action_mode)

            if not blocking:
                wait_s = (last_exec + args.step_duration) - time.time()
                if wait_s > 0:
                    time.sleep(wait_s)

            step_status = client.step_action(env_action, blocking=blocking)
            last_exec = time.time()
            if step_status != WidowXStatus.SUCCESS:
                raise RuntimeError(
                    "WidowX step_action failed: status="
                    f"{status_name(step_status, WidowXStatus)}, "
                    f"env_action={np.asarray(env_action).tolist()}")

            print(f"[{step:03d}] norm={np.round(na, 3)} -> "
                  f"env_action={np.round(env_action, 5)}")

            if log_fh is not None:
                import cv2
                np.save(args.log_dir / "raw" / f"{step:04d}.npy",
                        np.ascontiguousarray(raw))
                cv2.imwrite(str(args.log_dir / "fed" / f"{step:04d}.png"),
                            cv2.cvtColor(list(frame_buf)[-1], cv2.COLOR_RGB2BGR))
                st = raw_obs.get("state")
                log_fh.write(json.dumps({
                    "step": step,
                    "t": time.time(),
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
        try:
            client.stop()
        except Exception:
            pass
        print(f"Stopped after {step + 1} steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
