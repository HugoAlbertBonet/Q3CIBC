#!/usr/bin/env python3

"""
Real-robot WidowX eval for Q3CIBC (combinedv2 CP-cloud) Push-T checkpoints.

This is a port of the condBFNPol reference eval
(https://github.com/Abha2001/condBFNPol/blob/main/scripts/eval/eval_widowx.py)
with the policy layer swapped for our q3c checkpoints. The robot-facing half
(env params, workspace bounds, init-with-retry, neutral move, action adaptation,
safety clipping, sticky gripper, control loop, video save) is kept as close to
the reference as possible so that any remaining difference in behaviour is
attributable to the policy, not the harness.

Differences from the reference, and why (see DEVIATIONS at the bottom too):
  * Policy loading/prediction: q3c CP-cloud (control_point_generator +
    q_estimator .pt files, config.json, norm_stats.pt) instead of a hydra
    diffusion/BFN .ckpt. There is no shape_meta; obs geometry comes from
    config.json.
  * Model input: q3c takes **uint8 [0,255], channel-CONCATENATED**
    (3*frame_stack, H, W) frames, not float[0,1] stacked on a new axis. This
    matches PushTRealPixelsDataset and is not optional.
  * Actions are min-max denormalized with norm_stats (act_min/act_max) before
    being sent; the reference's policies emit raw actions.
  * initial_eep defaults to the measured demo start, NOT the reference's
    [0.3, 0.0, 0.15]. Our policy emits only (dx,dy) with dz identically zero in
    all 49463 demo transitions, so it can never descend; starting 13cm above the
    table would make the task impossible. Pass --initial_eep 0.3 0.0 0.15 to use
    the reference value.
  * Added: stale-frame guard and forensic logging (both flagged, see below).

Example:
    python scripts/deploy_pusht_real_v2.py \
      --seed_dir checkpoints/pusht_real_combinedv2/seed_0011 \
      --ip localhost --port 5556 \
      --camera_topics /blue/image_raw \
      --device cpu \
      --blocking \
      --act_exec_horizon 1 \
      --robot_action_dim 7 \
      --sticky_gripper_num_steps 0 \
      --max_delta_translation 0.01 \
      --max_delta_rotation 0.05 \
      --num_timesteps 200 \
      --print_action_debug --log_dir results/run_v2
"""

import os
import sys
import json
import time
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None
try:
    from PIL import Image
except ImportError:
    Image = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in [
    PROJECT_ROOT,
    PROJECT_ROOT.parent / "bridge_data_robot" / "widowx_envs",
    PROJECT_ROOT.parent / "bridge_data_robot" / "widowx_envs" / "multicam_server" / "src",
    Path.home() / "bridge_data_robot" / "widowx_envs",
    Path.home() / "bridge_data_robot" / "widowx_envs" / "multicam_server" / "src",
]:
    sp = str(p)
    if p.is_dir() and sp not in sys.path:
        sys.path.insert(0, sp)

# --- reference constants (verbatim) ----------------------------------------
STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 0
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]

# --- q3c-specific -----------------------------------------------------------
# Mean EEF start pose over all 110 demos (std ~0). The reference default of
# [0.3, 0.0, 0.15] is generic bridge boilerplate and sits above the table.
DEMO_INITIAL_EEP = [0.117, -0.019, 0.02]


def _build_env_params(camera_topics: List[str]) -> Dict:
    topics = [{"name": t} for t in camera_topics]
    return {
        "camera_topics": topics,
        "override_workspace_boundaries": WORKSPACE_BOUNDS,
        "move_duration": STEP_DURATION,
    }


def _stdin_has_data() -> bool:
    try:
        import select

        return select.select([sys.stdin], [], [], 0)[0] != []
    except Exception:
        return False


def _prep_device(arg: str) -> torch.device:
    if arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resize_hwc_float01(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if img.shape[:2] == (target_h, target_w):
        return img

    if cv2 is not None:
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if Image is not None:
        pil = Image.fromarray(np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8))
        pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)
        return np.asarray(pil).astype(np.float32) / 255.0

    src_h, src_w = img.shape[:2]
    y_idx = np.linspace(0, src_h - 1, target_h).astype(np.int32)
    x_idx = np.linspace(0, src_w - 1, target_w).astype(np.int32)
    return img[y_idx][:, x_idx]


def _rgb_to_bgr_u8(rgb_u8: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    return rgb_u8[..., ::-1].copy()


def _as_hwc_float01(image: np.ndarray, im_size: int) -> np.ndarray:
    arr = np.asarray(image)

    if arr.ndim == 1:
        expected = 3 * im_size * im_size
        if arr.size != expected:
            raise ValueError(
                f"Cannot reshape flat image of size {arr.size} into 3x{im_size}x{im_size}"
            )
        arr = arr.reshape(3, im_size, im_size).transpose(1, 2, 0)
    elif arr.ndim == 3:
        # CHW -> HWC
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = arr.transpose(1, 2, 0)
        elif arr.shape[-1] not in (1, 3):
            raise ValueError(f"Unsupported image shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image rank: {arr.ndim}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def _select_rgb_source(
    obs: Dict,
    im_size: int,
    target_hw: Tuple[int, int],
    prefer_full_image: bool,
    preferred_keys: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    target_h, target_w = target_hw

    if prefer_full_image:
        key_order = ["full_image", "external_img", "over_shoulder_img", "wrist_img", "image"]
    else:
        key_order = ["image", "full_image", "external_img", "over_shoulder_img", "wrist_img"]

    if preferred_keys:
        ordered = []
        seen = set()
        for key in list(preferred_keys) + key_order:
            if key not in seen:
                ordered.append(key)
                seen.add(key)
        key_order = ordered

    candidates = []
    for key in key_order:
        if obs.get(key) is None:
            continue
        try:
            img = _as_hwc_float01(obs[key], im_size)
            candidates.append((key, img))
        except Exception:
            continue

    if not candidates:
        obs_keys = sorted(list(obs.keys()))
        raise RuntimeError(
            f"Observation has no valid RGB image source. "
            f"Tried keys={key_order}, available_keys={obs_keys}"
        )

    best_img = None
    best_cost = None
    for key, img in candidates:
        h, w = img.shape[:2]
        cost = abs(h - target_h) + abs(w - target_w)
        if best_cost is None or cost < best_cost:
            best_img = img
            best_cost = cost

    src = best_img

    if src.shape[:2] != (target_h, target_w):
        src = _resize_hwc_float01(src, (target_h, target_w))

    chw = np.transpose(src, (2, 0, 1)).astype(np.float32)
    u8 = np.clip(src * 255.0, 0.0, 255.0).astype(np.uint8)
    return chw, u8


def _extract_state(obs: Dict) -> np.ndarray:
    state = obs.get("state", None)
    if state is None:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(state, dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------
# q3c policy layer (replaces the reference's hydra/diffusion checkpoint code)
# ---------------------------------------------------------------------------

def _load_v1_module():
    """Reuse the proven model build / weight load / CP-cloud selection code."""
    spec = importlib.util.spec_from_file_location(
        "deploy_v1", PROJECT_ROOT / "scripts" / "deploy_pusht_real.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["deploy_v1"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_policy_q3c(seed_dir: Path, device: torch.device, no_ema: bool) -> Dict:
    v1 = _load_v1_module()

    seed_dir = Path(seed_dir).expanduser().resolve()
    with (seed_dir / "config.json").open() as fh:
        config = json.load(fh)
    env_cfg = config["environments"][config["active_env"]]

    frame_stack = int(env_cfg.get("frame_stack", 2))
    cams = tuple(env_cfg.get("camera_streams", ["images1"]))
    hw = (int(env_cfg.get("image_height", 240)), int(env_cfg.get("image_width", 320)))
    in_channels = 3 * len(cams) * frame_stack

    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu",
                            weights_only=False)

    cp_gen, q_net = v1.build_models(env_cfg, in_channels, device)
    suffix = "" if no_ema else "_ema"
    v1.load_weights(cp_gen, seed_dir / f"control_point_generator{suffix}.pt", device)
    v1.load_weights(q_net, seed_dir / f"q_estimator{suffix}.pt", device)

    return {
        "v1": v1,
        "cp_gen": cp_gen,
        "q_net": q_net,
        "n_obs_steps": frame_stack,
        "cameras": cams,
        "hw": hw,
        "in_channels": in_channels,
        "act_min": np.asarray(norm_stats["act_min"], np.float32),
        "act_max": np.asarray(norm_stats["act_max"], np.float32),
        "norm_range": tuple(norm_stats.get("action_norm_range", (-1.0, 1.0))),
        "cp_selection": str(norm_stats.get("cp_selection", "argmax")),
        "cp_temperature": float(norm_stats.get("cp_selection_temperature", 1.0)),
        "action_dim": 2,
        "loaded_from": str(seed_dir),
    }


def _build_obs_frame_q3c(
    raw_obs: Dict,
    target_hw: Tuple[int, int],
    im_size: int,
    prefer_full_image: bool,
    preferred_keys: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """One frame -> (H,W,3) uint8 for the model, plus a display copy.

    q3c consumes uint8 [0,255]; PushTRealPixelsDataset resizes with AREA and
    rounds back to uint8, which is what _select_rgb_source's float path does
    before we scale back up.
    """
    _, rgb_u8 = _select_rgb_source(
        obs=raw_obs,
        im_size=im_size,
        target_hw=target_hw,
        prefer_full_image=prefer_full_image,
        preferred_keys=preferred_keys,
    )
    return rgb_u8, rgb_u8


def _stack_obs_q3c(frames: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """oldest->newest list of (H,W,3) uint8 -> (1, 3*fs, H, W) uint8 tensor.

    Channel concatenation, matching PushTRealPixelsDataset.__getitem__ (which
    stacks oldest->newest on the channel axis then transposes to CHW). This is
    NOT the reference's new-axis stacking.
    """
    stacked = np.concatenate(frames, axis=-1)          # (H, W, 3*fs)
    stacked = np.transpose(stacked, (2, 0, 1))          # (3*fs, H, W)
    return torch.from_numpy(np.ascontiguousarray(stacked)).unsqueeze(0).to(device)


@torch.no_grad()
def _predict_action_sequence_q3c(policy: Dict, obs_u8: torch.Tensor) -> np.ndarray:
    """CP-cloud selection -> denormalized (dx,dy) metres, shaped [1, 2]."""
    v1 = policy["v1"]
    na = v1.select_action(
        policy["cp_gen"], policy["q_net"], obs_u8,
        policy["cp_selection"], policy["cp_temperature"],
    )
    act = v1.unnormalize(na, policy["act_min"], policy["act_max"], policy["norm_range"])
    return np.asarray(act, np.float32).reshape(1, -1), np.asarray(na, np.float32)


# ---------------------------------------------------------------------------
# reference action post-processing (verbatim)
# ---------------------------------------------------------------------------

def _adapt_action_dim(action: np.ndarray, target_dim: int, gripper_value: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)

    if action.size == target_dim:
        return action
    if action.size > target_dim:
        return action[:target_dim].copy()

    out = np.zeros((target_dim,), dtype=np.float32)
    out[: action.size] = action

    # WidowX commonly expects gripper at index 6. The reference hardcodes 1.0;
    # our demos record 0.0 for every transition, so this is configurable.
    if target_dim >= 7 and action.size < 7:
        out[6] = gripper_value

    return out


def _postprocess_action(action: np.ndarray, no_pitch_roll: bool, no_yaw: bool) -> np.ndarray:
    if no_pitch_roll and action.size > 4:
        action[3] = 0.0
        action[4] = 0.0
    if no_yaw and action.size > 5:
        action[5] = 0.0
    return action


def _clip_delta_action(
    action: np.ndarray,
    max_delta_translation: float,
    max_delta_rotation: float,
) -> np.ndarray:
    """Additional safety clipping before sending delta commands to robot."""
    action = action.copy()
    if max_delta_translation > 0 and action.size >= 3:
        action[:3] = np.clip(action[:3], -max_delta_translation, max_delta_translation)

    if max_delta_rotation > 0:
        if action.size >= 6:
            action[3:6] = np.clip(action[3:6], -max_delta_rotation, max_delta_rotation)
        elif action.size == 5:
            action[3] = np.clip(action[3], -max_delta_rotation, max_delta_rotation)

    # Gripper is absolute in WidowX SDK.
    if action.size in (4, 5):
        action[-1] = np.clip(action[-1], 0.0, 1.0)
    elif action.size >= 7:
        action[6] = np.clip(action[6], 0.0, 1.0)

    return action


def _get_display_bgr(raw_obs: Dict, fallback_rgb_u8, im_size: int):
    try:
        _, rgb_u8 = _select_rgb_source(
            obs=raw_obs, im_size=im_size, target_hw=(im_size, im_size),
            prefer_full_image=True,
        )
        return _rgb_to_bgr_u8(rgb_u8)
    except Exception:
        pass
    if fallback_rgb_u8 is not None:
        try:
            return _rgb_to_bgr_u8(fallback_rgb_u8)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# reference WidowX plumbing (verbatim)
# ---------------------------------------------------------------------------

def _load_widowx_sdk():
    try:
        from experiments.widowx_envs.widowx_env_service import (
            WidowXClient, WidowXConfigs, WidowXStatus,
        )
        return WidowXClient, WidowXConfigs, WidowXStatus
    except ImportError:
        pass

    try:
        from widowx_envs.widowx_env_service import (
            WidowXClient, WidowXConfigs, WidowXStatus,
        )
        return WidowXClient, WidowXConfigs, WidowXStatus
    except ImportError as e:
        raise ModuleNotFoundError(
            "Missing WidowX SDK. Tried "
            "'experiments.widowx_envs.widowx_env_service' and "
            "'widowx_envs.widowx_env_service'. "
            "Install widowx_envs or set PYTHONPATH to bridge_data_robot/widowx_envs."
        ) from e


def _status_ok(status, WidowXStatus) -> bool:
    success = getattr(WidowXStatus, "SUCCESS", None)
    if success is None:
        return True
    return status == success


def _status_name(status, WidowXStatus) -> str:
    for name in ("SUCCESS", "NO_CONNECTION", "EXECUTION_FAILURE", "NOT_INITIALIZED"):
        if hasattr(WidowXStatus, name) and status == getattr(WidowXStatus, name):
            return name
    return str(status)


def _normalize_action_client_config(widowx_client) -> None:
    """Work around edgeml config mutation (list -> set) that breaks JSON serialization."""
    try:
        action_client = getattr(widowx_client, "_WidowXClient__client", None)
        if action_client is None:
            return
        cfg = getattr(action_client, "config", None)
        if cfg is None:
            return
        for key in ("observation_keys", "action_keys"):
            value = getattr(cfg, key, None)
            if isinstance(value, set):
                setattr(cfg, key, sorted(value))
    except Exception:
        pass


def _set_reqrep_timeout_ms(widowx_client, timeout_ms: int) -> None:
    try:
        action_client = getattr(widowx_client, "_WidowXClient__client", None)
        if action_client is None:
            return
        reqrep_client = getattr(action_client, "client", None)
        if reqrep_client is None:
            return
        reqrep_client.timeout_ms = int(timeout_ms)
        reqrep_client.reset_socket()
    except Exception:
        pass


def _wait_for_widowx_observation(widowx_client, timeout_s: float = 20.0,
                                 poll_s: float = 0.5) -> bool:
    end_t = time.time() + timeout_s
    while time.time() < end_t:
        try:
            obs = widowx_client.get_observation()
        except Exception:
            obs = None
        if obs is not None:
            return True
        time.sleep(poll_s)
    return False


def _init_widowx_with_retry(
    WidowXClient, WidowXConfigs, WidowXStatus,
    host: str, port: int, env_params: Dict, image_size: int,
    timeout_s: float = 180.0, retry_interval_s: float = 2.0,
    normalize_client_config: bool = False,
):
    """
    normalize_client_config: the reference rewrites the edgeml action config's
    observation_keys/action_keys from set -> sorted list to keep them JSON
    serializable. On this rig that CHANGES THE CLIENT CONFIG HASH and the server
    rejects the handshake with "Incompatible config with hash with server", so
    it is off by default (deploy_pusht_real.py never touched the config either).
    """
    deadline = time.time() + timeout_s
    attempt = 0
    last_status = None
    last_error = None
    client = None

    while time.time() < deadline:
        attempt += 1
        try:
            if client is None:
                if normalize_client_config:
                    cfg = getattr(WidowXConfigs, "DefaultActionConfig", None)
                    if cfg is not None:
                        for key in ("observation_keys", "action_keys"):
                            value = getattr(cfg, key, None)
                            if isinstance(value, set):
                                setattr(cfg, key, sorted(value))
                client = WidowXClient(host=host, port=port)
                if normalize_client_config:
                    _normalize_action_client_config(client)
                _set_reqrep_timeout_ms(client, timeout_ms=120_000)

            status = client.init(env_params, image_size=image_size)
            last_status = status
            if _status_ok(status, WidowXStatus):
                _set_reqrep_timeout_ms(client, timeout_ms=2_000)
                return client

            print(f"[WARN] WidowX init attempt {attempt} returned "
                  f"{_status_name(status, WidowXStatus)}; retrying...")
            _set_reqrep_timeout_ms(client, timeout_ms=2_000)
            if _wait_for_widowx_observation(client, timeout_s=10.0, poll_s=0.5):
                print("[INFO] WidowX observation stream is ready after init timeout.")
                return client
            _set_reqrep_timeout_ms(client, timeout_ms=120_000)
        except Exception as e:
            last_error = e
            print(f"[WARN] WidowX init attempt {attempt} failed: {e}; retrying...")
            client = None

        time.sleep(retry_interval_s)

    if last_error is not None:
        raise RuntimeError(f"WidowX init failed after {attempt} attempts: {last_error}")
    raise RuntimeError(
        f"WidowX init failed after {attempt} attempts "
        f"(last status={_status_name(last_status, WidowXStatus)})"
    )


def _go_to_neutral(widowx_client, initial_eep: List[float]) -> None:
    if hasattr(widowx_client, "go_to_neutral"):
        widowx_client.go_to_neutral()
        return
    if hasattr(widowx_client, "reset"):
        widowx_client.reset()
        return
    if hasattr(widowx_client, "move"):
        pose = np.asarray(
            [initial_eep[0], initial_eep[1], initial_eep[2], 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        widowx_client.move(pose, duration=1.5, blocking=True)
        return
    raise AttributeError("WidowX client has no go_to_neutral/reset/move method")


def main():
    parser = argparse.ArgumentParser(
        description="Real-robot WidowX eval for Q3CIBC Push-T checkpoints "
                    "(port of the condBFNPol eval_widowx.py harness)"
    )
    # --- q3c checkpoint selection (replaces --ckpt_path/--save_dir/--runs_root)
    parser.add_argument("--seed_dir", type=str, required=True,
                        help="q3c seed directory containing control_point_generator*.pt, "
                             "q_estimator*.pt, config.json, norm_stats.pt")
    parser.add_argument("--no_ema", action="store_true",
                        help="Load raw weights instead of the EMA copy")

    # --- reference args (same names/defaults unless noted) -------------------
    parser.add_argument("--im_size", type=int, default=256,
                        help="image_size passed to WidowX init. Reference default is 96; "
                             "256 is what our rig has been driven with.")
    parser.add_argument("--video_save_path", type=str, default=None)
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument("--blocking", action="store_true")
    parser.add_argument("--initial_eep", type=float, nargs=3, default=DEMO_INITIAL_EEP,
                        help="EEF start pose. Default is the measured demo start "
                             f"{DEMO_INITIAL_EEP}; the reference default is 0.3 0.0 0.15, "
                             "which is 13cm above the table and unreachable for a "
                             "policy whose dz is identically zero.")
    parser.add_argument("--act_exec_horizon", type=int, default=1,
                        help="q3c predicts a single action per observation "
                             "(dataset action_chunk=1), so >1 replays the same action.")
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--camera_topics", type=str, nargs="+",
                        default=[x["name"] for x in CAMERA_TOPICS])
    parser.add_argument("--show_image", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--robot_action_dim", type=int, default=7,
                        help="Action dimension sent to WidowX")
    parser.add_argument("--gripper_value", type=float, default=0.0,
                        help="Gripper value when padding to 7 dims. Reference hardcodes "
                             "1.0; all 49463 demo transitions record 0.0.")
    parser.add_argument("--action_noise_std", type=float, default=0.0)
    parser.add_argument("--sticky_gripper_num_steps", type=int,
                        default=STICKY_GRIPPER_NUM_STEPS)
    parser.add_argument("--max_delta_translation", type=float, default=0.01,
                        help="Safety clip for |dx,dy,dz| (m). Reference default is 0.03 "
                             "but its own Push-T invocation uses 0.01; our actions never "
                             "exceed 0.008.")
    parser.add_argument("--max_delta_rotation", type=float, default=0.05)
    parser.add_argument("--print_action_debug", action="store_true")
    parser.add_argument("--no_pitch_roll", action="store_true", default=NO_PITCH_ROLL)
    parser.add_argument("--no_yaw", action="store_true", default=NO_YAW)
    parser.add_argument("--prefer_full_image", dest="prefer_full_image",
                        action="store_true")
    parser.add_argument("--no_prefer_full_image", dest="prefer_full_image",
                        action="store_false")
    parser.set_defaults(prefer_full_image=True)

    # --- q3c overrides -------------------------------------------------------
    parser.add_argument("--cp_selection", choices=["argmax", "sample"], default=None,
                        help="Override CP-cloud selection (default: from norm_stats)")
    parser.add_argument("--cp_temperature", type=float, default=None)

    # --- DEVIATIONS from the reference (both default ON / opt-out) -----------
    parser.add_argument("--no_require_fresh", action="store_true",
                        help="DEVIATION: disable the stale-frame guard. The server "
                             "sometimes repeats images; a duplicate in the frame stack "
                             "means zero inter-frame motion, which measurably degrades "
                             "this policy (offline MAE 0.02 -> 0.22).")
    parser.add_argument("--fresh_timeout", type=float, default=0.5)
    parser.add_argument("--normalize_client_config", action="store_true",
                        help="Apply the reference's edgeml set->list config rewrite. "
                             "OFF by default: on this rig it changes the client "
                             "config hash and the server rejects init with "
                             "'Incompatible config with hash with server'.")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="DEVIATION: forensic log dir (raw/*.npy, fed/*.png, "
                             "steps.jsonl with EEF proprio state)")

    args = parser.parse_args()

    WidowXClient, WidowXConfigs, WidowXStatus = _load_widowx_sdk()

    if args.show_image and cv2 is None:
        print("[WARN] OpenCV is not installed; disabling --show_image")
        args.show_image = False

    device = _prep_device(args.device)

    policy = _load_policy_q3c(Path(args.seed_dir), device, args.no_ema)
    if args.cp_selection is not None:
        policy["cp_selection"] = args.cp_selection
    if args.cp_temperature is not None:
        policy["cp_temperature"] = args.cp_temperature

    n_obs_steps = policy["n_obs_steps"]
    target_hw = policy["hw"]
    print(f"Loaded q3c policy from {policy['loaded_from']} "
          f"({'raw' if args.no_ema else 'EMA'} weights)")
    print(f"  n_obs_steps={n_obs_steps} cameras={policy['cameras']} "
          f"model_hw={target_hw} in_channels={policy['in_channels']}")
    print(f"  cp_selection={policy['cp_selection']} (temp={policy['cp_temperature']})  "
          f"device={device}")
    print(f"  act_min={policy['act_min']} act_max={policy['act_max']} "
          f"norm_range={policy['norm_range']}")

    print("Initializing WidowX connection...")
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(_build_env_params(args.camera_topics))
    env_params["start_state"] = list(np.concatenate([args.initial_eep, [0, 0, 0, 1]]))
    print(f"WidowX camera topics: {args.camera_topics}")
    print(f"Initial EEP: {args.initial_eep}")
    widowx_client = _init_widowx_with_retry(
        WidowXClient=WidowXClient,
        WidowXConfigs=WidowXConfigs,
        WidowXStatus=WidowXStatus,
        host=args.ip,
        port=args.port,
        env_params=env_params,
        image_size=args.im_size,
        normalize_client_config=args.normalize_client_config,
    )
    if args.normalize_client_config:
        _normalize_action_client_config(widowx_client)
    print("WidowX connection established.")

    _go_to_neutral(widowx_client, args.initial_eep)
    time.sleep(0.5)

    # forensic logging (deviation)
    log_fh = None
    log_dir = None
    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
        (log_dir / "raw").mkdir(parents=True, exist_ok=True)
        (log_dir / "fed").mkdir(parents=True, exist_ok=True)
        log_fh = (log_dir / "steps.jsonl").open("w")
        print(f"Forensic log -> {log_dir}")

    require_fresh = not args.no_require_fresh
    last_raw = {"v": None}

    def _grab_obs():
        """get_observation with the stale-frame guard (deviation)."""
        raw_obs = widowx_client.get_observation()
        if raw_obs is None:
            return None
        if not require_fresh or last_raw["v"] is None:
            return raw_obs
        t0 = time.time()
        while (time.time() - t0) < args.fresh_timeout:
            probe = raw_obs.get("full_image", raw_obs.get("external_img"))
            if probe is None or not np.array_equal(probe, last_raw["v"]):
                return raw_obs
            time.sleep(0.05)
            nxt = widowx_client.get_observation()
            if nxt is not None:
                raw_obs = nxt
        print(f"[warn] stale frame: server repeated image "
              f"(no fresh frame within {args.fresh_timeout}s)")
        return raw_obs

    last_tstep = time.time()
    t = 0
    images = []
    obs_hist = deque(maxlen=n_obs_steps)
    is_gripper_closed = False
    consecutive_gripper_change = 0

    if args.show_image:
        obs = widowx_client.get_observation()
        while obs is None:
            print("Waiting for observations...")
            time.sleep(1)
            obs = widowx_client.get_observation()
        bgr = _get_display_bgr(obs, None, args.im_size)
        if bgr is not None:
            cv2.imshow("img_view", bgr)
            cv2.waitKey(100)

    input("Press [Enter] to start evaluation.")

    try:
        while t < args.num_timesteps:
            if time.time() <= last_tstep + STEP_DURATION and not args.blocking:
                continue

            raw_obs = _grab_obs()
            if raw_obs is None:
                print("WARNING: retrying get_observation...")
                continue

            try:
                frame_u8, frame_rgb_u8 = _build_obs_frame_q3c(
                    raw_obs=raw_obs,
                    target_hw=target_hw,
                    im_size=args.im_size,
                    prefer_full_image=args.prefer_full_image,
                    preferred_keys=None,
                )
            except RuntimeError as e:
                print(f"[WARN] {e}")
                time.sleep(0.2)
                continue

            last_raw["v"] = raw_obs.get("full_image", raw_obs.get("external_img"))

            if args.show_image:
                bgr_img = _get_display_bgr(raw_obs, frame_rgb_u8, args.im_size)
                if bgr_img is not None:
                    cv2.imshow("img_view", bgr_img)
                    key = cv2.waitKey(10) & 0xFF
                    if key in (ord("r"), ord("R")):
                        print("[INFO] Reset requested via 'R' key")
                        break
            else:
                if _stdin_has_data():
                    try:
                        line = sys.stdin.readline().strip()
                        if line.lower() == "r":
                            print("[INFO] Reset requested via stdin")
                            break
                    except Exception:
                        pass

            if len(obs_hist) == 0:
                obs_hist.extend([frame_u8] * obs_hist.maxlen)
            else:
                obs_hist.append(frame_u8)

            obs_u8 = _stack_obs_q3c(list(obs_hist), device)

            last_tstep = time.time()
            actions, na = _predict_action_sequence_q3c(policy, obs_u8)
            exec_horizon = min(max(1, args.act_exec_horizon), actions.shape[0])
            action_seq = actions[:exec_horizon].copy()

            for i in range(exec_horizon):
                action = action_seq[i].copy()

                if args.action_noise_std > 0:
                    action += np.random.normal(
                        loc=0.0, scale=args.action_noise_std, size=action.shape,
                    ).astype(np.float32)

                action = _adapt_action_dim(action, args.robot_action_dim,
                                           args.gripper_value)

                if args.sticky_gripper_num_steps > 0 and action.size > 6:
                    if (action[6] < 0.5) != is_gripper_closed:
                        consecutive_gripper_change += 1
                    else:
                        consecutive_gripper_change = 0
                    if consecutive_gripper_change >= args.sticky_gripper_num_steps:
                        is_gripper_closed = not is_gripper_closed
                        consecutive_gripper_change = 0
                    action[6] = 0.0 if is_gripper_closed else 1.0

                action = _postprocess_action(action, no_pitch_roll=args.no_pitch_roll,
                                             no_yaw=args.no_yaw)
                action = _clip_delta_action(
                    action,
                    max_delta_translation=args.max_delta_translation,
                    max_delta_rotation=args.max_delta_rotation,
                )

                if args.print_action_debug:
                    print(f"[action] step={t} exec_idx={i} norm={np.round(na, 3)} "
                          f"cmd={np.array2string(action, precision=5, suppress_small=True)}")

                status = widowx_client.step_action(action, blocking=args.blocking)
                if not _status_ok(status, WidowXStatus):
                    raise RuntimeError(
                        f"WidowX step_action failed with "
                        f"status={_status_name(status, WidowXStatus)}"
                    )

                if log_fh is not None:
                    rawimg = last_raw["v"]
                    if rawimg is not None:
                        np.save(log_dir / "raw" / f"{t:04d}.npy",
                                np.ascontiguousarray(rawimg))
                    if cv2 is not None:
                        cv2.imwrite(str(log_dir / "fed" / f"{t:04d}.png"),
                                    cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))
                    st = _extract_state(raw_obs)
                    log_fh.write(json.dumps({
                        "step": t,
                        "t": time.time(),
                        "norm": [float(x) for x in np.ravel(na)],
                        "action": [float(x) for x in np.ravel(action)],
                        "state": st.tolist() if st.size else None,
                    }) + "\n")
                    log_fh.flush()

                if frame_rgb_u8 is not None:
                    images.append(frame_rgb_u8)

                t += 1
                if t >= args.num_timesteps:
                    break

    except KeyboardInterrupt:
        print("[INFO] Ctrl+C received; moving robot to neutral...", file=sys.stderr)
        try:
            _go_to_neutral(widowx_client, args.initial_eep)
        except Exception as e:
            print(f"[WARN] Failed to move to neutral: {e}", file=sys.stderr)
    except Exception as e:
        print(str(e), file=sys.stderr)
    finally:
        if log_fh is not None:
            log_fh.close()

    if args.video_save_path is not None and images:
        try:
            import imageio
        except ImportError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'imageio'. Install it or disable --video_save_path."
            ) from e
        os.makedirs(args.video_save_path, exist_ok=True)
        curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(args.video_save_path, f"{curr_time}_widowx_eval.mp4")
        imageio.mimsave(save_path, images, fps=max(1.0, 1.0 / STEP_DURATION))
        print(f"Video saved to: {save_path}")

    print(f"Stopped after {t} steps.")


if __name__ == "__main__":
    main()
