#!/usr/bin/env python3
"""Evaluate IBC (EBM + DFO) Push-T checkpoints on the real WidowX arm.

Control flow, WidowX handling, rollout termination, video capture and latency
reporting are adapted from data/eval_widowx_bfn.py, which is the harness known
to work on this rig. What changed is the policy: instead of a hydra/dill
workspace checkpoint with an action-chunk sampler, this loads the per-run IBC
seed directory written by scripts/train_pusht_real_ibc.py and selects actions
with DFO.

Policy
------
    seed_dir/
        config.json      immutable per-run config (model + inference params)
        norm_stats.pt    act_min/act_max, frame_stack, camera streams
        q_estimator.pt   PixelEBM weights (ConvMaxpoolEncoder + DenseResnetValue)

Action selection follows google-research/ibc's optimal Pushing-Pixels policy
(configs/pushing_pixels/pixel_ebm_best.gin + mcmc.iterative_dfo defaults):
encode the frame stack once (late fusion), draw 2048 uniform action samples in
[-1-buf, 1+buf], then 3 rounds of score -> softmax resample -> shrinking
Gaussian jitter, and take the argmax of the final scores.

Differences from the BFN harness that are specific to this policy
-----------------------------------------------------------------
* Observation is a channel-stacked uint8 image, not a float obs dict. The
  stack is [oldest/cam0, ..., newest/camN] exactly as
  utils.datasets.PushTRealPixelsDataset builds it, resized with INTER_AREA and
  left in [0, 255] because the conv encoder normalizes internally.
* Single-step policy: there is no action chunk, so the executed horizon is
  always 1 and --act_exec_horizon does not apply.
* Duplicate-frame guard (--no-require-fresh to disable). The server sometimes
  repeats an image; a repeat inside a 2-frame stack means zero inter-frame
  motion, which is far out of distribution and previously produced a straight
  diagonal runaway. See PUSHT_DEPLOY_HANDOFF.md.
* The arm is moved to the pose all demos start from before each rollout.
  Skipping it leaves the arm ~17cm off the demo start, which is out of
  distribution and stalls the policy.

Usage (Alienware, WidowX server already running):

    # Always dry-run first: no motion, dumps the frames actually fed to the
    # model so you can confirm the T renders RED before enabling control.
    python scripts/deploy_pusht_real_ibc.py \
        --seed_dir checkpoints/pusht_real_ibc/seed_0029 --dry_run

    python scripts/deploy_pusht_real_ibc.py \
        --seed_dir checkpoints/pusht_real_ibc/seed_0029 \
        --action_mode 2trans \
        --step_duration 0.05 \
        --widowx_init_timeout_ms 180000 --widowx_init_retries 8 \
        --video_save_path /data/ibc_results
"""

from __future__ import annotations

import json
import os
import re
import select
import socket
import sys
import termios
import time
import traceback
import tty
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
for p in [
    ROOT,
    Path.home() / "bridge_data_robot" / "widowx_envs",
    Path.home() / "bridge_data_robot" / "widowx_envs" / "multicam_server" / "src",
    Path.home() / "bridge_data_robot_pusht" / "widowx_envs",
    Path.home() / "bridge_data_robot_pusht" / "widowx_envs" / "multicam_server" / "src",
]:
    sp = str(p)
    if p.is_dir() and sp not in sys.path:
        sys.path.insert(0, sp)

from absl import app, flags, logging
import cv2
import imageio
import numpy as np
import torch

from utils import ibc_policy


FLAGS = flags.FLAGS

# ── Policy ────────────────────────────────────────────────────────────────
flags.DEFINE_string(
    "seed_dir",
    None,
    "IBC seed directory containing config.json, norm_stats.pt, q_estimator.pt.",
    required=True,
)
flags.DEFINE_integer(
    "ckpt_step",
    -1,
    "Load q_estimator_step{N:06d}.pt instead of the final q_estimator.pt. "
    "Use for a seed whose training has not finished.",
)
flags.DEFINE_string("device", "cuda:0", "Torch device, e.g. cuda:0 or cpu")
flags.DEFINE_integer(
    "dfo_samples", -1, "Override DFO sample count (default: per-run config, 2048)"
)
flags.DEFINE_integer(
    "dfo_iterations", -1, "Override DFO iteration count (default: per-run config, 3)"
)
flags.DEFINE_integer("policy_seed", 0, "RNG seed for the stochastic DFO sampler")
flags.DEFINE_integer(
    "inference_warmup_steps",
    10,
    "Number of warmup inference calls to exclude from latency statistics.",
)

# ── Observation ───────────────────────────────────────────────────────────
flags.DEFINE_integer("im_size", 480, "WidowX service image height")
flags.DEFINE_integer("im_width", 640, "WidowX service image width (0 = same as im_size)")
flags.DEFINE_bool(
    "swap_rgb",
    False,
    "Swap channels before feeding the model. Default off: this rig's server "
    "frame already has red in channel 0, matching the tf-decoded training JPEGs.",
)
flags.DEFINE_bool(
    "require_fresh",
    True,
    "Reject a frame byte-identical to the previous one and re-fetch, so the "
    "frame stack never carries a stale zero-motion slot.",
)
flags.DEFINE_float(
    "fresh_timeout",
    0.5,
    "Max seconds to wait for a non-duplicate frame before using the stale one.",
)

# ── Rollout control ───────────────────────────────────────────────────────
flags.DEFINE_integer("num_rollouts", 1, "Number of rollouts; <=0 means infinite")
flags.DEFINE_float("step_duration", 0.05, "Control period in seconds")
flags.DEFINE_float("max_duration", 120.0, "Max duration for each rollout in seconds.")
flags.DEFINE_float(
    "robot_exec_hz",
    0.0,
    "Robot command rate after linear interpolation. 0 (default) means match the "
    "policy rate, i.e. one step_action per inference. When higher, each delta is "
    "divided by the number of substeps so the commanded displacement per policy "
    "step is unchanged -- these are per-step deltas, not target poses.",
)
flags.DEFINE_bool("blocking", True, "Use blocking controller")
flags.DEFINE_float(
    "settle",
    0.0,
    "Extra seconds to wait after each blocking step before capturing the next "
    "frame.",
)

# ── Start pose ────────────────────────────────────────────────────────────
flags.DEFINE_bool(
    "move_to_demo_start",
    True,
    "Move the EEF to the pose all training demos start from before each "
    "rollout. Disabling leaves the arm ~17cm out of distribution.",
)
flags.DEFINE_string(
    "start_eep_npy",
    str(ROOT / "scripts" / "assets" / "pusht_start_eep.npy"),
    "4x4 EEF start transform (mean of demo starts).",
)
flags.DEFINE_float("start_move_duration", 1.5, "Duration of the initial move.")
flags.DEFINE_integer("max_initial_move_retries", 5, "Retries for the initial move.")

# ── Termination ───────────────────────────────────────────────────────────
flags.DEFINE_spaceseplist(
    "term_pose_xy",
    [],
    "Optional fixed termination XY. Empty auto-aligns to the post-reset pose.",
)
flags.DEFINE_float(
    "term_dist_thresh", 0.03, "EEF XY distance to count as termination area (m)."
)
flags.DEFINE_float(
    "term_hold_sec", 0.4, "Dwell time in the termination area before stopping (s)."
)
flags.DEFINE_float(
    "timeout_recovery_lift_z", 0.1, "Post-timeout: lift the gripper to this z."
)
flags.DEFINE_float(
    "timeout_recovery_stage_duration", 1.5, "Seconds per post-timeout recovery stage."
)
flags.DEFINE_spaceseplist(
    "timeout_recovery_target_joints",
    [
        "-0.11965051",
        "-0.4954758",
        "1.29621375",
        "-0.02914564",
        "0.65194184",
        "-0.13038836",
    ],
    "Joint target used by the post-timeout recovery sequence.",
)

# ── WidowX connection ─────────────────────────────────────────────────────
flags.DEFINE_string("ip", "localhost", "Robot IP")
flags.DEFINE_integer("port", 5556, "Robot port")
flags.DEFINE_integer("widowx_init_retries", 5, "Init retries on non-success status.")
flags.DEFINE_integer("widowx_init_timeout_ms", 120000, "Init RPC timeout (ms).")
flags.DEFINE_integer("widowx_rpc_timeout_ms", 10000, "Post-init RPC timeout (ms).")
flags.DEFINE_float("widowx_init_retry_sleep", 1.0, "Seconds between init retries.")
flags.DEFINE_bool("widowx_force_fresh_init", False, "Force server-side env reinit.")
flags.DEFINE_bool(
    "widowx_reuse_existing_env",
    True,
    "Reuse a live server env instead of calling init(), avoiding the cached "
    "init->reset(itraj=None) path.",
)
flags.DEFINE_integer("widowx_reset_retries", 3, "Reset retries on non-success status.")
flags.DEFINE_integer("widowx_reset_timeout_ms", 30000, "Reset RPC timeout (ms).")
flags.DEFINE_float("widowx_reset_retry_sleep", 1.0, "Seconds between reset retries.")

# ── Env params ────────────────────────────────────────────────────────────
flags.DEFINE_enum(
    "action_mode",
    "2trans",
    ["2trans"],
    "WidowX action_mode. The IBC checkpoints are trained on 2-D planar deltas, "
    "so only 2trans is meaningful here.",
)
flags.DEFINE_bool("lock_z", True, "Enable Z locking in env params")
flags.DEFINE_float("fixed_z_height", 0.02, "Fixed z height used when lock_z is True")
flags.DEFINE_float("neutral_z_height", 0.02, "Neutral z height")
flags.DEFINE_float("fixed_gripper", 0.0, "Fixed gripper command for 2trans env mode")
flags.DEFINE_spaceseplist(
    "camera_topics",
    ["/blue/image_raw"],
    "ROS RGB image topics passed to WidowX env init. The trained camera is "
    "images1 == /blue/image_raw; the D435 is no longer on the rig.",
)
flags.DEFINE_bool("skip_move_to_neutral", False, "Pass through to WidowX env init.")
flags.DEFINE_float(
    "safety_max_xy_delta",
    0.02,
    "Safety clip for |dx|,|dy| in meters. Training deltas span +/-0.008, so "
    "0.02 is a loose backstop against a malformed command. <=0 disables.",
)

# ── Output ────────────────────────────────────────────────────────────────
flags.DEFINE_bool("show_image", False, "Show camera image")
flags.DEFINE_string("video_save_path", None, "Directory to save rollout videos")
flags.DEFINE_string(
    # Not "log_dir": absl.logging already owns that flag name.
    "forensic_log_dir",
    None,
    "If set, forensic-log every step: raw + fed frames and a steps.jsonl row.",
)
flags.DEFINE_bool(
    "dry_run",
    False,
    "No motion: dump the frames fed to the model and print predicted actions.",
)
flags.DEFINE_integer("dry_run_steps", 20, "Frames to capture in --dry_run.")
flags.DEFINE_string(
    "dump_dir", str(ROOT / "deploy_dryrun_ibc"), "Where --dry_run writes frames."
)

WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]


# ═══════════════════════════════════════════════════════════════════════════
# WidowX plumbing (adapted from data/eval_widowx_bfn.py)
# ═══════════════════════════════════════════════════════════════════════════

def _load_widowx_dependencies():
    try:
        from widowx_envs.widowx_env_service import (  # type: ignore
            WidowXClient,
            WidowXConfigs,
            WidowXStatus,
        )
    except Exception as exc:
        raise ImportError(
            "Failed to import widowx_envs. Install it into this env "
            "(pip install -e ~/bridge_data_robot/widowx_envs)."
        ) from exc
    return WidowXClient, WidowXStatus, WidowXConfigs


def _status_name(status: Any, WidowXStatus: Any) -> str:
    for name in ("SUCCESS", "NO_CONNECTION", "EXECUTION_FAILURE", "NOT_INITIALIZED"):
        if hasattr(WidowXStatus, name) and status == getattr(WidowXStatus, name):
            return name
    return str(status)


def _set_widowx_reqrep_timeout_ms(widowx_client: Any, timeout_ms: int) -> None:
    """Best-effort update of the req/rep timeout used by widowx_envs."""
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
        # Keep compatibility across SDK variants that hide these internals.
        pass


def _tcp_port_reachable(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=max(0.05, float(timeout_s))):
            return True
    except Exception:
        return False


def _widowx_server_has_live_env(
    widowx_client: Any, max_wait_sec: float = 1.0, poll_interval_sec: float = 0.1
) -> bool:
    """Best-effort probe for an already initialized server-side env."""
    deadline = time.monotonic() + max(0.1, float(max_wait_sec))
    poll_interval_sec = max(0.01, float(poll_interval_sec))
    while time.monotonic() < deadline:
        try:
            raw_obs = widowx_client.get_observation()
        except Exception:
            raw_obs = None
        if raw_obs is not None:
            state = raw_obs.get("state", None)
            if state is None:
                time.sleep(poll_interval_sec)
                continue
            if isinstance(state, dict):
                return len(state) > 0
            try:
                state_vec = np.asarray(state, dtype=np.float64).reshape(-1)
            except Exception:
                state_vec = np.array([], dtype=np.float64)
            if state_vec.size > 0:
                return True
        time.sleep(poll_interval_sec)
    return False


def _init_widowx_with_retry(
    widowx_client: Any, env_params: Dict[str, Any], image_size: int, WidowXStatus: Any
) -> Any:
    retries = max(1, int(FLAGS.widowx_init_retries))
    retry_sleep = max(0.0, float(FLAGS.widowx_init_retry_sleep))
    init_timeout_ms = max(1, int(FLAGS.widowx_init_timeout_ms))
    rpc_timeout_ms = max(1, int(FLAGS.widowx_rpc_timeout_ms))

    _set_widowx_reqrep_timeout_ms(widowx_client, init_timeout_ms)
    last_status = None

    for attempt in range(1, retries + 1):
        print(
            f"[INFO] WidowX init attempt {attempt}/{retries} "
            f"(timeout={init_timeout_ms} ms, server={FLAGS.ip}:{FLAGS.port})"
        )
        t0 = time.time()
        last_status = widowx_client.init(env_params, image_size=image_size)
        elapsed = time.time() - t0
        if last_status == WidowXStatus.SUCCESS:
            _set_widowx_reqrep_timeout_ms(widowx_client, rpc_timeout_ms)
            return last_status

        print(
            f"[WARN] WidowX init attempt {attempt}/{retries} failed with "
            f"status={_status_name(last_status, WidowXStatus)} after {elapsed:.2f}s."
        )
        if last_status == getattr(WidowXStatus, "NO_CONNECTION", None):
            print(
                "[HINT] No response from the WidowX action server. Make sure "
                "`widowx_env_service --server` is running and reachable at "
                f"{FLAGS.ip}:{FLAGS.port}."
            )
        if attempt < retries and retry_sleep > 0.0:
            time.sleep(retry_sleep)

    _set_widowx_reqrep_timeout_ms(widowx_client, rpc_timeout_ms)
    return last_status


def _reset_widowx_with_retry(
    widowx_client: Any, WidowXStatus: Any, i_traj: int | None = None
) -> Any:
    retries = max(1, int(FLAGS.widowx_reset_retries))
    retry_sleep = max(0.0, float(FLAGS.widowx_reset_retry_sleep))
    reset_timeout_ms = max(1, int(FLAGS.widowx_reset_timeout_ms))
    rpc_timeout_ms = max(1, int(FLAGS.widowx_rpc_timeout_ms))

    _set_widowx_reqrep_timeout_ms(widowx_client, max(reset_timeout_ms, rpc_timeout_ms))
    last_status = None
    warned_itraj_fallback = False
    for attempt in range(1, retries + 1):
        if i_traj is None:
            last_status = widowx_client.reset()
        else:
            try:
                last_status = widowx_client.reset(itraj=int(i_traj))
            except TypeError:
                if not warned_itraj_fallback:
                    print(
                        "[WARN] widowx_client.reset(itraj=...) is unsupported by this "
                        "widowx_envs version; falling back to reset()."
                    )
                    warned_itraj_fallback = True
                last_status = widowx_client.reset()
        if last_status == WidowXStatus.SUCCESS:
            break
        print(
            f"[WARN] WidowX reset attempt {attempt}/{retries} failed with "
            f"status={_status_name(last_status, WidowXStatus)}."
        )
        if attempt < retries and retry_sleep > 0.0:
            time.sleep(retry_sleep)

    _set_widowx_reqrep_timeout_ms(widowx_client, rpc_timeout_ms)
    return last_status


def _build_camera_topics_from_flags() -> List[Dict[str, str]]:
    topics: List[Dict[str, str]] = []
    for topic in FLAGS.camera_topics:
        for piece in re.split(r"[,\s]+", str(topic).strip()):
            topic_str = piece.strip()
            if topic_str:
                topics.append({"name": topic_str})
    if not topics:
        raise ValueError("--camera_topics must include at least one non-empty topic.")
    return topics


def _run_timeout_recovery_sequence(widowx_client: Any, blocking: bool = True) -> Any:
    target_joints = np.array(
        [float(v) for v in FLAGS.timeout_recovery_target_joints], dtype=np.float64
    ).reshape(-1)
    if target_joints.size != 6:
        raise ValueError(
            "--timeout_recovery_target_joints must contain exactly 6 joint values."
        )
    if not hasattr(widowx_client, "timeout_recovery_move"):
        raise AttributeError(
            "WidowXClient does not expose timeout_recovery_move(); this needs the "
            "patched widowx_envs tree."
        )
    print(
        "[INFO] Running post-timeout recovery: lift to z="
        f"{float(FLAGS.timeout_recovery_lift_z):.4f}, move above target joints, "
        "then descend."
    )
    return widowx_client.timeout_recovery_move(
        target_joints=target_joints,
        lift_z=float(FLAGS.timeout_recovery_lift_z),
        duration=float(FLAGS.timeout_recovery_stage_duration),
        blocking=bool(blocking),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Observation handling
# ═══════════════════════════════════════════════════════════════════════════

def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _reshape_flattened_image(flat: np.ndarray, im_size: int, im_width: int = 0) -> np.ndarray:
    h = int(im_size)
    w = int(im_width) if im_width > 0 else h
    if h > 0 and flat.size == 3 * h * w:
        return _to_uint8_rgb(flat.reshape(3, h, w).transpose(1, 2, 0))
    if h > 0 and w != h and flat.size == 3 * h * h:
        return _to_uint8_rgb(flat.reshape(3, h, h).transpose(1, 2, 0))
    side = int(round(np.sqrt(flat.size / 3.0)))
    if side > 0 and 3 * side * side == flat.size:
        return _to_uint8_rgb(flat.reshape(3, side, side).transpose(1, 2, 0))
    raise ValueError(
        f"Cannot reshape flattened image of size {flat.size} with "
        f"im_size={im_size}, im_width={im_width}"
    )


def _extract_widowx_rgb_sources(raw_obs: Dict[str, Any], im_size: int) -> Dict[str, np.ndarray]:
    sources: Dict[str, np.ndarray] = {}
    for key in ("external_img", "over_shoulder_img", "wrist_img"):
        if raw_obs.get(key) is not None:
            sources[key] = _to_uint8_rgb(np.asarray(raw_obs[key]))

    if raw_obs.get("full_image") is not None:
        full_image = np.asarray(raw_obs["full_image"])
        if full_image.ndim == 4:
            for i in range(full_image.shape[0]):
                sources[f"full_image_{i}"] = _to_uint8_rgb(full_image[i])
        elif full_image.ndim == 3:
            sources["full_image_0"] = _to_uint8_rgb(full_image)

    if raw_obs.get("image") is not None:
        image_arr = np.asarray(raw_obs["image"])
        if image_arr.ndim == 1:
            sources.setdefault(
                "image_0", _reshape_flattened_image(image_arr, im_size, FLAGS.im_width)
            )
        elif image_arr.ndim == 2:
            for i in range(image_arr.shape[0]):
                try:
                    sources.setdefault(
                        f"image_{i}",
                        _reshape_flattened_image(image_arr[i], im_size, FLAGS.im_width),
                    )
                except Exception:
                    continue
        elif image_arr.ndim == 3:
            if image_arr.shape[-1] == 3 or image_arr.shape[0] == 3:
                sources.setdefault("image_0", _to_uint8_rgb(image_arr))
            else:
                for i in range(image_arr.shape[0]):
                    try:
                        sources.setdefault(f"image_{i}", _to_uint8_rgb(image_arr[i]))
                    except Exception:
                        continue
        elif image_arr.ndim == 4:
            for i in range(image_arr.shape[0]):
                try:
                    sources.setdefault(f"image_{i}", _to_uint8_rgb(image_arr[i]))
                except Exception:
                    continue

    if not sources:
        raise ValueError(
            "WidowX observation contains none of {external_img, over_shoulder_img, "
            "wrist_img, full_image, image}"
        )
    return sources


def _select_source_for_camera_stream(
    stream: str, rgb_sources: Dict[str, np.ndarray]
) -> np.ndarray:
    """Map a training camera stream name (images0/images1) to a live source.

    The training streams are indexed, and the server exposes the cameras in
    camera_topics order: index 0 arrives as external_img, index 1 as
    over_shoulder_img. On the current single-camera rig the D435 is gone, so
    blue (images1) is full_image[0] and therefore arrives as external_img --
    hence external_img is a fallback for index 1 as well.
    """
    match = re.search(r"(\d+)$", stream)
    cam_idx = int(match.group(1)) if match else None

    if cam_idx == 0:
        preferred = ["external_img", "full_image_0", "image_0", "over_shoulder_img"]
    elif cam_idx == 1:
        preferred = ["over_shoulder_img", "full_image_1", "image_1", "external_img"]
    elif cam_idx == 2:
        preferred = ["wrist_img", "full_image_2", "image_2", "external_img"]
    else:
        preferred = []

    for key in preferred:
        if key in rgb_sources:
            return rgb_sources[key]
    return rgb_sources[sorted(rgb_sources.keys())[0]]


def _preprocess_frame(rgb: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Live frame -> (H, W, 3) uint8, matching PushTRealPixelsDataset.

    Training decoded the JPEGs and resized to (image_height, image_width) with
    an AREA filter, keeping uint8 in [0, 255]; the conv encoder divides by 255
    itself. Reproduce exactly that -- do NOT scale to [0, 1] here.
    """
    if FLAGS.swap_rgb:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    H, W = out_hw
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def _build_stack_frame(
    raw_obs: Dict[str, Any], camera_streams: List[str], out_hw: Tuple[int, int]
) -> np.ndarray:
    """One timestep of the stack: cameras concatenated on the channel axis.

    Camera order matches PushTRealPixelsDataset, which iterates camera_streams
    inside each timestep.
    """
    rgb_sources = _extract_widowx_rgb_sources(raw_obs, FLAGS.im_size)
    per_cam = [
        _preprocess_frame(_select_source_for_camera_stream(s, rgb_sources), out_hw)
        for s in camera_streams
    ]
    return np.concatenate(per_cam, axis=-1)  # (H, W, 3*ncam)


def _stack_to_tensor(frame_buf: deque, device: torch.device) -> torch.Tensor:
    """frame_buf oldest->newest, each (H, W, 3*ncam) -> (1, C, H, W) uint8."""
    stacked = np.concatenate(list(frame_buf), axis=-1)   # (H, W, 3*ncam*fs)
    stacked = np.transpose(stacked, (2, 0, 1))           # (C, H, W)
    return torch.from_numpy(np.ascontiguousarray(stacked)).unsqueeze(0).to(device)


def _vis_rgb(raw_obs: Dict[str, Any]) -> np.ndarray:
    sources = _extract_widowx_rgb_sources(raw_obs, FLAGS.im_size)
    for key in ("external_img", "over_shoulder_img"):
        if key in sources:
            return sources[key]
    return sources[sorted(sources.keys())[0]]


def _extract_two_camera_rgb(
    raw_obs: Dict[str, Any]
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    rgb_sources = _extract_widowx_rgb_sources(raw_obs, FLAGS.im_size)
    cam0 = None
    for key in ("external_img", "full_image_0", "image_0", "over_shoulder_img", "wrist_img"):
        if key in rgb_sources:
            cam0 = rgb_sources[key]
            break
    if cam0 is None and rgb_sources:
        cam0 = rgb_sources[sorted(rgb_sources.keys())[0]]

    cam1 = None
    for key in ("over_shoulder_img", "full_image_1", "image_1", "external_img", "wrist_img"):
        if key in rgb_sources and rgb_sources[key] is not cam0:
            cam1 = rgb_sources[key]
            break
    return cam0, cam1


def _extract_eef_xy(raw_obs: Dict[str, Any]) -> np.ndarray | None:
    for key in ("eef_pos", "ee_pos", "agent_pos", "state", "proprio"):
        if raw_obs.get(key) is None:
            continue
        arr = np.asarray(raw_obs[key], dtype=np.float64).reshape(-1)
        if arr.size >= 2:
            return arr[:2]
    return None


def _get_current_eef_xy_with_retry(
    widowx_client: Any, max_wait_sec: float = 2.0, poll_interval_sec: float = 0.05
) -> np.ndarray | None:
    deadline = time.monotonic() + max(0.1, float(max_wait_sec))
    poll_interval_sec = max(0.01, float(poll_interval_sec))
    while time.monotonic() < deadline:
        raw_obs = widowx_client.get_observation()
        if raw_obs is not None:
            eef_xy = _extract_eef_xy(raw_obs)
            if eef_xy is not None:
                return np.asarray(eef_xy, dtype=np.float64).reshape(-1)[:2]
        time.sleep(poll_interval_sec)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# IBC policy: load + DFO inference
# ═══════════════════════════════════════════════════════════════════════════

def _load_ibc_policy(seed_dir: Path, device: torch.device) -> ibc_policy.IBCPolicy:
    """Load the seed dir, applying any CLI overrides to the DFO settings."""
    return ibc_policy.load_policy(
        seed_dir,
        device,
        ckpt_step=int(FLAGS.ckpt_step),
        dfo_overrides={
            "samples": int(FLAGS.dfo_samples) if FLAGS.dfo_samples > 0 else None,
            "iterations": (
                int(FLAGS.dfo_iterations) if FLAGS.dfo_iterations > 0 else None
            ),
        },
    )


def _select_action_dfo(
    loaded: ibc_policy.IBCPolicy, obs_u8: torch.Tensor
) -> Tuple[np.ndarray, float]:
    """DFO action selection, timed. Returns (normalized action, latency ms)."""
    device = obs_u8.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    action = ibc_policy.dfo_select(loaded, obs_u8)   # (1, A)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_time_ms = (time.perf_counter() - t_start) * 1000.0
    return action[0].cpu().numpy(), inference_time_ms


def _unnormalize(norm_action: np.ndarray, loaded: ibc_policy.IBCPolicy) -> np.ndarray:
    return ibc_policy.unnormalize(norm_action, loaded)


# ═══════════════════════════════════════════════════════════════════════════
# Action execution
# ═══════════════════════════════════════════════════════════════════════════

def _safety_clip_xy(action_xy: np.ndarray) -> np.ndarray:
    action = np.asarray(action_xy, dtype=np.float64).reshape(-1)[:2].copy()
    if FLAGS.safety_max_xy_delta > 0:
        lim = float(FLAGS.safety_max_xy_delta)
        action = np.clip(action, -lim, lim)
    return action


def _split_delta_into_substeps(delta_xy: np.ndarray, num_substeps: int) -> List[np.ndarray]:
    """Split one policy delta into num_substeps commands at the robot rate.

    These are per-step displacements, not target poses, so the substeps must
    SUM to the policy delta. Repeating the full delta once per substep, as a
    pose-interpolating harness would, multiplies the commanded motion by the
    substep count.
    """
    num_substeps = max(1, int(num_substeps))
    if num_substeps == 1:
        return [np.asarray(delta_xy, dtype=np.float64).reshape(-1)[:2].copy()]
    share = np.asarray(delta_xy, dtype=np.float64).reshape(-1)[:2] / float(num_substeps)
    return [share.copy() for _ in range(num_substeps)]


def _enter_terminal_cbreak_mode() -> Tuple[int | None, Any]:
    if not sys.stdin.isatty():
        return None, None
    try:
        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return fd, old_attrs
    except Exception:
        return None, None


def _restore_terminal_mode(fd: int | None, old_attrs: Any) -> None:
    if fd is None or old_attrs is None:
        return
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
    except Exception:
        pass


def _poll_stop_key(show_image: bool, stdin_fd: int | None) -> bool:
    if show_image:
        if (cv2.waitKey(1) & 0xFF) in (ord("s"), ord("S")):
            return True
    if stdin_fd is not None:
        try:
            ready, _, _ = select.select([stdin_fd], [], [], 0.0)
        except Exception:
            ready = []
        if ready:
            try:
                ch = os.read(stdin_fd, 1).decode("utf-8", errors="ignore")
            except Exception:
                ch = ""
            if ch.lower() == "s":
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Output: video + timing
# ═══════════════════════════════════════════════════════════════════════════

def _append_rollout_video_frame(
    raw_obs: Dict[str, Any],
    cam0_images: List[np.ndarray],
    cam1_images: List[np.ndarray],
    frame_timestamps_s: List[float],
    warned_missing_camera_stream: bool,
    fallback_rgb: np.ndarray | None = None,
) -> bool:
    cam0_rgb, cam1_rgb = _extract_two_camera_rgb(raw_obs)
    if fallback_rgb is None and (cam0_rgb is None or cam1_rgb is None):
        fallback_rgb = _vis_rgb(raw_obs)
    if cam0_rgb is None:
        cam0_rgb = fallback_rgb
    if cam1_rgb is None:
        if not warned_missing_camera_stream:
            print(
                "[WARN] Second camera stream not found; cam1 video will mirror cam0."
            )
            warned_missing_camera_stream = True
        cam1_rgb = cam0_rgb
    if cam0_rgb is None:
        return warned_missing_camera_stream

    frame_timestamps_s.append(time.monotonic())
    cam0_images.append(cam0_rgb)
    cam1_images.append(cam1_rgb)
    return warned_missing_camera_stream


def _save_rollout_video(
    images: List[np.ndarray], frame_timestamps_s: List[float], policy_name: str
) -> None:
    if FLAGS.video_save_path is None or len(images) == 0:
        return
    os.makedirs(FLAGS.video_save_path, exist_ok=True)
    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = policy_name.replace("/", "_").replace(" ", "_")
    save_path = os.path.join(FLAGS.video_save_path, f"{curr_time}_{safe_name}.mp4")

    # Encode using real frame intervals so playback speed reflects robot speed.
    export_fps = 30.0
    export_frames: List[np.ndarray] = []
    n = len(images)
    if n == 1:
        export_frames = [images[0]]
    elif len(frame_timestamps_s) == n:
        default_dt = 1.0 / export_fps
        for i in range(n - 1):
            dt = float(frame_timestamps_s[i + 1] - frame_timestamps_s[i])
            if (not np.isfinite(dt)) or dt <= 0.0:
                dt = default_dt
            export_frames.extend([images[i]] * max(1, int(round(dt * export_fps))))
        last_dt = float(frame_timestamps_s[-1] - frame_timestamps_s[-2])
        if (not np.isfinite(last_dt)) or last_dt <= 0.0:
            last_dt = default_dt
        export_frames.extend([images[-1]] * max(1, int(round(last_dt * export_fps))))
    else:
        export_frames = images

    imageio.mimsave(save_path, export_frames, fps=export_fps)
    print(f"Saved video to: {save_path}")


def _print_inference_timing_stats(
    inference_times_ms: List[float], policy_name: str, warmup_steps: int,
    nfe_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    total_calls = len(inference_times_ms)
    if total_calls == 0:
        print("[TIMING] No inference calls recorded.")
        return {}

    all_times = np.array(inference_times_ms)
    if warmup_steps > 0 and total_calls > warmup_steps:
        measured = all_times[warmup_steps:]
        print(
            f"[TIMING] Total inference calls: {total_calls}, "
            f"warmup excluded: {warmup_steps}, measured: {len(measured)}"
        )
    else:
        measured = all_times
        print(
            f"[TIMING] Total inference calls: {total_calls} "
            f"(warmup={warmup_steps} not excluded: insufficient calls)"
        )

    mean_ms = float(np.mean(measured))
    stats = {
        "policy_name": policy_name,
        "total_calls": total_calls,
        "warmup_excluded": min(warmup_steps, total_calls),
        "measured_calls": len(measured),
        "mean_ms": mean_ms,
        "std_ms": float(np.std(measured)),
        "median_ms": float(np.median(measured)),
        "min_ms": float(np.min(measured)),
        "max_ms": float(np.max(measured)),
        "p5_ms": float(np.percentile(measured, 5)),
        "p95_ms": float(np.percentile(measured, 95)),
        "p99_ms": float(np.percentile(measured, 99)),
        "hz": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    }
    nfe_str = ""
    if nfe_info and nfe_info.get("nfe") is not None:
        stats["nfe"] = nfe_info["nfe"]
        stats["nfe_policy_type"] = nfe_info.get("policy_type", "unknown")
        stats["nfe_details"] = nfe_info.get("details", "")
        stats["nfe_breakdown"] = nfe_info.get("breakdown", {})
        if mean_ms > 0 and nfe_info["nfe"] > 0:
            stats["ms_per_nfe"] = round(mean_ms / nfe_info["nfe"], 6)
        nfe_str = f"  NFE:    {nfe_info['nfe']:>8d}      ({nfe_info.get('details','')})\n"
        if stats.get("ms_per_nfe"):
            nfe_str += f"  ms/NFE: {stats['ms_per_nfe']:.6f} ms\n"

    print(
        f"[TIMING] ========== Inference Latency: {policy_name} ==========\n"
        f"  Mean:   {stats['mean_ms']:8.2f} ms  (Hz: {stats['hz']:.1f})\n"
        f"  Std:    {stats['std_ms']:8.2f} ms\n"
        f"  Median: {stats['median_ms']:8.2f} ms\n"
        f"  Min:    {stats['min_ms']:8.2f} ms\n"
        f"  Max:    {stats['max_ms']:8.2f} ms\n"
        f"  P95:    {stats['p95_ms']:8.2f} ms\n"
        f"  P99:    {stats['p99_ms']:8.2f} ms\n"
        + nfe_str
        + "[TIMING] =================================================="
    )
    return stats


def _save_timing_json(stats: Dict[str, Any], policy_name: str, save_dir: str | None) -> None:
    if not stats:
        return
    save_dir = save_dir or "."
    os.makedirs(save_dir, exist_ok=True)
    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = policy_name.replace("/", "_").replace(" ", "_")
    json_path = os.path.join(save_dir, f"{curr_time}_{safe_name}_timing.json")
    payload = dict(stats)
    payload["timestamp"] = curr_time
    payload["device"] = FLAGS.device
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[TIMING] Saved timing stats to: {json_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(_):
    np.set_printoptions(suppress=True)
    logging.set_verbosity(logging.WARNING)

    if FLAGS.step_duration <= 0:
        raise ValueError("--step_duration must be > 0")
    if FLAGS.max_duration <= 0:
        raise ValueError("--max_duration must be > 0")

    policy_hz = 1.0 / float(FLAGS.step_duration)
    if FLAGS.robot_exec_hz > 0:
        interp_substeps = max(1, int(round(float(FLAGS.robot_exec_hz) / policy_hz)))
    else:
        interp_substeps = 1
    robot_step_duration = float(FLAGS.step_duration) / float(interp_substeps)
    print(
        "[INFO] Control rate: "
        f"policy_hz={policy_hz:.4f}, substeps={interp_substeps}, "
        f"robot_step_duration={robot_step_duration:.6f}s"
    )
    if interp_substeps > 1:
        print(
            f"[INFO] Each policy delta is divided by {interp_substeps} across the "
            "substeps so the per-policy-step displacement is unchanged."
        )

    term_pose_xy_override: np.ndarray | None = None
    if len(FLAGS.term_pose_xy) > 0:
        term_pose_xy_override = np.array(
            [float(v) for v in FLAGS.term_pose_xy], dtype=np.float64
        ).reshape(-1)
        if term_pose_xy_override.size < 2:
            raise ValueError("--term_pose_xy must contain at least two numbers: x y")
        term_pose_xy_override = term_pose_xy_override[:2]

    device = torch.device(FLAGS.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    torch.manual_seed(int(FLAGS.policy_seed))
    np.random.seed(int(FLAGS.policy_seed))

    # ── policy ─────────────────────────────────────────────────────────────
    seed_dir = Path(FLAGS.seed_dir).expanduser().resolve()
    loaded = _load_ibc_policy(seed_dir, device)
    print(f"Loaded policy: {loaded.name}")
    print(
        f"  cameras={loaded.camera_streams}  frame_stack={loaded.frame_stack}  "
        f"input={loaded.image_hw[0]}x{loaded.image_hw[1]}  device={device}"
    )
    print(f"  action range {loaded.act_min} -> {loaded.act_max}  norm={loaded.norm_range}")
    print(
        f"  DFO {loaded.dfo['samples']} samples x {loaded.dfo['iterations']} iters, "
        f"std={loaded.dfo['iteration_std']} (x{loaded.dfo['std_decay']}/iter), "
        f"buffer={loaded.dfo['boundary_buffer']}"
    )
    nfe_info = loaded.nfe_info()
    print(f"[NFE] {nfe_info['details']}")

    n_expected_cams = len(loaded.camera_streams)
    camera_topics = _build_camera_topics_from_flags()
    if len(camera_topics) < n_expected_cams:
        print(
            f"[WARN] checkpoint expects {n_expected_cams} camera stream(s) "
            f"{loaded.camera_streams} but only {len(camera_topics)} topic(s) were "
            "configured; a stream will be duplicated from another camera."
        )
    print(f"[INFO] Using camera topics: {[c['name'] for c in camera_topics]}")

    WidowXClient, WidowXStatus, WidowXConfigs = _load_widowx_dependencies()

    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(
        {
            "camera_topics": camera_topics,
            "override_workspace_boundaries": WORKSPACE_BOUNDS,
            "move_duration": robot_step_duration,
            "action_mode": FLAGS.action_mode,
            "skip_move_to_neutral": bool(FLAGS.skip_move_to_neutral),
            "move_to_rand_start_freq": -1,
            "fix_zangle": 0.1,
            "adaptive_wait": True,
            "fixed_z_height": float(FLAGS.fixed_z_height),
            "neutral_z_height": float(FLAGS.neutral_z_height),
            "z_lock_feedback_gain": 0.2,
            "z_lock_max_delta": 0.0015,
            "z_lock_deadband": 0.002,
            "xy_action_deadband": 0.0015,
            "vr_vertical_reject_ratio": 0.6,
            "vr_xy_step_deadband": 0.0015,
            "vr_xy_step_clip": 0.008,
            "vr_xy_scale": 0.9,
            "fixed_gripper": float(FLAGS.fixed_gripper),
            "lock_z": bool(FLAGS.lock_z),
            "action_clipping": None,
        }
    )

    if not _tcp_port_reachable(FLAGS.ip, FLAGS.port, timeout_s=1.0):
        print(
            f"[WARN] Preflight TCP check failed for WidowX server at "
            f"{FLAGS.ip}:{FLAGS.port}. Init may time out unless the action server "
            "is started."
        )

    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    reuse_existing_env = False
    reused_env_prepared_for_first_rollout = False
    if FLAGS.widowx_reuse_existing_env and not FLAGS.widowx_force_fresh_init:
        reuse_existing_env = _widowx_server_has_live_env(widowx_client, 1.0, 0.1)
        if reuse_existing_env:
            print(
                "[INFO] Reusing existing WidowX env on the server; skipping init() "
                "to avoid the cached init->reset(itraj=None) path."
            )

    if reuse_existing_env:
        _set_widowx_reqrep_timeout_ms(widowx_client, max(1, int(FLAGS.widowx_rpc_timeout_ms)))
        init_status = WidowXStatus.SUCCESS
        reused_reset_status = _reset_widowx_with_retry(widowx_client, WidowXStatus, i_traj=0)
        if reused_reset_status != WidowXStatus.SUCCESS:
            raise RuntimeError(
                "Failed to reset reused WidowX env with i_traj=0; "
                f"status={_status_name(reused_reset_status, WidowXStatus)}."
            )
        reused_env_prepared_for_first_rollout = True
    else:
        init_status = _init_widowx_with_retry(
            widowx_client, env_params, FLAGS.im_size, WidowXStatus
        )
    if init_status != WidowXStatus.SUCCESS:
        raise RuntimeError(
            f"WidowX init failed after {max(1, int(FLAGS.widowx_init_retries))} "
            f"attempts with status={_status_name(init_status, WidowXStatus)}. "
            f"Check server reachability at {FLAGS.ip}:{FLAGS.port}."
        )

    start_T = None
    if FLAGS.move_to_demo_start:
        start_path = Path(FLAGS.start_eep_npy).expanduser()
        if not start_path.is_file():
            raise FileNotFoundError(
                f"--start_eep_npy not found: {start_path}. Pass "
                "--nomove_to_demo_start to skip the initial move (not recommended: "
                "the arm then starts ~17cm out of distribution)."
            )
        start_T = np.load(start_path).astype(np.float32)

    # ── frame acquisition helpers ──────────────────────────────────────────
    last_raw = {"v": None}   # newest RAW frame of the first stream, for dup detection
    last_obs = {"v": None}   # newest full observation, for proprio logging

    def grab_obs(retries: int = 25) -> Dict[str, Any]:
        # get_observation() can transiently return None (server busy/restarting).
        for _ in range(retries):
            o = widowx_client.get_observation()
            if o is not None:
                last_obs["v"] = o
                return o
            time.sleep(0.2)
        raise RuntimeError("no observation from server after retries (server down?)")

    def refresh_obs() -> Dict[str, Any]:
        """Fetch an observation, rejecting a byte-identical repeat of the last.

        A repeated image inside the frame stack means zero inter-frame motion,
        which the policy maps to garbage. Bounded by --fresh_timeout so a
        genuinely static scene cannot hang the loop.
        """
        o = grab_obs()
        raw = _vis_rgb(o)
        if FLAGS.require_fresh and last_raw["v"] is not None:
            t0 = time.time()
            while np.array_equal(raw, last_raw["v"]) and (time.time() - t0) < FLAGS.fresh_timeout:
                time.sleep(0.05)
                o = grab_obs()
                raw = _vis_rgb(o)
            if np.array_equal(raw, last_raw["v"]):
                print(
                    f"[WARN] stale frame: server repeated image (no fresh frame "
                    f"within {FLAGS.fresh_timeout}s)"
                )
        last_raw["v"] = raw
        return o

    # ── dry run: no motion ─────────────────────────────────────────────────
    if FLAGS.dry_run:
        dump_dir = Path(FLAGS.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"DRY RUN: capturing {FLAGS.dry_run_steps} frames to {dump_dir} "
            "(no step_action). Confirm the T renders RED in fed_000.png."
        )
        frame_buf: deque = deque(maxlen=loaded.frame_stack)
        first = _build_stack_frame(grab_obs(), loaded.camera_streams, loaded.image_hw)
        for _ in range(loaded.frame_stack):
            frame_buf.append(first)
        for i in range(FLAGS.dry_run_steps):
            o = refresh_obs()
            np.save(dump_dir / f"raw_{i:03d}.npy", np.ascontiguousarray(_vis_rgb(o)))
            frame_buf.append(
                _build_stack_frame(o, loaded.camera_streams, loaded.image_hw)
            )
            obs_u8 = _stack_to_tensor(frame_buf, device)
            na, ms = _select_action_dfo(loaded, obs_u8)
            act = _unnormalize(na, loaded)
            newest = list(frame_buf)[-1][:, :, :3]
            cv2.imwrite(
                str(dump_dir / f"fed_{i:03d}.png"), cv2.cvtColor(newest, cv2.COLOR_RGB2BGR)
            )
            print(
                f"[{i:03d}] norm={np.round(na,3)} -> action(dx,dy)={np.round(act,5)} "
                f"({ms:.1f} ms)"
            )
            time.sleep(FLAGS.step_duration)
        widowx_client.stop()
        print(f"Dry run done. Inspect {dump_dir}/fed_000.png before live control.")
        return

    # ── rollouts ───────────────────────────────────────────────────────────
    rollout_idx = 0
    stop_requested = False
    while FLAGS.num_rollouts <= 0 or rollout_idx < FLAGS.num_rollouts:
        if FLAGS.show_image:
            o = grab_obs()
            cv2.imshow("img_view", cv2.cvtColor(_vis_rgb(o), cv2.COLOR_RGB2BGR))
            cv2.waitKey(100)

        input("Press [Enter] to start rollout.")

        if reused_env_prepared_for_first_rollout and rollout_idx == 0:
            print("[INFO] First rollout reuses the itraj=0 reset; skipping duplicate.")
            reset_status = WidowXStatus.SUCCESS
            reused_env_prepared_for_first_rollout = False
        else:
            reset_status = _reset_widowx_with_retry(
                widowx_client, WidowXStatus, i_traj=rollout_idx
            )
        if reset_status != WidowXStatus.SUCCESS:
            print(
                f"[ERROR] Reset failed with "
                f"status={_status_name(reset_status, WidowXStatus)}, skipping rollout."
            )
            rollout_idx += 1
            continue

        # Move to the pose all demos start from. Without it the arm sits near
        # neutral (~x=0.29) vs the demo start (~x=0.117), which is out of
        # distribution: the policy drifts to a workspace corner and stalls.
        if start_T is not None:
            print(
                f"[INFO] Moving EEF to demo start pose (x={start_T[0,3]:.3f}, "
                f"y={start_T[1,3]:.3f}, z={start_T[2,3]:.3f})..."
            )
            move_status, tries = None, 0
            while move_status != WidowXStatus.SUCCESS and tries < int(
                FLAGS.max_initial_move_retries
            ):
                move_status = widowx_client.move(
                    start_T, duration=float(FLAGS.start_move_duration)
                )
                tries += 1
            if move_status != WidowXStatus.SUCCESS:
                print(
                    f"[WARN] initial move did not report SUCCESS after {tries} tries "
                    f"(status={_status_name(move_status, WidowXStatus)}); continuing."
                )

        if term_pose_xy_override is not None:
            rollout_term_pose_xy = term_pose_xy_override.copy()
            print(f"[INFO] Fixed termination XY: {rollout_term_pose_xy.tolist()}")
        else:
            rollout_term_pose_xy = _get_current_eef_xy_with_retry(widowx_client)
            if rollout_term_pose_xy is None:
                print("[WARN] Could not read post-reset EEF XY; term-area stop disabled.")
            else:
                print(
                    "[INFO] Termination XY auto-aligned to post-reset pose: "
                    f"{np.array2string(rollout_term_pose_xy, precision=6, suppress_small=True)}"
                )

        # ── logging setup ──────────────────────────────────────────────────
        log_fh = None
        log_dir = (
            Path(FLAGS.forensic_log_dir) / f"rollout_{rollout_idx:03d}"
            if FLAGS.forensic_log_dir
            else None
        )
        if log_dir is not None:
            (log_dir / "raw").mkdir(parents=True, exist_ok=True)
            (log_dir / "fed").mkdir(parents=True, exist_ok=True)
            log_fh = (log_dir / "steps.jsonl").open("w")
            print(f"[INFO] Forensic log -> {log_dir}")

        cam0_images: List[np.ndarray] = []
        cam1_images: List[np.ndarray] = []
        frame_timestamps_s: List[float] = []
        inference_times_ms: List[float] = []
        frame_buf = deque(maxlen=loaded.frame_stack)
        last_raw["v"] = None

        # Pad the episode start with the first frame, as training padded
        # step 0 by clamping the history index.
        first_obs = grab_obs()
        first_frame = _build_stack_frame(first_obs, loaded.camera_streams, loaded.image_hw)
        for _ in range(loaded.frame_stack):
            frame_buf.append(first_frame)
        last_raw["v"] = _vis_rgb(first_obs)

        t = 0
        stop_this_rollout = False
        stop_reason: str | None = None
        rollout_t_start = time.monotonic()
        term_area_start_timestamp = float("inf")
        warned_missing_camera_stream = False
        stdin_fd, stdin_old_attrs = _enter_terminal_cbreak_mode()
        print(
            f"[INFO] Closed-loop control @ {policy_hz:.1f}Hz, blocking={FLAGS.blocking}. "
            "Stop with 's', --max_duration, or the termination area. "
            "Keep a hand on the E-stop."
        )
        try:
            while True:
                raw_obs = refresh_obs()

                frame_buf.append(
                    _build_stack_frame(raw_obs, loaded.camera_streams, loaded.image_hw)
                )
                vis = _vis_rgb(raw_obs)
                warned_missing_camera_stream = _append_rollout_video_frame(
                    raw_obs=raw_obs,
                    cam0_images=cam0_images,
                    cam1_images=cam1_images,
                    frame_timestamps_s=frame_timestamps_s,
                    warned_missing_camera_stream=warned_missing_camera_stream,
                    fallback_rgb=vis,
                )

                if FLAGS.show_image:
                    cv2.imshow("img_view", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                if _poll_stop_key(FLAGS.show_image, stdin_fd):
                    print("[INFO] Stop requested by user; ending rollout early.")
                    stop_this_rollout = True
                    stop_requested = True
                    stop_reason = "manual"
                    break
                if (time.monotonic() - rollout_t_start) > float(FLAGS.max_duration):
                    print("[INFO] Terminated by timeout.")
                    stop_this_rollout = True
                    stop_reason = "timeout"
                    break

                if rollout_term_pose_xy is not None:
                    eef_xy = _extract_eef_xy(raw_obs)
                    if eef_xy is not None:
                        dist = float(np.linalg.norm(eef_xy - rollout_term_pose_xy))
                        now = time.monotonic()
                        if dist < float(FLAGS.term_dist_thresh):
                            if term_area_start_timestamp > now:
                                term_area_start_timestamp = now
                            elif (now - term_area_start_timestamp) > float(FLAGS.term_hold_sec):
                                print("[INFO] Terminated by reaching termination area.")
                                stop_this_rollout = True
                                stop_reason = "term_area"
                                break
                        else:
                            term_area_start_timestamp = float("inf")

                t_step_start = time.time()
                obs_u8 = _stack_to_tensor(frame_buf, device)
                na, infer_ms = _select_action_dfo(loaded, obs_u8)
                inference_times_ms.append(infer_ms)
                act = _unnormalize(na, loaded)

                if len(inference_times_ms) <= FLAGS.inference_warmup_steps:
                    print(f"  [TIMING] step {len(inference_times_ms):4d} (warmup): {infer_ms:.2f} ms")
                elif len(inference_times_ms) % 10 == 0:
                    recent = inference_times_ms[FLAGS.inference_warmup_steps:]
                    print(
                        f"  [TIMING] step {len(inference_times_ms):4d}: {infer_ms:.2f} ms "
                        f"(running avg: {np.mean(recent):.2f} ms)"
                    )

                action_xy = _safety_clip_xy(act)
                for sub_action in _split_delta_into_substeps(action_xy, interp_substeps):
                    step_status = widowx_client.step_action(
                        np.asarray(sub_action, np.float32), blocking=FLAGS.blocking
                    )
                    if step_status == getattr(WidowXStatus, "NO_CONNECTION", None):
                        print("[ERROR] Lost connection to the server. Stopping.")
                        stop_this_rollout = True
                        stop_reason = "no_connection"
                        break
                    if step_status != WidowXStatus.SUCCESS:
                        raise RuntimeError(
                            "WidowX step_action failed: status="
                            f"{_status_name(step_status, WidowXStatus)}, "
                            f"action={np.asarray(sub_action).tolist()}"
                        )
                    if _poll_stop_key(FLAGS.show_image, stdin_fd):
                        print("[INFO] Stop requested by user; ending rollout early.")
                        stop_this_rollout = True
                        stop_requested = True
                        stop_reason = "manual"
                        break
                if stop_this_rollout:
                    break

                if FLAGS.settle > 0:
                    time.sleep(FLAGS.settle)

                if log_fh is not None:
                    np.save(log_dir / "raw" / f"{t:04d}.npy", np.ascontiguousarray(vis))
                    fed_rgb = list(frame_buf)[-1][:, :, :3]
                    cv2.imwrite(
                        str(log_dir / "fed" / f"{t:04d}.png"),
                        cv2.cvtColor(fed_rgb, cv2.COLOR_RGB2BGR),
                    )
                    state = (last_obs["v"] or {}).get("state")
                    log_fh.write(
                        json.dumps(
                            {
                                "step": t,
                                "t": time.time(),
                                "norm": [float(x) for x in np.ravel(na)],
                                "action": [float(x) for x in np.ravel(act)],
                                "inference_ms": infer_ms,
                                "state": (
                                    np.ravel(state).astype(float).tolist()
                                    if state is not None
                                    else None
                                ),
                            }
                        )
                        + "\n"
                    )
                    log_fh.flush()

                print(
                    f"[{t:04d}] norm={np.round(na, 3)} -> "
                    f"action(dx,dy)={np.round(act, 5)}"
                )
                t += 1

                dt = time.time() - t_step_start
                if dt < FLAGS.step_duration:
                    time.sleep(FLAGS.step_duration - dt)

        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
        finally:
            _restore_terminal_mode(stdin_fd, stdin_old_attrs)
            if log_fh is not None:
                log_fh.close()

        if stop_reason == "timeout":
            try:
                recovery_status = _run_timeout_recovery_sequence(widowx_client, blocking=False)
                if recovery_status != WidowXStatus.SUCCESS:
                    print(
                        "[WARN] Post-timeout recovery failed with status="
                        f"{_status_name(recovery_status, WidowXStatus)}."
                    )
            except Exception as exc:
                print(f"[WARN] Post-timeout recovery could not run: {exc}")

        timing_stats = _print_inference_timing_stats(
            inference_times_ms=inference_times_ms,
            policy_name=loaded.name,
            warmup_steps=FLAGS.inference_warmup_steps,
            nfe_info=nfe_info,
        )
        _save_timing_json(timing_stats, loaded.name, FLAGS.video_save_path)
        _save_rollout_video(cam0_images, frame_timestamps_s, f"{loaded.name}_cam0")
        _save_rollout_video(cam1_images, frame_timestamps_s, f"{loaded.name}_cam1")

        print(f"[INFO] Rollout {rollout_idx} ended after {t} steps (reason={stop_reason}).")
        rollout_idx += 1
        if stop_requested:
            print("[INFO] Evaluation stopped by user request.")
            break

    widowx_client.stop()


if __name__ == "__main__":
    app.run(main)
