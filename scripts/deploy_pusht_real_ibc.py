#!/usr/bin/env python3
"""Deploy a trained IBC (EBM + DFO) Push-T policy on the real WidowX arm.

IBC counterpart of scripts/deploy_pusht_real.py: identical server/client
split, camera handling, preprocessing, fresh-frame guard, start-pose move,
dry-run, and forensic logging — only the model and action selection differ.

Action selection follows the official google-research/ibc optimal policy for
Pushing-Pixels (ibc/configs/pushing_pixels/pixel_ebm_best.gin + the
mcmc.iterative_dfo defaults):

    1. Encode the frame stack ONCE (late fusion, cached 256-D features).
    2. Draw 2048 uniform action samples in [-1-buf, 1+buf] (buffer 0.05).
    3. 3 DFO iterations: score -> softmax resample (bincount-ordered, the
       tf.gather+tf.repeat convention) -> Gaussian jitter (std 0.33, halved
       per iteration) -> clip to the action box.
    4. argmax of the final scores picks the action.

The normalized action is clipped to [-1, 1] before denormalization: the
boundary buffer lets DFO propose up to 5% outside the training range, which
must not be sent to the hardware.

Run on the Alienware (localhost) with the server already up:

    python scripts/deploy_pusht_real_ibc.py \
        --seed-dir checkpoints/pusht_real_ibc/seed_0000 --dry-run
    # then, once the dry-run frames look right and the arm is clear:
    python scripts/deploy_pusht_real_ibc.py \
        --seed-dir checkpoints/pusht_real_ibc/seed_0000
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Deploy-time env params for the server's robot env init — identical to
# deploy_pusht_real.py (planar 2trans control, z locked at table height).
FIXED_Z_HEIGHT = 0.02
DEPLOY_ENV_PARAMS = {
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
                   help="checkpoint dir with q_estimator.pt, norm_stats.pt, config.json")
    p.add_argument("--ip", default="localhost", help="robot server host")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--steps", type=int, default=985,
                   help="max control steps (default = longest training episode)")
    p.add_argument("--hz", type=float, default=5.0, help="control loop rate")
    p.add_argument("--dfo-samples", type=int, default=None,
                   help="override DFO sample count (default: config, best gin 2048)")
    p.add_argument("--dfo-iterations", type=int, default=None,
                   help="override DFO iteration count (default: config, best gin 3)")
    p.add_argument("--non-blocking", action="store_true",
                   help="send actions non-blocking (legacy). Default is BLOCKING: "
                        "wait for the arm to finish each delta before grabbing the "
                        "next frame so the 2-frame stack carries real inter-frame "
                        "motion, as in training.")
    p.add_argument("--settle", type=float, default=0.0,
                   help="extra seconds to wait after each (blocking) step before "
                        "capturing the next frame")
    p.add_argument("--no-require-fresh", action="store_true",
                   help="disable the duplicate-frame guard (server occasionally "
                        "repeats images; a stale slot means zero-motion obs)")
    p.add_argument("--fresh-timeout", type=float, default=0.5,
                   help="max seconds to wait for a non-duplicate frame before "
                        "proceeding with the stale one (logs a warning)")
    p.add_argument("--no-initial-move", action="store_true",
                   help="skip moving the EEF to the demo start pose before the "
                        "rollout (without it the arm starts ~17cm out of the "
                        "training distribution)")
    p.add_argument("--start-eep-npy", type=Path,
                   default=ROOT / "scripts" / "assets" / "pusht_start_eep.npy",
                   help="4x4 EEF start transform to move to (mean of demo starts)")
    p.add_argument("--start-move-duration", type=float, default=1.5)
    p.add_argument("--log-dir", type=Path, default=None,
                   help="if set, forensic-log every closed-loop step here: raw + "
                        "fed frames and a steps.jsonl row")
    p.add_argument("--ckpt-step", type=int, default=None,
                   help="load q_estimator_step{N:06d}.pt instead of the final "
                        "q_estimator.pt")
    p.add_argument("--swap-rgb", action="store_true",
                   help="swap channels before feeding the model (default OFF: this "
                        "rig's server frame already has red in channel 0)")
    p.add_argument("--obs-key", default="auto",
                   choices=["auto", "external_img", "over_shoulder_img"],
                   help="which get_observation() field holds the blue frame")
    p.add_argument("--dry-run", action="store_true",
                   help="no motion: dump fed frames + print predicted actions")
    p.add_argument("--dry-run-steps", type=int, default=20)
    p.add_argument("--dump-dir", type=Path, default=ROOT / "deploy_dryrun_ibc",
                   help="where --dry-run writes the fed RGB frames")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the stochastic DFO sampler")
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


def build_model(env: dict, in_channels: int, device):
    """Reconstruct the PixelEBM exactly as train_pusht_real_ibc.py built it."""
    from utils.models import PixelQEstimator

    m = env["model"]
    ebm = PixelQEstimator(
        action_dim=int(env.get("action_dim", 2)),
        in_channels=in_channels,
        encoder_target_height=int(env.get("encoder_target_height", 180)),
        encoder_target_width=int(env.get("encoder_target_width", 240)),
        value_width=int(m.get("value_width", 1024)),
        value_num_blocks=int(m.get("value_num_blocks", 1)),
        cond_dim=0,
        encoder_kind=m.get("encoder_kind", "conv_maxpool"),
    ).to(device).eval()
    return ebm


def load_weights(model, path: Path, device):
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint weights: {path}")
    # weights_only=False: our own trusted checkpoints (state dicts + numpy).
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    return model


def pick_blue_frame(obs: dict, obs_key: str) -> np.ndarray:
    """Return the blue-camera BGR frame from a get_observation() dict."""
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
    """Live blue frame (H,W,3 uint8) -> (H',W',3 uint8), channels as training."""
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
def select_action_dfo(ebm, obs_u8, num_samples: int, num_iterations: int,
                      iteration_std: float, std_decay: float,
                      boundary_buffer: float, action_bounds) -> np.ndarray:
    """iterative_dfo port (mirrors bench_inference_pixels.make_ibc_dfo).

    Encoder runs once; the DFO loop scores cached features. Resampling uses
    the bincount-ordered gather so the final argmax indexes samples the same
    way IBC's MappedCategorical.mode() does.
    """
    a_lo, a_hi = action_bounds
    action_dim = ebm.action_dim
    device = obs_u8.device

    features = ebm.encode(obs_u8)  # (1, 256) — cached for the whole loop

    buf = (a_hi - a_lo) * boundary_buffer
    actions = torch.empty(1, num_samples, action_dim, device=device).uniform_(
        a_lo - buf, a_hi + buf
    )
    std = iteration_std
    log_probs = None
    for it in range(num_iterations):
        log_probs = ebm.score(features, actions).squeeze(-1)  # (1, N)
        probs = torch.softmax(log_probs.squeeze(0), dim=-1)
        idx = torch.multinomial(probs, num_samples, replacement=True)
        counts = torch.bincount(idx, minlength=num_samples)
        repeat_idx = torch.repeat_interleave(
            torch.arange(num_samples, device=device), counts
        )
        actions = actions[:, repeat_idx, :]
        if it < num_iterations - 1:
            actions = actions + torch.randn_like(actions) * std
            actions = actions.clamp(a_lo, a_hi)
            std *= std_decay
    sel = log_probs.argmax(dim=1)
    action = actions[0, sel[0], :]
    # Hardware safety: the boundary buffer admits samples up to 5% outside the
    # training range; never send those to the arm.
    return action.clamp(a_lo, a_hi).cpu().numpy()


def unnormalize(norm_action, act_min, act_max, norm_range):
    lo, hi = norm_range
    scale = (act_max - act_min) / (hi - lo)
    return (act_min + (np.asarray(norm_action, np.float32) - lo) * scale).astype(np.float32)


def main() -> int:
    args = parse_args()
    seed_dir = args.seed_dir.resolve()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- checkpoint metadata ------------------------------------------------
    env = load_run_config(seed_dir)
    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu",
                            weights_only=False)
    act_min = np.asarray(norm_stats["act_min"], np.float32)
    act_max = np.asarray(norm_stats["act_max"], np.float32)
    norm_range = tuple(norm_stats.get("action_norm_range", (-1.0, 1.0)))
    frame_stack = int(norm_stats.get("frame_stack", env.get("frame_stack", 2)))
    a_lo, a_hi = env.get("action_bounds", [-1.0, 1.0])

    inf = env.get("inference", {})
    dfo_samples = args.dfo_samples or int(inf.get("dfo_samples", 2048))
    dfo_iters = args.dfo_iterations or int(inf.get("dfo_iterations", 3))
    dfo_std = float(inf.get("dfo_iteration_std", 0.33))
    dfo_decay = float(inf.get("dfo_std_decay", 0.5))
    boundary_buffer = float(inf.get("uniform_boundary_buffer", 0.05))

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
    print(f"Cameras:      {cams} (deploy stream = blue)")
    print(f"Frame stack:  {frame_stack}  in_channels={in_channels}  "
          f"input={image_h}x{image_w}")
    print(f"Action range: {act_min} -> {act_max}  norm={norm_range}")
    print(f"DFO:          {dfo_samples} samples x {dfo_iters} iters, "
          f"std={dfo_std} (x{dfo_decay}/iter), buffer={boundary_buffer}  "
          f"device={device}")

    # --- model --------------------------------------------------------------
    ebm = build_model(env, in_channels, device)
    ckpt_name = ("q_estimator.pt" if args.ckpt_step is None
                 else f"q_estimator_step{args.ckpt_step:06d}.pt")
    load_weights(ebm, seed_dir / ckpt_name, device)
    print(f"Loaded weights: {ckpt_name}")

    # --- connect to robot ---------------------------------------------------
    from widowx_envs.widowx_env_service import WidowXClient, WidowXStatus

    client = WidowXClient(host=args.ip, port=args.port)
    client.init(DEPLOY_ENV_PARAMS, image_size=256)
    # init only constructs the env; reset() homes the arm AND starts the
    # control loop that step_action depends on.
    print("Resetting robot (home + start control loop)...")
    client.reset()

    if not args.no_initial_move:
        start_T = np.load(args.start_eep_npy).astype(np.float32)
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
    blue0 = pick_blue_frame(obs, args.obs_key)
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
        for _ in range(retries):
            o = client.get_observation()
            frame = None if o is None else pick_blue_frame(o, args.obs_key)
            if frame is not None:
                last_obs["v"] = o
                return frame
            time.sleep(0.2)
        raise RuntimeError("no observation from server after retries (server down?)")

    def refresh_frame():
        # Reject byte-identical repeated frames so the 2-frame stack never has
        # a stale zero-motion slot (see deploy_pusht_real.py rationale).
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

    def predict(obs_u8):
        return select_action_dfo(
            ebm, obs_u8,
            num_samples=dfo_samples,
            num_iterations=dfo_iters,
            iteration_std=dfo_std,
            std_decay=dfo_decay,
            boundary_buffer=boundary_buffer,
            action_bounds=(float(a_lo), float(a_hi)),
        )

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
            raw = None
            while raw is None:
                o = client.get_observation()
                raw = None if o is None else pick_blue_frame(o, args.obs_key)
                if raw is None:
                    time.sleep(0.2)
            np.save(args.dump_dir / f"raw_{i:03d}.npy", np.ascontiguousarray(raw))
            frame_buf.append(preprocess(raw, (image_h, image_w), args.swap_rgb))
            obs_u8 = stack_to_tensor(frame_buf, device)
            na = predict(obs_u8)
            act = unnormalize(na, act_min, act_max, norm_range)
            newest_rgb = list(frame_buf)[-1]
            cv2.imwrite(str(args.dump_dir / f"fed_{i:03d}.png"),
                        cv2.cvtColor(newest_rgb, cv2.COLOR_RGB2BGR))
            print(f"[{i:03d}] norm={np.round(na,3)} -> action(dx,dy)={np.round(act,4)}")
            time.sleep(period)
        client.stop()
        print(f"Dry run done. Inspect {args.dump_dir}/fed_000.png before live control.")
        return 0

    # --- closed loop --------------------------------------------------------
    mode = "non-blocking (legacy)" if args.non_blocking else "blocking"
    print(f"Closed-loop control up to {args.steps} steps @ {args.hz}Hz, "
          f"step={mode}, settle={args.settle}s. "
          f"Keep a hand on the E-stop. Ctrl-C to stop.")
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
            na = predict(obs_u8)
            act = unnormalize(na, act_min, act_max, norm_range)
            res = client.step_action(np.asarray(act, np.float32),
                                     blocking=not args.non_blocking)
            if args.settle > 0:
                time.sleep(args.settle)
            log_step(step, na, act, list(frame_buf)[-1])
            print(f"[{step:03d}] norm={np.round(na, 3)} -> action(dx,dy)={np.round(act, 4)}")
            if res == WidowXStatus.NO_CONNECTION:
                print("Lost connection to server. Stopping.")
                break
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
