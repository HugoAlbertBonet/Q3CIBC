#!/usr/bin/env python3
"""Deploy a trained Diffusion-Policy Push-T checkpoint on the real WidowX arm.

DP counterpart of deploy_pusht_real.py. The robot-facing half is IDENTICAL —
every server call, safety clip, z-hold, approach floor, cond, dry-run and
forensic-log path is imported from deploy_pusht_real.py so the two clients agree
bit-for-bit. Only the policy differs:

  * Model: a PixelDiffusionDenoiser (ConvMaxpool encoder + DenseResnet head)
    plus a GaussianDiffusion sampler, rebuilt from the checkpoint's
    config.json / norm_stats exactly as train_pusht_real_dp.py wrote them.
  * Action: instead of the Q3C CP-cloud argmax, we SAMPLE the fitted action
    distribution once per step via DDPM (or DDIM with --sampler ddim). Both are
    inference schedules over the same denoiser.

Weights: denoiser_ema.pt (default) or denoiser.pt (--no-ema).

Usage (server already up):

    python scripts/deploy_pusht_real_dp.py \
        --seed-dir checkpoints/pusht_real_dp/d03_vpred_s11_dropzero \
        --device cpu --dry-run
    python scripts/deploy_pusht_real_dp.py \
        --seed-dir checkpoints/pusht_real_dp/d03_vpred_s11_dropzero \
        --device cpu --steps 200 --log-dir results/run_dp
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
    p.add_argument("--camera-topics", nargs="+", default=d.CAMERA_TOPICS)

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
    p.add_argument("--safety-max-xy-delta", type=float, default=d.SAFETY_MAX_XY_DELTA)
    p.add_argument("--min-step-xy", type=float, default=0.0,
                   help="metres. If >0, any nonzero |dx|/|dy| below this is "
                        "snapped UP to it (sign kept); exact 0 stays 0.")
    p.add_argument("--lock-z", dest="lock_z", action="store_true", default=True)
    p.add_argument("--no-lock-z", dest="lock_z", action="store_false")
    p.add_argument("--fixed-z-height", type=float, default=d.FIXED_Z_HEIGHT)
    p.add_argument("--neutral-z-height", type=float, default=d.NEUTRAL_Z_HEIGHT)
    p.add_argument("--z-hold", type=float, default=0.0,
                   help="metres. If >0, inject a per-step dz to hold EEF z "
                        "(needs a z-carrying action_mode). See deploy_pusht_real.py.")
    p.add_argument("--z-hold-gain", type=float, default=1.0)
    p.add_argument("--z-hold-max", type=float, default=0.01)
    p.add_argument("--fixed-gripper", type=float, default=d.FIXED_GRIPPER,
                   help="gripper target (0.0 = CLOSED, 1.0 = OPEN).")
    p.add_argument("--gripper-command", type=float, default=0.0,
                   help="explicitly actuate gripper after reset (0.0 = closed). "
                        "Negative to skip.")
    p.add_argument("--skip-move-to-neutral", action="store_true")
    p.add_argument("--i-traj", type=int, default=0,
                   help="trajectory index passed to reset(itraj=N).")

    # --- initial pose --------------------------------------------------------
    p.add_argument("--move-to-demo-start", dest="move_to_demo_start",
                   action="store_true", default=True)
    p.add_argument("--no-move-to-demo-start", dest="move_to_demo_start",
                   action="store_false")
    p.add_argument("--start-eep-npy", type=Path,
                   default=ROOT / "scripts" / "assets" / "pusht_start_eep.npy",
                   help="4x4 EEF start transform (mean of demo starts, x~0.117).")
    p.add_argument("--start-move-duration", type=float, default=1.5)
    p.add_argument("--max-initial-move-retries", type=int, default=5)

    # --- HARD approach guard -------------------------------------------------
    p.add_argument("--approach-floor", dest="approach_floor",
                   action="store_true", default=True)
    p.add_argument("--no-approach-floor", dest="approach_floor",
                   action="store_false")
    p.add_argument("--approach-floor-x", type=float, default=None)

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
    p.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"],
                   help="inference schedule over the trained denoiser.")
    p.add_argument("--ddim-steps", type=int, default=None,
                   help="DDIM sub-sampled steps (default: ddim_eval_steps[0] "
                        "from norm_stats).")
    p.add_argument("--ddim-eta", type=float, default=None,
                   help="DDIM stochasticity (default: ddim_eta from norm_stats).")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="if set, torch.manual_seed before sampling for a "
                        "reproducible dry run.")

    # --- diagnostics ---------------------------------------------------------
    p.add_argument("--dry-run", action="store_true",
                   help="no motion: dump fed frames + print sampled actions")
    p.add_argument("--dry-run-steps", type=int, default=20)
    p.add_argument("--dump-dir", type=Path, default=ROOT / "deploy_dryrun_dp")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="per-step forensic log: raw/*.npy, fed/*.png, steps.jsonl")
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
        encoder_target_height=enc_h, encoder_target_width=enc_w, device=device)
    diffusion = build_diffusion(dp, device, (-1.0, 1.0))
    return denoiser, diffusion, dp


@torch.no_grad()
def dp_sample_action(diffusion, denoiser, obs_u8, sampler, ddim_steps, ddim_eta,
                     cond=None):
    """One draw of the (normalized) action. obs_u8: (1,C,H,W) uint8 -> (2,)."""
    if cond is not None:
        denoiser._cond = cond
    state = obs_u8.float()
    if sampler == "ddim":
        a = diffusion.ddim_sample(denoiser, state, action_dim=2,
                                  num_steps=ddim_steps, eta=ddim_eta)
    else:
        a = diffusion.ddpm_sample(denoiser, state, action_dim=2)
    return a[0].detach().cpu().numpy()


def main() -> int:
    args = parse_args()
    if args.z_hold > 0 and args.action_mode == "2trans":
        raise SystemExit(
            "--z-hold needs an action_mode that sends z "
            "(3trans/3trans1rot/3trans3rot); got 2trans.")
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
                                env_cfg.get("camera_streams", ["images1"])))
    image_hw = norm_stats.get("image_hw",
                              (int(env_cfg.get("image_height", 240)),
                               int(env_cfg.get("image_width", 320))))
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    in_channels = int(norm_stats.get("in_channels", 3 * len(cams) * frame_stack))

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

    ddim_steps = args.ddim_steps
    if ddim_steps is None:
        ev = norm_stats.get("ddim_eval_steps", dp.get("ddim_eval_steps", [10]))
        ddim_steps = int(ev[0]) if ev else 10
    ddim_eta = args.ddim_eta
    if ddim_eta is None:
        ddim_eta = float(norm_stats.get("ddim_eta", dp.get("ddim_eta", 0.0)))
    if args.sample_seed is not None:
        torch.manual_seed(args.sample_seed)

    print(f"Loaded denoiser ({'raw' if args.no_ema else 'EMA'}) from {seed_dir}")
    print(f"  frame_stack={frame_stack} cameras={cams} model_hw=({image_h},{image_w}) "
          f"in_channels={in_channels}")
    print(f"  sampler={args.sampler}"
          + (f" ({ddim_steps} steps, eta={ddim_eta})" if args.sampler == "ddim" else "")
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
        return dp_sample_action(diffusion, denoiser, obs_u8, args.sampler,
                                ddim_steps, ddim_eta, cond=make_cond(raw_obs))

    # --- connect -------------------------------------------------------------
    WidowXClient, WidowXConfigs, WidowXStatus = d.load_widowx_dependencies(
        args.widowx_envs_path)
    print(f"WidowX SDK: {WidowXClient.__module__} "
          f"({getattr(sys.modules.get(WidowXClient.__module__), '__file__', '?')})")

    env_params = d.build_env_params(args, WidowXConfigs)
    print(f"Camera topics: {args.camera_topics}")
    print(f"action_mode={args.action_mode} lock_z={args.lock_z} "
          f"fixed_z_height={args.fixed_z_height} move_duration={args.step_duration}")

    client = WidowXClient(host=args.ip, port=args.port)

    reuse_existing_env = False
    if args.reuse_existing_env and not args.force_fresh_init:
        reuse_existing_env = d.widowx_server_has_live_env(client, max_wait_sec=1.0)
        if reuse_existing_env:
            print("[INFO] Server already has a live env; reusing it (skipping init()).")
            print("[WARN] The live env keeps its FIRST env_params; restart the "
                  "server if action_mode/lock_z/etc. must change.")

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

    first_obs = grab_obs()
    first = d.extract_blue_frame(first_obs)
    print(f"Blue frame: raw {first.shape}")
    for _ in range(frame_stack):
        frame_buf.append(d.preprocess(first, (image_h, image_w)))

    # --- dry run -------------------------------------------------------------
    if args.dry_run:
        import cv2
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"DRY RUN: dumping {args.dry_run_steps} frames to {args.dump_dir} "
              f"(no step_action). Confirm the T renders RED.")
        for i in range(args.dry_run_steps):
            raw_obs = grab_obs()
            raw = d.extract_blue_frame(raw_obs)
            np.save(args.dump_dir / f"raw_{i:03d}.npy", np.ascontiguousarray(raw))
            frame_buf.append(d.preprocess(raw, (image_h, image_w)))
            obs_u8 = d.stack_to_tensor(frame_buf, device)
            na = sample(obs_u8, raw_obs)
            act = d.unnormalize(na, act_min, act_max, norm_range)
            cv2.imwrite(str(args.dump_dir / f"fed_{i:03d}.png"),
                        cv2.cvtColor(list(frame_buf)[-1], cv2.COLOR_RGB2BGR))
            print(f"[{i:03d}] norm={np.round(na, 3)} -> action(dx,dy)={np.round(act, 4)}")
            time.sleep(args.step_duration)
        client.stop()
        print("Dry run done. Inspect the fed_000.png before live control.")
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
            raw = d.extract_blue_frame(raw_obs)
            frame_buf.append(d.preprocess(raw, (image_h, image_w)))
            obs_u8 = d.stack_to_tensor(frame_buf, device)

            na = sample(obs_u8, raw_obs)
            act_xy = d.unnormalize(na, act_min, act_max, norm_range)

            act_xy, snapped = d.apply_min_step(act_xy, args.min_step_xy)
            if snapped:
                print(f"[min-step] snapped {np.round(na, 3)} -> dx,dy={np.round(act_xy, 4)}")

            cur_x = d.eef_x_from_obs(raw_obs)
            act_xy, floored = d.apply_approach_floor(act_xy, cur_x, approach_floor_x)
            if floored:
                print(f"[SAFETY] approach floor: clipped dx at x={cur_x:.4f} "
                      f"(floor={approach_floor_x:.4f})")

            action_7d = d.to_action_7d(act_xy, args.fixed_gripper)
            action_7d = d.safety_clip_action(action_7d, args.action_mode,
                                             args.safety_max_xy_delta)
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
