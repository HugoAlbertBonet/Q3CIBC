#!/usr/bin/env python3
"""Open-loop replay of one recorded Push-T episode, with the policy in SHADOW.

Purpose: split the two failure modes that a live rollout confounds.

  1. Is the RIG still able to reproduce a demonstration? We command the expert's
     recorded (dx, dy) sequence open-loop, through the exact same action pipeline
     the deploy client uses (min-step snap -> approach floor -> 7-D lift ->
     safety clip -> action-mode projection -> step_action). If the T does not end
     where the demo's last frame shows it, the problem is the rig / scene / start
     alignment, not the policy.
  2. Are the POLICY's actions reasonable? At every step the trained checkpoint
     sees the live frame stack and samples an action exactly as its deploy
     client would, but that action is only RECORDED — never sent. Afterwards we
     plot expert vs shadow action per timestep.

--policy picks the shadow family: `dp` samples the DP denoiser
(deploy_pusht_real_dp.dp_sample_action, --sampler/--ddim-*), `q3c` runs the
energy model's CP selection (deploy_pusht_real.select_action, --cp-selection /
--inference / --refine-iters, EEF conditioning included). `auto` (default) reads
the weight files in --seed-dir.

Everything policy-side is imported from scripts/deploy_pusht_real_dp.py and
scripts/deploy_pusht_real.py (build_dp_policy, dp_sample_action, build_models,
select_action, preprocess, build_stack_frame, stack_to_tensor, unnormalize,
apply_min_step, apply_approach_floor, to_action_7d, safety_clip_action,
project_action_to_env_mode, the init/reset retry helpers). No logic is
re-implemented here, so if the replay works the only remaining difference to a
live rollout is *which* action gets sent.

Cameras: identical to the deploy client. Topics are registered in the
collection's order (D435 then blue), each camera is read from its position in
that list, and the policy stack is built from the checkpoint's camera_streams --
one camera (g01/g02/g03) or two (g04) with no flag change. The alignment gate
below shows whichever cameras --align-cameras names, independently of that.

Deviations from the deploy client, both deliberate and both printed at startup:
  * The start pose is this EPISODE's first robot_eef_pose (translation) instead
    of the mean-of-demos asset, so the arm starts where the demo started.
  * The policy's action is recorded, not sent; the expert's is sent.

Usage (server already up):

    # dry run: no motion, but full alignment + shadow policy + plot
    python scripts/replay_pusht_episode.py \
        --seed-dir checkpoints/pusht_real_dp_2026_07/g01_resnet18_s11_350k \
        --archive data/pusht_2026_07_zarr.zip --episode 0 \
        --device cpu --sampler ddim --dry-run --log-dir results/replay_ep0

    # live replay
    python scripts/replay_pusht_episode.py \
        --seed-dir checkpoints/pusht_real_dp_2026_07/g01_resnet18_s11_350k \
        --archive data/pusht_2026_07_zarr.zip --episode 0 \
        --device cpu --sampler ddim --log-dir results/replay_ep0

    # same episode, Q3C energy model in the shadow instead
    python scripts/replay_pusht_episode.py \
        --seed-dir checkpoints/pusht_real_combinedv2/seed_0011 --policy q3c \
        --archive data/pusht_2026_07_zarr.zip --episode 0 \
        --device cpu --inference langevin --log-dir results/replay_ep0_q3c
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The DP deploy client is the reference implementation; loading it also loads the
# Q3C client it shares every robot-facing helper with (exposed as `dp.d`).
_spec = importlib.util.spec_from_file_location(
    "deploy_dp", ROOT / "scripts" / "deploy_pusht_real_dp.py")
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)
d = dp.d


# ---------------------------------------------------------------------------
# Archive readers (loader-external on purpose: no frame cache is built)
# ---------------------------------------------------------------------------

def _zarr_prefix(archive: zipfile.ZipFile) -> str:
    for name in archive.namelist():
        idx = name.find("replay_buffer.zarr/")
        if idx != -1:
            return name[: idx + len("replay_buffer.zarr/")]
    raise SystemExit("replay_buffer.zarr not found in the archive")


def load_lowdim(archive_path: Path):
    """Return (actions[N,7], eef[N,7], episode_ends[E]) from the zarr block."""
    import zarr

    with zipfile.ZipFile(archive_path, "r") as ar:
        prefix = _zarr_prefix(ar)
        members = [n for n in ar.namelist() if n.startswith(prefix)]
        tmp = tempfile.mkdtemp(prefix="replay_zarr_")
        try:
            ar.extractall(tmp, members=members)
            root = zarr.open(os.path.join(tmp, prefix.rstrip("/")), mode="r")
            actions = np.asarray(root["data/action"][:], dtype=np.float64)
            eef = np.asarray(root["data/robot_eef_pose"][:], dtype=np.float64)
            ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return actions, eef, ends


def load_episode_frames(archive_path: Path, episode: int, camera: int,
                        indices) -> dict[int, np.ndarray]:
    """Decode selected frames of videos/<episode>/<camera>.mp4 as (H,W,3) RGB."""
    import imageio.v3 as iio

    wanted = sorted({int(i) for i in indices})
    with zipfile.ZipFile(archive_path, "r") as ar:
        root = _zarr_prefix(ar).split("replay_buffer.zarr/")[0]
        member = f"{root}videos/{episode}/{camera}.mp4"
        if member not in set(ar.namelist()):
            raise SystemExit(f"missing video in archive: {member}")
        scratch = tempfile.mkdtemp(prefix="replay_vid_")
        try:
            ar.extract(member, scratch)
            path = os.path.join(scratch, member)
            out: dict[int, np.ndarray] = {}
            for i, frame in enumerate(iio.imiter(path)):
                if i in wanted:
                    out[i] = np.asarray(frame, dtype=np.uint8)
                if wanted and i >= wanted[-1]:
                    break
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    missing = [i for i in wanted if i not in out]
    if missing:
        raise SystemExit(
            f"episode {episode} camera {camera}: frames {missing} not in the video")
    return out


def resolve_backend(policy: str, seed_dir: Path) -> str:
    """"auto" -> "dp" | "q3c", decided by which weight files the run wrote."""
    if policy != "auto":
        return policy
    has_dp = any((seed_dir / f"denoiser{s}.pt").is_file() for s in ("_ema", ""))
    has_q3c = any((seed_dir / f"control_point_generator{s}.pt").is_file()
                  for s in ("_ema", ""))
    if has_dp and not has_q3c:
        return "dp"
    if has_q3c and not has_dp:
        return "q3c"
    if has_dp and has_q3c:
        raise SystemExit(
            f"{seed_dir} holds BOTH denoiser*.pt and control_point_generator*.pt; "
            "pass --policy dp or --policy q3c.")
    raise SystemExit(
        f"no policy weights in {seed_dir}: expected denoiser{'{_ema,}'}.pt (DP) "
        f"or control_point_generator{'{_ema,}'}.pt + q_estimator*.pt (Q3C).")


# ---------------------------------------------------------------------------
# Alignment gate
# ---------------------------------------------------------------------------

def alignment_panel(ref_rgb: np.ndarray, live_rgb: np.ndarray, alpha: float,
                    label: str) -> np.ndarray:
    """[ref | live | blend] as one BGR strip, annotated. Shapes are matched."""
    import cv2

    ref = ref_rgb
    if ref.shape[:2] != live_rgb.shape[:2]:
        ref = cv2.resize(ref, (live_rgb.shape[1], live_rgb.shape[0]),
                         interpolation=cv2.INTER_AREA)
    blend = cv2.addWeighted(ref, alpha, live_rgb, 1.0 - alpha, 0.0)
    strip = np.concatenate([ref, live_rgb, blend], axis=1)
    strip = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    w = live_rgb.shape[1]
    for i, text in enumerate((f"{label} DEMO frame 0", f"{label} LIVE",
                              f"{label} BLEND")):
        cv2.putText(strip, text, (10 + i * w, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return strip


def wait_for_alignment(grab_obs, live_frame, ref_frames: dict[int, np.ndarray],
                       alpha: float, use_gui: bool, dump_dir: Path | None) -> None:
    """Block until the operator confirms the T matches the demo's first frame."""
    import cv2

    cams = sorted(ref_frames)
    print("\n=== ALIGNMENT ===")
    print(f"Place the T so the LIVE view matches the DEMO frame 0 on cameras {cams}.")
    if use_gui:
        print("  [Enter]/[space] = confirmed, aligned    [s] = save png    [q] = abort")
    else:
        print(f"  headless: panels are written to {dump_dir}; refresh and inspect.")

    saved = 0
    while True:
        raw_obs = grab_obs()
        strips = [alignment_panel(ref_frames[c], live_frame(raw_obs, c),
                                  alpha, f"cam{c}") for c in cams]
        width = max(s.shape[1] for s in strips)
        strips = [s if s.shape[1] == width else
                  cv2.resize(s, (width, int(s.shape[0] * width / s.shape[1])))
                  for s in strips]
        view = np.concatenate(strips, axis=0)

        if dump_dir is not None:
            cv2.imwrite(str(dump_dir / "alignment.png"), view)

        if not use_gui:
            ans = input("Aligned? [y = start replay / n = re-check / q = abort]: ")
            ans = ans.strip().lower()
            if ans in ("y", "yes", ""):
                return
            if ans in ("q", "quit", "abort"):
                raise SystemExit("aborted at the alignment gate")
            continue

        cv2.imshow("pusht replay alignment", view)
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 10, 32):            # Enter / space
            cv2.destroyAllWindows()
            return
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("aborted at the alignment gate")
        if key == ord("s") and dump_dir is not None:
            out = dump_dir / f"alignment_{saved:02d}.png"
            cv2.imwrite(str(out), view)
            saved += 1
            print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_expert_vs_policy(expert, policy, executed, eef_meas, eef_demo,
                          out_path: Path, title: str) -> None:
    """expert/policy/executed: (T,2) metres. eef_*: (T,2) or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(expert)
    t = np.arange(T)
    valid = np.isfinite(policy).all(axis=1)
    err = policy[valid] - expert[valid]
    mae = np.abs(err).mean(axis=0) if valid.any() else np.array([np.nan, np.nan])

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    for ax, ax_i, name in ((axes[0, 0], 0, "dx"), (axes[0, 1], 1, "dy")):
        ax.plot(t, expert[:, ax_i] * 1000, lw=1.2, label="expert (executed)")
        ax.plot(t[valid], policy[valid, ax_i] * 1000, lw=1.0, alpha=0.85,
                label="policy (shadow)")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel("timestep")
        ax.set_ylabel(f"{name} [mm]")
        ax.set_title(f"{name}: MAE {mae[ax_i] * 1000:.2f} mm")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax = axes[1, 0]
    exp_path = np.cumsum(executed, axis=0)
    ax.plot(exp_path[:, 0] * 100, exp_path[:, 1] * 100, lw=1.5,
            label="expert commands (cumsum)")
    if valid.any():
        # Zero-order hold across un-sampled steps (--shadow-every > 1), so the
        # hypothetical path is not artificially shortened by the missing draws.
        idx = np.where(valid, np.arange(T), 0)
        pol = policy[np.maximum.accumulate(idx)]
        pol[: int(np.argmax(valid))] = 0.0
        pol_path = np.cumsum(pol, axis=0)
        ax.plot(pol_path[:, 0] * 100, pol_path[:, 1] * 100, lw=1.2, alpha=0.85,
                label="policy commands (cumsum, hypothetical)")
    if eef_meas is not None and len(eef_meas):
        m = eef_meas - eef_meas[0]
        ax.plot(m[:, 0] * 100, m[:, 1] * 100, lw=1.2, ls="--",
                label="measured EEF (live)")
    if eef_demo is not None and len(eef_demo):
        dm = eef_demo - eef_demo[0]
        ax.plot(dm[:, 0] * 100, dm[:, 1] * 100, lw=1.2, ls=":",
                label="measured EEF (demo)")
    ax.set_xlabel("dx from start [cm]")
    ax.set_ylabel("dy from start [cm]")
    ax.set_title("planar path")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1, 1]
    if valid.any():
        lim = 1000 * max(np.abs(expert).max(), np.abs(policy[valid]).max(), 1e-6)
        for ax_i, name, marker in ((0, "dx", "o"), (1, "dy", "x")):
            e, p = expert[valid, ax_i] * 1000, policy[valid, ax_i] * 1000
            corr = (np.corrcoef(e, p)[0, 1]
                    if e.std() > 1e-9 and p.std() > 1e-9 else np.nan)
            sign = float((np.sign(e) == np.sign(p)).mean())
            ax.scatter(e, p, s=6, alpha=0.35, marker=marker,
                       label=f"{name}: r={corr:.2f}, sign match={sign:.0%}")
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.legend(fontsize=8)
    ax.set_xlabel("expert [mm]")
    ax.set_ylabel("policy [mm]")
    ax.set_title("per-step agreement")
    ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Plot -> {out_path}")


# ---------------------------------------------------------------------------
# CLI (a copy of deploy_pusht_real_dp.parse_args + the replay-specific knobs)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-ema", action="store_true",
                   help="use raw denoiser instead of the EMA copy")
    p.add_argument("--ip", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--widowx-envs-path", type=Path, default=None)
    p.add_argument("--camera-topics", nargs="+", default=d.CAMERA_TOPICS,
                   help="ROS topics, registered in THIS order. Default = the "
                        "order the training data was collected in "
                        f"({d.DATASET_CAMERA_TOPICS}).")
    p.add_argument("--topic-camera-ids", nargs="+", type=int, default=None,
                   help="dataset camera id of each --camera-topics entry "
                        "(default 0,1,...). Blue-only rig: `--camera-topics "
                        "/blue/image_raw --topic-camera-ids 1 --align-cameras 1`.")

    # --- episode source ------------------------------------------------------
    p.add_argument("--archive", type=Path,
                   default=ROOT / "data" / "pusht_2026_07_zarr.zip",
                   help="zarr_video archive the checkpoint was trained on.")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None,
                   help="truncate the replay (default: the whole episode).")
    p.add_argument("--align-cameras", type=int, nargs="+", default=[0, 1],
                   help="camera indices shown at the alignment gate.")
    p.add_argument("--align-alpha", type=float, default=0.5,
                   help="blend weight of the DEMO frame in the blend panel.")
    p.add_argument("--no-gui", action="store_true",
                   help="no cv2 window: write alignment.png and prompt on stdin.")
    p.add_argument("--skip-alignment", action="store_true",
                   help="skip the alignment gate entirely (not recommended).")

    # --- service image geometry ---------------------------------------------
    p.add_argument("--im-size", type=int, default=480)
    p.add_argument("--im-width", type=int, default=640)

    # --- control -------------------------------------------------------------
    p.add_argument("--step-duration", type=float, default=d.STEP_DURATION,
                   help="control period; also the env move_duration. Default is "
                        "the collection's move_duration (20 Hz).")
    p.add_argument("--non-blocking", action="store_true")
    p.add_argument("--action-mode", default="2trans",
                   choices=["2trans", "3trans", "3trans1rot", "3trans3rot"])
    p.add_argument("--safety-max-xy-delta", type=float, default=d.SAFETY_MAX_XY_DELTA)
    p.add_argument("--min-step-xy", type=float, default=0.0)
    p.add_argument("--match-exposure", action="store_true",
                   help="OFF by default, same as the deploy client.")
    p.add_argument("--exposure-gains", type=float, nargs=3, default=[1.22, 1.18, 1.17],
                   metavar=("R", "G", "B"))
    p.add_argument("--lock-z", dest="lock_z", action="store_true", default=True)
    p.add_argument("--no-lock-z", dest="lock_z", action="store_false")
    p.add_argument("--fixed-z-height", type=float, default=d.FIXED_Z_HEIGHT)
    p.add_argument("--neutral-z-height", type=float, default=d.NEUTRAL_Z_HEIGHT)
    p.add_argument("--z-hold", type=float, default=0.0)
    p.add_argument("--z-hold-gain", type=float, default=1.0)
    p.add_argument("--z-hold-max", type=float, default=0.01)
    p.add_argument("--fixed-gripper", type=float, default=d.FIXED_GRIPPER)
    p.add_argument("--gripper-command", type=float, default=0.0)
    p.add_argument("--skip-move-to-neutral", action="store_true")
    p.add_argument("--i-traj", type=int, default=0)

    # --- initial pose --------------------------------------------------------
    p.add_argument("--move-to-demo-start", dest="move_to_demo_start",
                   action="store_true", default=True)
    p.add_argument("--no-move-to-demo-start", dest="move_to_demo_start",
                   action="store_false")
    p.add_argument("--start-eep-npy", type=Path,
                   default=ROOT / "scripts" / "assets" / "pusht_start_eep.npy",
                   help="4x4 EEF transform; its ROTATION is always used, and its "
                        "translation only when --no-start-from-episode.")
    p.add_argument("--start-from-episode", dest="start_from_episode",
                   action="store_true", default=True,
                   help="translate to THIS episode's first robot_eef_pose.")
    p.add_argument("--no-start-from-episode", dest="start_from_episode",
                   action="store_false")
    p.add_argument("--start-move-duration", type=float, default=1.5)
    p.add_argument("--max-initial-move-retries", type=int, default=5)

    # --- HARD approach guard -------------------------------------------------
    p.add_argument("--approach-floor", dest="approach_floor",
                   action="store_true", default=True)
    p.add_argument("--no-approach-floor", dest="approach_floor",
                   action="store_false")
    p.add_argument("--approach-floor-x", type=float, default=None)

    # --- init / reset robustness --------------------------------------------
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

    # --- which policy runs in the shadow -------------------------------------
    p.add_argument("--policy", default="auto", choices=["auto", "dp", "q3c"],
                   help="shadow policy family. auto = infer from the weight "
                        "files in --seed-dir (denoiser*.pt -> dp, "
                        "control_point_generator*.pt -> q3c).")

    # --- DP sampler knobs (deploy_pusht_real_dp.py) --------------------------
    p.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--ddim-steps", type=int, default=None)
    p.add_argument("--ddim-eta", type=float, default=None)
    p.add_argument("--sample-seed", type=int, default=None)

    # --- Q3C selection knobs (deploy_pusht_real.py) --------------------------
    p.add_argument("--cp-selection", choices=["argmax", "sample"], default=None,
                   help="default: the checkpoint's cp_selection in norm_stats.")
    p.add_argument("--cp-temperature", type=float, default=None,
                   help="default: the checkpoint's cp_selection_temperature.")
    p.add_argument("--inference", choices=["argmax", "sample", "langevin", "dfo"],
                   default="argmax",
                   help="argmax/sample rank the CP cloud; langevin matches the "
                        "TRAINING sampler; dfo is the cheap derivative-free "
                        "refinement. Same semantics as deploy_pusht_real.py.")
    p.add_argument("--refine-iters", type=int, default=50)
    p.add_argument("--langevin-lr-init", type=float, default=0.1)
    p.add_argument("--langevin-lr-final", type=float, default=1e-5)
    p.add_argument("--dfo-noise-init", type=float, default=0.1)
    p.add_argument("--dfo-noise-decay", type=float, default=0.8)

    # --- shadow / diagnostics ------------------------------------------------
    p.add_argument("--shadow-every", type=int, default=1,
                   help="sample the policy every N steps. DP's DDPM at T=100 and "
                        "Q3C's langevin/dfo refinement can both exceed a 0.05 s "
                        "control period on CPU; raise this (or use --sampler "
                        "ddim / --inference argmax) if the loop cannot keep up.")
    p.add_argument("--no-shadow", action="store_true",
                   help="pure open-loop replay, no policy at all (rig check).")
    p.add_argument("--dry-run", action="store_true",
                   help="everything except step_action: no motion.")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="per-step log: steps.jsonl, replay.npz, plot, frames.")
    p.add_argument("--save-frames", action="store_true",
                   help="also dump the fed frames (needs --log-dir).")
    p.add_argument("--plot-out", type=Path, default=None,
                   help="plot path (default: <log-dir>/expert_vs_policy.png).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.z_hold > 0 and args.action_mode == "2trans":
        raise SystemExit(
            "--z-hold needs an action_mode that sends z "
            "(3trans/3trans1rot/3trans3rot); got 2trans.")
    seed_dir = args.seed_dir.resolve()
    log_dir = args.log_dir
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    # --- episode -------------------------------------------------------------
    actions, eef, ends = load_lowdim(args.archive)
    if not 0 <= args.episode < len(ends):
        raise SystemExit(f"--episode must be in [0, {len(ends)}); got {args.episode}")
    ep_start = int(ends[args.episode - 1]) if args.episode > 0 else 0
    ep_end = int(ends[args.episode])
    ep_actions = np.asarray(actions[ep_start:ep_end, :2], dtype=np.float64)
    ep_eef = np.asarray(eef[ep_start:ep_end, :3], dtype=np.float64)
    n_steps = len(ep_actions)
    if args.max_steps is not None:
        n_steps = min(n_steps, int(args.max_steps))
    zero_frac = float((np.linalg.norm(ep_actions, axis=1) == 0).mean())
    print(f"Episode {args.episode} of {args.archive.name}: rows "
          f"[{ep_start}, {ep_end}) = {ep_end - ep_start} steps "
          f"(replaying {n_steps}), zero-action share {zero_frac:.1%}")
    print(f"  demo EEF start={np.round(ep_eef[0], 4)} end={np.round(ep_eef[-1], 4)}")

    # --- checkpoint metadata (identical to the matching deploy client) -------
    backend = resolve_backend(args.policy, seed_dir)
    env_cfg = d.load_run_config(seed_dir)
    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu",
                            weights_only=False)
    act_min = np.asarray(norm_stats["act_min"], np.float32)
    act_max = np.asarray(norm_stats["act_max"], np.float32)
    norm_range = tuple(norm_stats.get("action_norm_range", (-1.0, 1.0)))
    frame_stack = int(norm_stats.get("frame_stack", env_cfg.get("frame_stack", 2)))
    image_hw = norm_stats.get("image_hw",
                              (int(env_cfg.get("image_height", 240)),
                               int(env_cfg.get("image_width", 320))))
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    # The DP trainer records camera_streams as ("video1",); the Q3C runs carry
    # ("images1",) in their run config. Both index the same dataset cameras.
    cams = tuple(norm_stats.get("camera_streams",
                                env_cfg.get("camera_streams", ["video1"])))
    cam_ids = d.camera_ids_from_streams(cams)
    topic_camera_ids = d.resolve_topic_camera_ids(args.camera_topics,
                                                  args.topic_camera_ids)
    in_channels = int(norm_stats.get("in_channels",
                                     3 * len(cam_ids) * frame_stack))
    expected_channels = 3 * len(cam_ids) * frame_stack
    if expected_channels != in_channels:
        raise SystemExit(
            f"checkpoint says in_channels={in_channels} but its camera_streams "
            f"{cams} x frame_stack {frame_stack} imply {expected_channels}.")
    if act_min.size != 2:
        raise SystemExit(
            f"act_min has {act_min.size} entries: this checkpoint predicts an "
            "action chunk, which neither deploy client executes.")

    # EEF (x, y) conditioning: Q3C runs trained with --cond-eef-xy carry it; the
    # DP client has no conditioned path (deploy_pusht_real_dp.build_dp_policy
    # raises), so mirror that limitation instead of inventing one here.
    cond_dim = int(norm_stats.get("cond_dim", 0))
    cond_min = cond_max = None
    if cond_dim:
        if backend == "dp":
            raise SystemExit("conditioned DP checkpoints are not wired up "
                             "(same limitation as deploy_pusht_real_dp.py).")
        if str(norm_stats.get("cond_kind", "")) != "eef_xy":
            raise SystemExit(
                f"cond_dim={cond_dim} cond_kind={norm_stats.get('cond_kind')!r}; "
                "only eef_xy is known")
        cond_min = np.asarray(norm_stats["cond_min"], np.float32)
        cond_max = np.asarray(norm_stats["cond_max"], np.float32)

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    suffix = "" if args.no_ema else "_ema"
    which = "raw" if args.no_ema else "EMA"
    denoiser = diffusion = cp_gen = q_net = None
    ddim_steps, ddim_eta = args.ddim_steps, args.ddim_eta
    cp_selection = args.cp_selection or str(norm_stats.get("cp_selection", "argmax"))
    cp_temp = (args.cp_temperature if args.cp_temperature is not None
               else float(norm_stats.get("cp_selection_temperature", 1.0)))

    if args.no_shadow:
        print("[INFO] --no-shadow: policy never loaded; pure open-loop rig check.")
    elif backend == "dp":
        denoiser, diffusion, dpar = dp.build_dp_policy(
            env_cfg, norm_stats, in_channels, device)
        denoiser.load_state_dict(torch.load(seed_dir / f"denoiser{suffix}.pt",
                                            map_location=device, weights_only=True))
        denoiser.eval()
        if ddim_steps is None:
            ev = norm_stats.get("ddim_eval_steps", dpar.get("ddim_eval_steps", [10]))
            ddim_steps = int(ev[0]) if ev else 10
        if ddim_eta is None:
            ddim_eta = float(norm_stats.get("ddim_eta", dpar.get("ddim_eta", 0.0)))
        if args.sample_seed is not None:
            torch.manual_seed(args.sample_seed)
        print(f"Shadow policy: DP denoiser ({which}) from {seed_dir}")
        print(f"  sampler={args.sampler}"
              + (f" ({ddim_steps} steps, eta={ddim_eta})"
                 if args.sampler == "ddim" else "")
              + f"  pred={dpar.get('prediction_type')} "
                f"T={dpar.get('num_train_timesteps')}")
    else:
        cp_gen, q_net = d.build_models(env_cfg, in_channels, device,
                                       cond_dim=cond_dim)
        d.load_weights(cp_gen, seed_dir / f"control_point_generator{suffix}.pt",
                       device)
        d.load_weights(q_net, seed_dir / f"q_estimator{suffix}.pt", device)
        if args.sample_seed is not None:
            torch.manual_seed(args.sample_seed)
        print(f"Shadow policy: Q3C energy model ({which}) from {seed_dir}")
        print(f"  cp_selection={cp_selection} (temp={cp_temp}) "
              f"inference={args.inference}"
              + (f" ({args.refine_iters} iters)"
                 if args.inference in ("langevin", "dfo") else ""))
    if not args.no_shadow:
        print(f"  frame_stack={frame_stack} cameras={cams} (ids {cam_ids}) "
              f"model_hw=({image_h},{image_w}) in_channels={in_channels} "
              f"device={device}")
        print(f"  act_min={act_min} act_max={act_max} norm_range={norm_range} "
              f"cond_dim={cond_dim}")

    def make_cond(raw_obs):
        """Live EEF (x,y) -> (1,2) normalized, mirroring the Q3C deploy client."""
        if not cond_dim:
            return None
        st = None if raw_obs is None else raw_obs.get("state")
        if st is None:
            raise RuntimeError(
                "checkpoint needs EEF conditioning but the observation has no "
                "'state' field")
        xy = np.asarray(st, np.float32).reshape(-1)[:2]
        span = np.where(cond_max == cond_min, np.ones_like(cond_max),
                        cond_max - cond_min)
        z = np.clip(-1.0 + 2.0 * (xy - cond_min) / span, -1.0, 1.0)
        return torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)

    def sample(obs_u8, raw_obs):
        if backend == "dp":
            return dp.dp_sample_action(diffusion, denoiser, obs_u8, args.sampler,
                                       ddim_steps, ddim_eta, cond=None)
        return d.select_action(
            cp_gen, q_net, obs_u8, cp_selection, cp_temp,
            cond=make_cond(raw_obs), inference=args.inference,
            refine_iters=args.refine_iters,
            langevin_lr=(args.langevin_lr_init, args.langevin_lr_final),
            dfo_noise=(args.dfo_noise_init, args.dfo_noise_decay))

    # --- reference frames ----------------------------------------------------
    align_cams = sorted({int(c) for c in args.align_cameras})
    ref_frames = {}
    if not args.skip_alignment:
        unavailable = [c for c in align_cams if c not in topic_camera_ids]
        if unavailable:
            raise SystemExit(
                f"--align-cameras {unavailable} are not among the registered "
                f"topics (which map to cameras {topic_camera_ids}).")
        for cam in align_cams:
            ref_frames[cam] = load_episode_frames(args.archive, args.episode,
                                                  cam, [0])[0]
        print(f"Demo frame 0 loaded for cameras {align_cams} "
              f"({ref_frames[align_cams[0]].shape})")

    # --- connect (verbatim from deploy_pusht_real_dp.main) -------------------
    WidowXClient, WidowXConfigs, WidowXStatus = d.load_widowx_dependencies(
        args.widowx_envs_path)
    env_params = d.build_env_params(args, WidowXConfigs)
    print(f"Camera topics: {args.camera_topics} -> dataset camera ids "
          f"{topic_camera_ids}; policy reads {cam_ids}")
    print(f"action_mode={args.action_mode} lock_z={args.lock_z} "
          f"fixed_z_height={args.fixed_z_height} move_duration={args.step_duration}")

    client = WidowXClient(host=args.ip, port=args.port)

    reuse_existing_env = False
    if args.reuse_existing_env and not args.force_fresh_init:
        reuse_existing_env = d.widowx_server_has_live_env(client, max_wait_sec=1.0)
        if reuse_existing_env:
            print("[INFO] Server already has a live env; reusing it (skipping init()).")
            print("[WARN] The live env keeps its FIRST env_params -- including its "
                  "camera_topics. If it was started with a different topic list, "
                  "the camera indices below are WRONG; restart the server or pass "
                  "--no-reuse-existing-env.")

    if reuse_existing_env:
        d.set_reqrep_timeout_ms(client, max(1, args.rpc_timeout_ms))
    else:
        init_status = d.init_widowx_with_retry(
            client, env_params, args.im_size, WidowXStatus, args)
        if init_status != WidowXStatus.SUCCESS:
            raise RuntimeError(
                f"WidowX init failed after {args.init_retries} attempts with "
                f"status={d.status_name(init_status, WidowXStatus)}.")
    print("WidowX connection established.")

    reset_status = d.reset_widowx_with_retry(client, WidowXStatus, args, args.i_traj)
    if reset_status != WidowXStatus.SUCCESS:
        raise RuntimeError(
            f"WidowX reset failed with status={d.status_name(reset_status, WidowXStatus)}")
    print(f"Reset done (itraj={args.i_traj}).")

    if args.gripper_command >= 0.0 and hasattr(client, "move_gripper"):
        try:
            gstatus = client.move_gripper(float(args.gripper_command))
            print(f"Gripper commanded to {args.gripper_command} "
                  f"(0=closed,1=open); status={d.status_name(gstatus, WidowXStatus)}")
            time.sleep(1.0)
        except Exception as exc:
            print(f"[WARN] move_gripper({args.gripper_command}) failed: {exc}")

    # --- move to the start pose ---------------------------------------------
    start_T = None
    if args.move_to_demo_start:
        start_path = Path(args.start_eep_npy).expanduser()
        if not start_path.is_file():
            raise FileNotFoundError(f"--start-eep-npy not found: {start_path}")
        start_T = np.load(start_path).astype(np.float32)
        if args.start_from_episode:
            xyz = ep_eef[0]
            if not np.all(np.isfinite(xyz)) or np.abs(xyz).sum() == 0:
                print("[WARN] episode's first robot_eef_pose is all-zero "
                      "(2 episodes in the 2026-07 archive are); using the "
                      "mean-of-demos asset instead.")
            else:
                start_T = start_T.copy()
                start_T[:3, 3] = xyz.astype(np.float32)
        print(f"[INFO] Moving EEF to start pose (x={start_T[0,3]:.4f}, "
              f"y={start_T[1,3]:.4f}, z={start_T[2,3]:.4f})...")
        move_status, tries = None, 0
        while move_status != WidowXStatus.SUCCESS and tries < args.max_initial_move_retries:
            move_status = client.move(start_T, duration=args.start_move_duration)
            tries += 1
        if move_status != WidowXStatus.SUCCESS:
            print(f"[WARN] initial move not SUCCESS after {tries} tries "
                  f"(status={d.status_name(move_status, WidowXStatus)}); continuing.")

    # --- approach floor ------------------------------------------------------
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
        demo_min_x = float(ep_eef[:, 0].min())
        print(f"[SAFETY] Approach floor ARMED: EEF x never below "
              f"{approach_floor_x:.4f} m (this demo's min x is {demo_min_x:.4f} m).")
        if demo_min_x < approach_floor_x - 1e-4:
            print("[WARN] the demo itself goes below the floor; those expert "
                  "steps WILL be clipped. Pass --no-approach-floor for a "
                  "faithful replay.")

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

    def live_frame(raw_obs, cam_id):
        return d.frame_for_camera(raw_obs, cam_id, topic_camera_ids)

    def policy_frames(raw_obs):
        return d.build_stack_frame(raw_obs, cam_ids, topic_camera_ids,
                                   (image_h, image_w), gains=exposure_gains)

    # --- alignment gate ------------------------------------------------------
    if not args.skip_alignment:
        wait_for_alignment(grab_obs, live_frame, ref_frames,
                           float(args.align_alpha), use_gui=not args.no_gui,
                           dump_dir=log_dir)
        print("Alignment confirmed.")

    # --- warm up the frame buffer (deploy semantics: pad with the first frame)
    frame_buf = collections.deque(maxlen=frame_stack)
    first = policy_frames(grab_obs())
    print(f"Stacked frame per timestep: {first.shape} (cameras {cam_ids})")
    for _ in range(frame_stack):
        frame_buf.append(first)

    # --- replay --------------------------------------------------------------
    log_fh = None
    if log_dir is not None:
        log_fh = (log_dir / "steps.jsonl").open("w")
        if args.save_frames:
            (log_dir / "fed").mkdir(parents=True, exist_ok=True)
        print(f"Log -> {log_dir}")

    blocking = not args.non_blocking
    print(f"\nOpen-loop replay of {n_steps} EXPERT steps, blocking={blocking}, "
          f"step_duration={args.step_duration}s"
          + ("  [DRY RUN: no motion]" if args.dry_run else "")
          + ("" if args.no_shadow else
             f", {backend.upper()} policy in shadow every "
             f"{args.shadow_every} step(s)")
          + ". Keep a hand on the E-stop.")
    input("Press [Enter] to start.")

    expert_log = np.full((n_steps, 2), np.nan)
    executed_log = np.full((n_steps, 2), np.nan)
    policy_log = np.full((n_steps, 2), np.nan)
    eef_log = np.full((n_steps, 2), np.nan)
    sample_ms: list[float] = []
    step = -1
    last_exec = time.time()

    try:
        for step in range(n_steps):
            raw_obs = grab_obs()
            frame_buf.append(policy_frames(raw_obs))

            # --- SHADOW: the policy sees exactly what it would at deploy -----
            na = None
            if not args.no_shadow and step % max(1, args.shadow_every) == 0:
                obs_u8 = d.stack_to_tensor(frame_buf, device)
                t0 = time.time()
                na = sample(obs_u8, raw_obs)
                sample_ms.append((time.time() - t0) * 1000.0)
                policy_log[step] = d.unnormalize(na, act_min, act_max, norm_range)

            # --- EXECUTED: the expert action, through the deploy pipeline ----
            act_xy = ep_actions[step].copy()
            expert_log[step] = act_xy

            act_xy, snapped = d.apply_min_step(act_xy, args.min_step_xy)
            cur_x = d.eef_x_from_obs(raw_obs)
            act_xy, floored = d.apply_approach_floor(act_xy, cur_x, approach_floor_x)
            if floored:
                print(f"[SAFETY] approach floor: clipped dx at x={cur_x:.4f} "
                      f"(floor={approach_floor_x:.4f})")

            action_7d = d.to_action_7d(act_xy, args.fixed_gripper)
            action_7d = d.safety_clip_action(action_7d, args.action_mode,
                                             args.safety_max_xy_delta)
            if args.z_hold > 0:
                action_7d[2] = d.z_hold_dz(d.z_from_obs(raw_obs), args.z_hold,
                                           args.z_hold_gain, args.z_hold_max)
            env_action = d.project_action_to_env_mode(action_7d, args.action_mode)
            executed_log[step] = action_7d[:2]

            st = raw_obs.get("state")
            if st is not None:
                sv = np.ravel(np.asarray(st, dtype=np.float64))
                if sv.size >= 2:
                    eef_log[step] = sv[:2]

            if not args.dry_run:
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
            else:
                time.sleep(args.step_duration)
                last_exec = time.time()

            if step % 20 == 0 or snapped or floored:
                pol = ("      -" if not np.isfinite(policy_log[step]).all()
                       else str(np.round(policy_log[step] * 1000, 2)))
                print(f"[{step:04d}/{n_steps}] expert(mm)="
                      f"{np.round(expert_log[step] * 1000, 2)} "
                      f"policy(mm)={pol} env_action={np.round(env_action, 5)}")

            if log_fh is not None:
                if args.save_frames:
                    d.save_fed_png(log_dir / "fed" / f"{step:04d}",
                                   list(frame_buf)[-1], cam_ids)
                log_fh.write(json.dumps({
                    "step": step,
                    "t": time.time(),
                    "expert": expert_log[step].tolist(),
                    "executed": executed_log[step].tolist(),
                    "policy_norm": (None if na is None
                                    else [float(x) for x in np.ravel(na)]),
                    "policy": (None if not np.isfinite(policy_log[step]).all()
                               else policy_log[step].tolist()),
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

    done = step + 1
    print(f"\nReplay stopped after {done} steps.")
    if done <= 0:
        return 1

    expert_log = expert_log[:done]
    executed_log = executed_log[:done]
    policy_log = policy_log[:done]
    eef_log = eef_log[:done]

    if sample_ms:
        arr = np.asarray(sample_ms)
        print(f"Shadow sampling: {arr.mean():.1f} ms mean / {arr.max():.1f} ms max "
              f"over {len(arr)} draws (control period "
              f"{args.step_duration * 1000:.0f} ms)")
        if arr.mean() > args.step_duration * 1000:
            cheaper = ("--sampler ddim" if backend == "dp" else "--inference argmax")
            print("[WARN] sampling is slower than the control period, so the "
                  f"replay ran slower than the demo. Use {cheaper} or raise "
                  "--shadow-every for a rate-faithful replay.")

    clipped = int(np.sum(np.abs(executed_log - expert_log) > 1e-9))
    print(f"Expert steps altered by the deploy pipeline: {clipped}")
    valid = np.isfinite(policy_log).all(axis=1)
    if valid.any():
        err = policy_log[valid] - expert_log[valid]
        print(f"Shadow policy vs expert over {int(valid.sum())} steps: "
              f"MAE dx={np.abs(err[:, 0]).mean() * 1000:.2f} mm, "
              f"dy={np.abs(err[:, 1]).mean() * 1000:.2f} mm; "
              f"expert |a| mean={np.linalg.norm(expert_log[valid], axis=1).mean() * 1000:.2f} mm, "
              f"policy |a| mean={np.linalg.norm(policy_log[valid], axis=1).mean() * 1000:.2f} mm")

    eef_meas = eef_log if np.isfinite(eef_log).all(axis=1).any() else None
    if eef_meas is not None:
        eef_meas = eef_meas[np.isfinite(eef_meas).all(axis=1)]
    plot_out = args.plot_out
    if plot_out is None:
        plot_out = ((log_dir or ROOT / "results" / f"replay_ep{args.episode}")
                    / "expert_vs_policy.png")
    plot_expert_vs_policy(
        expert_log, policy_log, executed_log, eef_meas, ep_eef[:done, :2], plot_out,
        f"episode {args.episode} of {args.archive.name} — expert (executed) vs "
        f"{backend.upper()} {seed_dir.name} (shadow)")

    if log_dir is not None:
        np.savez(log_dir / "replay.npz", expert=expert_log, executed=executed_log,
                 policy=policy_log, eef_live=eef_log, eef_demo=ep_eef[:done],
                 policy_backend=backend,
                 episode=args.episode, step_duration=args.step_duration)
        print(f"Arrays -> {log_dir / 'replay.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
