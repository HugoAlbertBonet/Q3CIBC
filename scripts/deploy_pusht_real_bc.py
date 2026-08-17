#!/usr/bin/env python3
"""Deploy a trained EXPLICIT BC (MSE) Push-T policy on the real WidowX arm.

Explicit-BC counterpart of deploy_pusht_real.py (Q3C). The robot-facing half --
everything from the WidowX handshake down to the safety clipping -- is that
script UNCHANGED, so a BC rollout and a Q3C rollout differ only in what produced
the (dx, dy). What is gone is the entire search apparatus: there is no energy
net to rank candidates, so no --inference mode, no --refine-iters, no Langevin
or DFO refinement, and no argmax stall escape (the escape existed because
ranking a finite cloud has no way out of a fixed point; a regression head simply
outputs whatever it outputs). One forward pass per action, always.

Checkpoints come from scripts/train_pusht_real_bc.py: bc_policy_ema.pt (default;
--no-ema for the raw weights), rebuilt as PixelControlPointGenerator with
control_points=1 -- the same encoder + head module the Q3C client builds, with a
cloud of one.

Stalling, for the record: an MSE policy trained on an archive where 24% of the
actions are (0,0) can regress toward zero and freeze the arm just as the energy
policies could. --idle-filter drop_zero at TRAIN time is the fix; at deploy time
--min-step-xy is the only lever here, and it only rescues commands that are
small-but-nonzero.

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
  * ``reset()`` is what places the arm. NOTE: ``--i-traj`` cannot reach the
    server -- ``WidowXClient.reset()`` takes no argument, so the server always
    calls ``bridge_env.reset()`` with ``itraj=None``, which runs
    ``move_to_startstate()`` -> ``env_params["start_state"]``. The collection
    ran the OTHER branch (``reset(itraj=int)`` with ``move_to_rand_start_freq
    == -1``), which never moves off neutral, so its demos start at the neutral
    pose, x~0.117. We therefore set ``start_state`` to that same pose (see
    ``demo_start_state``) instead of leaving the ``WidowXConfigs`` default of
    (0.3, 0.0), which is ~18 cm further from the base than any demo.
  * ``blocking=True`` by default.

Policy-specific (differs from the BFN reference by necessity):
  * Observation: uint8 [0,255], resized to (image_height, image_width) with
    INTER_AREA and channel-CONCATENATED oldest->newest into (3*frame_stack,H,W),
    reproducing utils.datasets.PushTRealPixelsDataset. Not float[0,1], not
    stacked on a new axis.
  * The policy emits a normalized (dx, dy); it is min-max denormalized with
    norm_stats (act_min/act_max) before being sent.
  * Camera: the checkpoint's ``camera_streams`` index IS the dataset camera id
    (``images1``/``video1`` -> 1 -> ``/blue/image_raw``). Topics are registered
    in the collection's order (D435 first, blue second), and the live frame is
    picked by that camera's POSITION in the topic list -- index 0 arrives as
    ``external_img``, index 1 as ``over_shoulder_img``. One- and two-camera
    checkpoints both work; a rig without the D435 needs
    ``--camera-topics /blue/image_raw --topic-camera-ids 1``.

Chunked checkpoints: ``--action-chunk 16`` runs predict 16 planar deltas at once
(``act_min`` has size 2*K). ``--exec-horizon`` picks how many of them execute
open-loop before re-predicting; it is clipped to the checkpoint's chunk length,
so it is a no-op on a single-step checkpoint.

Usage (server already up):

    python scripts/deploy_pusht_real_bc.py \
        --seed-dir checkpoints/pusht_real_bc/BC2c_ch16 \
        --device cpu --dry-run
    python scripts/deploy_pusht_real_bc.py \
        --seed-dir checkpoints/pusht_real_bc/BC2c_ch16 \
        --steps 200 --exec-horizon 8 --measure --start-position top

Scoring: ``--measure`` reads the final frame of every registered camera through
measure_target_coverage.py and appends one row per episode to
``results/pusht/experiments.csv`` (``--results-csv``), recording the algorithm
(``bc``, the one this script deploys), the checkpoint, the inference settings,
whether ``--control-z`` was on, ``--start-position``, the coverage each camera
saw, and the centroid error. The table is shared with the Q3C / IBC / DP
clients, which is the whole point: the ``inference`` column reads
``deterministic`` and ``refine_iters`` 0 for every BC row, because there is
nothing to tune there.
Repeating a parameter combination adds a row with the next ``trial`` number
instead of overwriting the previous one. Scoring runs even if the episode is
interrupted, and a failure there never fails the run.

Inference cost: every run also times the policy call itself -- just the call, so
robot I/O and camera reads stay out of it, with a CUDA synchronize on each side
when the model is on the GPU -- and appends one row per episode to
``results/pusht/inference_speed.csv`` (``--speed-csv``, off with
``--no-speed-csv``). The row carries the ms/inference distribution (mean, median,
p95, std, warm-up call reported separately), the amortized ms per control step,
the sequential net evaluations behind one action (always 1 here -- that is the
headline number for this baseline), GFLOPs per action (``--no-flops`` skips the
one extra call this costs, after the arm has stopped)
and the parameter count. It is a separate table from experiments.csv on purpose:
that one is keyed by start position and its existing rows predate any timing, so
speed columns there would be blank everywhere and only a full re-run could fill
them. These rows repeat the seven key columns plus ``trial``, so a join puts a
timing back next to the episode that produced it.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import statistics
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
# Inherited box. It does NOT contain the Push-T demonstrations: measured over
# the 2026-07 collection (151 episodes, all-zero capture rows excluded) the
# recorded EEF spans x [0.080, 0.479], y [-0.360, 0.293], z [0.020, 0.022], so
# 23.4% of demo steps sit outside the y limits and 1.7% outside x. Commanding
# the arm past the boundary pins it against the wall. Worse, the z floor here is
# -0.01 m, i.e. 30 mm BELOW the 0.02 m working height, so this guard does
# nothing to stop the wrist sagging onto the table.
LEGACY_WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0],
                           [0.45, 0.25, 0.25, 1.57, 0]]
# Default: the measured demo envelope plus ~2 cm of xy margin, with the z floor
# raised to just under the working height. Every pose inside this box was
# actually executed by this arm during collection.
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0],
                           [0.45, 0.25, 0.25, 1.57, 0]]
# Camera topics in the order the DATA COLLECTION registered them (the archive's
# metadata.json "provenance" block). The list index IS the dataset camera id:
# 0 == images0/video0 == D435, 1 == images1/video1 == blue scene cam. Register
# them in this order at deploy so a checkpoint's camera_streams index means the
# same thing live as it did in training. If a camera is physically absent, drop
# it AND say which ids remain with --topic-camera-ids (e.g. blue-only rig:
# `--camera-topics /blue/image_raw --topic-camera-ids 1`).
DATASET_CAMERA_TOPICS = ["/D435/color/image_raw", "/blue/image_raw"]
CAMERA_TOPICS = list(DATASET_CAMERA_TOPICS)
FIXED_Z_HEIGHT = 0.026
NEUTRAL_Z_HEIGHT = FIXED_Z_HEIGHT
FIXED_GRIPPER = 0.0
# The demo archive's actions are ±0.008 in x/y; the working script clips at the
# same magnitude via vr_xy_step_clip.
SAFETY_MAX_XY_DELTA = 0.008
# 4x4 EEF transform of the demo start pose (mean over the archive's episodes;
# x~0.117, y~-0.019). This is the collection's neutral pose: every demo's first
# robot_eef_pose sits within a millimetre of it.
START_EEP_NPY = ROOT / "scripts" / "assets" / "pusht_start_eep.npy"
# The collection loop ran at move_duration = 0.05 s (20 Hz) -- see the archive
# provenance and PUSHT_DATA_COLLECTION_RUNBOOK.md. Deploying at 0.1 s replays
# every learned delta at half the demonstrated speed, so the default matches the
# data.
STEP_DURATION = 0.05
# Where --measure appends its rows.
RESULTS_CSV = ROOT / "results" / "pusht" / "experiments.csv"
RESULTS_COLUMNS = ["algorithm", "seed_dir", "inference", "refine_iters",
                   "exec_horizon", "control_z", "start_position", "trial",
                   "coverage_cam0", "coverage_cam1", "dist_centroid"]
# A row's identity: another run with all seven equal gets the next trial number
# rather than overwriting the old one. exec_horizon belongs here because a
# re-predict every step and a re-predict every 8 steps are different control
# policies over the same weights, not repeats of one condition.
RESULTS_KEY = ("algorithm", "seed_dir", "inference", "refine_iters",
               "exec_horizon", "control_z", "start_position")
# What to write into a column that a pre-existing table has never heard of.
# Every row already in such a table was written before the flag was recorded,
# by this script, with the z loop off and one action executed per prediction
# (--exec-horizon's default, and the only value an unchunked checkpoint allows
# since chunk_len is then 1 and the flag is clamped to it).
RESULTS_LEGACY_DEFAULTS: Dict[str, Any] = {"algorithm": "q3c", "control_z": False,
                                           "exec_horizon": 1}
# This script deploys the explicit BC (MSE) baseline. Q3C is in
# deploy_pusht_real.py, IBC in deploy_pusht_real_ibc.py, DP in
# deploy_pusht_real_dp.py; the column keeps all four readable from one table.
ALGORITHM = "bc"
# What goes in the shared table's `inference` column. An explicit policy has one
# inference mode -- a forward pass -- so this is a constant rather than a flag,
# and `refine_iters` is pinned to 0 alongside it.
INFERENCE_LABEL = "deterministic"
REFINE_ITERS = 0
# Sequential net evaluations behind one action. One encoder+head pass; no
# candidate ranking, no refinement loop.
NET_EVALS_PER_ACTION = 1

# Where --measure appends the cost of the policy call. Deliberately NOT more
# columns on experiments.csv: that table is one row per scored rollout and its
# 300+ existing rows were all written before any clock ran here, so new columns
# there would be blank for every one of them and could only be filled by
# re-running the entire sweep -- to recover a number that does not depend on the
# start position or on the block ending up in the goal. Speed rows carry the
# full RESULTS_KEY plus `trial`, so joining on those eight columns reattaches a
# timing to the episode that produced it whenever both were written.
SPEED_CSV = ROOT / "results" / "pusht" / "inference_speed.csv"
SPEED_COLUMNS = ["algorithm", "seed_dir", "inference", "refine_iters",
                 "control_z", "start_position", "trial", "device",
                 "n_infer", "n_steps", "exec_horizon",
                 "ms_per_infer_mean", "ms_per_infer_median", "ms_per_infer_p95",
                 "ms_per_infer_std", "ms_first_infer", "ms_per_step",
                 "infer_hz", "net_evals_per_infer", "gflops_per_infer",
                 "params_m"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-ema", action="store_true",
                   help="use raw weights instead of the EMA copy")
    p.add_argument("--ip", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--widowx-envs-path", type=Path, default=None,
                   help="OPTIONAL path prepended to sys.path before importing "
                        "widowx_envs. Default: do NOT touch sys.path and import "
                        "whatever is installed in this env (pip -e), which is what "
                        "the previously-working client did. Only set this if the "
                        "installed package is NOT the one the server runs -- "
                        "pointing it at a second copy changes "
                        "WidowXConfigs.DefaultActionConfig and the edgeml handshake "
                        "then fails with 'Incompatible config with hash'.")
    p.add_argument("--camera-topics", nargs="+", default=CAMERA_TOPICS,
                   help="ROS topics, registered in THIS order. Default = the "
                        "order the training data was collected in "
                        f"({DATASET_CAMERA_TOPICS}).")
    p.add_argument("--topic-camera-ids", nargs="+", type=int, default=None,
                   help="dataset camera id of each --camera-topics entry "
                        "(default 0,1,...). The checkpoint's camera_streams "
                        "(images1/video1 -> id 1) are resolved through this map, "
                        "so a blue-only rig needs `--camera-topics "
                        "/blue/image_raw --topic-camera-ids 1`.")

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
    p.add_argument("--step-duration", type=float, default=STEP_DURATION,
                   help="control period; also used as env move_duration. Default "
                        "is the collection's move_duration (20 Hz).")
    p.add_argument("--non-blocking", action="store_true",
                   help="the working reference uses blocking=True; this opts out")
    p.add_argument("--action-mode", default="2trans",
                   choices=["2trans", "3trans", "3trans1rot", "3trans3rot"])
    p.add_argument("--safety-max-xy-delta", type=float, default=SAFETY_MAX_XY_DELTA)
    p.add_argument("--workspace-xyz", type=float, nargs=6, default=None,
                   metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                   help="override the server's workspace box (metres). Default "
                        f"{[WORKSPACE_BOUNDS[0][:3], WORKSPACE_BOUNDS[1][:3]]} = "
                        "the measured demo envelope + margin. The legacy box "
                        f"was {[LEGACY_WORKSPACE_BOUNDS[0][:3], LEGACY_WORKSPACE_BOUNDS[1][:3]]}, "
                        "which excluded 23%% of the demo steps and let z sag "
                        "30 mm below the working height. Only applied on init().")
    p.add_argument("--min-step-xy", type=float, default=0.0,
                   help="metres. If >0, any nonzero |dx|/|dy| below this is "
                        "snapped UP to it (sign kept); exact 0 stays 0. The "
                        "expert teleop is bang-bang (0 or >=1.5mm; measured "
                        "dead zone in (0,1.5mm)), and an MSE policy regressing "
                        "the mean of that bimodal target lands inside the dead "
                        "zone -- a command the arm can't execute, so it locks. "
                        "Suggested 0.0015. Default 0 = off.")
    p.add_argument("--lock-z", dest="lock_z", action="store_true", default=True)
    p.add_argument("--no-lock-z", dest="lock_z", action="store_false")
    p.add_argument("--fixed-z-height", type=float, default=FIXED_Z_HEIGHT)
    p.add_argument("--neutral-z-height", type=float, default=NEUTRAL_Z_HEIGHT)
    p.add_argument("--control-z", type=float, nargs="?", const=FIXED_Z_HEIGHT,
                   default=None, metavar="HEIGHT",
                   help="hold the MEASURED EEF z at HEIGHT metres (bare flag = "
                        "--fixed-z-height) with a client-side integrating loop. "
                        "Implies action_mode=3trans and lock_z=False, and forces "
                        "a fresh server init. Rationale, from the widowx_envs "
                        "source: the env's z lock rebuilds locked_z from "
                        "fixed_z_height every step, so it is a pure P term whose "
                        "total authority is capped at fixed_z_height + "
                        "z_lock_max_delta and which leaves s/(1+gain) of any "
                        "droop s. With lock_z=False the target z instead "
                        "accumulates through _previous_target_qpos, so a per-step "
                        "dz integrates -- that is the only path in this stack "
                        "with integral action, and the only one that can zero an "
                        "x-dependent droop. NOTE 3trans makes the env raise "
                        "Environment_Exception if the EEF leaves the plane "
                        "(|roll| or |pitch| > 0.2 rad), and the env skips its own "
                        "xy_action_deadband outside 2trans, so this client "
                        "applies it instead.")
    p.add_argument("--control-z-gain", type=float, default=0.5,
                   help="gain on (target - measured z) for --control-z. The loop "
                        "integrates through the env's target z and the "
                        "measurement lags one control step, so the error "
                        "dynamics are lambda^2 - lambda + gain = 0: |lambda| = "
                        "sqrt(gain) for complex roots, i.e. STABLE ONLY BELOW "
                        "1.0. Simulated at gain 1.0 this rings at 11 mm "
                        "peak-to-peak forever; 0.5 settles clean. Do not raise "
                        "this above ~0.8.")
    p.add_argument("--control-z-max-dz", type=float, default=0.001,
                   help="metres, per-step |dz| clip for --control-z. A true rate "
                        "limit, not a cap on the achievable correction: the "
                        "target accumulates, so this only sets how FAST z can "
                        "climb (0.001 at 20 Hz = 20 mm/s). Sizing: the measured "
                        "droop is ~82 mm per metre of x, so an arm crossing "
                        "x=0.117->0.47 at the max demo rate (8 mm/step) needs "
                        "0.64 mm/step just to keep up -- below ~0.0007 the loop "
                        "falls behind during a fast traverse and never catches "
                        "up while moving. 0.002 doubles the margin at no cost; "
                        "0.001 tracks with the rate limit active ~53%% of a "
                        "worst-case traverse. Recovering 34 mm from a standing "
                        "start takes 67 steps (3.4 s) at 0.001, 1.7 s at 0.002.")
    p.add_argument("--control-z-windup", type=float, default=0.04,
                   help="metres. Anti-windup: the commanded z target is clamped "
                        "to --control-z +/- this, which also caps the largest "
                        "droop the loop can cancel. Default 0.04 covers the "
                        "~34 mm droop measured at x=0.47 with margin. Without "
                        "the clamp, an arm that cannot reach the target (torque "
                        "saturation) integrates upward without bound and leaps "
                        "when the load releases.")
    p.add_argument("--z-hold", type=float, default=0.0,
                   help="metres. If >0, inject a per-step dz to actively hold "
                        "the measured EEF z at this target, compensating the "
                        "x-dependent cantilever droop (measured corr(x,z)=-0.97; "
                        "demos held ~0.0197). REQUIRES an action_mode that sends "
                        "z (3trans/3trans1rot/3trans3rot) AND a widowx_env_service "
                        "relaunched with a matching 3-dim action space -- the live "
                        "env asserts action shape (2,) otherwise. dz is injected, "
                        "NOT from the 2trans-trained policy. Suggested 0.0197. "
                        "Default 0 = off.")
    p.add_argument("--z-hold-gain", type=float, default=1.0,
                   help="proportional gain on (z_target - cur_z) for --z-hold.")
    p.add_argument("--z-hold-max", type=float, default=0.01,
                   help="metres, per-step |dz| clip for --z-hold.")
    p.add_argument("--fixed-gripper", type=float, default=FIXED_GRIPPER,
                   help="gripper target for the 2trans env (0.0 = CLOSED, 1.0 = "
                        "OPEN, per the WidowX SDK convention).")
    p.add_argument("--gripper-command", type=float, default=0.0,
                   help="explicitly actuate the gripper to this value after reset "
                        "(0.0 = closed to hold the pusher). The env's fixed_gripper "
                        "only sets the target; reset can leave the gripper open, so "
                        "this move_gripper() call is what physically closes the "
                        "clamp. Set to a negative number to skip the command.")
    p.add_argument("--skip-move-to-neutral", action="store_true")
    p.add_argument("--i-traj", type=int, default=0,
                   help="trajectory index passed to reset(itraj=N).")

    # --- initial pose (matches deploy_pusht_real_ibc.py) --------------------
    p.add_argument("--move-to-demo-start", dest="move_to_demo_start",
                   action="store_true", default=True,
                   help="after reset, move the EEF to the demo start pose in "
                        "--start-eep-npy (same as the ibc deploy).")
    p.add_argument("--no-move-to-demo-start", dest="move_to_demo_start",
                   action="store_false")
    p.add_argument("--start-eep-npy", type=Path, default=START_EEP_NPY,
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
    p.add_argument("--force-fresh-init", action="store_true",
                   help="always call init(), even if the server already has a live "
                        "env. Only works if the server's cached env_params match "
                        "ours; otherwise it fails the config-hash check and the "
                        "server must be restarted.")
    p.add_argument("--no-reuse-existing-env", dest="reuse_existing_env",
                   action="store_false", default=True,
                   help="disable reusing an already-initialized server env")

    # --- policy --------------------------------------------------------------
    # An explicit policy has no inference knobs: the action IS the forward pass.
    # (Q3C's --inference / --refine-iters / --cp-selection / stall-escape flags
    # are deliberately absent rather than accepted-and-ignored, so a command
    # line copied from the Q3C client fails loudly instead of silently doing
    # something else.)
    p.add_argument("--ckpt-step", type=int, default=0,
                   help="load bc_policy[_ema]_step{N:06d}.pt instead of the "
                        "final bc_policy[_ema].pt (for a seed still training).")
    p.add_argument("--match-exposure", action="store_true",
                   help="lift the live frame to the training white point/exposure "
                        "(deploy scene measured ~16%% dimmer; washes out the red "
                        "T). Applies per-channel gains in preprocess. Default gains "
                        "(1.22,1.18,1.17) = train_board/deploy_board; override with "
                        "--exposure-gains.")
    p.add_argument("--exposure-gains", type=float, nargs=3, default=[1.22, 1.18, 1.17],
                   metavar=("R", "G", "B"),
                   help="per-channel gains for --match-exposure.")
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
                   help="no motion: dump fed frames + print predicted actions")
    p.add_argument("--dry-run-steps", type=int, default=20)
    p.add_argument("--dump-dir", type=Path, default=ROOT / "deploy_dryrun")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="per-step forensic log: raw/*.npy, fed/*.png, steps.jsonl")
    p.add_argument("--measure", action="store_true",
                   help="after the episode, score the final frame with "
                        "measure_target_coverage.py and append a row to "
                        "--results-csv")
    p.add_argument("--start-position", default="top",
                   help="where the block started; recorded in the results CSV")
    p.add_argument("--algorithm", default=ALGORITHM,
                   help="algorithm label written as the first CSV column")
    p.add_argument("--results-csv", type=Path, default=RESULTS_CSV,
                   help="results table appended to by --measure (created, with "
                        "its parent directories, if missing)")
    p.add_argument("--speed-csv", type=Path, default=SPEED_CSV,
                   help="inference-cost table: one row per episode with the "
                        "ms/inference distribution, ms/control-step, GFLOPs and "
                        "parameter count. Joins to --results-csv on the seven key "
                        "columns plus trial")
    p.add_argument("--no-speed-csv", action="store_true",
                   help="time the policy for the console line but do not append "
                        "a row to --speed-csv")
    p.add_argument("--no-flops", action="store_true",
                   help="skip the FLOP count, which costs one extra policy call "
                        "after the episode ends")
    return p.parse_args()


# ---------------------------------------------------------------------------
# WidowX plumbing (mirrors data/eval_widowx_bfn.py)
# ---------------------------------------------------------------------------

def load_widowx_dependencies(widowx_envs_path: Path | None):
    # Default: import whatever widowx_envs is installed in this environment.
    # Prepending a source directory can shadow it with a second copy whose
    # DefaultActionConfig hashes differently than the server's.
    if widowx_envs_path is not None:
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


def demo_start_state(args) -> List[float] | None:
    """``start_state`` that makes the server's reset() land on the demo start.

    The server always resets with ``itraj=None`` (see the module docstring), so
    ``move_to_startstate()`` runs and drives the arm to ``start_state``. The
    ``WidowXConfigs`` default is ``[0.3, 0.0, 0.15, ...]`` -- ~18 cm further
    from the base than any demo -- so the arm swings out across the board and
    only comes back on the ``--move-to-demo-start`` move that follows.

    The state layout is ``[x, y, z, roll, pitch, yaw, gripper]`` and
    ``state2transform`` builds the rotation as ``euler(rpy) @ default_rot``.
    The asset's rotation IS ``default_rot`` (``DEFAULT_ROTATION``, with
    ``workspace_rotation_angle_z == 0``), so zero rpy reproduces the asset's
    transform exactly -- this is the same target the explicit ``client.move``
    sends, just applied one step earlier. ``move_to_startstate`` overwrites z
    with ``fixed_z_height`` while ``lock_z`` is on; we pass it anyway so the
    value is right if z-lock is ever disabled.

    Returns None to keep the stock default (flag off, or asset missing).
    """
    # Callers that share this builder (deploy_pusht_real_dp.py,
    # replay_pusht_episode.py) have no --demo-start-state of their own, so they
    # follow their --move-to-demo-start instead.
    if not getattr(args, "demo_start_state",
                   getattr(args, "move_to_demo_start", True)):
        return None
    path = Path(getattr(args, "start_eep_npy", None) or START_EEP_NPY).expanduser()
    if not path.is_file():
        print(f"[WARN] {path} not found; reset() keeps the stock start_state "
              "(0.3, 0.0), ~18 cm outside the demos' start distribution.")
        return None
    xyz = np.load(path).astype(np.float64)[:3, 3]
    start_z = float(xyz[2])
    control_z = getattr(args, "control_z", None)
    if control_z:
        # --control-z turns lock_z off, and move_to_startstate only rewrites the
        # start z when lock_z is on (widowx_env.py:76-77). Start at the height we
        # intend to hold, or the integrator has to climb there from the asset's z.
        start_z = float(control_z)
    return [float(xyz[0]), float(xyz[1]), start_z, 0.0, 0.0, 0.0,
            float(getattr(args, "fixed_gripper", FIXED_GRIPPER))]


def workspace_bounds_from_args(args) -> list:
    """Bounds to send to the server; --workspace-xyz overrides the xyz limits."""
    bounds = [list(WORKSPACE_BOUNDS[0]), list(WORKSPACE_BOUNDS[1])]
    xyz = getattr(args, "workspace_xyz", None)
    if xyz:
        if len(xyz) != 6:
            raise SystemExit(
                "--workspace-xyz takes 6 numbers: x0 y0 z0 x1 y1 z1")
        for i in range(3):
            bounds[0][i], bounds[1][i] = float(xyz[i]), float(xyz[3 + i])
    for i, ax in enumerate("xyz"):
        if bounds[0][i] >= bounds[1][i]:
            raise SystemExit(
                f"workspace {ax} lower bound {bounds[0][i]} >= upper "
                f"{bounds[1][i]}")
    return bounds


def build_env_params(args, WidowXConfigs) -> Dict[str, Any]:
    """Exactly the dict the confirmed-working BFN eval sends."""
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update({
        "camera_topics": [{"name": t} for t in args.camera_topics],
        "override_workspace_boundaries": workspace_bounds_from_args(args),
        "move_duration": args.step_duration,
        "action_mode": args.action_mode,
        "skip_move_to_neutral": bool(args.skip_move_to_neutral),
        "move_to_rand_start_freq": -1,
        "fix_zangle": 0.1,
        "adaptive_wait": True,
        "fixed_z_height": float(args.fixed_z_height),
        "neutral_z_height": float(args.neutral_z_height),
        # The COLLECTION values, read from the per-session config.json inside
        # data/pusht_2026_07.zip -- NOT from conf_clam_pusht.py, which is the
        # stale snapshot (it also claims move_duration 0.08 when the data
        # measures 0.0503). 15 of 16 sessions ran gain 0.4 / max_delta 0.02 /
        # deadband 0.001; only the first (2026-07-27) used 0.2 / 0.0015 / 0.002.
        # Do not "restore" those older numbers.
        #
        # Note the correction does NOT accumulate: locked_z is rebuilt from
        # fixed_z_height every step, so max_delta caps the TOTAL correction at
        # fixed_z_height + max_delta, it is not a per-step rate limit.
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
    start_state = demo_start_state(args)
    if start_state is not None:
        env_params["start_state"] = start_state
    return env_params


def widowx_server_has_live_env(client, max_wait_sec: float = 1.0,
                               poll_interval_sec: float = 0.1) -> bool:
    """Best-effort probe for an already-initialized server-side env.

    The server caches the env it was initialized with. Calling init() again with
    DIFFERENT env_params is rejected with "Incompatible config with hash with
    server" -- the stale config lives in the server, not the client, so no
    client-side change can fix it. If an env is already live we skip init() and
    just reset, exactly like data/eval_widowx_bfn.py does.
    """
    deadline = time.monotonic() + max(0.1, float(max_wait_sec))
    poll_interval_sec = max(0.01, float(poll_interval_sec))
    while time.monotonic() < deadline:
        try:
            raw_obs = client.get_observation()
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


# --- camera resolution by dataset id ---------------------------------------
# widowx_env_service returns the registered cameras positionally: the FIRST
# camera_topic arrives as external_img, the second as over_shoulder_img, the
# third as wrist_img (see data/eval_widowx_bfn.py::_select_rgb_source_for_obs_key
# and the same mapping in scripts/deploy_pusht_real_ibc.py). There is
# deliberately NO cross-position fallback here: with both collection topics
# registered, falling back from over_shoulder_img to external_img would silently
# feed the D435 to a checkpoint trained on the blue camera.
_POSITION_KEYS = (
    ("external_img", "full_image_0", "image_0"),
    ("over_shoulder_img", "full_image_1", "image_1"),
    ("wrist_img", "full_image_2", "image_2"),
)


def camera_ids_from_streams(streams) -> List[int]:
    """["video1"] / ["images0", "images1"] -> [1] / [0, 1]."""
    ids = []
    for s in streams:
        digits = "".join(ch for ch in str(s) if ch.isdigit())
        if not digits:
            raise ValueError(
                f"camera stream {s!r} carries no index; cannot map it to a "
                "camera topic. Pass --topic-camera-ids explicitly.")
        ids.append(int(digits))
    return ids


def resolve_topic_camera_ids(camera_topics, topic_camera_ids) -> List[int]:
    """Dataset camera id of each registered topic (default: positional 0,1,...)."""
    if topic_camera_ids is None:
        return list(range(len(camera_topics)))
    ids = [int(i) for i in topic_camera_ids]
    if len(ids) != len(camera_topics):
        raise ValueError(
            f"--topic-camera-ids has {len(ids)} entries but --camera-topics has "
            f"{len(camera_topics)}; they must line up one-to-one.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"--topic-camera-ids must be unique, got {ids}")
    return ids


def frame_for_camera(raw_obs: Dict[str, Any], cam_id: int,
                     topic_camera_ids: List[int]) -> np.ndarray:
    """Live frame of DATASET camera `cam_id` as (H,W,3) uint8 RGB."""
    if cam_id not in topic_camera_ids:
        raise RuntimeError(
            f"the checkpoint reads camera {cam_id} but the registered topics map "
            f"to cameras {topic_camera_ids}. Add the topic (in collection order) "
            "or correct --topic-camera-ids.")
    pos = topic_camera_ids.index(cam_id)
    if pos >= len(_POSITION_KEYS):
        raise RuntimeError(f"no known observation key for camera position {pos}")

    full = raw_obs.get("full_image")
    full_arr = np.asarray(full) if full is not None else None
    for key in _POSITION_KEYS[pos]:
        if key.startswith("full_image_"):
            i = int(key.rsplit("_", 1)[1])
            if full_arr is not None and full_arr.ndim == 4 and full_arr.shape[0] > i:
                return to_uint8_rgb(full_arr[i])
            if full_arr is not None and full_arr.ndim == 3 and i == 0:
                return to_uint8_rgb(full_arr)
            continue
        if raw_obs.get(key) is not None:
            return to_uint8_rgb(np.asarray(raw_obs[key]))
    raise RuntimeError(
        f"no frame for camera {cam_id} (topic position {pos}) in the observation "
        f"(keys={sorted(raw_obs.keys())}). Is the camera publishing?")


def build_stack_frame(raw_obs: Dict[str, Any], cam_ids: List[int],
                      topic_camera_ids: List[int], out_hw, gains=None) -> np.ndarray:
    """One timestep of the stack: (H, W, 3*len(cam_ids)), cameras in cam_ids order.

    Matches PushTWidowXVideoDataset.__getitem__, which iterates the cameras
    INSIDE each stack offset, so channel-concatenating these per-timestep blocks
    oldest->newest reproduces the training layout
    [cam0_oldest, cam1_oldest, ..., cam0_newest, cam1_newest].
    """
    per_cam = [preprocess(frame_for_camera(raw_obs, c, topic_camera_ids),
                          out_hw, gains=gains) for c in cam_ids]
    return np.concatenate(per_cam, axis=-1)


# ---------------------------------------------------------------------------
# end-of-episode scoring (--measure)
# ---------------------------------------------------------------------------

def score_final_frames(frames: Dict[int, np.ndarray]) -> Dict[str, Any]:
    """Coverage per camera and centroid error, from measure_target_coverage.

    ``frames`` maps DATASET camera id -> final RGB frame. That script carries
    the target outline and the projection bias as compiled-in constants for
    this rig, so nothing here needs a dataset or a calibration directory.
    """
    import cv2

    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import measure_target_coverage as mtc

    out: Dict[str, Any] = {"coverage_cam0": None, "coverage_cam1": None,
                           "dist_centroid": None}
    dists = []
    for cam_id, rgb in sorted(frames.items()):
        camera = f"images{cam_id}"
        if camera not in mtc.TARGET_POLYGONS:
            print(f"[measure] no target polygon for {camera}; skipping it")
            continue
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        target = mtc.builtin_mask(camera, bgr.shape[:2])
        background = mtc.load_background(None, camera, bgr.shape[:2])
        res, _ = mtc.measure_frame(
            bgr, target, background=background,
            goal_offset=mtc.load_goal_offset(None, camera))
        out[f"coverage_cam{cam_id}"] = round(float(res["covered_frac"]), 4)
        if background is not None:
            shift = mtc.ShiftChecker(background, target)(bgr, mtc.red_mask(bgr))
            if shift > 3.0:
                print(f"[measure] WARNING {camera} has moved {shift:.1f} px since "
                      f"the target was calibrated; the score is unreliable")
        if "centroid_dist_px" in res:
            dists.append(res["centroid_dist_px"])
    if dists:
        # Worst camera, matching measure_target_coverage's own combined figure:
        # the two agree closely near the goal and diverge when the block is far,
        # where each view foreshortens the error differently.
        out["dist_centroid"] = round(float(max(dists)), 2)
    return out


def csv_text(value: Any) -> str:
    """CSV form of one cell. ``None`` is blank; ``False`` is "False", not blank."""
    return "" if value is None else str(value)


def append_result_row(csv_path: Path, row: Dict[str, Any]) -> int:
    """Append one experiment, numbering trials within its parameter combo.

    A table written before a column existed is rewritten with the current
    header rather than having wider rows appended under it, which would leave
    a file no reader can parse.
    """
    import csv as _csv

    csv_path = Path(csv_path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing: List[Dict[str, str]] = []
    header: List[str] = []
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        with csv_path.open(newline="") as fh:
            reader = _csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            existing = [dict(old) for old in reader]

    stale_header = bool(header) and header != RESULTS_COLUMNS
    if stale_header:
        # Backfill BEFORE counting trials: a row from an older table is missing
        # the newer key columns, so without this it matches nothing and the
        # trial counter restarts, silently producing two rows numbered 0 for
        # the same combination.
        for old in existing:
            for column, fallback in RESULTS_LEGACY_DEFAULTS.items():
                if not old.get(column):
                    old[column] = fallback

    trial = 0
    for old in existing:
        if all(csv_text(old.get(k)) == csv_text(row[k]) for k in RESULTS_KEY):
            try:
                trial = max(trial, int(old.get("trial", 0)) + 1)
            except (TypeError, ValueError):
                continue
    row = dict(row, trial=trial)

    def as_text(record: Dict[str, Any]) -> Dict[str, str]:
        return {k: csv_text(record.get(k)) for k in RESULTS_COLUMNS}

    if stale_header:
        with csv_path.open("w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS)
            writer.writeheader()
            writer.writerows(as_text(old) for old in existing)
            writer.writerow(as_text(row))
        print(f"[measure] {csv_path} predates the "
              f"{sorted(set(RESULTS_COLUMNS) - set(header))} column(s); rewrote it "
              f"with the current header")
        return trial

    with csv_path.open("a", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS)
        if not header:
            writer.writeheader()
        writer.writerow(as_text(row))
    return trial


class InferenceTimer:
    """Wall clock around the policy call only -- no camera read, no robot I/O.

    CUDA queues kernels asynchronously, so a bare ``perf_counter`` around a GPU
    forward measures how long it took to *launch* the work, not to do it. Every
    sample therefore brackets the region with a synchronize when the model is on
    the GPU, which is the difference between a plausible-looking sub-millisecond
    number and the truth.
    """

    def __init__(self, device: Any) -> None:
        self.cuda = str(getattr(device, "type", device)) == "cuda"
        self.samples_ms: List[float] = []

    @contextlib.contextmanager
    def measure(self):
        if self.cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.cuda:
                torch.cuda.synchronize()
            self.samples_ms.append((time.perf_counter() - t0) * 1e3)

    def summary(self, n_steps: int, exec_horizon: int,
                warmup: int = 1) -> Dict[str, Any]:
        """Per-call and per-control-step cost of the samples collected so far.

        The first call pays for cuDNN autotuning and lazy kernel loading and can
        run an order of magnitude slow. It is reported on its own as
        ``ms_first_infer`` and dropped from the distribution rather than left to
        inflate the mean of a short episode.
        """
        out: Dict[str, Any] = {"n_infer": len(self.samples_ms),
                               "n_steps": int(n_steps),
                               "exec_horizon": int(exec_horizon)}
        if not self.samples_ms:
            return out
        out["ms_first_infer"] = round(self.samples_ms[0], 3)
        warm = self.samples_ms[warmup:] or self.samples_ms
        mean = statistics.fmean(warm)
        ordered = sorted(warm)
        # Nearest-rank p95, which is defined for the handful of samples a short
        # episode yields; statistics.quantiles needs more points than that.
        rank = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
        out["ms_per_infer_mean"] = round(mean, 3)
        out["ms_per_infer_median"] = round(statistics.median(warm), 3)
        out["ms_per_infer_p95"] = round(ordered[max(0, rank)], 3)
        out["ms_per_infer_std"] = round(
            statistics.stdev(warm) if len(warm) > 1 else 0.0, 3)
        # What control actually pays per step: one inference is amortized over
        # the exec_horizon steps that consume its chunk, so a policy 8x slower
        # per call than another can still be cheaper per step.
        out["ms_per_step"] = round(mean * len(self.samples_ms) / max(1, n_steps), 3)
        out["infer_hz"] = round(1000.0 / mean, 2) if mean > 0 else None
        return out


def count_params_m(*modules: Any) -> float | None:
    """Millions of distinct parameters across `modules`, or None if given none.

    De-duplicated by identity (BC passes a single module, but the helper is
    shared verbatim with the Q3C client, whose CP generator and energy net share
    an image encoder that would otherwise be counted twice).
    """
    seen: set = set()
    total = 0
    found = False
    for module in modules:
        if module is None or not hasattr(module, "parameters"):
            continue
        found = True
        for p in module.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += int(p.numel())
    return round(total / 1e6, 4) if found else None


def count_gflops(call) -> float | None:
    """GFLOPs of one policy call, or None when they cannot be counted.

    Wall time only compares runs on the same machine under the same load; FLOPs
    compare BC, Q3C, IBC and DP anywhere, which is what makes "how much
    computation" answerable rather than rig-specific. For BC this is the floor
    of the comparison: one encoder + head pass, no sampler, no refinement.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as exc:                        # torch < 2.0
        print(f"[speed] this torch has no FLOP counter: {exc!r}")
        return None
    try:
        counter = FlopCounterMode(display=False)
        with counter:
            call()
        return round(counter.get_total_flops() / 1e9, 4)
    except Exception as exc:
        print(f"[speed] FLOP count failed, leaving the column blank: {exc!r}")
        return None


def energy_net_evals(inference: str = INFERENCE_LABEL,
                     refine_iters: int = REFINE_ITERS) -> int:
    """Sequential net passes behind one action: 1, unconditionally.

    Kept as a function (rather than inlining the constant) so this client writes
    the same column through the same call the Q3C/IBC ones do -- there the count
    varies with the refinement mode, here there is no refinement to vary.
    """
    return NET_EVALS_PER_ACTION


def append_speed_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append one inference-cost row, rewriting a stale header in place.

    Same contract as append_result_row, minus the trial counter: the trial
    number is copied from the results table so the two join, and a run scored
    without --measure simply leaves it blank.
    """
    import csv as _csv

    csv_path = Path(csv_path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing: List[Dict[str, str]] = []
    header: List[str] = []
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        with csv_path.open(newline="") as fh:
            reader = _csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            existing = [dict(old) for old in reader]

    def as_text(record: Dict[str, Any]) -> Dict[str, str]:
        return {k: csv_text(record.get(k)) for k in SPEED_COLUMNS}

    if header and header != SPEED_COLUMNS:
        with csv_path.open("w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=SPEED_COLUMNS)
            writer.writeheader()
            writer.writerows(as_text(old) for old in existing)
            writer.writerow(as_text(row))
        print(f"[speed] {csv_path} predates the "
              f"{sorted(set(SPEED_COLUMNS) - set(header))} column(s); rewrote it "
              f"with the current header")
        return

    with csv_path.open("a", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=SPEED_COLUMNS)
        if not header:
            writer.writeheader()
        writer.writerow(as_text(row))


def report_inference_cost(csv_path: Path, timer: InferenceTimer,
                          key: Dict[str, Any], n_steps: int, exec_horizon: int,
                          net_evals: int | None = None,
                          gflops: float | None = None,
                          params_m: float | None = None) -> None:
    """Print the inference-cost line and append it to `csv_path`.

    `key` supplies the join columns (RESULTS_KEY plus trial and device); the
    measured fields are filled in from `timer`.
    """
    stats = timer.summary(n_steps, exec_horizon)
    append_speed_row(csv_path, dict(key, **stats,
                                    net_evals_per_infer=net_evals,
                                    gflops_per_infer=gflops,
                                    params_m=params_m))
    if "ms_per_infer_mean" not in stats:
        print(f"[speed] no inference was timed; wrote an empty row -> {csv_path}")
        return
    print(f"[speed] {stats['ms_per_infer_mean']} ms/inference "
          f"(median {stats['ms_per_infer_median']}, "
          f"p95 {stats['ms_per_infer_p95']}, n={stats['n_infer']}), "
          f"{stats['ms_per_step']} ms/control-step, {stats['infer_hz']} Hz"
          + (f", {net_evals} net evals" if net_evals is not None else "")
          + (f", {gflops} GFLOPs/inference" if gflops is not None else "")
          + f" -> {csv_path}")


def save_fed_png(path_stem: Path, block: np.ndarray, cam_ids: List[int]) -> None:
    """Write the fed frame(s) of one timestep. Multi-camera -> one png per cam."""
    import cv2

    for i, cam in enumerate(cam_ids):
        img = block[:, :, 3 * i:3 * (i + 1)]
        out = (path_stem.with_suffix(".png") if len(cam_ids) == 1
               else path_stem.with_name(f"{path_stem.name}_cam{cam}.png"))
        cv2.imwrite(str(out), cv2.cvtColor(np.ascontiguousarray(img),
                                           cv2.COLOR_RGB2BGR))


def eef_x_from_obs(raw_obs: Dict[str, Any]) -> float | None:
    """Current end-effector x (metres). state[0] is x on this rig."""
    if raw_obs is None:
        return None
    for key in ("eef_pos", "ee_pos", "state", "proprio", "agent_pos"):
        v = raw_obs.get(key)
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        if arr.size >= 1:
            return float(arr[0])
    return None


def apply_approach_floor(act_xy: np.ndarray, cur_x: float | None,
                         floor_x: float | None) -> tuple[np.ndarray, bool]:
    """Clip dx so the EEF never moves closer to the robot than floor_x.

    Smaller x == closer to the robot base. If the commanded dx would take
    (cur_x + dx) below floor_x, reduce dx so the arm stops AT the floor (never
    past it). Returns (possibly-clipped action, was_clipped).
    """
    if floor_x is None or cur_x is None:
        return act_xy, False
    act = np.asarray(act_xy, np.float64).copy()
    max_neg_dx = floor_x - cur_x        # most negative dx allowed this step
    if act[0] < max_neg_dx:
        act[0] = max_neg_dx             # >=0 if already at/below the floor
        return act, True
    return act, False


def preprocess(frame: np.ndarray, out_hw, gains=None) -> np.ndarray:
    """(H,W,3) uint8 RGB -> (H',W',3) uint8, as PushTRealPixelsDataset does.

    The training pipeline decodes to RGB and resizes with AREA, keeping uint8
    (the conv encoder does the /255 itself).

    `gains`: optional per-channel (R,G,B) multipliers applied BEFORE resize to
    match the training white point / exposure (see --match-exposure). The deploy
    scene was measured ~16% dimmer than training (board 0.82-0.86x), which washes
    out the salient red T (peak redness 79 vs 110). A per-channel gain lifts the
    whole image to the training exposure. Clipped to [0,255].
    """
    import cv2

    if gains is not None:
        frame = np.clip(frame.astype(np.float32) * np.asarray(gains, np.float32),
                        0, 255).astype(np.uint8)
    H, W = out_hw
    if frame.shape[:2] != (H, W):
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)


def stack_to_tensor(frame_buf, device) -> torch.Tensor:
    """oldest->newest (H,W,3*ncam) blocks -> (1, 3*ncam*fs, H, W) uint8 tensor."""
    stacked = np.concatenate(list(frame_buf), axis=-1)     # (H, W, 3*ncam*fs)
    stacked = np.transpose(stacked, (2, 0, 1))             # (3*ncam*fs, H, W)
    return torch.from_numpy(np.ascontiguousarray(stacked)).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Explicit BC policy
# ---------------------------------------------------------------------------

def load_run_config(seed_dir: Path) -> dict:
    with (seed_dir / "config.json").open() as fh:
        config = json.load(fh)
    return config["environments"][config["active_env"]]


def build_policy(env: dict, in_channels: int, device, cond_dim: int = 0,
                 norm_stats: dict | None = None):
    """Reconstruct the BC policy exactly as train_pusht_real_bc.py built it.

    That trainer builds a PixelControlPointGenerator with control_points=1: the
    same encoder + head module the Q3C client rebuilds, emitting one action
    instead of a cloud. Every knob the run recorded must be forwarded here or
    load_state_dict fails on a shape mismatch -- the ResNet-18 lines
    (batches/pushtWidowXbc.txt: --encoder-kind resnet18 --encoder-norm-kind gn
    --encoder-num-kp 128) differ from the conv_maxpool defaults in the encoder
    stem, the norm layers AND the SpatialSoftmax keypoint count.

    `cond_dim` comes from norm_stats: 0 for the pixels-only checkpoints, 2 when
    the run was trained with EEF (x, y) conditioning (--cond-eef-xy).

    `norm_stats`, when given, wins over the config: the trainer writes the head
    geometry and the encoder block there, and it fixes the action width, which
    is 2 * action_chunk rather than the config's nominal 2.
    """
    from utils.models import PixelControlPointGenerator

    m = env.get("model", {})
    ns = norm_stats or {}

    def pick(key, default, cast=None):
        v = ns.get(key, m.get(key, default))
        return cast(v) if cast is not None else v

    enc_h = int(ns.get("encoder_target_height", env.get("encoder_target_height", 180)))
    enc_w = int(ns.get("encoder_target_width", env.get("encoder_target_width", 240)))
    a_lo, a_hi = env.get("action_bounds", [-1.0, 1.0])
    # The head width is the ACTION SHAPE the trainer used, i.e. 2 * action_chunk.
    action_dim = int(env.get("action_dim", 2))
    if "act_min" in ns:
        action_dim = int(np.asarray(ns["act_min"]).size)

    # A BC checkpoint is single-headed by construction. Read it anyway rather
    # than hardcoding 1, so a mislabelled directory (a Q3C seed handed to this
    # client) fails on the shape it actually has instead of loading a truncated
    # head.
    control_points = int(pick("control_points", 1, int))
    if control_points != 1:
        raise ValueError(
            f"{__file__} deploys explicit BC policies (control_points=1) but "
            f"this checkpoint records control_points={control_points}. If it is "
            f"a Q3C seed, use scripts/deploy_pusht_real.py.")
    num_neurons = int(m.get("num_neurons", 512))
    num_hidden_layers = int(m.get("num_hidden_layers", 4))
    cp_width = int(pick("cp_width", num_neurons, int))
    cp_depth = int(pick("cp_depth", num_hidden_layers, int))
    cp_network_kind = pick("cp_network_kind", "mlp", str)
    cp_use_spectral_norm = bool(m.get("cp_use_spectral_norm", False))

    # Encoder block -- the part that silently breaks a resnet18 checkpoint.
    encoder_kind = pick("encoder_kind", "conv_maxpool", str)
    encoder_feature_dim = pick("encoder_feature_dim", 256, int)
    # Weights come from the state_dict; `pretrained` only affects train-time
    # init, but it must still match because it selects the same module tree.
    encoder_pretrained = pick("encoder_pretrained", True, bool)
    encoder_num_kp = pick("encoder_num_kp", 64, int)
    encoder_norm_kind = pick("encoder_norm_kind", "bn", str)
    encoder_per_camera = pick("encoder_per_camera", False, bool)
    cond_fusion = pick("cond_fusion", "concat", str)
    goal_dim = int(ns.get("goal_emb_dim", 0))

    policy = PixelControlPointGenerator(
        output_dim=action_dim,
        control_points=1,
        hidden_dims=[cp_width for _ in range(cp_depth)],
        action_bounds=(float(a_lo), float(a_hi)),
        network_kind=cp_network_kind,
        width=cp_width,
        depth=cp_depth,
        use_spectral_norm=cp_use_spectral_norm,
        in_channels=in_channels,
        encoder_target_height=enc_h,
        encoder_target_width=enc_w,
        encoder_feature_dim=encoder_feature_dim,
        cond_dim=cond_dim,
        encoder_kind=encoder_kind,
        encoder_pretrained=encoder_pretrained,
        encoder_num_kp=encoder_num_kp,
        encoder_norm_kind=encoder_norm_kind,
        encoder_per_camera=encoder_per_camera,
        cond_fusion=cond_fusion,
        goal_dim=goal_dim,
    ).to(device).eval()
    return policy


def load_weights(model, path: Path, device):
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint weights: {path}")
    # weights_only=False: our own trusted checkpoints (state dicts + numpy).
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    return model


def resolve_checkpoint(seed_dir: Path, ckpt_step: int, use_ema: bool) -> Path:
    """Which weights to load: EMA by default (--no-ema for the raw copy).

    An explicit --ckpt-step wins; otherwise the final bc_policy[_ema].pt,
    falling back to the newest matching step snapshot so a seed that is still
    training is still deployable rather than failing outright.
    """
    stem = "bc_policy_ema" if use_ema else "bc_policy"
    if ckpt_step > 0:
        return seed_dir / f"{stem}_step{ckpt_step:06d}.pt"
    final = seed_dir / f"{stem}.pt"
    if final.is_file():
        return final
    snapshots = sorted(seed_dir.glob(f"{stem}_step*.pt"))
    if not snapshots:
        raise FileNotFoundError(
            f"no {stem}.pt and no {stem}_step*.pt snapshots in {seed_dir}")
    print(f"[WARN] no final {stem}.pt in {seed_dir}; using snapshot "
          f"{snapshots[-1].name} (training likely still in progress)")
    return snapshots[-1]


@torch.no_grad()
def select_action(policy, obs_u8, cond: "torch.Tensor | None" = None):
    """The whole policy: one forward pass, one action.

    `cond` is the (1, cond_dim) conditioning vector, read off `_cond` exactly as
    in training. The module emits (1, 1, action_dim) because it is a
    control-point generator with a cloud of one; index the single row out and
    return the normalized action as (action_dim,).

    No candidate ranking, no refinement, no stall escape -- the deliberate cost
    (and the deliberate limitation) of an explicit policy.
    """
    policy._cond = cond
    act = policy(obs_u8)[0, 0]                        # (action_dim,) normalized
    return act.detach().cpu().numpy()


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


def apply_min_step(act_xy: np.ndarray, min_step: float,
                   eps: float = 1e-5) -> tuple[np.ndarray, bool]:
    """Snap sub-min-step nonzero components up to the min real step.

    The expert action distribution is bang-bang per axis: exactly 0, or a real
    step >= ~1.5mm, with an empty dead zone in between. A policy interpolating
    between the 0-spike and the min-step cluster can output a value inside that
    dead zone -- too small for the arm to execute, so it freezes at a fixed
    point (measured on c09/c10 rollouts). MSE regression lands there MORE
    readily than an energy argmax, not less: fitting the conditional mean of a
    bimodal target is exactly how a value between the modes gets produced. This
    forces any
    nonzero-but-tiny command onto the supported grid. Exact 0 (a genuine hold)
    is preserved.
    """
    if min_step <= 0:
        return act_xy, False
    act = np.asarray(act_xy, np.float64).copy()
    snapped = False
    for i in range(len(act)):
        v = act[i]
        if eps < abs(v) < min_step:
            act[i] = math.copysign(min_step, v)
            snapped = True
    return act, snapped


def z_hold_dz(cur_z: float | None, z_target: float, gain: float,
              max_dz: float) -> float:
    """Proportional dz to drive measured z toward z_target (G4 droop hold).

    Returns 0 if z_target<=0 (disabled) or cur_z is unavailable.
    """
    if z_target <= 0 or cur_z is None:
        return 0.0
    dz = gain * (z_target - cur_z)
    return float(np.clip(dz, -max_dz, max_dz))


def control_z_step(z_cmd: float, measured_z: float | None, target: float,
                   gain: float, max_dz: float,
                   windup: float) -> tuple[float, float]:
    """One step of the client-side integrating z loop (--control-z).

    ``z_cmd`` mirrors the server's target z. That mirror is exact: with
    action_mode=3trans the env zeroes the rotation dims, so
    ``action2transform_local`` returns a pure translation D, and
    ``next_transform = D . prev_transform`` moves the target's z by exactly dz.
    The env then stores that back into ``_previous_target_qpos``, so target z
    accumulates -- which is what turns this proportional law into an integrator
    on the MEASURED z, and why it can zero a droop the env's own z lock cannot.

    Anti-windup is structural: the mirror is clamped to target +/- windup and dz
    is derived from the clamped mirror, so the integrator state can never run
    away past the clamp while the arm is saturated.

    Returns (dz to send, new mirror value).
    """
    if measured_z is None:
        return 0.0, z_cmd
    desired = float(np.clip(gain * (target - measured_z), -max_dz, max_dz))
    new_cmd = float(np.clip(z_cmd + desired, target - windup, target + windup))
    return new_cmd - z_cmd, new_cmd


def z_from_obs(raw_obs) -> float | None:
    """Measured EEF z = state[2], mirroring eef_x_from_obs's state[0]."""
    if raw_obs is None:
        return None
    st = raw_obs.get("state")
    if st is None or len(st) < 3:
        return None
    return float(st[2])


def safety_clip_action(action_7d: np.ndarray, action_mode: str,
                       max_xy_delta: float) -> np.ndarray:
    action = np.asarray(action_7d, dtype=np.float64).copy()
    # The xy clip is the safety limit and must hold in EVERY mode -- it used to
    # apply only to 2trans, which would have silently dropped it under
    # --control-z (3trans).
    if max_xy_delta > 0:
        action[:2] = np.clip(action[:2], -max_xy_delta, max_xy_delta)
    if action_mode == "2trans":
        # 2trans should only use planar translation deltas.
        action[2:6] = 0.0
    elif action_mode == "3trans":
        # Keep dz (the --control-z / --z-hold channel), drop the rotations the
        # env would zero anyway.
        action[3:6] = 0.0
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
    if args.control_z is not None:
        if args.z_hold > 0:
            raise SystemExit(
                "--control-z and --z-hold both inject dz; pick one. "
                "--control-z is the integrating version and supersedes --z-hold.")
        if args.control_z <= 0:
            raise SystemExit("--control-z HEIGHT must be > 0")
        args.action_mode = "3trans"
        args.lock_z = False
        # A cached server env keeps the action_mode it was FIRST built with, and
        # the env asserts action.shape[0] == _base_adim -- reusing a 2trans env
        # would reject our 4-dim action outright.
        args.reuse_existing_env = False
        if args.control_z_gain >= 1.0:
            raise SystemExit(
                f"--control-z-gain {args.control_z_gain} is at or past the "
                "stability limit. The loop integrates through the env's target z "
                "with one step of measurement lag, so |lambda| = sqrt(gain); at "
                "1.0 it rings indefinitely (~11 mm peak-to-peak in simulation). "
                "Use <= 0.8.")
        print(f"[control-z] target={args.control_z:.4f} m "
              f"gain={args.control_z_gain} max_dz={args.control_z_max_dz} "
              f"windup=+/-{args.control_z_windup} -> action_mode=3trans, "
              f"lock_z=False, fresh server init forced.")
    if args.z_hold > 0 and args.action_mode == "2trans":
        raise SystemExit(
            "--z-hold needs an action_mode that sends z "
            "(3trans/3trans1rot/3trans3rot); got 2trans. The injected dz would "
            "be dropped. Re-run with e.g. --action-mode 3trans."
        )
    seed_dir = args.seed_dir.resolve()

    # --- checkpoint metadata -------------------------------------------------
    env_cfg = load_run_config(seed_dir)
    norm_stats = torch.load(seed_dir / "norm_stats.pt", map_location="cpu",
                            weights_only=False)
    act_min = np.asarray(norm_stats["act_min"], np.float32)
    act_max = np.asarray(norm_stats["act_max"], np.float32)
    norm_range = tuple(norm_stats.get("action_norm_range", (-1.0, 1.0)))

    frame_stack = int(env_cfg.get("frame_stack", 2))
    cams = tuple(norm_stats.get("camera_streams",
                                env_cfg.get("camera_streams", ["images1"])))
    image_h = int(env_cfg.get("image_height", 240))
    image_w = int(env_cfg.get("image_width", 320))
    in_channels = 3 * len(cams) * frame_stack
    cam_ids = camera_ids_from_streams(cams)
    topic_camera_ids = resolve_topic_camera_ids(args.camera_topics,
                                                args.topic_camera_ids)

    # EEF (x, y) conditioning: present only for runs trained with --cond-eef-xy.
    # cond_min/cond_max are the TRAINING workspace bounds; the live proprio must
    # be normalized with these exact numbers or the conditioning is off-scale.
    cond_dim = int(norm_stats.get("cond_dim", 0))
    cond_min = cond_max = None
    if cond_dim:
        if str(norm_stats.get("cond_kind", "")) != "eef_xy":
            raise ValueError(
                f"checkpoint has cond_dim={cond_dim} but cond_kind="
                f"{norm_stats.get('cond_kind')!r}; this client only knows eef_xy"
            )
        cond_min = np.asarray(norm_stats["cond_min"], np.float32)
        cond_max = np.asarray(norm_stats["cond_max"], np.float32)

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    policy = build_policy(env_cfg, in_channels, device, cond_dim=cond_dim,
                          norm_stats=norm_stats)
    ckpt_path = resolve_checkpoint(seed_dir, args.ckpt_step, use_ema=not args.no_ema)
    load_weights(policy, ckpt_path, device)
    print(f"Loaded weights ({'raw' if args.no_ema else 'EMA'}) from {ckpt_path}")
    print(f"  frame_stack={frame_stack} cameras={cams} model_hw=({image_h},{image_w}) "
          f"in_channels={in_channels}")
    print(f"  inference={INFERENCE_LABEL} (1 forward pass/action)  device={device}")
    print(f"  act_min={act_min} act_max={act_max} norm_range={norm_range}")
    print(f"  cond_dim={cond_dim}"
          + (f" (eef_xy, min={cond_min} max={cond_max})" if cond_dim else " (pixels only)"))

    def make_cond(raw_obs):
        """Live observation -> (1, cond_dim) normalized EEF x/y, or None.

        Mirrors PushTWidowXVideoDataset.normalize_cond: min-max to [-1,1] over
        the training workspace, CLIPPED, because the arm can leave the
        demonstrated region and an unbounded vector would be a far larger shift
        than a saturated one. The server's state is
        [x, y, z, r0, r1, r2, gripper] -- x/y are dims 0:2.
        """
        if not cond_dim:
            return None
        st = None if raw_obs is None else raw_obs.get("state")
        if st is None:
            raise RuntimeError(
                "checkpoint needs EEF conditioning but the observation has no "
                "'state' field"
            )
        xy = np.asarray(st, np.float32).reshape(-1)[:2]
        span = np.where(cond_max == cond_min, np.ones_like(cond_max), cond_max - cond_min)
        z = np.clip(-1.0 + 2.0 * (xy - cond_min) / span, -1.0, 1.0)
        return torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)

    # --- connect -------------------------------------------------------------
    WidowXClient, WidowXConfigs, WidowXStatus = load_widowx_dependencies(
        args.widowx_envs_path)
    print(f"WidowX SDK: {WidowXClient.__module__} "
          f"({getattr(sys.modules.get(WidowXClient.__module__), '__file__', '?')})")

    env_params = build_env_params(args, WidowXConfigs)
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
        reuse_existing_env = widowx_server_has_live_env(client, max_wait_sec=1.0)
        if reuse_existing_env:
            print("[INFO] Server already has a live env; skipping init() and "
                  "reusing it. (Re-initializing with different env_params is what "
                  "triggers 'Incompatible config with hash with server'.)")
            print("[WARN] The live env keeps the env_params it was FIRST "
                  "initialized with -- not the ones printed above. If the robot "
                  "behaves as though action_mode/lock_z/etc. differ, restart "
                  "`widowx_env_service --server` and re-run to apply ours.")

    if reuse_existing_env:
        set_reqrep_timeout_ms(client, max(1, args.rpc_timeout_ms))
    else:
        init_status = init_widowx_with_retry(
            client, env_params, args.im_size, WidowXStatus, args)
        if init_status != WidowXStatus.SUCCESS:
            raise RuntimeError(
                f"WidowX init failed after {args.init_retries} attempts with "
                f"status={status_name(init_status, WidowXStatus)}.\n"
                f"If this is a config-hash error, the server is holding a cached "
                f"env from a previous run with different env_params: RESTART "
                f"`widowx_env_service --server` (and the docker container) and "
                f"re-run. Otherwise check reachability at {args.ip}:{args.port} "
                f"and that --widowx-envs-path ({args.widowx_envs_path}) matches "
                f"the server's widowx_envs.")
    print("WidowX connection established.")

    # Reset: move_to_neutral, then move_to_startstate -> env_params["start_state"],
    # which --demo-start-state has set to the demo start pose. --i-traj never
    # reaches the server (WidowXClient.reset takes no argument), so the itraj
    # branch the collection used is unreachable from here.
    reset_status = reset_widowx_with_retry(client, WidowXStatus, args, args.i_traj)
    if reset_status != WidowXStatus.SUCCESS:
        raise RuntimeError(
            f"WidowX reset failed with "
            f"status={status_name(reset_status, WidowXStatus)}")
    print(f"Reset done (itraj={args.i_traj}).")

    # Physically actuate the clamp. In 2trans mode the gripper dim is never sent
    # (step_action gets action[:2]), and reset can leave the clamp open, so we
    # command it explicitly here. 0.0 = closed to grip the pusher.
    if args.gripper_command >= 0.0:
        if hasattr(client, "move_gripper"):
            try:
                gstatus = client.move_gripper(float(args.gripper_command))
                print(f"Gripper commanded to {args.gripper_command} "
                      f"(0=closed,1=open); status={status_name(gstatus, WidowXStatus)}")
                time.sleep(1.0)
            except Exception as exc:
                print(f"[WARN] move_gripper({args.gripper_command}) failed: {exc}")
        else:
            print("[WARN] WidowXClient has no move_gripper(); cannot actuate the "
                  "clamp explicitly. The env fixed_gripper target is "
                  f"{args.fixed_gripper}.")

    # --- move to the demo start pose (same as the ibc deploy) ---------------
    start_T = None
    if args.move_to_demo_start:
        start_path = Path(args.start_eep_npy).expanduser()
        if not start_path.is_file():
            raise FileNotFoundError(
                f"--start-eep-npy not found: {start_path}. Pass "
                "--no-move-to-demo-start to skip (not recommended: the arm then "
                "starts ~17cm out of distribution).")
        start_T = np.load(start_path).astype(np.float32)
        if args.control_z is not None:
            # Same reason as in demo_start_state: with lock_z off nothing lifts
            # this to the hold height, and __move re-syncs the env's target z to
            # wherever this leaves the arm.
            start_T = start_T.copy()
            start_T[2, 3] = float(args.control_z)
        print(f"[INFO] Moving EEF to demo start pose (x={start_T[0,3]:.3f}, "
              f"y={start_T[1,3]:.3f}, z={start_T[2,3]:.3f})...")
        move_status, tries = None, 0
        while move_status != WidowXStatus.SUCCESS and tries < args.max_initial_move_retries:
            move_status = client.move(start_T, duration=args.start_move_duration)
            tries += 1
        if move_status != WidowXStatus.SUCCESS:
            print(f"[WARN] initial move did not report SUCCESS after {tries} tries "
                  f"(status={status_name(move_status, WidowXStatus)}); continuing.")

    # --- resolve the HARD approach floor (x closest-to-robot the arm may go) -
    # Priority: explicit override, else the start pose's x, else post-reset EEF x.
    approach_floor_x = None
    if args.approach_floor:
        if args.approach_floor_x is not None:
            approach_floor_x = float(args.approach_floor_x)
        elif start_T is not None:
            approach_floor_x = float(start_T[0, 3])
        else:
            eef0 = None
            try:
                st = client.get_observation()
                eef0 = eef_x_from_obs(st)
            except Exception:
                eef0 = None
            approach_floor_x = eef0
        if approach_floor_x is None:
            raise RuntimeError(
                "Approach guard is ON but the x floor could not be determined "
                "(no --start-eep-npy and no readable EEF x). Pass "
                "--approach-floor-x <metres> or --no-approach-floor.")
        print(f"[SAFETY] Approach floor ARMED: EEF x will never go below "
              f"{approach_floor_x:.4f} m (closer to the robot than the start).")

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

    def policy_frames(raw_obs) -> np.ndarray:
        return build_stack_frame(raw_obs, cam_ids, topic_camera_ids,
                                 (image_h, image_w), gains=exposure_gains)

    def raw_frame(raw_obs) -> np.ndarray:
        return frame_for_camera(raw_obs, cam_ids[0], topic_camera_ids)

    first_obs = grab_obs()
    first = policy_frames(first_obs)
    print(f"Policy cameras {cam_ids} (topics {args.camera_topics} -> ids "
          f"{topic_camera_ids}); stacked frame {first.shape}")
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
            obs_u8 = stack_to_tensor(frame_buf, device)
            na = select_action(policy, obs_u8, cond=make_cond(raw_obs))
            act = unnormalize(na, act_min, act_max, norm_range)
            save_fed_png(args.dump_dir / f"fed_{i:03d}", list(frame_buf)[-1], cam_ids)
            print(f"[{i:03d}] norm={np.round(na, 3)} -> action(dx,dy)={np.round(act, 4)}")
            time.sleep(args.step_duration)
        client.stop()
        print("Dry run done. Inspect deploy_dryrun/fed_000.png before live control.")
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
                cur_x = eef_x_from_obs(raw_obs)
                act_xy2, floored = apply_approach_floor(act_xy, cur_x, approach_floor_x)
                a7 = safety_clip_action(to_action_7d(act_xy2, args.fixed_gripper),
                                        args.action_mode, args.safety_max_xy_delta)
                env_action = project_action_to_env_mode(a7, args.action_mode)
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

    # One entry point for the policy so the control loop, the timer and the
    # post-episode FLOP count all measure the identical call.
    def predict(obs_u8, raw_obs):
        return select_action(policy, obs_u8, cond=make_cond(raw_obs))

    timer = InferenceTimer(device)
    last_obs_u8 = None
    last_raw_obs = None

    step = 0
    last_exec = time.time()
    pending: list[np.ndarray] = []   # unexecuted (dx,dy) tail of the chunk
    pending_norm: list[np.ndarray] = []
    chunk_idx = 0
    # Mirror of the server's target z for --control-z. _reset_previous_qpos sets
    # that target to the MEASURED state in 3trans (it only forces z to
    # fixed_z_height in 2trans), so seeding from the first observation below is
    # exact. dz stays 0 until then.
    z_cmd: float | None = None
    dz = 0.0
    try:
        for step in range(args.steps):
            raw_obs = grab_obs()
            raw = raw_frame(raw_obs)
            frame_buf.append(policy_frames(raw_obs))
            obs_u8 = stack_to_tensor(frame_buf, device)
            last_obs_u8, last_raw_obs = obs_u8, raw_obs

            if not pending:
                with timer.measure():
                    na_full = predict(obs_u8, raw_obs)
                act_full = unnormalize(na_full, act_min, act_max, norm_range)
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
            act_xy, snapped = apply_min_step(act_xy, args.min_step_xy)
            if snapped:
                print(f"[min-step] snapped {np.round(na, 3)} -> "
                      f"dx,dy={np.round(act_xy, 4)}")

            # HARD SAFETY: never move closer to the robot than the start pose.
            cur_x = eef_x_from_obs(raw_obs)
            act_xy, floored = apply_approach_floor(act_xy, cur_x, approach_floor_x)
            if floored:
                print(f"[SAFETY] approach floor: clipped dx at x={cur_x:.4f} "
                      f"(floor={approach_floor_x:.4f})")

            action_7d = to_action_7d(act_xy, args.fixed_gripper)
            action_7d = safety_clip_action(action_7d, args.action_mode,
                                           args.safety_max_xy_delta)
            # The env applies xy_action_deadband only in 2trans
            # (robot_base_env.py:246-247), so outside 2trans the client has to
            # apply it or the xy contract silently differs from collection.
            deadband = float(env_params.get("xy_action_deadband", 0.0))
            if args.action_mode != "2trans" and deadband > 0:
                small = np.abs(action_7d[:2]) < deadband
                action_7d[:2] = np.where(small, 0.0, action_7d[:2])
            # z-droop compensation: inject dz AFTER safety_clip (which zeros
            # dims 2-6 in 2trans) so it survives, and only when the mode carries
            # z. Startup guards reject both flags with action_mode=2trans.
            if args.control_z is not None:
                cur_z = z_from_obs(raw_obs)
                if z_cmd is None and cur_z is not None:
                    z_cmd = cur_z
                if z_cmd is not None:
                    dz, z_cmd = control_z_step(
                        z_cmd, cur_z, args.control_z, args.control_z_gain,
                        args.control_z_max_dz, args.control_z_windup)
                    action_7d[2] = dz
            elif args.z_hold > 0:
                dz = z_hold_dz(z_from_obs(raw_obs), args.z_hold,
                               args.z_hold_gain, args.z_hold_max)
                action_7d[2] = dz
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

            zmsg = ""
            if args.control_z is not None:
                mz = z_from_obs(raw_obs)
                zmsg = (f" z={'n/a' if mz is None else f'{mz:.4f}'}"
                        f" cmd={'n/a' if z_cmd is None else f'{z_cmd:.4f}'}"
                        f" dz={dz:+.5f}")
            print(f"[{step:03d}] chunk[{chunk_idx}/{exec_horizon - 1}] "
                  f"norm={np.round(na, 3)} -> "
                  f"env_action={np.round(env_action, 5)}{zmsg}")

            if log_fh is not None:
                np.save(args.log_dir / "raw" / f"{step:04d}.npy",
                        np.ascontiguousarray(raw))
                save_fed_png(args.log_dir / "fed" / f"{step:04d}",
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
                    **({"z_cmd": z_cmd, "dz": dz}
                       if args.control_z is not None else {}),
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
        key = {
            "algorithm": args.algorithm,
            "seed_dir": str(Path(args.seed_dir).expanduser().resolve()),
            # Constants for this client: an explicit policy has exactly one
            # inference mode and nothing to refine. They stay in the row so BC
            # episodes join the shared table on the same seven key columns.
            "inference": INFERENCE_LABEL,
            "refine_iters": REFINE_ITERS,
            # The RESOLVED horizon, not args.exec_horizon: an unchunked
            # checkpoint clamps it to chunk_len, and the table has to record
            # what ran rather than what was asked for.
            "exec_horizon": exec_horizon,
            "control_z": bool(args.control_z),
            "start_position": args.start_position,
        }
        trial = None
        if args.measure:
            try:
                final_obs = grab_obs()
                frames = {cam: frame_for_camera(final_obs, cam, topic_camera_ids)
                          for cam in topic_camera_ids}
                scores = score_final_frames(frames)
                trial = append_result_row(args.results_csv, dict(key, **scores))
                print(f"[measure] trial {trial}: "
                      f"coverage cam0={scores['coverage_cam0']} "
                      f"cam1={scores['coverage_cam1']} "
                      f"centroid={scores['dist_centroid']} px "
                      f"-> {args.results_csv}")
            except Exception as exc:
                print(f"[measure] FAILED, no row written: {exc!r}")
        # Inference cost. Runs whether or not the episode was scored, and never
        # takes the run down with it -- a timing is not worth losing a rollout.
        if timer.samples_ms and not args.no_speed_csv:
            try:
                gflops = None
                if not args.no_flops and last_obs_u8 is not None:
                    # The arm is already idle here, so the extra policy call the
                    # counter needs cannot perturb the episode it is measuring.
                    gflops = count_gflops(
                        lambda: predict(last_obs_u8, last_raw_obs))
                report_inference_cost(
                    args.speed_csv, timer, dict(key, trial=trial,
                                                device=str(device)),
                    n_steps=step + 1, exec_horizon=exec_horizon,
                    net_evals=energy_net_evals(),
                    gflops=gflops,
                    params_m=count_params_m(policy))
            except Exception as exc:
                print(f"[speed] FAILED, no row written: {exc!r}")
        try:
            client.stop()
        except Exception:
            pass
        print(f"Stopped after {step + 1} steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
