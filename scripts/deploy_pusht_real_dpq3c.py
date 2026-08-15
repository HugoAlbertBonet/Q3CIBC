#!/usr/bin/env python3
"""Deploy a DP+Q3C hybrid Push-T policy on the real WidowX arm.

This is ``scripts/deploy_pusht_real.py`` with ONE substitution: the control-point
cloud that the Q estimator ranks is no longer produced by the Q3C control-point
generator, it is produced by a trained Diffusion Policy denoiser. Everything
else -- every WidowX server call, safety clip, approach floor, z loop, receding
horizon, dry run, calibration, scoring, forensic log, stall escape and the
langevin/DFO refinement path -- is the Q3C client's, imported from it so the two
agree bit-for-bit.

    Q3C:    cp_gen(obs) -> (1, P, A)  ->  q_net.score  ->  argmax / refine
    DP+Q3C: DP sampler  -> (1, P, A)  ->  q_net.score  ->  argmax / refine
            (P = --cp draws from the SAME denoiser, one encoder pass, shared
             image features, different noise seeds)

Two checkpoints, two directories:
  ``--dp-dir``  a train_pusht_real_dp.py run (``denoiser_ema.pt`` + config.json
                + norm_stats.pt) -- supplies the control points.
  ``--q-dir``   a Q3C run (``q_estimator_ema.pt`` + config.json + norm_stats.pt)
                -- supplies the scorer. Its control-point generator is never
                loaded.

They must agree on the things that make a candidate action MEAN the same thing
to both nets: action width (2 * chunk length), act_min/act_max, the action
normalization range, frame_stack, camera streams and the model image size. A
mismatch is a hard error unless ``--allow-mismatch`` downgrades it to a warning
(the Q values are then scored in a different action space than the one the
samples were drawn in, and the ranking is meaningless).

New flags (everything else is deploy_pusht_real.py's):
  ``--cp N``        how many control points the DP draws per prediction. Default:
                    the Q run's own ``model.control_points``, so the cloud the Q
                    net ranks is the size it was trained to rank.
  ``--dp-method``   ``ddim`` (default) or ``ddpm`` -- the sampler schedule.
  ``--dp-iters K``  denoising iterations run before the cloud is scored. DDIM:
                    the sub-sampled step count. DDPM: a respaced ancestral chain
                    of K steps (K >= num_train_timesteps runs the full chain).
                    Default: the checkpoint's ddim_eval_steps[0] for ddim, the
                    full training chain for ddpm.

``--inference`` keeps its Q3C meaning and stacks on top: ``argmax``/``sample``
rank the DP cloud directly, ``dfo``/``langevin`` use the DP cloud as the INITIAL
cloud and refine it against the Q estimator for ``--refine-iters`` rounds. The
argmax stall escape is unchanged.

Four optional ways of spending the Q estimator, ALL OFF BY DEFAULT. With none of
them set this client is exactly draw-a-cloud-then-rank-it, which is the baseline
each of them has to beat:

  ``--cp-selection sample --cp-temperature B``
        Draw the executed candidate from ``softmax(Q/B)`` over the cloud instead
        of taking the argmax. B->0 recovers argmax, B->inf ignores Q and
        recovers plain DP. Softer than argmax against a miscalibrated Q, which
        by construction is worst exactly at the candidate argmax selects.
  ``--cp-score-norm zscore|rank``
        Rescale the scores WITHIN the cloud before that softmax, so one
        temperature means the same thing on every frame. NO EFFECT under argmax
        (both maps are monotone) -- the client warns and continues.
  ``--cascade-iters K --cascade-topk k``
        Denoise all ``--cp`` candidates only K steps, rank that cheap cloud,
        keep k, finish the rest for the survivors only. Same denoiser budget
        searches many more initial noises. The filter ranks the predicted CLEAN
        sample, never the noisy iterate -- the Q net has only ever seen clean
        actions. k defaults to ``--cp // 4``.
  ``--q-guidance A [--q-guidance-schedule linear|const]``
        Classifier guidance: nudge the predicted clean sample along the Q
        gradient at every denoising step, so the Q signal is spent THROUGHOUT
        the chain instead of only at the end. The denoiser is not retrained and
        never sees the gradient -- it is added outside the network, between the
        model output and the sampler's update. Only the Q value head is
        differentiated (image features are cached and detached), so this costs
        one small-MLP forward+backward per denoising step -- cheaper than
        ``--inference langevin --refine-iters 50``, which already does 50 such
        gradient steps after the fact. The gradient is unit-normalized per
        candidate, so A is a step size in normalized action units (try
        0.01-0.1). The default ``linear`` schedule ramps A from ~0 at high noise
        to full at t=0, because the predicted clean sample is a blurry
        conditional mean early in the chain.

Combining ``--q-guidance`` with ``--inference dfo|langevin`` applies the Q
estimator both inside the chain and again afterwards; it runs, and the client
warns, but the two effects are confounded -- sweep guidance under argmax/sample.

Usage (server already up):

    python scripts/deploy_pusht_real_dpq3c.py \
        --dp-dir checkpoints/pusht_real_dp_2026_07/g01_resnet18_s11_350k \
        --q-dir  checkpoints/pusht_real_combinedv2/seed_0011 \
        --device cpu --dry-run
    python scripts/deploy_pusht_real_dpq3c.py \
        --dp-dir checkpoints/pusht_real_dp_2026_07/g01_resnet18_s11_350k \
        --q-dir  checkpoints/pusht_real_combinedv2/seed_0011 \
        --cp 64 --dp-method ddim --dp-iters 10 \
        --steps 700 --measure --start-position top

Scoring: ``--measure`` writes to the SAME ``results/pusht/experiments.csv`` the
Q3C, IBC and DP clients write, with ``algorithm=dpq3c``. That table's columns are
fixed and three other scripts append to it, so rather than migrate its header the
two checkpoints share the ``seed_dir`` cell as ``<dp-dir>|<q-dir>`` and the
sampler settings ride in the ``inference`` cell. Only NON-DEFAULT knobs appear
there, so the all-off baseline keeps the short label::

    ddim10x64+argmax                  baseline
    ddim10x256c3k64+argmax            --cascade-iters 3 --cascade-topk 64
    ddim10x64g0.05l+argmax            --q-guidance 0.05 (linear schedule)
    ddim10x64+argmaxt0.5nz            --cp-selection sample -> temp 0.5, zscore

Without that encoding, two runs differing only in ``--cp`` or ``--dp-iters``
would land on the same key and be recorded as repeat trials of one condition.
``refine_iters`` keeps its Q3C meaning (langevin/DFO rounds).

Inference cost: as in the Q3C client, one row per episode in
``results/pusht/inference_speed.csv``. ``net_evals_per_infer`` counts the
sequential passes behind one action: ``--dp-iters`` denoiser passes (the cloud is
drawn in ONE batch, so the candidate count does not add sequential depth), plus
one batched Q pass, plus the refinement rounds.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse every WidowX/server/safety/preprocess/scoring helper from the Q3C deploy
# client so this client and the Q3C client behave identically off-policy.
_spec = importlib.util.spec_from_file_location(
    "deploy", ROOT / "scripts" / "deploy_pusht_real.py")
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)

from utils.diffusion import build_diffusion, build_pixel_denoiser, resolve_dp_params

# One row written by this file is a DP+Q3C row by construction. Q3C lives in
# deploy_pusht_real.py, IBC in deploy_pusht_real_ibc.py, plain DP in
# deploy_pusht_real_dp.py; all four append to the same results table.
ALGORITHM = "dpq3c"


def parse_args() -> argparse.Namespace:
    # Full copy of deploy_pusht_real.parse_args. --seed-dir is split into
    # --dp-dir / --q-dir and the DP sampler knobs (--cp/--dp-method/--dp-iters)
    # are added; every other flag is deliberately identical.
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dp-dir", type=Path, required=True,
                   help="Diffusion-Policy checkpoint dir (config.json, "
                        "norm_stats.pt, denoiser[_ema].pt). Generates the "
                        "control points.")
    p.add_argument("--q-dir", type=Path, required=True,
                   help="Q3C checkpoint dir (config.json, norm_stats.pt, "
                        "q_estimator[_ema].pt). Scores the control points; its "
                        "control-point generator is never loaded.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-ema", action="store_true",
                   help="use raw weights instead of the EMA copy (BOTH nets)")
    p.add_argument("--allow-mismatch", action="store_true",
                   help="downgrade the DP/Q checkpoint compatibility check to a "
                        "warning. The Q net then scores actions in an action "
                        "space it was not trained on; the ranking is not "
                        "meaningful. Diagnostics only.")
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
    p.add_argument("--camera-topics", nargs="+", default=d.CAMERA_TOPICS,
                   help="ROS topics, registered in THIS order. Default = the "
                        "order the training data was collected in "
                        f"({d.DATASET_CAMERA_TOPICS}).")
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
    p.add_argument("--step-duration", type=float, default=d.STEP_DURATION,
                   help="control period; also used as env move_duration. Default "
                        "is the collection's move_duration (20 Hz).")
    p.add_argument("--non-blocking", action="store_true",
                   help="the working reference uses blocking=True; this opts out")
    p.add_argument("--action-mode", default="2trans",
                   choices=["2trans", "3trans", "3trans1rot", "3trans3rot"])
    p.add_argument("--safety-max-xy-delta", type=float,
                   default=d.SAFETY_MAX_XY_DELTA)
    p.add_argument("--workspace-xyz", type=float, nargs=6, default=None,
                   metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                   help="override the server's workspace box (metres). Default "
                        f"{[d.WORKSPACE_BOUNDS[0][:3], d.WORKSPACE_BOUNDS[1][:3]]} = "
                        "the measured demo envelope + margin. The legacy box "
                        f"was {[d.LEGACY_WORKSPACE_BOUNDS[0][:3], d.LEGACY_WORKSPACE_BOUNDS[1][:3]]}, "
                        "which excluded 23%% of the demo steps and let z sag "
                        "30 mm below the working height. Only applied on init().")
    p.add_argument("--min-step-xy", type=float, default=0.0,
                   help="metres. If >0, any nonzero |dx|/|dy| below this is "
                        "snapped UP to it (sign kept); exact 0 stays 0. The "
                        "expert teleop is bang-bang (0 or >=1.5mm; measured "
                        "dead zone in (0,1.5mm)), so the policy can emit "
                        "sub-min-step OOD actions that the arm can't execute and "
                        "it locks. Suggested 0.0015. Default 0 = off.")
    p.add_argument("--lock-z", dest="lock_z", action="store_true", default=True)
    p.add_argument("--no-lock-z", dest="lock_z", action="store_false")
    p.add_argument("--fixed-z-height", type=float, default=d.FIXED_Z_HEIGHT)
    p.add_argument("--neutral-z-height", type=float, default=d.NEUTRAL_Z_HEIGHT)
    p.add_argument("--control-z", type=float, nargs="?", const=d.FIXED_Z_HEIGHT,
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
    p.add_argument("--fixed-gripper", type=float, default=d.FIXED_GRIPPER,
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
    p.add_argument("--force-fresh-init", action="store_true",
                   help="always call init(), even if the server already has a live "
                        "env. Only works if the server's cached env_params match "
                        "ours; otherwise it fails the config-hash check and the "
                        "server must be restarted.")
    p.add_argument("--no-reuse-existing-env", dest="reuse_existing_env",
                   action="store_false", default=True,
                   help="disable reusing an already-initialized server env")

    # --- control-point generation (DP; replaces the Q3C cp generator) --------
    p.add_argument("--cp", type=int, default=None,
                   help="how many control points the diffusion policy draws per "
                        "prediction. They come from ONE batched sampler call on "
                        "shared image features, so the cost is one encoder pass "
                        "plus --dp-iters denoiser passes at batch size --cp. "
                        "Default: the Q run's model.control_points, i.e. the "
                        "cloud size its scorer was trained against.")
    p.add_argument("--dp-method", choices=["ddim", "ddpm"], default="ddim",
                   help="diffusion sampler schedule used to draw the control "
                        "points. ddim (default) is the sub-sampled deterministic "
                        "chain; at --ddim-eta 0 the ONLY thing separating the --cp "
                        "draws is the initial noise. ddpm is the ancestral chain.")
    p.add_argument("--dp-iters", type=int, default=None,
                   help="denoising iterations run before the cloud is handed to "
                        "the Q estimator. ddim: the sub-sampled step count. ddpm: "
                        "a respaced ancestral chain with this many steps (>= the "
                        "training T runs the full chain). Default: the "
                        "checkpoint's ddim_eval_steps[0] for ddim, the full "
                        "training chain for ddpm.")
    p.add_argument("--ddim-eta", type=float, default=None,
                   help="DDIM stochasticity (default: ddim_eta from norm_stats). "
                        "0 = deterministic given the initial noise.")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="if set, torch.manual_seed before sampling for a "
                        "reproducible dry run.")

    # --- optional hybrid knobs. ALL OFF BY DEFAULT: with none of them set the
    # client is exactly draw-a-DP-cloud-then-rank-it, which is the baseline
    # every one of these has to beat. ------------------------------------------
    p.add_argument("--cp-score-norm", choices=["none", "zscore", "rank"],
                   default="none",
                   help="OFF by default. Rescale the Q scores WITHIN the cloud "
                        "before the softmax. Raw Q magnitude swings with the "
                        "state, so one --cp-temperature is near-greedy on some "
                        "frames and near-uniform on others; zscore fixes the "
                        "scale, rank also removes the influence of a single "
                        "wildly overestimated candidate. NO EFFECT under "
                        "--cp-selection argmax: both maps are monotone.")
    p.add_argument("--cascade-iters", type=int, default=None,
                   help="OFF by default. Two-stage sampling: denoise all --cp "
                        "candidates only this many steps, rank that cheap cloud "
                        "with the Q estimator, keep --cascade-topk, and finish "
                        "the remaining --dp-iters steps for the survivors only. "
                        "Same denoiser budget searches many more initial noises. "
                        "The filter ranks the predicted CLEAN sample, not the "
                        "noisy iterate. Sensible: 2-4 of a 10-step DDIM chain.")
    p.add_argument("--cascade-topk", type=int, default=None,
                   help="survivors kept by --cascade-iters. Default when the "
                        "cascade is on: --cp // 4 (min 1). Ignored otherwise.")
    p.add_argument("--q-guidance", type=float, default=0.0,
                   help="OFF by default (0). Classifier guidance: at every "
                        "denoising step, nudge the predicted clean sample along "
                        "the Q gradient before the sampler's own update. The "
                        "denoiser is NOT retrained and does not see the "
                        "gradient -- it is added outside the network. Only the "
                        "Q value head is differentiated (image features are "
                        "cached and detached), so the cost is one small MLP "
                        "forward+backward per denoising step. The gradient is "
                        "unit-normalized per candidate, so this is a step size "
                        "in normalized action units: try 0.01-0.1.")
    p.add_argument("--q-guidance-schedule", choices=["const", "linear"],
                   default="linear",
                   help="how --q-guidance scales over the chain. linear "
                        "(default when guidance is on) ramps from ~0 at high "
                        "noise to the full value at t=0, because the predicted "
                        "clean sample is a blurry conditional mean early on and "
                        "the Q estimator was trained on clean actions. const "
                        "applies the same weight throughout. Ignored when "
                        "--q-guidance is 0.")

    # --- policy (Q-side selection; unchanged from deploy_pusht_real.py) -----
    p.add_argument("--cp-selection", choices=["argmax", "sample"], default=None,
                   help="override CP-cloud selection (default: from the Q run's "
                        "norm_stats)")
    p.add_argument("--cp-temperature", type=float, default=None)
    p.add_argument("--inference", choices=["argmax", "sample", "langevin", "dfo"],
                   default="argmax",
                   help="how the Q estimator turns the DP cloud into an action. "
                        "argmax/sample = pure cloud ranking (default). langevin = "
                        "refine the DP cloud with Langevin MCMC against the Q "
                        "energy. dfo = derivative-free iterative refinement "
                        "(cheaper). argmax also uses --refine-iters as a stall "
                        "escape, see --argmax-stall-steps.")
    p.add_argument("--refine-iters", type=int, default=50,
                   help="langevin/dfo refinement iterations (train used 50). "
                        "Under --inference argmax this is the number of DFO "
                        "iterations the stall escape runs; 0 disables it.")
    p.add_argument("--argmax-stall-steps", type=int, default=10,
                   help="--inference argmax only: how many consecutive idle "
                        "actions (see --argmax-stall-action) are returned before "
                        "--refine-iters DFO iterations are run on top of the "
                        "cloud. Refinement stays on until a returned action is "
                        "no longer idle. 0 disables the escape.")
    p.add_argument("--argmax-stall-action", type=float, default=0.19,
                   help="an action is idle when every component it will execute "
                        "is below this in absolute value (NORMALIZED units, so "
                        "the same scale as the [-1,1] action bounds).")
    p.add_argument("--langevin-lr-init", type=float, default=0.1)
    p.add_argument("--langevin-lr-final", type=float, default=1e-5)
    p.add_argument("--dfo-noise-init", type=float, default=0.1)
    p.add_argument("--dfo-noise-decay", type=float, default=0.8)
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
    p.add_argument("--dump-dir", type=Path, default=ROOT / "deploy_dryrun_dpq3c")
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
    p.add_argument("--results-csv", type=Path, default=d.RESULTS_CSV,
                   help="results table appended to by --measure (created, with "
                        "its parent directories, if missing)")
    p.add_argument("--speed-csv", type=Path, default=d.SPEED_CSV,
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
# DP control-point generator (replaces PixelControlPointGenerator)
# ---------------------------------------------------------------------------

def build_dp_denoiser(env_cfg: dict, norm_stats: dict, in_channels: int, device):
    """Rebuild the denoiser + diffusion sampler exactly as the trainer did.

    Same body as deploy_pusht_real_dp.build_dp_policy; duplicated rather than
    imported so this client does not depend on that file's module-level import
    of deploy_pusht_real.py (which would execute it twice).
    """
    dp = resolve_dp_params(env_cfg)
    # norm_stats is the authority on what was actually trained; let it win.
    for k in ("num_train_timesteps", "beta_schedule", "prediction_type",
              "time_emb_dim", "denoiser_network_kind", "denoiser_width",
              "denoiser_depth"):
        if k in norm_stats:
            dp[k] = norm_stats[k]

    cond_dim = int(norm_stats.get("cond_dim", 0))
    if cond_dim:
        # Would need CondPixelDiffusionDenoiser from train_pusht_real_dp.py.
        raise NotImplementedError(
            f"DP checkpoint has cond_dim={cond_dim}: conditioned DP deploy is "
            "not wired up (the pushtWidowXdp batch is pixels-only).")

    enc_h = int(norm_stats.get("encoder_target_height",
                               env_cfg.get("encoder_target_height", 180)))
    enc_w = int(norm_stats.get("encoder_target_width",
                               env_cfg.get("encoder_target_width", 240)))
    # The head's width is the ACTION SHAPE the trainer used: 2 * action_chunk,
    # not 2. Reading it from act_min is the only reliable source -- the config's
    # `action_dim` stays 2 even for a chunked run (same reasoning as
    # deploy_pusht_real.build_models). Hardcoding 2 here silently rejects every
    # chunked checkpoint with a state_dict shape mismatch.
    action_dim = int(np.asarray(norm_stats["act_min"]).size)
    denoiser = build_pixel_denoiser(
        action_dim, in_channels, dp,
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


def _timestep_seq(diffusion, method: str, iters: int) -> list[int]:
    """Descending list of training timesteps the sampler will actually visit.

    DDIM: exactly the sub-sequence ``GaussianDiffusion.ddim_sample`` builds.
    DDPM: the same construction, de-duplicated and capped at num_timesteps --
    the ancestral chain cannot take more steps than it was trained with.
    """
    T = diffusion.num_timesteps
    if method == "ddim":
        idx = torch.linspace(0, T - 1, int(iters),
                             device=diffusion.device).round().long()
        return list(reversed(idx.tolist()))
    idx = torch.linspace(0, T - 1, min(int(iters), T),
                         device=diffusion.device).round().long()
    return list(reversed(sorted(set(idx.tolist()))))


def _q_grad(q_net, q_features, x0: torch.Tensor) -> torch.Tensor:
    """d Q / d action at `x0`, unit-normalized per candidate.

    Only the value HEAD is differentiated: `q_features` is the already-computed,
    detached output of the conv tower, so the backward pass runs through
    DenseResnetValue alone. That is what makes guidance affordable -- one small
    MLP forward+backward per denoising step, against a ResNet-18 that ran once
    for the whole control step.

    The per-candidate unit normalization makes --q-guidance a step size in
    normalized action units instead of a number that has to be retuned every
    time the Q head's output scale drifts.
    """
    with torch.enable_grad():
        a = x0.detach().requires_grad_(True)
        # 3-D action path: features (1,F) broadcast over (1,N,A). Reusing it
        # keeps the module's own cond broadcasting rather than duplicating it.
        score = q_net.score(q_features, a.unsqueeze(0)).sum()
        grad, = torch.autograd.grad(score, a)
    return grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-12)


@torch.no_grad()
def _run_chain(diffusion, head, feats, x, seq, method: str, eta: float,
               q_net=None, q_features=None, guidance: float = 0.0,
               guidance_schedule: str = "const"):
    """Walk `seq` of the reverse chain. Returns (x, last predicted x0).

    One implementation for both samplers so the cascade can stop and resume at
    any point and so guidance enters at exactly one place. With `guidance` 0 and
    `seq` the full schedule this reproduces ``GaussianDiffusion.ddim_sample`` /
    ``ddpm_sample``.

    Guidance is applied in x0 SPACE (reconstruction guidance): the Q gradient
    nudges the predicted clean sample, and the ordinary posterior/DDIM update
    then runs on the nudged x0. Guiding x_t directly would ask the Q net to
    score a noisy action, which it has never seen.
    """
    T = diffusion.num_timesteps
    B = x.shape[0]
    x0 = x
    guided = guidance > 0.0 and q_net is not None
    for i, t in enumerate(seq):
        t_batch = torch.full((B,), t, device=diffusion.device, dtype=torch.float32)
        out = head(feats, x, t_batch)
        x0 = diffusion._model_out_to_x0(x, t, out)
        if guided:
            # Ramp: x0 is a blurry conditional mean at high noise, so a linear
            # schedule spends the Q signal where the estimate is worth guiding.
            alpha_t = (guidance if guidance_schedule == "const"
                       else guidance * (1.0 - t / max(T - 1, 1)))
            if alpha_t > 0:
                x0 = x0 + alpha_t * _q_grad(q_net, q_features, x0)
                if diffusion.clip_sample:
                    x0 = x0.clamp(diffusion.action_low, diffusion.action_high)
        t_prev = seq[i + 1] if i + 1 < len(seq) else -1
        acp_t = diffusion.alphas_cumprod[t]
        acp_prev = (diffusion.alphas_cumprod[t_prev] if t_prev >= 0
                    else torch.ones((), device=diffusion.device))
        if method == "ddim":
            eps = diffusion._x0_to_eps(x, t, x0)
            sigma = (eta * torch.sqrt((1 - acp_prev) / (1 - acp_t))
                     * torch.sqrt(1 - acp_t / acp_prev))
            dir_xt = torch.sqrt((1 - acp_prev - sigma ** 2).clamp(min=0.0)) * eps
            x = torch.sqrt(acp_prev) * x0 + dir_xt
            if eta > 0 and t_prev >= 0:
                x = x + sigma * torch.randn_like(x)
        else:
            # Respaced ancestral posterior: for the jump t -> t_prev the
            # ordinary DDPM coefficients hold with alpha = acp_t / acp_prev and
            # beta = 1 - alpha. seq == the full chain reproduces ddpm_sample.
            alpha = acp_t / acp_prev
            beta = 1.0 - alpha
            mean = ((beta * torch.sqrt(acp_prev) / (1.0 - acp_t)) * x0
                    + ((1.0 - acp_prev) * torch.sqrt(alpha) / (1.0 - acp_t)) * x)
            if t_prev >= 0:
                var = beta * (1.0 - acp_prev) / (1.0 - acp_t)
                x = mean + torch.sqrt(var.clamp(min=0.0)) * torch.randn_like(x)
            else:
                x = mean
    return x, x0


@torch.no_grad()
def dp_control_points(diffusion, denoiser, obs_u8, n_cp: int, method: str,
                      iters: int, eta: float, action_dim: int,
                      q_net=None, q_features=None,
                      guidance: float = 0.0, guidance_schedule: str = "const",
                      cascade_iters: int | None = None,
                      cascade_topk: int | None = None) -> torch.Tensor:
    """Draw control points from the diffusion policy for ONE observation.

    Returns (1, P, action_dim) in the normalized action space -- the exact shape
    and semantics PixelControlPointGenerator returns in the Q3C client, so
    everything downstream (Q scoring, DFO/Langevin refinement, the stall escape)
    is untouched. P is `n_cp`, or `cascade_topk` when the cascade is on.

    The conv tower runs ONCE: the encoder output is the same for all n_cp draws
    and for every denoising step, so it is computed once and broadcast, and only
    the small denoiser head runs inside the sampling loop (the late-fusion trick
    PixelDiffusionDenoiser's own docstring prescribes). Cost is one encoder pass
    + `iters` head passes at batch n_cp, not n_cp * iters encoder passes.

    `cascade_iters`: draw n_cp candidates but only denoise them `cascade_iters`
    steps, rank that cheap cloud, keep `cascade_topk`, and finish only those.
    Same denoiser budget searches a far larger set of initial noises. The filter
    ranks the predicted CLEAN sample x0, never the partially-denoised x_t.
    """
    feats = denoiser.encode(obs_u8).expand(int(n_cp), -1)   # (n_cp, F)
    head = denoiser.denoiser                                # flat DiffusionDenoiser
    seq = _timestep_seq(diffusion, method, iters)
    x = torch.randn(int(n_cp), action_dim, device=diffusion.device)

    if cascade_iters:
        if q_net is None or q_features is None:
            raise ValueError("the cascade needs the Q estimator to filter with")
        k0 = min(int(cascade_iters), len(seq))
        x, x0 = _run_chain(diffusion, head, feats, x, seq[:k0], method, eta,
                           q_net, q_features, guidance, guidance_schedule)
        scores = q_net.score(q_features, x0.unsqueeze(0)).squeeze(-1).squeeze(0)
        k = max(1, min(int(cascade_topk), scores.numel()))
        keep = torch.topk(scores, k).indices
        x, feats = x[keep], feats[keep]
        seq = seq[k0:]

    if seq:
        x, _ = _run_chain(diffusion, head, feats, x, seq, method, eta,
                          q_net, q_features, guidance, guidance_schedule)
    if diffusion.clip_sample:
        x = x.clamp(diffusion.action_low, diffusion.action_high)
    return x.unsqueeze(0)                                   # (1, P, A)


def normalize_scores(logits: torch.Tensor, kind: str) -> torch.Tensor:
    """Rescale the Q scores WITHIN one cloud before the softmax.

    Raw Q magnitude swings with the state, so a single --cp-temperature acts
    near-greedy on one frame and near-uniform on the next. Both transforms are
    strictly monotone, so neither can change an argmax -- they exist for
    --cp-selection sample.

      zscore  (q - mean) / std over the cloud. Keeps relative gaps.
      rank    replace scores by their rank, mapped to [-1, 1]. Scale-free and
              immune to a single wildly overestimated candidate, at the cost of
              discarding how much better the winner actually is.
    """
    if kind == "zscore":
        return (logits - logits.mean()) / (logits.std() + 1e-6)
    if kind == "rank":
        order = torch.argsort(torch.argsort(logits)).to(logits.dtype)
        n = max(1, logits.numel() - 1)
        return 2.0 * order / n - 1.0
    return logits


def select_action(diffusion, denoiser, q_net, obs_u8, n_cp: int, dp_method: str,
                  dp_iters: int, ddim_eta: float, action_dim: int,
                  cp_selection: str, temperature: float,
                  cond: "torch.Tensor | None" = None, inference: str = "argmax",
                  refine_iters: int = 50, langevin_lr=(0.1, 1e-5),
                  dfo_noise=(0.1, 0.8),
                  stall: "d.ArgmaxStallDetector | None" = None,
                  score_norm: str = "none", guidance: float = 0.0,
                  guidance_schedule: str = "const",
                  cascade_iters: int | None = None,
                  cascade_topk: int | None = None):
    """deploy_pusht_real.select_action with the DP cloud in place of cp_gen.

    Line for line the Q3C selection logic; the ONLY change is where `cps` comes
    from. `cond` is the (1, cond_dim) conditioning vector for the Q net (the DP
    denoiser here is unconditioned by construction -- build_dp_denoiser rejects a
    conditioned checkpoint).

    At their defaults (`score_norm="none"`, `guidance=0`, `cascade_iters=None`)
    the four optional knobs are entirely inert and this is the plain
    draw-a-cloud-then-rank path.
    """
    q_net._cond = cond
    features = q_net.encode(obs_u8)                   # (1, feat)
    cps = dp_control_points(diffusion, denoiser, obs_u8, n_cp, dp_method,
                            dp_iters, ddim_eta, action_dim,
                            q_net=q_net, q_features=features,
                            guidance=guidance,
                            guidance_schedule=guidance_schedule,
                            cascade_iters=cascade_iters,
                            cascade_topk=cascade_topk)        # (1, P, A)

    def refine_dfo():
        A = cps.shape[-1]
        amin = torch.full((A,), -1.0, device=cps.device)
        amax = torch.full((A,), 1.0, device=cps.device)   # normalized bounds
        return d._refine_dfo(q_net, features, cps, amin, amax,
                             refine_iters, dfo_noise[0], dfo_noise[1])

    if inference in ("langevin", "dfo"):
        if inference == "langevin":
            A = cps.shape[-1]
            amin = torch.full((A,), -1.0, device=cps.device)
            amax = torch.full((A,), 1.0, device=cps.device)
            act = d._refine_langevin(q_net, features, cps, amin, amax,
                                     refine_iters, langevin_lr[0], langevin_lr[1])
        else:
            act = refine_dfo()
        return act.detach().cpu().numpy()
    logits = q_net.score(features, cps).squeeze(-1)   # (1, P)
    if cp_selection == "sample":
        scores = normalize_scores(logits.squeeze(0), score_norm)
        probs = torch.softmax(scores / max(temperature, 1e-6), dim=-1)
        idx = int(torch.multinomial(probs, 1).item())
    else:
        idx = int(logits.squeeze(0).argmax().item())
    act_np = cps[0, idx].detach().cpu().numpy()       # normalized action
    if stall is None or refine_iters <= 0:
        return act_np
    if stall.refine_now():
        if stall.kicks == 1 or stall.idle == stall.patience:
            # Only the transition, so a long stall does not flood the step log.
            print(f"[stall] {stall.patience} action(s) with every executed "
                  f"component under {stall.threshold}; refining with "
                  f"{refine_iters} DFO iteration(s) until one reaches it")
        act_np = refine_dfo().detach().cpu().numpy()
    # Judge what is actually returned, refined or not: the latch releases only
    # when a command large enough to move the arm goes out.
    stall.observe(act_np)
    return act_np


def check_compatible(dp_meta: dict, q_meta: dict, allow: bool) -> None:
    """The two checkpoints must agree on what a candidate action MEANS.

    The DP draws in its own normalized action space and the Q net scores in its
    own; if the two disagree on the action width, the min-max bounds, the
    normalization range or the observation the encoders were fed, the argmax is
    over numbers that mean different things in the two nets and the whole hybrid
    is noise. Hard error by default.
    """
    problems = []
    for name in ("action_dim", "norm_range", "frame_stack", "cam_ids",
                 "image_hw", "in_channels"):
        if dp_meta[name] != q_meta[name]:
            problems.append(f"  {name}: dp={dp_meta[name]!r} q={q_meta[name]!r}")
    for name in ("act_min", "act_max"):
        a, b = np.asarray(dp_meta[name]), np.asarray(q_meta[name])
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-6):
            problems.append(f"  {name}: dp={a.tolist()} q={b.tolist()}")
    if not problems:
        return
    msg = ("DP and Q3C checkpoints are not compatible:\n" + "\n".join(problems)
           + "\nThe Q estimator would rank DP samples drawn in a different "
             "action/observation space than the one it was trained on.")
    if not allow:
        raise SystemExit(msg + "\nPass --allow-mismatch to run anyway "
                               "(diagnostics only).")
    print("[WARN] " + msg + "\n[WARN] --allow-mismatch: continuing; the Q "
          "ranking is NOT meaningful.")


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
    dp_dir = args.dp_dir.resolve()
    q_dir = args.q_dir.resolve()

    # --- checkpoint metadata -------------------------------------------------
    q_cfg = d.load_run_config(q_dir)
    q_ns = torch.load(q_dir / "norm_stats.pt", map_location="cpu",
                      weights_only=False)
    dp_cfg = d.load_run_config(dp_dir)
    dp_ns = torch.load(dp_dir / "norm_stats.pt", map_location="cpu",
                       weights_only=False)

    # The Q run is the authority on the deployed observation/action contract:
    # it is the net whose ranking decides the command, and the results table's
    # Q3C rows were written from these fields.
    act_min = np.asarray(q_ns["act_min"], np.float32)
    act_max = np.asarray(q_ns["act_max"], np.float32)
    norm_range = tuple(q_ns.get("action_norm_range", (-1.0, 1.0)))
    cp_selection = args.cp_selection or str(q_ns.get("cp_selection", "argmax"))
    cp_temp = (args.cp_temperature if args.cp_temperature is not None
               else float(q_ns.get("cp_selection_temperature", 1.0)))

    frame_stack = int(q_cfg.get("frame_stack", 2))
    cams = tuple(q_ns.get("camera_streams", q_cfg.get("camera_streams", ["images1"])))
    image_h = int(q_cfg.get("image_height", 240))
    image_w = int(q_cfg.get("image_width", 320))
    in_channels = 3 * len(cams) * frame_stack
    cam_ids = d.camera_ids_from_streams(cams)
    topic_camera_ids = d.resolve_topic_camera_ids(args.camera_topics,
                                                  args.topic_camera_ids)

    dp_frame_stack = int(dp_ns.get("frame_stack", dp_cfg.get("frame_stack", 2)))
    dp_cams = tuple(dp_ns.get("camera_streams",
                              dp_cfg.get("camera_streams", ["video1"])))
    dp_hw = dp_ns.get("image_hw", (int(dp_cfg.get("image_height", 240)),
                                   int(dp_cfg.get("image_width", 320))))
    dp_in_channels = int(dp_ns.get("in_channels",
                                   3 * len(dp_cams) * dp_frame_stack))
    check_compatible(
        dict(action_dim=int(np.asarray(dp_ns["act_min"]).size),
             norm_range=tuple(dp_ns.get("action_norm_range", (-1.0, 1.0))),
             frame_stack=dp_frame_stack,
             cam_ids=tuple(d.camera_ids_from_streams(dp_cams)),
             image_hw=(int(dp_hw[0]), int(dp_hw[1])),
             in_channels=dp_in_channels,
             act_min=dp_ns["act_min"], act_max=dp_ns["act_max"]),
        dict(action_dim=int(act_min.size),
             norm_range=tuple(norm_range),
             frame_stack=frame_stack,
             cam_ids=tuple(cam_ids),
             image_hw=(image_h, image_w),
             in_channels=in_channels,
             act_min=act_min, act_max=act_max),
        args.allow_mismatch)

    # EEF (x, y) conditioning: present only for Q runs trained with --cond-eef-xy.
    # cond_min/cond_max are the TRAINING workspace bounds; the live proprio must
    # be normalized with these exact numbers or the conditioning is off-scale.
    cond_dim = int(q_ns.get("cond_dim", 0))
    cond_min = cond_max = None
    if cond_dim:
        if str(q_ns.get("cond_kind", "")) != "eef_xy":
            raise ValueError(
                f"Q checkpoint has cond_dim={cond_dim} but cond_kind="
                f"{q_ns.get('cond_kind')!r}; this client only knows eef_xy")
        cond_min = np.asarray(q_ns["cond_min"], np.float32)
        cond_max = np.asarray(q_ns["cond_max"], np.float32)

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")

    # Q side. build_models also constructs a control-point generator; it is
    # discarded here -- the DP replaces it -- but the call is kept whole so the
    # Q estimator is built with byte-identical arguments to the Q3C client.
    _cp_gen, q_net = d.build_models(q_cfg, in_channels, device, cond_dim=cond_dim,
                                    norm_stats=q_ns)
    del _cp_gen
    suffix = "" if args.no_ema else "_ema"
    d.load_weights(q_net, q_dir / f"q_estimator{suffix}.pt", device)

    # DP side.
    denoiser, diffusion, dp = build_dp_denoiser(dp_cfg, dp_ns, in_channels, device)
    dp_weights = torch.load(dp_dir / f"denoiser{suffix}.pt", map_location=device,
                            weights_only=True)
    denoiser.load_state_dict(dp_weights)
    denoiser.eval()

    action_dim = int(act_min.size)
    n_cp = args.cp
    if n_cp is None:
        n_cp = int(q_cfg.get("model", {}).get("control_points", 50))
    if n_cp < 1:
        raise SystemExit("--cp must be >= 1")
    num_train_timesteps = int(dp.get("num_train_timesteps", 100))
    dp_iters = args.dp_iters
    if dp_iters is None:
        if args.dp_method == "ddim":
            ev = dp_ns.get("ddim_eval_steps", dp.get("ddim_eval_steps", [10]))
            dp_iters = int(ev[0]) if ev else 10
        else:
            dp_iters = num_train_timesteps
    if dp_iters < 1:
        raise SystemExit("--dp-iters must be >= 1")
    if args.dp_method == "ddpm" and dp_iters > num_train_timesteps:
        print(f"[WARN] --dp-iters {dp_iters} > the training chain "
              f"{num_train_timesteps}; clipped (ddpm cannot take more steps "
              f"than it was trained with).")
        dp_iters = num_train_timesteps
    ddim_eta = args.ddim_eta
    if ddim_eta is None:
        ddim_eta = float(dp_ns.get("ddim_eta", dp.get("ddim_eta", 0.0)))
    if args.sample_seed is not None:
        torch.manual_seed(args.sample_seed)

    # --- optional knobs: resolve, validate, and say what is actually on ------
    cascade_iters = args.cascade_iters
    cascade_topk = args.cascade_topk
    if cascade_iters is not None:
        if cascade_iters < 1:
            raise SystemExit("--cascade-iters must be >= 1 (omit it to disable)")
        if cascade_iters >= dp_iters:
            raise SystemExit(
                f"--cascade-iters {cascade_iters} >= --dp-iters {dp_iters}: the "
                "cascade would filter after the chain has already finished, "
                "which is just a smaller cloud. Use a value strictly below "
                f"{dp_iters} (2-4 is the useful range for a 10-step chain).")
        if cascade_topk is None:
            cascade_topk = max(1, n_cp // 4)
        if cascade_topk < 1 or cascade_topk > n_cp:
            raise SystemExit(f"--cascade-topk must be in [1, --cp]={n_cp}")
        print(f"[cascade] ON: {n_cp} candidates x {cascade_iters} step(s) -> "
              f"keep top {cascade_topk} -> finish {dp_iters - cascade_iters} "
              f"more step(s). Filter ranks the predicted clean sample.")
    elif cascade_topk is not None:
        print("[WARN] --cascade-topk given without --cascade-iters; ignored.")
        cascade_topk = None

    if args.q_guidance < 0:
        raise SystemExit("--q-guidance must be >= 0 (0 disables it)")
    if args.q_guidance > 0:
        print(f"[guidance] ON: alpha={args.q_guidance} "
              f"schedule={args.q_guidance_schedule} -- one Q value-head "
              f"forward+backward per denoising step ({dp_iters} per action).")
        if args.inference in ("dfo", "langevin"):
            print(f"[WARN] --q-guidance with --inference {args.inference} "
                  "applies the Q estimator BOTH inside the denoising chain and "
                  "again as post-hoc refinement. Not wrong, but the two effects "
                  "are confounded; sweep guidance with argmax/sample.")

    if args.cp_score_norm != "none" and cp_selection != "sample":
        print(f"[WARN] --cp-score-norm {args.cp_score_norm} has NO EFFECT under "
              f"--cp-selection {cp_selection}: both normalizations are monotone "
              "so the argmax is unchanged. Pass --cp-selection sample.")

    # experiments.csv has a fixed column set shared with the Q3C/IBC/DP clients,
    # so the sampler settings ride in the `inference` cell rather than as new
    # columns (adding columns would rewrite the header of a table those three
    # scripts also append to). Without this, two runs differing only in --cp or
    # --dp-iters would share a key and be recorded as repeat trials of one
    # condition. Only NON-DEFAULT knobs are appended, so the all-off baseline
    # keeps the short label. `refine_iters` keeps its Q3C meaning.
    inference_label = f"{args.dp_method}{dp_iters}x{n_cp}"
    if cascade_iters is not None:
        inference_label += f"c{cascade_iters}k{cascade_topk}"
    if args.q_guidance > 0:
        inference_label += f"g{args.q_guidance:g}{args.q_guidance_schedule[0]}"
    inference_label += f"+{args.inference}"
    if cp_selection == "sample":
        inference_label += f"t{cp_temp:g}"
        if args.cp_score_norm != "none":
            inference_label += f"n{args.cp_score_norm[0]}"

    print(f"Loaded weights ({'raw' if args.no_ema else 'EMA'}):")
    print(f"  DP denoiser  <- {dp_dir}")
    print(f"  Q estimator  <- {q_dir}")
    print(f"  frame_stack={frame_stack} cameras={cams} model_hw=({image_h},{image_w}) "
          f"in_channels={in_channels}")
    print(f"  control points: {n_cp} draws via {args.dp_method} "
          f"({dp_iters} denoising iters"
          + (f", eta={ddim_eta}" if args.dp_method == "ddim" else "")
          + f", pred={dp.get('prediction_type')}, T={num_train_timesteps})")
    print(f"  Q selection: inference={args.inference} cp_selection={cp_selection} "
          f"(temp={cp_temp}) refine_iters={args.refine_iters}  device={device}")
    _opts = []
    if args.cp_score_norm != "none":
        _opts.append(f"score-norm={args.cp_score_norm}")
    if cascade_iters is not None:
        _opts.append(f"cascade={cascade_iters}->top{cascade_topk}")
    if args.q_guidance > 0:
        _opts.append(f"guidance={args.q_guidance}({args.q_guidance_schedule})")
    print(f"  optional knobs: {', '.join(_opts) if _opts else 'none (baseline)'}")
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
            print("[INFO] Server already has a live env; skipping init() and "
                  "reusing it. (Re-initializing with different env_params is what "
                  "triggers 'Incompatible config with hash with server'.)")
            print("[WARN] The live env keeps the env_params it was FIRST "
                  "initialized with -- not the ones printed above. If the robot "
                  "behaves as though action_mode/lock_z/etc. differ, restart "
                  "`widowx_env_service --server` and re-run to apply ours.")

    if reuse_existing_env:
        d.set_reqrep_timeout_ms(client, max(1, args.rpc_timeout_ms))
    else:
        init_status = d.init_widowx_with_retry(
            client, env_params, args.im_size, WidowXStatus, args)
        if init_status != WidowXStatus.SUCCESS:
            raise RuntimeError(
                f"WidowX init failed after {args.init_retries} attempts with "
                f"status={d.status_name(init_status, WidowXStatus)}.\n"
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
    reset_status = d.reset_widowx_with_retry(client, WidowXStatus, args, args.i_traj)
    if reset_status != WidowXStatus.SUCCESS:
        raise RuntimeError(
            f"WidowX reset failed with "
            f"status={d.status_name(reset_status, WidowXStatus)}")
    print(f"Reset done (itraj={args.i_traj}).")

    # Physically actuate the clamp. In 2trans mode the gripper dim is never sent
    # (step_action gets action[:2]), and reset can leave the clamp open, so we
    # command it explicitly here. 0.0 = closed to grip the pusher.
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
                  f"(status={d.status_name(move_status, WidowXStatus)}); continuing.")

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
                eef0 = d.eef_x_from_obs(st)
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
        return d.build_stack_frame(raw_obs, cam_ids, topic_camera_ids,
                                   (image_h, image_w), gains=exposure_gains)

    def raw_frame(raw_obs) -> np.ndarray:
        return d.frame_for_camera(raw_obs, cam_ids[0], topic_camera_ids)

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
            obs_u8 = d.stack_to_tensor(frame_buf, device)
            na = select_action(diffusion, denoiser, q_net, obs_u8, n_cp,
                               args.dp_method, dp_iters, ddim_eta, action_dim,
                               cp_selection, cp_temp,
                               cond=make_cond(raw_obs), inference=args.inference,
                               refine_iters=args.refine_iters,
                               langevin_lr=(args.langevin_lr_init, args.langevin_lr_final),
                               dfo_noise=(args.dfo_noise_init, args.dfo_noise_decay),
                               score_norm=args.cp_score_norm,
                               guidance=args.q_guidance,
                               guidance_schedule=args.q_guidance_schedule,
                               cascade_iters=cascade_iters,
                               cascade_topk=cascade_topk)
            act = d.unnormalize(na, act_min, act_max, norm_range)
            d.save_fed_png(args.dump_dir / f"fed_{i:03d}", list(frame_buf)[-1], cam_ids)
            print(f"[{i:03d}] norm={np.round(na, 3)} -> action(dx,dy)={np.round(act, 4)}")
            time.sleep(args.step_duration)
        client.stop()
        print(f"Dry run done. Inspect {args.dump_dir}/fed_000.png before live "
              f"control.")
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

    # argmax on its own cannot leave a fixed point, so --refine-iters doubles as
    # its escape hatch instead of being silently ignored. The other inference
    # modes already refine on every call and need no detector. NOTE the DP cloud
    # is redrawn from fresh noise every call, so a hybrid argmax is less prone to
    # the deterministic fixed point than Q3C's -- the escape is kept anyway
    # because a confidently-idle Q landscape still traps it.
    stall_detector = None
    if args.inference == "argmax" and args.argmax_stall_steps > 0:
        if args.refine_iters > 0:
            stall_detector = d.ArgmaxStallDetector(args.argmax_stall_steps,
                                                   args.argmax_stall_action,
                                                   n_exec=2 * exec_horizon)
            print(f"[stall] argmax escape armed: {args.refine_iters} DFO "
                  f"iteration(s) after {args.argmax_stall_steps} action(s) with "
                  f"every executed component under "
                  f"{args.argmax_stall_action} (normalized)")
        else:
            print("[stall] --refine-iters 0: argmax stall escape disabled")

    # One entry point for the policy so the control loop, the timer and the
    # post-episode FLOP count all measure the identical call.
    def predict(obs_u8, raw_obs, stall=None):
        return select_action(
            diffusion, denoiser, q_net, obs_u8, n_cp, args.dp_method, dp_iters,
            ddim_eta, action_dim, cp_selection, cp_temp,
            cond=make_cond(raw_obs), inference=args.inference,
            refine_iters=args.refine_iters,
            langevin_lr=(args.langevin_lr_init, args.langevin_lr_final),
            dfo_noise=(args.dfo_noise_init, args.dfo_noise_decay),
            stall=stall,
            score_norm=args.cp_score_norm,
            guidance=args.q_guidance,
            guidance_schedule=args.q_guidance_schedule,
            cascade_iters=cascade_iters,
            cascade_topk=cascade_topk)

    timer = d.InferenceTimer(device)
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
            obs_u8 = d.stack_to_tensor(frame_buf, device)
            last_obs_u8, last_raw_obs = obs_u8, raw_obs

            if not pending:
                with timer.measure():
                    na_full = predict(obs_u8, raw_obs, stall_detector)
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
                print(f"[min-step] snapped {np.round(na, 3)} -> "
                      f"dx,dy={np.round(act_xy, 4)}")

            # HARD SAFETY: never move closer to the robot than the start pose.
            cur_x = d.eef_x_from_obs(raw_obs)
            act_xy, floored = d.apply_approach_floor(act_xy, cur_x, approach_floor_x)
            if floored:
                print(f"[SAFETY] approach floor: clipped dx at x={cur_x:.4f} "
                      f"(floor={approach_floor_x:.4f})")

            action_7d = d.to_action_7d(act_xy, args.fixed_gripper)
            action_7d = d.safety_clip_action(action_7d, args.action_mode,
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
                cur_z = d.z_from_obs(raw_obs)
                if z_cmd is None and cur_z is not None:
                    z_cmd = cur_z
                if z_cmd is not None:
                    dz, z_cmd = d.control_z_step(
                        z_cmd, cur_z, args.control_z, args.control_z_gain,
                        args.control_z_max_dz, args.control_z_windup)
                    action_7d[2] = dz
            elif args.z_hold > 0:
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

            zmsg = ""
            if args.control_z is not None:
                mz = d.z_from_obs(raw_obs)
                zmsg = (f" z={'n/a' if mz is None else f'{mz:.4f}'}"
                        f" cmd={'n/a' if z_cmd is None else f'{z_cmd:.4f}'}"
                        f" dz={dz:+.5f}")
            print(f"[{step:03d}] chunk[{chunk_idx}/{exec_horizon - 1}] "
                  f"norm={np.round(na, 3)} -> "
                  f"env_action={np.round(env_action, 5)}{zmsg}")

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
            # experiments.csv has one seed_dir cell and this run has two
            # checkpoints; the pair is what identifies the condition.
            "seed_dir": f"{dp_dir}|{q_dir}",
            "inference": inference_label,
            "refine_iters": args.refine_iters,
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
                frames = {cam: d.frame_for_camera(final_obs, cam, topic_camera_ids)
                          for cam in topic_camera_ids}
                scores = d.score_final_frames(frames)
                trial = d.append_result_row(args.results_csv, dict(key, **scores))
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
                    # No detector is passed: this measures the ordinary path, and
                    # feeding it one more proposal would also skew the kick count.
                    gflops = d.count_gflops(
                        lambda: predict(last_obs_u8, last_raw_obs))
                # Sequential depth of one action: dp_iters denoiser passes (the
                # whole cloud is drawn in ONE batch, so --cp adds width, not
                # depth), then the batched Q pass, then the refinement rounds.
                # Guidance adds one Q value-head pass per denoising step; the
                # cascade adds its one filtering pass.
                net_evals = dp_iters + d.energy_net_evals(args.inference,
                                                          args.refine_iters)
                if args.q_guidance > 0:
                    net_evals += dp_iters
                if cascade_iters is not None:
                    net_evals += 1
                if stall_detector is not None and timer.samples_ms:
                    # argmax costs one Q pass; the stall escape adds refine_iters
                    # more on the calls where it fires, so the per-call figure is
                    # only honest as the average over the episode.
                    base = dp_iters + 1
                    if args.q_guidance > 0:
                        base += dp_iters
                    if cascade_iters is not None:
                        base += 1
                    net_evals = round(
                        base + args.refine_iters * stall_detector.kicks
                        / len(timer.samples_ms), 3)
                    print(f"[stall] escape fired on {stall_detector.kicks} of "
                          f"{len(timer.samples_ms)} inference(s)")
                d.report_inference_cost(
                    args.speed_csv, timer, dict(key, trial=trial,
                                                device=str(device)),
                    n_steps=step + 1, exec_horizon=exec_horizon,
                    net_evals=net_evals,
                    gflops=gflops,
                    params_m=d.count_params_m(denoiser, q_net))
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
