# Push-T Real-Robot Deployment — File Map & Health-Check Guide

Purpose: let another agent do a full health check of the Push-T real-robot
deploy (Q3CIBC combinedv2 policy on the LiraLab WidowX). Lists every file
related to **training**, **deployment**, and **forensics**, what it is, and how
to use it. Read `PUSHT_DEPLOY_HANDOFF.md` first for the running status; this doc
is the file-level index.

## TL;DR current state (2026-07)

Pipeline is verified correct end-to-end; the residual failure is
**behavioral-cloning covariate shift**, not a script bug. Fixed so far:
zero-motion collapse (blocking step + fresh-frame guard) and OOD start pose
(auto-move to the demo start). From a correct start the arm still drifts off the
demo manifold and freezes (learned "hold" ≈ 0 action). Robot executes commands
faithfully (corr(command, EEF-delta) ≈ +0.9/+0.96), policy is healthy offline
(MAE 0.02 vs GT). See "Findings" at the bottom.

---

## 1. Training

| File | What it is |
|---|---|
| `scripts/train_pusht_real.py` | Launcher. Writes an immutable per-run config and points the trainer (`combinedv2_cpascounter_training`) at it via `Q3C_CONFIG_PATH`. One config per seed so Slurm array tasks don't race. **v2 (2026-07-17): defaults retuned to the best pushing_pixels hyperparam-search recipe (trial 95, sr 0.99: lr 3e-4 both nets, clamp 10, cosine_warm_restarts t0=50k, 150k steps, batch 128, 0 uniform/langevin negatives — v1 seeds had lr 1e-3, clamp 20, plain cosine, 100k, batch 64, and 32+32 negatives from silent trainer defaults) + train-time appearance augmentation ON (`image_aug`, `--no-aug` to disable). Output root now `checkpoints/pusht_real_combinedv2_v2` so v1 seeds stay intact.** |
| `scripts/train_pusht_real_array.sbatch` | Slurm array job that trains the seeds (11/29/47/83). |
| `scripts/submit_pusht_real.sh` | Convenience wrapper that `sbatch`es the array with log dirs set. |
| `combinedv2_cpascounter_training.py` | The actual trainer (CP-cloud + Q-estimator, "combinedv2"). The policy the deploy client loads. |
| `utils/datasets.py` → `PushTRealPixelsDataset` | Dataset loader. Reads the zip, decodes `images1` JPEGs with TensorFlow (`tf.io.decode_jpeg`, AREA resize to (240,320)), channel-stacks `frame_stack` frames oldest→newest, min-max normalizes the 2-D action to (-1,1). **Action target = `policy_out.pkl` step `actions[:2]` = planar EEF delta (x,y) metres.** This is the ground truth for what the model outputs and what deploy must send. **`augment=True` (v2 training) applies per-sample appearance augmentation: photometric (per-channel gain 0.7–1.3, brightness ±0.15, contrast 0.7–1.3, saturation 0.6–1.4) + random crop-zoom 0.85–1.0, all drawn once and shared across the frame stack (so no fake inter-frame motion); gaussian sensor noise per frame. Ranges cover the measured deploy shift (T at ~0.67× training red). Deploy/diagnose unaffected (default off).** |
| `models.py`, `loss.py`, `normalizations.py` | Model/loss/normalization building blocks used by the trainer. |
| `hyperparams_search.py` | script used to test a hyperparameter combination for different simulation environments. |
| `submit_experiments.sh`, `batches/*` | Scripts used to run a batch of experiments for the hyperparameter search in the simulation environments. |

### Training data
| File | What it is |
|---|---|
| `data/03-23-pusht-data.zip` (~9 GB) | The demonstration archive. 110 trajectories under `raw/traj_group0/traj*/`. Each has `images1/im_*.jpg` (blue Logitech = deploy input stream), `images0/*` (D435, unused — removed from rig), and `policy_out.pkl` (per-step `actions` (7,), `new_robot_transform` (4×4 EEF pose), `delta_robot_transform`). **All demos start at a near-identical EEF pose x≈0.117, y≈−0.019, z≈0.02 (rotation std ≈ 0).** |
| `data/bridge_data_robot_pusht.zip` (~4 GB) | Raw bridge-format robot data (superset / original capture). |
| `data/example_images1_blue.jpg` | Single reference blue frame (== `scripts/assets/pusht_images1_ref.jpg`). Upright red T top-left, target outline center-left, small square bottom-left. |
| `data/example_images0_D435.jpg` | Reference D435 frame (unused camera). |
| `data/example_episode/im_*.jpg` | **Full images1 sequence of one random training episode (traj82, 445 frames)** + `SOURCE.txt`. For eyeballing how a real demo looks over time (T motion, pusher visibility, lighting) vs deploy captures. |
| `data/eval_example.py` | Reference eval script from a **different** project (bridge/jaxrl_m, z-score norm, 7-DoF). Not ours — kept for comparison. It is the source of the "move EEF to `initial_eep` before rollout" pattern we adopted. |

### Checkpoints
| Path | What it is |
|---|---|
| `checkpoints/pusht_real_combinedv2/seed_00{11,29,47,83}/` | Trained policies. Each dir: `control_point_generator{,_ema}.pt`, `q_estimator{,_ema}.pt`, `config.json` (env/model), `norm_stats.pt` (`act_min/act_max`, `action_norm_range`, `cp_selection`). **Note: empty on this (WSL/cluster) checkout — checkpoints live on the cluster / Alienware.** |

---

## 2. Deployment

| File | What it is |
|---|---|
| `scripts/deploy_pusht_real.py` | **The deploy client.** Connects to the WidowX server (`widowx_env_service`), captures the blue frame, preprocesses (RGB, AREA resize to (240,320), no channel swap by default — server frame already has red in ch0), channel-stacks, runs CP-cloud argmax, denormalizes to metres, sends `(dx,dy)` via `step_action`. Key mechanics: **blocking step** (default; arm finishes each delta before next frame — non-blocking starved inter-frame motion → `(-,-)` collapse), **fresh-frame guard** (rejects a frame byte-identical to previous; server sometimes repeats images), **initial-move** (moves EEF to the demo start pose from the asset before the loop), **forensic `--log-dir`**. Env config `DEPLOY_ENV_PARAMS`: `action_mode:"2trans"` (2-DoF planar), `lock_z:True`, `fixed_z_height:0.02`, `move_duration:0.08`. Flags: `--dry-run`, `--swap-rgb`, `--obs-key`, `--hz`, `--steps`, `--non-blocking`, `--settle`, `--no-require-fresh`, `--fresh-timeout`, `--no-initial-move`, `--start-eep-npy`, `--log-dir`. |
| `scripts/align_pusht_camera.py` | Live camera-alignment helper. Overlays the live blue stream on a training reference frame so you can physically nudge the Logitech until the view matches training. Misalignment is a classic silent deploy failure. Needs the robot server up. |
| `scripts/assets/pusht_images1_ref.jpg` | Reference training frame for alignment (== `data/example_images1_blue.jpg`). |
| `scripts/assets/pusht_start_eep.npy` | **4×4 EEF start transform = mean of all 110 demo start poses** (x≈0.117, y≈−0.019, z≈0.02). Loaded by `deploy_pusht_real.py` to position the arm in-distribution before a rollout. |
| `data/eval_example.py` | (see above) reference for the deploy pattern. |

### Robot server (NOT in this repo — on the Alienware)
`~/bridge_data_robot` — the `widowx_env_service` server/client and launch files.
Edited for the no-D435 single-camera rig (see `PUSHT_DEPLOY_HANDOFF.md` "Setup
facts"). Client conda env `q3c_deploy` needs numpy<2 + opencv-python==4.10.0.84.

---

## 3. Forensics / diagnostics

| File | What it is |
|---|---|
| `scripts/diagnose_pusht_actions.py` (+ `.sbatch`) | **Offline collapse check.** Runs each checkpoint over the training dataset, compares predicted vs GT actions (mean/std, corr(dx,dy), MAE, quadrant histogram). Proved all 4 seeds healthy offline (MAE 0.02, spread ++-leaning). `--zero-motion` flag duplicates the newest frame across the stack (kills inter-frame motion) — proved motion matters (MAE 0.02→0.22) but does NOT cause the collapse. Run on cluster (needs torch + tf). |
| `scripts/check_preproc_parity.py` | Sweeps geometric transforms (identity/flip_v/flip_h/rot180/chan_reverse/transpose) on captured raw frames, reports action + quadrant per transform. **Caveat: comparing to a dataset-*average* reference is invalid for a single/few T poses (dataset mean ≈ 0 by symmetry). Use only to see how sensitive the policy is to orientation, not to pick the "right" transform.** Geometry was ultimately ruled out by direct visual + red-box comparison. |
| `scripts/check_brightness_parity.py` | Measures live-frame red-T brightness (peak redness = ch0 − mean(ch1,ch2)) vs the demo target (peak ≈ 120) and prints OK / TOO DIM. Reusable lighting-tuning loop; runs on captured `raw_*.npy`, no robot needed. Deploy frames measured dim (peak 83 vs 120). |
| `deploy_pusht_real.py --dry-run` | Captures `deploy_dryrun/raw_*.npy` (exact server frame) + `fed_*.png` (what the model sees) with NO motion. Use to confirm the red T renders and to feed the parity/brightness tools. |
| `deploy_pusht_real.py --log-dir DIR` | Per-step forensic log: `DIR/raw/NNNN.npy`, `DIR/fed/NNNN.png`, `DIR/steps.jsonl` (step, timestamp, normalized action, metric action, **7-DoF EEF proprio state**). The primary artifact for post-hoc rollout analysis. |

### Forensic result artifacts
| Path | What it is |
|---|---|
| `results/pusht_action_diagnostic.json` | Baseline (real 2-frame stack) diagnose output — healthy. |
| `results/pusht_action_diagnostic_zeromotion.json` | `--zero-motion` diagnose output — MAE 10× worse but no collapse. |
| `results/log.txt` | stdout of an early closed-loop run (pre-forensic-logging). |
| `results/run01/` | Forensic log, neutral start (pre initial-move fix). Arm drifts from the T to a workspace corner and freezes. |
| `results/run02/` | Forensic log, WITH initial-move to demo start. Start pose now correct; arm still drifts to corner and freezes → covariate shift. |

---

## 4. Findings (chronological, what's ruled in/out)

1. **`(-,-)` runaway root cause = zero-motion frame stack.** Non-blocking
   `step_action` grabbed frames before the commanded move landed → the 2-frame
   stack had ~zero inter-frame motion → OOD → collapse. Fixed: blocking step +
   duplicate-frame guard. Collapse gone; actions now spread across quadrants.
2. **Motion matters for accuracy** (`diagnose --zero-motion`: MAE 0.02→0.22) but
   does not by itself cause the straight-line collapse.
3. **OOD start pose.** Deploy reset to neutral (x≈0.29); all demos start at
   x≈0.117 (17 cm off). Fixed: auto-move EEF to `scripts/assets/pusht_start_eep.npy`.
4. **Geometry / channel / aspect all clean.** Raw is 480×640 (4:3) → 240×320
   (4:3), no distortion. Red is in ch0. T position matches demos. No flip.
5. **Robot mapping verified un-flipped.** corr(commanded, actual EEF delta) =
   +0.89 (x), +0.96 (y) from `results/run02`. No sign/axis bug.
6. **Residual = behavioral-cloning covariate shift.** From a correct start with a
   good obs match, the policy makes small correct-signed moves, fails to engage
   the T on the first approach, drifts off the demo manifold, and freezes at its
   learned near-zero "hold". Only measured OOD axis: T is dimmer (peak redness
   83 vs 120). 110 demos is small and BC has no recovery behavior.

7. **2026-07-17 audit refinements.** (a) T darkness is *object-local, not
   exposure*: deploy mat RGB matches training ref (136 vs 132) but T pixels are
   (93,20,34) vs (139,48,57) — fix is light-on-T / camera saturation / T paint,
   not room light. (b) The freeze is a *self-locking fixed point*: static stack
   → ~0 action → static stack; sensor noise makes frames byte-differ so the
   fresh-frame guard can't detect it. Needs stall-detect + perturb. (c) z sags
   to ~0.009 at extended poses (demos hold ~0.0197; `lock_z` commands 0.02 but
   arm droops ~1.1 cm) → contact height differs from demos. (d) Gripper closes
   during the first ~5 rollout steps (state[6] 1.0→0.05); demos have it closed
   from frame 0 — small early-rollout OOD transient. (e) Freeze position
   y=−0.22 is *inside* the demo workspace (demos reach y=−0.356).

8. **v2 retrain (2026-07-17).** Deployed v1 seeds trained with worse-than-best
   hyperparams (see `scripts/train_pusht_real.py` row above). v2 = best
   pushing_pixels search recipe + appearance augmentation targeting (a).
   Checkpoints → `checkpoints/pusht_real_combinedv2_v2/`.

### Suggested next levers (ML/setup, not script bugs)
1. On-robot corrective data (DAgger-lite): teleop a few pushes on the deploy rig
   (same camera/light), fine-tune. Highest payoff.
2. Tighten setup match: lighting + exact T size/pose so the first approach lands.
3. Try other seeds / `--cp-selection sample` to escape the frozen fixed point.
4. Reset-and-retry on stall + per-step delta clip.

---

## 5. Fast health-check recipe for a new agent

1. Read `PUSHT_DEPLOY_HANDOFF.md` (status) + this file (file map).
2. Offline sanity: `sbatch scripts/diagnose_pusht_actions.sbatch` — expect
   healthy (MAE ~0.02, spread quadrants). If collapsed → checkpoint problem.
3. Inspect a demo episode: `data/example_episode/im_*.jpg` (traj82) — how the T
   moves, pusher visibility, lighting.
4. On robot: `deploy_pusht_real.py --dry-run` → `check_brightness_parity.py`
   (lighting) and eyeball `fed_000.png` (red T, upright, matches
   `scripts/assets/pusht_images1_ref.jpg`).
5. Rollout with `--log-dir results/runNN`, then analyze `steps.jsonl`:
   EEF trajectory, action quadrants, stall detection, corr(command, EEF-delta).
6. Compare deploy `raw_000.npy` vs `data/example_episode/im_0.jpg` (red-box
   center/size/brightness) to catch any obs mismatch.
