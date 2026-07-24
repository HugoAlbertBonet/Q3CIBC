# Push-T WidowX — Failure-Mode Handoff

Diagnostic reference for the Q3C(IBC) Push-T policy that **does not work on the
real WidowX** after retraining on the new Diffusion-Policy dataset
(`data/pusht_widowx_data.zip`). None of the batch checkpoints
(`batches/pushtWidowX.txt`, tags `c01`–`c10`) produce a working push.

This document enumerates every plausible point of failure across the
data → training → checkpoint → deploy → robot → physical pipeline: what it is,
why it can break things, what kind of failure it produces, which files are
involved and how, and a concrete test to confirm or rule it out.

---

## FINAL CONCLUSION (investigation complete — read this first)

**Every section A–G was worked through. No single deploy/data/environment bug
explains the failure. The residual cause is closed-loop robustness of the
single-step IBC policy (behavioral-cloning covariate shift): offline-healthy,
brittle on the real robot.** The arm stalls the same way (random timestep,
position-dependent, occasionally nudges the T) regardless of inference method,
min-step snap, z-height, exposure match, or start tweaks.

### Symptom (established from logs, not assumed)
The arm drives to a premature fixed point and stops/orbits; **the T essentially
never moves** (red-T centroid frozen; only a single ~14 px nudge across all
rollouts). Because the T doesn't move, the scene is static → frame-diff pins at
the camera noise floor (~0.92) → the pixels-only motion signal is dead → the
policy gets no progress feedback → it settles into a dead-zone / near-zero
action. All the downstream symptoms (freeze, orbit, dead-zone commands) follow
from the arm not reliably driving to and moving the T.

### What was RULED OUT (each by a concrete test, not assumption)
| section | verdict | how |
|---|---|---|
| A data/loader | CLEAN | A1 action↔eef corr +0.8 all eps; frame-count parity exact; A2 camera=1 chain; A3 cache content reproduced; A4/D2 resize parity <1 gray level |
| B training | CLEAN | B1 no mode-collapse (pred spread), B2 fit (MAE~0.04), B4 act scale ±0.008 |
| C ckpt/config | CLEAN | loads without shape error; cond_dim correct |
| D deploy preprocess | CLEAN | D1 RGB (fed==model-input byte-exact, red T), D2 resize, D3 240×320, D4 no stale/dup frames |
| E conditioning | CLEAN | E1 layout correct (start-pose match), in-distribution, no saturation; probe: cond STRONG + balanced with vision |
| F mapping | CLEAN | F1 corr(cmd,eef) +0.8–0.9; F2 no clipping; frame↔image direction verified via `--calibrate` (+dx→right, +dy→up, matches training) |
| G1 camera pose | CLEAN | rig `align_pusht_camera.py` overlay, ≤1mm |
| G2 T appearance | corrected, still stalls | scene ~16% underexposed; `--match-exposure` lifts T redness 79→97; no fix |
| G3 start pose | MINOR OOD | deploy starts x=0.110 vs demo band [0.113,0.118]; ~7mm undershoot; not primary |
| G4 z/contact | MOOT | arm never reaches the T, so contact height is irrelevant |
| inference | no fix | argmax vs Langevin vs DFO all stall; Langevin/DFO make actions more decisive but same attractor |

### RECOMMENDED PATH FORWARD (retrain-side; not a deploy bug)
1. **Switch to the Diffusion Policy.** The d-series runs scored offline MAE
   **0.019–0.030 vs IBC's 0.038–0.16**, and Diffusion Policy executes **action
   chunks** (open-loop K-step), the single biggest known remedy for the
   compounding single-step drift that produces this stall on real pushing tasks.
2. **On-robot DAgger** for whichever policy — collect corrections in the drifted
   states and retrain; directly attacks covariate shift.
3. **Training augmentation** — brightness/saturation/color + spatial jitter for
   robustness to residual OOD (washed-out T, 7mm start offset, etc.).

### DEPLOY-SIDE PALLIATIVES ADDED (help symptoms, don't fix the root)
`deploy_pusht_real.py` flags: `--min-step-xy` (snap sub-min-step OOD actions),
`--inference langevin|dfo` (match training's energy-min refinement),
`--match-exposure` (white-point lift), `--z-hold` (needs a 3-dim server action
space), `--calibrate` (open-loop frame/contact check). Analysis tooling:
`analyze_rollout.py`, `probe_conditioning.py`, `check_action_image_frame.py`,
`check_frame_action_alignment.py`, `check_frame_count_parity.py`,
`check_resize_parity.py`, `plot_pusht_action_hist.py`.

---

## How to use this document

### Two priors that shrink the search

**Prior 1 — the master bisection.** The offline diagnostic runs the *exact*
deploy model-build + inference over the training data, no robot involved. Its
verdict splits the whole space in half:

- **Healthy offline** (low MAE vs GT, actions spread across quadrants, pred std
  ≈ GT std) **but fails on the robot** → the bug is **deploy-side or physical**
  (sections **D–G**). The weights are fine.
- **Collapsed offline** (predicts ~constant, ignores the image, high MAE) → the
  bug is in **data or training** (sections **A–C**). Do not spend robot time.

**Prior 2 — "none of them work".** The 10 configs differ in idle-filter,
hyperparameters, control-point count, and conditioning. If they **all fail the
same way**, the cause is almost certainly **common to every config** — the
dataset loader, the frame cache, deploy preprocessing, or the physical rig — and
almost certainly **not** the hyperparameters or the idle filter. Weight
**A, D, F** above **B, C, E**.

### The master test now supports the new checkpoints (FIXED)

`scripts/diagnose_pusht_actions.py` was previously hardcoded to the old bridge
dataset. It has been updated and can now evaluate the new checkpoints:

- **Switches on `env["data_format"]`** (`zarr_video` → `PushTWidowXVideoDataset`,
  `bridge_zip` → `PushTRealPixelsDataset`), mirroring the trainer's
  `load_dataset()`.
- **Auto-picks the archive** from each checkpoint's own `config.json`
  (`data_archive`), so a zarr checkpoint is scored against the zarr zip and a
  bridge checkpoint against the bridge zip — no more silent wrong-dataset
  scoring. `--dataset` still overrides.
- **Conditioning aware**: reads `cond_dim` from `norm_stats.pt`, rebuilds the
  nets with it, gathers the `cond` batch and feeds it through `predict_batch`
  (`_cond`). Handles c09/c10; unconditioned checkpoints skip it.
- **`--idle-filter` (default `none`)**: scores the FULL action distribution
  regardless of how the checkpoint was trained, so you can see whether the
  policy still emits the zero spike. Reuses the training `frame_cache_dir`.

Verified end-to-end on a fabricated conditioned zarr checkpoint (dataset load,
model build/load, cond feed, stats all run clean).

**One caveat (handled):** the base sbatch default `--seeds 11 29 47 83` expects
`seed_00NN` directories, but the batch checkpoints are `--tag`-named (`c01_...`,
all `--seed 11`, so a shared `seed_0011` slot would collide). Use
`scripts/diagnose_pusht_widowx.sbatch`, which loops the 10 tag dirs, symlinks a
`seed_0001` slot per tag, and omits `--dataset` so each run scores against its
own `data_archive` (zarr). It also passes `--dump-arrays` for the histograms.

---

## RESULTS — bisect complete (2026-07, all 10 configs)

**Verdict: HEALTHY OFFLINE → the bug is deploy-side or physical (sections
D–G). The weights are fine. Do not spend more time on A / B / C.**

### How it was run
`scripts/diagnose_pusht_widowx.sbatch` (raw weights, `PUSHT_DIAG_NO_EMA=1`),
3000 seed-0 samples per checkpoint, `--idle-filter none` (full action
distribution). Per-sample `pred`/`gt` arrays dumped via the new
`--dump-arrays results/diag_arrays_raw`; JSON summaries in
`results/diag_<tag>_raw.json`. Histograms: `scripts/plot_pusht_action_hist.py`
→ `results/pusht_action_hist_raw.png` (GT vs pred, dx & dy, per checkpoint).

### Numbers (raw weights, MAE on normalized [-1,1] actions)

| tag | MAE dx / dy | note |
|-----|-------------|------|
| c03_dropstatic     | **0.038 / 0.044** | best |
| c10_condxy_nofilter| **0.038 / 0.042** | best |
| c02_none_control   | 0.046 / 0.049 | stall control — healthy offline |
| c04_subsample25    | 0.065 / 0.069 | |
| c09_condxy_dropzero| 0.067 / 0.075 | |
| c06_hard_negatives | 0.067 / 0.076 | |
| c07_lowmse_cp100   | 0.069 / 0.076 | |
| c05_libero_recipe  | 0.074 / 0.079 | |
| c01_dropzero_base  | 0.082 / 0.085 | |
| c08_dropzero_aug   | 0.154 / 0.165 | outlier — augmentation over-spread; drop |

For every config except c08: `pred std ≈ GT std`, actions spread across all four
quadrants, low per-sample MAE. By the bisection rule this is **healthy offline**.

### What the histograms confirm
- **Not mode-collapse.** Pred is a *broad* bump around 0, not a delta spike at 0.
  A collapsed policy would show a tall narrow pred spike matching/exceeding the
  GT zero-spike; the opposite is seen. B1 is ruled out.
- **±1 pileup is real expert behavior, not a clip artifact.** GT sits *exactly*
  on ±1 for 6.7% of dx and 12.8% of dy steps (atol 1e-6) — a hard teleop
  velocity cap at the true data extremes (`act_min/max = ±0.008` = the data
  min/max, so nothing is clipped beyond). Pred matches the near-edge fraction
  (dx 5.6% vs 7.1%, dy 12.1% vs 13.5%) without emitting exact ±1 — well
  calibrated to the tails. **dy rides the cap ~2× more than dx** (12.8% vs 6.7%):
  the vertical axis is the one at max step, worth watching in the F1/B4 deploy
  sign/scale checks.
- **Policy never emits exact (0,0).** GT has a huge 0-spike; pred is smooth
  through 0. The model never commands a true "hold", so the robot stall cannot
  originate in the weights — it is a deploy-side effect (prime suspect **D4**,
  stale/duplicate stacked frame → learned "static stack ⇒ hold").
- **c02 (the stall control) is healthy offline too.** Offline does not reproduce
  the stall, which independently confirms the stall is deploy/robot, not the idle
  filter and not the weights.

### What the offline result does and does NOT clear
The diagnostic trains and tests through the **same loader**, so "healthy
offline" only proves the learned map is self-consistent with whatever that
loader serves. It cannot see a bug that corrupts frames and labels *together*,
or one that only appears at deploy.

- **Cleared by offline:** **B1** (mode-collapse — pred is spread, not a spike),
  **B2** (non-convergence — MAE ≈ 0.04 means it fit), **C1** (arch/cond load ran
  without shape error), **B4** scale sanity (`act_min/max = ±0.008` as expected).
- **NOT cleared by offline** (self-consistent or deploy-only, need loader-external
  checks): **A1** frame↔action, **A2** wrong camera, **A3** frame-cache,
  **A4/D2** tf-vs-cv2 resize.

### A1 — CLOSED separately (loader-external, local, no cluster)
Two direct checks on the raw zarr + source mp4s (both in the 292 MB zip):
- `scripts/check_frame_action_alignment.py`: per-episode
  `corr(action[t,:2], eef[t+1]-eef[t])` = **+0.81 dx / +0.84 dy, all 150
  episodes ≥ +0.61, residual |Δeef-action| ≈ 1.4 mm** (cap 8 mm). Lowdim
  action/eef/episode_ends are coherent; correct sign; benign lag0→+1 rise
  (actuation latency, not off-by-one).
- `scripts/check_frame_count_parity.py`: every video has **exactly** `L =
  end-start` frames (delta +0 for all 150). `_build_frame_cache` writes
  `frames[:L]` sequentially into `mm[start:end]`, so with zero truncation this is
  an **exact 1:1 frame↔flat-index** map — no leading/trailing shift possible.
  ep0 frame0 dumped to `results/a1_ep0_f0.png` shows the red T top-left in the
  expected cam-1 layout.

**A1 is not the bug.**

### A2 — CHECKED, consistent (software); only G1 (physical) remains
Train/deploy use the **same** camera, verified through the whole chain:
- `train_pusht_real.py --video-camera` default = **1**; the batch has **zero**
  overrides, so every c01–c10 trained on camera 1. All three diagnostic runs log
  `PushTWidowXVideoDataset: ... camera=1`. Config writes
  `camera_streams=["video1"]`, `video_camera=1`.
- mp4 camera 1 **is** the fixed blue scene camera: `results/a1_ep0_f0.png`
  (from `videos/0/1.mp4`) shows the red T top-left in the reference layout.
- Deploy reads that same blue camera — `images1 == /blue/image_raw`, the only
  camera left after the D435 was removed (`deploy_pusht_real.py` header +
  `extract_blue_frame`). So **train cam 1 == deploy blue frame**, no view swap.

Remaining camera risk is purely **physical mounting/aim** (→ **G1**, needs the
live rig), not a software mismatch.

### A3 — CLOSED (cache integrity)
- **Length/truncation**: loader asserts `_cache_len == n_frames` (72988)
  (`datasets.py:1452`); all three diagnostic runs loaded clean at
  `state_shape=(6,240,320)`, so a short/truncated cache is impossible.
- **H×W = 240×320 correct**: confirmed by the logged `state_shape` and by a local
  reproduction of the build's tf AREA resize.
- **No drop/dup**: frame-count parity (every video == L) + the sequential
  per-episode write leave no room.
- **Content correct**: reproduced `_build_frame_cache` on ep0 locally
  (`results/a3_ep0_f0_resized.png`) — 240×320 uint8, red T preserved top-left.
  Only unchecked bit is byte-identity of the on-cluster `.u8` vs this
  reproduction, which the deterministic build + the length assert make
  near-certain.

### A4/D2 — CLOSED (resize parity, negligible)
`scripts/check_resize_parity.py` (16 frames across ep 0/40/80/120, 640x480 ->
240x320): tf AREA(antialias) vs cv2 INTER_AREA agree to **mean abs diff
0.095/255, max 1, 0% of pixels off by >1** — pure rounding, signed bias
-0.095. No systematic train/deploy shift. Deploy native is 640x480 (its header),
same ratio as the cache, so this transfers. Visual: `results/a4_resize_diff.png`
(tf | cv2 | 10x|diff|).

### Section A — FULLY CLEARED
A1 (frame↔action) CLOSED, A2 (camera) consistent in software (physical → G1),
A3 (cache) CLOSED, A4/D2 (resize) CLOSED. The data → training supervision is
intact end-to-end; **the failure is deploy-runtime or physical (D/E/F/G)**,
which need the live rig. c08 aside (augmentation hurt), any of c03 / c10 / c02
is fine to deploy for the later tests.

### Loose ends
- **c08_dropzero_aug**: drop it, augmentation flattened the distribution.
- **EMA vs raw (B3)**: results above are raw. Re-run without `PUSHT_DIAG_NO_EMA`
  for the EMA copy if a deployed EMA checkpoint misbehaves.

---

## A. Data / dataset loader  *(new code, common to all configs → HIGH suspicion)*

All of section A lives in `utils/datasets.py` →
`class PushTWidowXVideoDataset`, plus the frame cache it writes.

### A1. Frame ↔ action misalignment
- **What / why.** The zarr arrays are one flat block of all 150 episodes
  concatenated (`data/action` is `(72988, 7)`); the videos are one MP4 per
  episode. Video frame `t` must be paired with action row `t` **within that
  episode's slice** `[episode_ends[i-1], episode_ends[i])`. Any off-by-one, a
  wrong episode boundary, or pairing the flat index against the wrong video
  desynchronizes image and label for a whole episode.
- **Failure type.** The image→action map is corrupted at the source. The model
  fits noise; **collapses or is inaccurate offline**. This is the single most
  damaging data bug and it is invisible without an explicit check.
- **Files.** `PushTWidowXVideoDataset._load_lowdim` (reads action/eef/ends),
  `._build_frame_cache` (extracts `videos/<ep>/<cam>.mp4`, writes frames into
  the per-episode slice `mm[start:end]`), `.__getitem__` (maps sample index →
  episode via `np.searchsorted(self._episode_ends, t)` and reads the stack).
- **Test.** The command action at step `t` should move the arm between frame `t`
  and `t+1`. On the raw zarr, per episode, compute
  `corr(action[t, :2], eef_pose[t+1, :2] − eef_pose[t, :2])`; it must be
  strongly **positive** (≈ +0.9, as measured on the old data). A near-zero or
  negative correlation for a whole episode = misalignment. Needs no training or
  robot — a standalone `check_frame_action_alignment.py` reading the zarr does
  it. Also visually: cached frame 0 of episode 0 should show the T in its known
  start pose.

### A2. Wrong camera
- **What / why.** Each episode has `0.mp4` and `1.mp4`. **Camera 1 is the fixed
  blue scene camera** the deploy client reads and every prior checkpoint used;
  camera 0 is a different viewpoint. Training on camera 0 makes the whole
  deployment out-of-distribution.
- **Failure type.** Can look fine offline (self-consistent) yet fail completely
  on the robot, since deploy feeds a different view than training.
- **Files.** `train_pusht_real.py --video-camera` → config `video_camera` →
  `combinedv2_cpascounter_training.py` load_dataset passes `camera=...` →
  `PushTWidowXVideoDataset.camera` / `.camera_streams = (f"video{camera}",)`.
- **Test.** Confirm `video_camera: 1` in the seed's `config.json`. Eyeball a
  cached frame against `scripts/assets/pusht_images1_ref.jpg` (T top-left,
  target outline center-left, small square bottom-left).

### A3. Frame-cache build error
- **What / why.** MP4 random access is seek-bound, so frames are decoded once
  into a uint8 memmap (`data/_frame_cache/*.u8`, ~17 GB). A wrong `H×W`, a
  dropped/duplicated frame, or a truncated build silently pairs wrong pixels
  with labels.
- **Failure type.** Same class as A1 (corrupted supervision), but originating in
  the cache rather than the indexing.
- **Files.** `PushTWidowXVideoDataset._ensure_frame_cache` (lock + meta),
  `._build_frame_cache` (decode + resize + `mm[start:end] = frames`), the
  sidecar `*.u8.json` (`n_frames`).
- **Test.** Cache meta `n_frames` must equal `episode_ends[-1]` (= 72988). The
  loader already asserts this and raises `frame cache has N frames but the
  replay buffer has M`. Spot-check: decode episode `k` fresh with imageio and
  compare frame 0 to `memmap[episode_starts[k]]` (after the same resize).

### A4. Resize algorithm mismatch (train vs deploy)
- **What / why.** The cache resizes 640×480 → 320×240 with **TensorFlow**
  `tf.image.resize(method=AREA, antialias=True)`. The deploy client resizes the
  live frame with **OpenCV** `cv2.resize(..., INTER_AREA)`. These are both
  "area" resamplers but are **not bit-identical**; the systematic pixel
  difference is a small train/deploy distribution shift on every frame.
- **Failure type.** Subtle, uniform OOD — plausibly enough to degrade a policy
  that is otherwise correct. Would show as "healthy offline, weak on robot".
- **Files.** Train: `PushTWidowXVideoDataset._build_frame_cache` (tf resize).
  Deploy: `scripts/deploy_pusht_real.py::preprocess` (line ~417,
  `cv2.resize(..., interpolation=cv2.INTER_AREA)`).
- **Test.** Take one raw 640×480 frame; resize with both paths; report max/mean
  abs difference. If large, make deploy use the tf path (or make the cache use
  cv2) so both sides match. Chunk-resize was already verified byte-identical
  *within* the tf path, so this is specifically a tf-vs-cv2 question.

### A5. Idle filter over/under-removing
- **What / why.** 24% of target actions are exactly (0,0) (teleop pauses); the
  filter thins them (`drop_zero` removes all, `drop_static` only the ones whose
  frame pair is also static, `subsample` keeps a fraction). A misconfigured
  filter either leaves the absorbing "hold" spike in (stall persists) or strips
  legitimate decelerations (jerky policy).
- **Failure type.** Behavioral — either the original stall (too little removed)
  or over-aggressive pushes with no slow-down (too much removed).
- **Files.** `PushTWidowXVideoDataset._build_samples` (the mask logic),
  `train_pusht_real.py --idle-filter/--idle-eps/--idle-move-eps/--idle-keep-frac`.
- **Test.** The loader prints `idle_filter=... -> kept X%, zero-share now Y%`
  on startup. Grep the job stdout. Expected: `drop_zero` → 0% zeros, ~76% kept.

---

## B. Training

Lives in `combinedv2_cpascounter_training.py`.

### B1. Mode collapse onto the zero spike
- **What / why.** The zero action is a delta spike holding ~24% of the mass at
  one point; real pushes spread over a 2-D continuum. `lossMSE`
  (`utils/loss.py`) pulls the nearest control point to the expert, and the Q
  estimator's argmax over the CP cloud preferentially selects the dense spike.
  This is the failure mode the idle filter and the negatives were added to
  break.
- **Failure type.** Offline: predicted std ≈ 0, actions cluster at ~0, high MAE
  in the moving regime. On robot: stall.
- **Files.** `combinedv2_cpascounter_training.py` loss assembly (loss_mse /
  loss_sep / InfoNCE), `utils/loss.py::lossMSE`,
  `utils/datasets.py` idle filter (mitigation), the `--uniform-negatives` /
  `--langevin-negatives` path (mitigation, teaches the Q net to lower energy at
  zero when the expert moved).
- **Test.** `diagnose_pusht_actions.py` (once zarr-enabled): quadrant histogram
  and pred std. Compare a `drop_zero` config (c01) against the `none` control
  (c02) — c02 is in the batch **precisely** to confirm this by reproducing the
  stall.

### B2. Non-convergence / wrong optimization
- **What / why.** LR, schedule, clamp, or batch size wrong for this data → the
  model never fits.
- **Failure type.** Offline high MAE across all regimes; wandb loss flat or
  diverging.
- **Files.** `train_pusht_real.py` hyperparameter group → per-run
  `config.json` training block → trainer reads via `env_training.get(...)`.
- **Test.** wandb loss curves; `diagnose` MAE. Defaults are the best
  pushing_pixels search recipe (trial 95, sr 0.99), so a total non-convergence
  would more likely indicate a data bug (A) feeding garbage.

### B3. EMA vs raw divergence
- **What / why.** The trainer keeps an EMA shadow (`ema_decay=0.999`) and saves
  both `*.pt` (raw) and `*_ema.pt`. Deploy/diagnose default to EMA. If one is
  bad (e.g. EMA lagging a late collapse, or raw noisier), picking the wrong one
  looks like "the checkpoint doesn't work".
- **Failure type.** Behavioral; one weight set works and the other doesn't.
- **Files.** `combinedv2_cpascounter_training.py::update_ema` / `save_checkpoints`;
  deploy `--no-ema`; diagnose `--no-ema`.
- **Test.** Run diagnose both ways (default and `--no-ema`); deploy the better
  one.

### B4. Wrong action normalization stats
- **What / why.** `norm_stats.pt` carries `act_min`/`act_max`; deploy
  denormalizes the policy's [-1,1] output back to metres with them. If they are
  wrong, every command is mis-scaled.
- **Failure type.** Actions systematically too large (unsafe) or too small
  (no motion) on the robot, even with a perfect policy.
- **Files.** `PushTWidowXVideoDataset` (computes act_min/max from the full
  action set), `persist_norm_stats()` in the trainer,
  `deploy_pusht_real.py::unnormalize`.
- **Test.** `python -c "import torch; print(torch.load('.../norm_stats.pt',
  weights_only=False))"`; act_min/max must be ≈ ±0.008.

---

## C. Checkpoint ↔ config consistency

### C1. Architecture / cond_dim mismatch
- **What / why.** Deploy and diagnose rebuild the nets from the seed's
  `config.json` model block + `norm_stats` `cond_dim`, then `load_state_dict`.
  If cp_width / value_num_blocks / **cond_dim** differ from what was trained,
  the load fails or (worse) loads a wrongly-shaped head.
- **Failure type.** Hard crash at load, or subtly wrong network. The conditioned
  runs (c09/c10) have `cond_dim=2`; the rest have 0 — deploying one as the other
  breaks.
- **Files.** `deploy_pusht_real.py::build_models(cond_dim=...)` and its
  `norm_stats["cond_dim"]` read; `train_pusht_real.py` model block;
  `combinedv2_cpascounter_training.py` model construction + `persist_norm_stats`
  (`cond_dim`, `cond_min`, `cond_max`, `cond_kind`).
- **Test.** If it loads without a shape error, this is mostly ruled out. Confirm
  `norm_stats["cond_dim"]` matches the intended config.

### C2. Stale / overwritten config.json
- **What / why.** The launcher writes `config.json` into the run dir at job
  start. A repeated `--tag` or a re-run can leave a `config.json` that does not
  describe the weights beside it. (The launcher now refuses to overwrite a dir
  that already holds `*.pt`, which mitigates this.)
- **Failure type.** Deploy rebuilds the wrong architecture → crash or silent
  wrongness.
- **Files.** `train_pusht_real.py::main` (run-dir guard + config write).
- **Test.** Diff the model block in the seed's `config.json` against the batch
  line that produced it (`batches/pushtWidowX.txt`).

---

## D. Deploy preprocessing  *(common to all configs → HIGH suspicion)*

Lives in `scripts/deploy_pusht_real.py`.

### D1. Channel order (RGB vs BGR) — CHECKED, CORRECT (not the bug)
Analyzed the existing capture `results/dry_03/` (raw_*.npy + fed_*.png):
- `raw_000.npy` most-chromatic pixel (the T) = RGB(94, 11, 20): **R ≫ B**
  (max redness 79 vs max blueness 15). Buffer is genuinely RGB.
- `fed_000.png` renders the T **red and upright**, correct layout. No swap.
- **Reversal ruled out:** `imageio.imread(fed_000.png)` (read as RGB) equals the
  exact model-input array `frame_buf[-1]` byte-for-byte (`np.array_equal ==
  True`); T pixel RGB(95,12,21), index0=R highest. The dump's `cvtColor(RGB2BGR)`
  is cancelled by `imwrite`'s BGR convention, so the file stores true RGB. A
  BGR buffer would have rendered the T **blue** — red proves RGB into the model.
- **Caveat:** valid only if `dry_03` came from the *current* rig/server; if the
  server changed since, re-capture with `--dry-run` and re-check.
- **Side finding → G2:** the deploy T is a dull maroon, peak redness **79** vs a
  training frame's ~133. The T is measurably darker/desaturated — the G2
  lighting OOD, visible in this same capture.

Original notes below.

- **What / why.** Training frames are tf-decoded RGB with the red T in channel
  0. The deploy client feeds the server frame as-is (no swap), assuming the same
  order. If this rig's server delivers BGR, the T's color channel is scrambled
  and every spatial feature the encoder learned is wrong.
- **Failure type.** Severe OOD; policy effectively blind. Can look like random
  or frozen behavior.
- **Files.** `extract_blue_frame` / `to_uint8_rgb` (line ~345),
  `preprocess` (line ~417, no channel swap), PNG dump uses
  `cv2.cvtColor(..., COLOR_RGB2BGR)` implying the buffer is treated as RGB.
- **Test.** `--dry-run` → open `deploy_dryrun/fed_000.png`; the T must render
  **red** and upright (imshow/imwrite assume BGR, so a correct RGB buffer saved
  via `RGB2BGR` shows red correctly). If the T is blue, channels are swapped.

### D2. Resize mismatch (deploy side) — CLOSED (negligible)
Same issue as A4, measured on **real deploy frames** (`dry_03/raw_*.npy`, 20
frames, 480x640 -> 240x320): `|cv2_deploy - tf_train|` mean **0.116/255, max 1,
0% of pixels off by >1**, signed bias +0.116. The deploy cv2 INTER_AREA output
matches the tf training cache to within rounding. Not a train/deploy shift.

### D3. Wrong target resolution — CLOSED (correct)
`dry_03/fed_000.png` is **240x320** = the model input size; the raw server frame
is 480x640 and `preprocess` downsizes it to 240x320. Also verified fed png ==
`frame_buf[-1]` (the exact model-input array) byte-for-byte, so what the model
ingests is confirmed 240x320. Matches training `state_shape=(6,240,320)`.

Original notes: training cache 240×320; deploy `preprocess(out_hw=(image_h,
image_w))` reads `image_height`/`image_width` from `config.json`; encoder then
bilinearly resizes to 180×240 internally.

### D4. Stale / duplicate / zero-motion frame — ANALYZED: stall REPRODUCED, but NOT a frame-grab bug
Analyzed `results/roll_03/` (120-step logged rollout of a **pixels-only** config
— `steps.jsonl` has no `cond` field, so c01–c08, not c09/c10):

- **No stale/duplicate frames.** 0/119 consecutive raw pairs are identical; the
  freshness guard is fine. The server never repeats an image. D4-as-originally-
  written (frame-grab freshness) is **clean**.
- **But the stall is real and self-locking.** Steps 0–~60 push healthily
  (x 0.110→0.210, |action| ~.005–.007); after ~step 65 the policy commands
  **exactly 0.0000** (40.8% of all steps, all in the back half) and the arm
  freezes at (0.218, 0.105) — short of the demo push range (x up to 0.49). A
  premature stall, not task success.
- **Root cause = razor-thin motion margin.** Inter-frame pixel motion while
  *moving* is only ~1.2–1.6 (0–255 scale); the camera **noise floor is 0.94**.
  The 8 mm max step barely clears sensor noise. As the arm decelerates near the
  push, real motion sinks *into* the noise → the 2-frame stack reads static →
  the pixels-only model fires its learned **"static pair ⇒ hold"** rule → arm
  stops → stack stays static → absorbing lock. `corr(frame-diff, next |action|)
  = +0.47` confirms: less scene motion → smaller next action.
- **Reconciles with "healthy offline".** Offline sampled *moving* transitions;
  the model correctly emits ~0 on static pairs, and near-target the deploy input
  *becomes* a static pair. This is the B1 / zero-action absorbing state
  manifesting at deploy, triggered physically — exactly what the idle filters +
  negatives were meant to break, and it still bites a pixels-only policy.

**Implication / leading fix.** A pixels-only policy has no motion signal but the
2-frame pixel diff, so it is structurally prone to this lock. **c09/c10 add
eef-(x,y) conditioning to supply a non-visual "where am I"**.

**UPDATE — c09/c10 tested (`roll_04`, `roll_05`), stall NOT solved, only
changed.** Conditioning removes the *exact-zero* collapse (0% zero-action vs
40.8% pixels-only) but all three still stall mid-task at a premature fixed
point, far from goal (demos push x→0.49; reached: pixels 0.218, **c09 0.303**,
c10 0.184):
- Pixels-only fixed point = *exact zero* action (static-stack → hold).
- c09/c10 fixed point = a small **constant nonzero** action (~0.001 m, pointing
  −dx toward base) that the arm does not execute (move ~0.4 mm) → position
  frozen → conditioning input frozen → same action out → fixed-point lock.
- `corr(cmd, actual) = +0.82…+0.94` (F1 sign fine); `action == env_action`
  through the whole stuck region (F2 approach-floor NOT firing). Neither is the
  cause.
- **G4 z-droop is the prime physical contributor** — see G4: `corr(x,z) = −0.97`,
  c09 sags to z=0.007 at x=0.303 vs the 0.0197 contact height. Wrong contact
  height → no purchase on the T → no progress → the policy settles.

Next: compensate the droop (raise `--fixed-z-height`, ideally x-dependent) and
re-test; separately check whether the reached fixed-point region is OOD (few
demos there) so the policy under-drives. (Deploy palliatives: minimum-step floor
on |action|, proprioceptive-velocity gating.)

- **Files.** `extract_blue_frame` / control-loop grab logic; `stack_to_tensor`
  (line ~431); analysis via `results/roll_*/raw/*.npy` + `steps.jsonl`.
- **Test (reusable).** Diff consecutive `raw/*.npy` (duplicates = freshness bug),
  and cross-plot `|action|` vs frame-diff vs EEF-move from `steps.jsonl` to catch
  the absorbing lock even when frames are technically fresh.

---

## E. Conditioning  *(new; only c09 / c10)*

### E — CLEARED BY TEST. Inputs correct (offline) AND conditioning proven effective (probe).
(See the E-EFFICACY PROBE result below: cond STRONG and balanced with vision.)

Checked offline from `roll_04` (c09) / `roll_05` (c10) `steps.jsonl` vs the
training workspace (zarr `robot_eef_pose[:, :2]`). This confirms the *inputs* to
conditioning are sane — NOT that conditioning is wired correctly through the
model or that it actually steers the policy. Those are only confirmable on the
rig by testing.
- **E1 layout correct.** Deploy start `state=(0.110, −0.017)` matches the known
  demo start `(0.117, −0.019)` → `state[0]=x, state[1]=y`. The "inferred from old
  logs, unconfirmed" worry is resolved for the layout.
- **In-distribution.** 100% of the fed x,y lie inside the training range
  (x∈[0.094,0.491], y∈[−0.376,0.338]).
- **E3 no saturation.** Normalized cond never pinned at ±1 (x∈[−0.92,+0.05],
  0% saturated). E2 already verified in-repo (`make_cond` == `normalize_cond` to
  1e-7).

**Open:** whether the cond head is correctly connected and whether the (x,y)
signal is actually driving the policy (vs decorative / under-driving) is NOT
proven offline. **Order: try the two deploy fixes below on the rig FIRST.** If
the stall clears, conditioning was fine. If it persists, do full E diagnostics —
e.g. ablate the cond input at deploy (feed constant vs true (x,y) and check the
action changes), and confirm the cond tensor reaches the value net with nonzero
learned weights.

### ROOT CAUSE (deepest): expert action dead zone → OOD sub-min-step stall
Measured on the raw zarr (normalized actions, ±0.008 → ±1):
- Per axis the expert is **bang-bang**: exactly 0, or a real step ≥ ~1.5 mm.
  dx is 0 in 46.4% of rows, dy in 38.1% (the (0,0) both-zero case is 24%). The
  smallest nonzero |dx| is 1.5 mm; **only 0.30% of nonzero dx fall below 1.5 mm**
  → an empty dead zone in (0, 1.5 mm).
- The stuck fixed-point actions (c09 dx≈−0.57 mm, c10 ≈−1 mm) sit **inside that
  dead zone** — values with ~zero training support. The energy/IBC policy
  interpolates between the 0-spike and the ≥1.5 mm cluster and emits an
  in-between action that (a) no expert made and (b) is below the arm's execution
  threshold → under-executes → freeze → same action out → lock.
- Explains BOTH modes: pixels-only interpolation collapses to the dominant
  0-spike (exact-zero hold); conditioned lands in the (0,1.5 mm) gap
  (sub-threshold creep).

### FIXES IMPLEMENTED — test on rig (ablation)
Two opt-in deploy flags in `scripts/deploy_pusht_real.py`:
1. **Min-step snap** `--min-step-xy 0.0015`: any nonzero |dx|/|dy| below the
   value is snapped up to it (sign kept); exact 0 preserved. Forces commands onto
   the supported bang-bang grid so they execute. (`apply_min_step`.)
2. **z-hold servo** `--z-hold 0.0197 [--z-hold-gain 1.0 --z-hold-max 0.01]`:
   injects a per-step `dz = clip(gain*(z_target−cur_z))` to actively hold z flat
   against the x-dependent droop. **Requires `--action-mode 3trans`** (startup
   guard rejects 2trans) and a server/arm that executes dz; dz is injected, not
   from the 2trans-trained policy. (`z_hold_dz`, `z_from_obs`.)

### ABLATION RESULTS (c09 on the rig) — neither fix solves it; z-hold blocked
Analyze with `scripts/analyze_rollout.py <log-dir>`.

| run | sub-min-step | outcome | x reached | note |
|-----|--------------|---------|-----------|------|
| base (this session) | 87% | FREEZE @step47 | 0.171 | z~0.017 flat (corr +0.04) |
| min-step snap | 0% | **ORBIT**, no stall | 0.231 | limit cycle: path 0.76 m / net 0.12 m |
| z-flat (`--fixed-z-height 0.026`) | 88% | FREEZE @step27 | 0.173 | barely changed base |
| z-hold servo (3trans) | — | **SERVER ERROR** | — | see below |

- **Min-step snap trades freeze for a limit cycle.** It removes the exact-zero /
  dead-zone freeze (0% sub-min-step, "no stall") but the arm then **orbits** the
  fixed point (~(0.21,0.02), heading flips >90° on 13% of steps) and never seats
  the T. Snap forces intended-near-zero commands to 1.5 mm in oscillating
  directions. Not a standalone fix.
- **z-hold servo is BLOCKED by the server, not our code.** The injected dz was
  correct (`env_action=[dx,dy,+0.0036,grip]`) but the live widowx_env_service env
  is hardwired to a **2-dim action space**:
  `AssertionError: Action should have shape (2,) but has shape (4,)`. Running it
  needs the server relaunched with a 3-dim action space (and the Push-T env to
  actually accept a z-delta) — a server-side change. In 2trans the only z lever
  is the flat `--fixed-z-height` (imperfect for x-dependent droop).
- **CONFOUND / correction to G4:** this session's baseline holds z≈0.017 **flat**
  (corr(x,z)=+0.04), NOT the droop-to-0.007 (corr −0.97) that `roll_04` showed —
  the server restart changed z handling. With decent z the policy **still**
  stalls at x≈0.17. So **G4 z-droop is real but NOT the primary blocker now.**

### E-EFFICACY PROBE — conditioning WORKS (not decorative, not over-conditioned)
`scripts/probe_conditioning.py` holds the image fixed, sweeps cond over the full
workspace (and the mirror: fixed cond, vary image):
- **c09:** cond-induced action range = 96% of the ±8 mm span → STRONG. Mirror:
  **cond 88% vs image 32%** — cond-leaning (~2.7×) but the image clearly matters.
- **c10:** STRONG. Mirror: **cond 54% vs image 52%** — essentially **balanced**.

Both policies genuinely use the image; c10 is well-balanced and still stalls. So
the stall is **not** "visually blind / over-conditioned." Combined with E1
(layout correct, in-distribution, no saturation) and E2 (verified in-repo),
**section E is CLEARED by test** — conditioning is wired, effective, balanced.

### PRIME REMAINING LEAD — deploy inference is degraded vs training (no Langevin)
`select_action` (deploy) does **pure argmax over the control-point cloud**;
training refines actions with **Langevin sampling** (`langevin_num_iterations`
default 50, lr schedule; `combinedv2_cpascounter_training.py:75`,
`utils/sampling.sample_langevin`). Comment: "langevin/DFO disabled for this
hardware." So deploy is limited to the generator's discrete candidates with no
gradient refinement toward the energy minimum — near contact the true optimum can
fall between control points → coarse actions → the orbit/dead-zone behavior.
Caveat: offline MAE with this same argmax-CP path was decent (~0.067), so Langevin
may sharpen rather than transform. **Next: re-enable Langevin/DFO at deploy as an
opt-in flag and re-test** (then F).

### WHERE THE DIAGNOSIS HAS CONVERGED
A/B/C/D/E and F(sign/clip) cleared as *bugs*. The failure is **policy robustness
in closed-loop deployment**: reasonable single-step behavior (offline MAE ok,
conditioning works) but it reaches x≈0.17–0.23 and fixed-points/orbits short of
the goal (x≈0.49). Concrete contributing levers, in priority:
1. **Deploy inference degraded (no Langevin/DFO)** — most actionable, testable.
2. **G2 T-darkness** OOD (peak redness 79 vs 133) — still open, physical.
3. Compounding covariate shift + contact dynamics — a fundamental BC limit that
   may need more/near-contact data or a stronger policy/inference.

### E1. Deploy state layout — is `state[:2]` really (x, y)?
- **What / why.** The conditioned policy expects the current EEF (x, y),
  normalized to the training workspace. Deploy pulls it from the server
  observation's `state` field as dims 0:2. This layout was **inferred from old
  `run02` logs, not confirmed on the current rig/server**. If the live `state`
  is ordered differently, the conditioning vector is plausible-but-wrong and the
  failure is silent.
- **Failure type.** Silent mis-conditioning; the policy is fed a wrong "where am
  I", degrading or destabilizing behavior with no error.
- **Files.** `deploy_pusht_real.py::make_cond` (reads `raw_obs["state"][:2]`),
  `eef_x_from_obs` (uses `state[0]` as x), the normalization mirror of
  `PushTWidowXVideoDataset.normalize_cond`.
- **Test.** `--dry-run` and print the raw `state`. Dim 0 (x) should sit in
  ≈ [0.10, 0.45] and dim 1 (y) in ≈ [−0.38, 0.34] (the training workspace,
  = `cond_min`/`cond_max` in norm_stats). If the values don't land there, the
  layout or units differ.

### E2. cond normalization mismatch (train vs deploy)
- **What / why.** Deploy must min-max the live x/y with the **exact**
  `cond_min`/`cond_max` computed at training time, or the conditioning is on a
  different scale than the network was trained for.
- **Failure type.** Off-scale conditioning; silent degradation.
- **Files.** `PushTWidowXVideoDataset.normalize_cond` (+ persisted cond bounds),
  `persist_norm_stats` (`cond_min`/`cond_max`/`cond_kind`),
  `deploy_pusht_real.py::make_cond`.
- **Test.** Already verified in-repo that `make_cond` reproduces
  `normalize_cond` to atol 1e-7. Just confirm `norm_stats` actually contains
  `cond_min`/`cond_max` for the conditioned checkpoints.

### E3. Live pose outside the training workspace
- **What / why.** If the arm leaves the demonstrated region, the normalized cond
  saturates to ±1 (intentional clip). Persistent saturation makes the
  conditioning uninformative exactly when the arm is most OOD.
- **Failure type.** Conditioning stops helping in the corner where recovery is
  needed.
- **Files.** the `np.clip(..., -1, 1)` in `normalize_cond` / `make_cond`.
- **Test.** Log cond values during a rollout; watch for stretches pinned at ±1.

---

## F. Robot mapping / action execution

### F — sign/clip cleared; frame check surfaced the REAL root: NO T CONTACT
- **F1 sign** clean: `corr(cmd, actual eef-Δ) = +0.82…+0.94` across roll_03/04/05.
- **F2 clip/floor** clean: `action == env_action` (0% clipped) in every stuck
  region; the approach floor fired only once at start.
- **F3 denorm** clean: commanded actions ≤ ±0.008 m = act_max.
- **Action↔IMAGE frame** (`scripts/check_action_image_frame.py`, optical flow):
  TRAINING is clean (+dx→image RIGHT/UP, +dy→LEFT/UP, R²≈0.2–0.4). INFERENCE is
  **unmeasurable from policy rollouts** — an early "FLIPPED" verdict was NOISE
  (negative R²): deploy per-step motion is too small for optical flow.

**THE PIVOTAL FINDING — the T never moves in deployment.** Red-T centroid is
frozen at (211,203) for all 200 frames of `roll_c09_base` (and only a single
~14 px nudge in `roll_04`), while the arm travels 6–19 cm in world space. So:
- The arm approaches the T (fed pngs show the gripper at the T's corner) but
  **does not push it** — ineffective contact.
- T static → whole scene static → **frame-diff pinned at the 0.92 noise floor**
  (this is *why* it sits at noise — nothing moves), so the pixels-only motion
  signal is dead → the policy gets no progress feedback → settles/orbits.
- Therefore the "policy attractor / dead-zone / freeze-or-orbit" is all
  **downstream of the arm failing to move the T.** The offline policy was fine;
  the deploy arm just isn't moving the object. **Root is physical CONTACT → G.**

**Definitive frame test tool added:** `deploy_pusht_real.py --calibrate`
(scripted open-loop ±dx/±dy, logs `raw/` + `steps.jsonl`). Run it + the frame
check for a clean action↔image direction check, AND to see whether a scripted
push moves the T at all (contact test), since the policy won't move it.

### F1. Sign / axis flip (command → motion)
- **What / why.** If the server maps +dx to −x (or swaps x/y), the policy fights
  itself: correct intentions produce wrong motion.
- **Failure type.** Runs the wrong way / oscillates.
- **Files.** `to_action_7d` (line ~547), `safety_clip_action` (line ~559), the
  server's `2trans` handling; forensic `state` logged per step.
- **Test.** From `--log-dir` `steps.jsonl`: `corr(commanded dx, actual EEF Δx)`
  and same for y. Both must be strongly positive (was +0.89 / +0.96 previously).

### F2. Safety clip / approach-floor masking real motion
- **What / why.** The rewritten deploy client adds an **approach floor**
  (`--approach-floor`, default ON): x is not allowed below `floor_x` (default =
  demo start x ≈ 0.117). It is a floor, not a ban — a step toward the base is
  allowed until x reaches the floor; at/below it, dx is forced ≥ 0. Plus
  `--safety-max-xy-delta` magnitude clipping.
  **Measured on the demo data, the floor should rarely fire:** the start x
  (0.117) is essentially the workspace minimum; demos push *outward* (x up to
  0.491, mean 0.275), only 1.3% of timesteps sit below the floor, and x dips a
  mean of 4.6 mm below its own start per episode. So in normal operation this is
  a benign ~1%-of-steps safety limit, **not** a likely "all fail" cause — UNLESS
  the policy is already misbehaving and driving the arm toward the base (then
  the floor pins it and masks the real symptom), or `--approach-floor-x` was set
  too high.
- **Failure type.** Only masks motion if the policy is *already* pushing toward
  the base against the demos' outward direction. Secondary suspect, not primary.
- **Files.** `apply_approach_floor` (line ~399), `safety_clip_action`
  (line ~559), the loop at line ~848 (`apply_approach_floor(act_xy, cur_x,
  approach_floor_x)`), floor init at lines ~746–766.
- **Test.** Log **commanded** action vs **post-clip** action each step (the loop
  already prints when it floors). If they diverge on most steps, the guard, not
  the policy, is the problem. Re-run once with `--no-approach-floor` and a large
  `--safety-max-xy-delta` to see the unmasked policy.

### F3. Denormalization magnitude
- Covered by **B4** from the deploy end: `unnormalize` maps [-1,1] → metres via
  act_min/max. Test: printed metric actions should be ≤ ~0.008 m.

---

## G. Physical / environment

These are setup-match issues, all testable without retraining.

### G1. Camera view — CLEAN. Verified aligned on the rig (≤1mm).
Ran `align_pusht_camera.py` live overlay vs `pusht_widowx_cam1_ref.jpg`: the live
view matched the reference (≤~1mm off, no nudge needed). Camera pose matches
training; world→pixel mapping is correct. NOT the cause. (Offline registration
below was inconclusive — kept for the record.)

### G1 (offline attempt) — INCONCLUSIVE, superseded by the rig check above
- **Why.** A moved/re-aimed Logitech changes world→pixel mapping; a SUBTLE shift
  puts the goal outline at slightly wrong pixels → the policy (trained to push T
  onto the outline at training image-coords) aims slightly wrong → drifts →
  position-dependent, random-timestep stalls (matches the observed symptom: with
  a different initial pose the arm goes elsewhere, nudges the T, then stalls).
- **Offline attempt (this session):** ECC registration of deploy vs training
  frames is UNRELIABLE on this near-featureless white board — train-to-train
  controls (same fixed camera, must be ~0) scatter 12–133 px (affine even worse:
  scale 2.3, 12°). So no offline conclusion. Visual side-by-side
  (`results/g1_compare.png`) shows no GROSS difference (not flipped/rotated/
  rescaled) but can't exclude a subtle shift.
- **Files.** `scripts/align_pusht_camera.py` (live overlay vs
  `scripts/assets/pusht_images1_ref.jpg` — on the rig, not in this checkout).
- **Test (definitive, needs rig).** Run the alignment overlay; nudge until the
  live view matches the reference. **Cleaner offline check possible:** capture a
  deploy frame with the T REMOVED (pure static board+outline+square), then it can
  be registered against a training frame without the moving red T corrupting it.

### G2. Lighting / T darkness — QUANTIFIED (global underexposure) + SOFTWARE FIX added
- **Measured (this session)** deploy `dry_03` vs training frames:
  - T: peak_red 79 vs 110 (0.71x), mean_red 60 vs 80, sat 0.70 vs 0.82,
    RGB(100,32,48) vs (113,27,39) — washed out, +20%% G / +22%% B contamination.
  - **Board: (155,152,158) vs (189,180,185) = 0.82–0.86x** — the WHOLE scene is
    ~16%% dimmer with neutral balance preserved. So it's a **global
    underexposure**, not object-local; the saturated red T just loses the most.
- **Software fix added:** `deploy_pusht_real.py --match-exposure` applies
  per-channel gains (default `(1.22,1.18,1.17)` = train_board/deploy_board) in
  `preprocess`, lifting the frame to the training white point. Verified: deploy T
  peak_red 79→97, mean_red 60→71 (near the 110/80 target). Does NOT fix
  saturation (0.70→0.67) — the gains are ~uniform (brightness match). A
  `--saturation-gain` (HSV) refinement can fix the T's saturation without
  touching the neutral board, if brightness-match alone is insufficient.
- **Physical alternative:** raise lighting/exposure ~+20%% (fixes board + T).
- **Test.** `--match-exposure --dry-run` (confirm redder T), then
  `--match-exposure --steps 200 --log-dir …` and `analyze_rollout.py`.

### G3. Start pose — CHECKED: ~7mm off a razor-tight demo start (minor OOD)
- **Demos start in an extremely tight cluster:** x=0.1168±0.0004, y=−0.0195±0.0009,
  z=0.0177±0.0002 (x range [0.113,0.118], <1mm std). Start asset
  `pusht_start_eep.npy` = (0.1171,−0.0193,0.020), matching.
- **Deploy settles at x≈0.110, y≈−0.016, z≈0.0163** in every rollout — ~7mm short
  in x, BELOW the demonstrated start band [0.113,0.118] → slightly OOD from step
  0. `--move-to-demo-start` commands the right pose but the arm **undershoots x by
  ~7mm**.
- **Side effect:** the approach floor is set to the commanded start x (0.1171)
  while the arm actually starts at 0.110 → floor immediately active, pinning −dx
  from step 0 ("clipped dx at x=0.1096"). Demos push outward (+x) so it likely
  doesn't block, but it's off-nominal.
- **Verdict:** genuine but MINOR OOD; not the primary cause (stalls happen at
  random LATER steps, not at start). Worth fixing the move-to-start undershoot
  (get x to 0.117) and lowering/removing the approach floor for the test, but not
  the smoking gun.

### CONTACT ROOT — the arm never REACHES the T (approach failure, wrong y). NOT height.
Corrects both the z-droop AND the "too high to contact" theories. Verified with
the `--calibrate` run + demo contact analysis:
- **Frame convention is correct** (calib open-loop: +dx→world +6mm & image RIGHT,
  +dy→image UP, matching training). Commands execute (6mm cmd → 6mm world).
- **The arm's operating region excludes the T.** During calibration the arm swept
  x∈[0.11,0.23], y∈[−0.02,0.095] in all 4 directions and the T centroid never
  moved (211,203). The deploy policy (roll_c09_base) reached only x≤0.171,
  y∈[−0.016,0.056].
- **Where the arm SHOULD go:** demo contact-onset eef (where the T first moves)
  is x̄=0.228, ȳ=−0.125 (y range −0.32…+0.17). Fitting demo T-start-image →
  contact-eef and evaluating at the deploy T (image 211,203) predicts the arm
  must reach **≈(x=0.27, y=−0.12)** to touch it. It actually goes to ~(0.17,
  +0.03): ~10 cm short in x and the **WRONG SIGN in y**.
- **=> The policy drives the arm to the wrong place** (stays near y≈0 instead of
  going to negative y where this T sits) → never contacts the T → T static →
  dead visual signal → freeze/orbit. An APPROACH/localization failure, not
  contact height. Behavior isn't even consistent across sessions (base went +y,
  roll_04 went −y and grazed the T), i.e. closed-loop is unstable/OOD-sensitive.
- **Likely cause:** perception OOD — the T/scene differs enough that the policy
  mislocalizes the approach. z-perspective is an unlikely driver (only ~4 mm z
  difference). Prime concrete lead now: **G2 (T darkness, redness 79 vs 133)** and
  general closed-loop covariate shift. NEXT: fix T appearance to match training
  (lighting/saturation) and re-test; if the arm then drives to negative-y toward
  the T, perception was the cause.

### G4. z height — deploy arm sits higher than demos at matched x (SECONDARY; not the root)
Kept for reference, but per the CONTACT ROOT above the arm never reaches the T,
so contact height is moot until the approach is fixed.
- **Earlier claim was WRONG** (compared deploy z to the demo *start* z 0.0197
  instead of the demo z at the *same x*). The demos are SUPPOSED to be low at
  reach — they deliberately press the gripper DOWN as the arm extends.
- **Demo z(x) profile** (zarr `robot_eef_pose[:,2]`):
  x~0.11→0.016, 0.17→0.0125, 0.25→0.0039, 0.35→−0.0071, 0.45→−0.0097
  (full range +0.0195 … −0.0155). Roughly `z(x) ≈ 0.016 − 0.076·(x−0.11)`.
- **Deploy holds z too high** at matched x: roll_c09_base z=0.0169 vs demo 0.0125
  @x=0.17 (**+4.4 mm high**); roll_04 z=0.0070 vs demo ≈−0.001 @x=0.30
  (**+8 mm high**). lock_z holds z ~constant while the demos descend → at
  extension the gripper is ABOVE the T's contact zone → no push → **T never
  moves** (confirmed: red-T centroid frozen, see F). This is the likely contact
  root of the whole failure.
- **The demos command dz=0** (deploy: "all demo transitions have dims 2-6
  exactly zero"), so their z-descent is NATURAL DROOP with z nominally fixed —
  and the demos succeeded WITH it (droop brings the gripper to the T at reach).
  Deploy also commands dz=0 but ends up HIGHER → `lock_z`/`fixed_z_height=0.02`
  holds the arm up more than the demos drooped.
- **Fix (INVERTED from before), 2trans-compatible, NO server change:** let the
  arm sit LOWER to match the demo droop — try **`--no-lock-z`** (droop freely
  like the demos) or **lower `--fixed-z-height`** (e.g. toward the demo mid-reach
  height ~0.004). The z-hold servo / raised `--fixed-z-height` we built earlier
  push the WRONG way. For an exact match, drive z along the demo profile
  `z_target(x) ≈ 0.016 − 0.076·(x−0.11)` (that needs the 3trans z channel the
  server currently rejects).
- **Verify on rig with `--calibrate`:** does a scripted push move the T at the
  demo-height vs at the current (higher) height? Watch the T centroid in `raw/`.

---

## Recommended order

**Steps 1–2 DONE (see RESULTS): master test run, bisect = healthy offline.
Offline clears B1/B2/C1; **section A FULLY CLEARED** (A1/A3/A4 closed, A2
software-consistent → physical G1). Data+training exonerated. Remaining: D–G
(deploy-runtime + physical), which need the live rig. Start at step 3.**

1. ~~Run the master test.~~ Done — `scripts/diagnose_pusht_widowx.sbatch`.
2. ~~Bisect.~~ Done — **healthy offline → clears B1/B2/C1, not A2/A3/A4.**
   ~~A1~~ CLOSED via `check_frame_action_alignment.py` +
   `check_frame_count_parity.py`.
3. **Front-load the common-mode DEPLOY causes** (these break *every* config
   regardless of hyperparameters, matching "all fail identically"):
   **D4** (stale/duplicate stacked frame → learned "static ⇒ hold"; prime
   stall suspect since the policy never emits exact (0,0) itself),
   **D1** (channel order RGB/BGR), **A4/D2** (tf-vs-cv2 resize mismatch),
   **F1** (command→motion sign/axis). (**F2** approach-floor is only a masker
   when the policy is already driving toward the base — ~1% of demo steps — so
   it is a secondary check, not a primary cause.)
4. Only after the above, look at per-config causes (**B1** collapse, **E**
   conditioning) and physical setup (**G**).

## Quick commands

```bash
# Offline master test. Zarr + conditioning supported; archive auto-picked from
# each checkpoint's config. Raw weights (PUSHT_DIAG_NO_EMA=1) = the faithful
# recipe; also try default (EMA). NOTE: the sbatch --seeds default expects
# seed_00NN dirs; for the --tag-named batch dirs, call the .py directly:
PUSHT_OUTPUT_ROOT=$PWD/checkpoints/<seed_named_root> PUSHT_DIAG_NO_EMA=1 \
    sbatch scripts/diagnose_pusht_actions.sbatch

# Batch (cNN tag) checkpoints: diagnose resolves seed_dir as
# <output-root>/seed_<NN:04d>, so it CANNOT target a --tag dir (c01_...) as-is.
# Either (a) symlink, e.g.  ln -s c01_dropzero_base <root>/seed_0001  and run
# with --output-root <root> --seeds 1, or (b) add a --seed-dirs mode to
# diagnose that takes explicit directory names (small change, ~10 lines).
# The loader itself reads data_format + data_archive from each checkpoint config.

# Deploy, no motion — confirms obs sanity (D1 color, E1 state layout, D3 size)
python scripts/deploy_pusht_real.py --seed-dir <ckpt> --device cpu \
    --dry-run --dry-run-steps 20 --dump-dir results/dry_<tag>

# Deploy, capped logged rollout — F1 mapping, F2 clip-masking, D4 freshness, G3/G4
python scripts/deploy_pusht_real.py --seed-dir <ckpt> --device cpu \
    --steps 120 --log-dir results/roll_<tag>

# Unmask the policy from the safety guards (F2)
python scripts/deploy_pusht_real.py --seed-dir <ckpt> --device cpu \
    --steps 120 --no-approach-floor --safety-max-xy-delta 0.02 \
    --log-dir results/roll_<tag>_noguard
```

## Key artifacts to analyze

- `results/roll_*/steps.jsonl` — per step: norm action, metric action, 7-DoF
  EEF `state`. Source for quadrant/collapse checks, corr(command, EEF-Δ), stall
  detection, commanded-vs-clipped comparison, z/start-pose checks.
- `results/dry_*/fed_*.png` + `raw_*.npy` — what the model sees with no motion;
  feeds the color/brightness/resize parity checks.
- The dataset loader's startup line (`idle_filter=... kept X%`) in the training
  job stdout — A5.
