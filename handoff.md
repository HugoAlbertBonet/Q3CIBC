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

### D1. Channel order (RGB vs BGR)
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

### D2. Resize mismatch (deploy side)
- Same underlying issue as **A4**, viewed from the deploy end. `preprocess`
  uses `cv2.INTER_AREA`; the cache used tf AREA. Test as in A4.

### D3. Wrong target resolution
- **What / why.** Training cache is 240×320; deploy must resize to the same
  `(image_height, image_width)` before the encoder (which then bilinearly
  resizes to 180×240 internally). A mismatch changes aspect/scale.
- **Failure type.** Geometric OOD.
- **Files.** deploy reads `image_height`/`image_width` from `config.json`;
  `preprocess(out_hw=(image_h, image_w))`.
- **Test.** Confirm `image_height=240`, `image_width=320` in `config.json`;
  `fed_*.png` should be 320×240.

### D4. Stale / duplicate / zero-motion frame
- **What / why.** The original `(-,-)` runaway was caused by a 2-frame stack
  with no inter-frame motion (server repeated an image, or the frame was grabbed
  before the commanded move landed). The stack is the *only* motion signal the
  pixels-only policy has; a static stack maps to the learned "hold".
- **Failure type.** Freeze or runaway; self-locking once the arm stalls.
- **Files.** `extract_blue_frame` / the control loop grab logic;
  `stack_to_tensor` (line ~431).
- **Test.** From a `--log-dir` run, diff consecutive `raw/*.npy`; they must
  differ by more than sensor noise. If identical frames appear in a stack, the
  freshness guard is failing.

---

## E. Conditioning  *(new; only c09 / c10)*

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

### G1. Camera view differs from training
- **Why.** A moved/re-aimed Logitech changes the mapping from world to pixels;
  the fixed-camera assumption breaks.
- **Files.** `scripts/align_pusht_camera.py` (live overlay vs
  `scripts/assets/pusht_images1_ref.jpg`).
- **Test.** Run the alignment overlay; nudge until the live view matches the
  reference.

### G2. Lighting / T darkness
- **Why.** Previously measured: the deploy T rendered ~33% darker than training
  (peak redness 120 → 83) while the mat matched — an object-local lighting/
  saturation shift, i.e. real OOD on the one salient object.
- **Files.** `scripts/check_brightness_parity.py` (peak-redness vs target ≈120),
  `deploy_pusht_real.py --dry-run` (captures `raw_*.npy`).
- **Test.** Dry-run capture → brightness parity; tune light-on-T / camera
  saturation until peak ≈ 120.

### G3. Start pose OOD
- **Why.** Demos all start at x≈0.117; starting elsewhere is OOD from step 0.
- **Files.** deploy `--move-to-demo-start` (default ON) + `--start-eep-npy`
  (`scripts/assets/pusht_start_eep.npy`); logged `state`.
- **Test.** `steps.jsonl` step 0 EEF vs the asset (x≈0.117, y≈−0.019).

### G4. z droop
- **Why.** `lock_z` commands z=0.02 but the arm was measured drooping to ~0.009
  at extended poses; demos held ~0.0197. Wrong contact height with the T.
- **Files.** deploy `--fixed-z-height` / `--lock-z`; logged `state[2]`.
- **Test.** Log z through a rollout; compare to 0.0197. Raise `--fixed-z-height`
  to compensate if it sags.

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
