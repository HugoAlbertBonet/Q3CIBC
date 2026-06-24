# Plan: Q3CIBC on LIBERO-Goal under the STANDARD (pixel) protocol

Handoff plan for an agent to implement the **standard LIBERO protocol** variant of
the `libero_goal` env, so Q3CIBC numbers land in the same column as published
baselines (OpenVLA, Diffusion Policy, Octo, …). Goal: get the **first batch**
(`liberoGoalStdA.txt`) running.

The state-based variant already works end-to-end (`active_env=libero_goal`,
batch `liberoGoalA.txt`). This plan adds a **pixel** variant; do NOT modify the
state-based path — add a parallel `libero_goal_pixels` env.

---

## What "standard procedure" means (target spec)

- **Obs**: per-camera RGB — `agentview_rgb` + `eye_in_hand_rgb` (128×128×3 each)
  — PLUS low-dim proprio (`ee_pos`(3) + `gripper_states`(2) + `joint_states`(7) = 12).
  **No `object-state`** (that's privileged; the whole point of pixels is the policy
  infers objects from images).
- **Task conditioning**: BERT/MiniLM embedding of the task instruction, same as the
  state variant (reuse `scripts/precompute_libero_goal_embs.py` cache).
- **Action**: 7-D, already in [-1, 1].
- **Metric**: success rate, mean over all 10 tasks × N rollouts (use 50 seeds like
  the state variant). Headline = per-suite mean success.

Standard baselines this makes us comparable to (LIBERO-Goal, multitask, image+lang):

| Method | LIBERO-Goal SR |
|---|---|
| Diffusion Policy (scratch) | 68.3% |
| OpenVLA | 79.2% |
| Octo (ft) | 84.6% |
| OpenVLA-OFT / Dream-VLA | ~97% |

(Encoder note: baselines use ResNet/ViT; our `ConvMaxpoolEncoder` is IBC-style.
Numbers are protocol-comparable but not encoder-matched — state that in the paper.)

---

## What already exists to REUSE (do not rewrite)

- `utils/models.py`: `ConvMaxpoolEncoder`, `PixelControlPointGenerator`,
  `PixelQEstimator`, `DenseResnetValue` — the full pixel CP/Q stack (from
  `pushing_pixels`). Late-fusion encode-once-per-state is already implemented.
- `utils/datasets.py`: `PushingPixelsDataset` — the lazy JPEG/uint8 channel-stack
  pattern + frame-stack index map. Model template for the LIBERO pixel dataset.
- `utils/libero.py`: benchmark/task accessors, goal-embedding cache I/O,
  `LIBERO_CONFIG_PATH` auto-config, `resolve_live_obs`, `_EXCLUDE_LOWDIM_KEYS`.
- `simulations/libero_goal_simulation.py`: `LiberoGoalSimulation` — renderless
  multitask env loop, goal concat, action denorm. The pixel sim is a close cousin
  (but needs rendering ON — see gotchas).
- `combinedv2_cpascounter_training.py`: the `active_env=="pushing_pixels"` branch
  already builds `PixelControlPointGenerator`/`PixelQEstimator` and skips the obs
  normalizer. Mirror it.
- `hyperparam_search.py`: `pushing_pixels` model-build + sim dispatch in
  `evaluate_q3c`. Mirror for `libero_goal_pixels`.

---

## NEW WORK (the actual task)

### 1. Goal/task-embedding fusion into the pixel nets  (utils/models.py)
The pixel CP/Q nets take only image features today. Add the goal embedding:
- `PixelControlPointGenerator.forward(images, goal_emb)`: encode image → 256-D,
  **concat `goal_emb` (384-D)** → 640-D → CP head. Add `goal_dim` ctor arg; head
  `input_dim = encoder_feature_dim + goal_dim`.
- `PixelQEstimator`: in `encode()` keep image→256-D; in `score(features, action,
  goal_emb)` concat `[features, goal_emb, action]` before the value net. Update
  `DenseResnetValue` `in_dim = feature_dim + goal_dim + action_dim`.
- Keep backward-compat (default `goal_dim=0` → behaves like pushing_pixels).
This is the core model change. Multi-camera: stack the two RGB streams
channel-wise (6→12 channels with frame_stack handled like pushing) OR run two
encoders and concat features. Start with **channel-stack agentview+wrist** (one
encoder, in_channels = 3*2*frame_stack) — simplest.

### 2. `LiberoGoalPixelsDataset`  (utils/datasets.py)
Model on `PushingPixelsDataset` + `LiberoGoalDataset`:
- Per task (benchmark order), open demo HDF5, read `obs/agentview_rgb` +
  `obs/eye_in_hand_rgb` (uint8, T×128×128×3) and proprio keys (ee_pos, gripper,
  joint — reuse `select_lowdim_obs_keys` minus object-state, or a fixed list).
- Channel-stack the two cameras (+ frame_stack) → (3*2*fs, 128, 128) uint8.
- Return `{"image": stacked_uint8, "proprio": vec, "goal": emb[task], "action": a}`.
  (Or fold proprio into the value net input alongside features+goal.)
- Min-max normalize actions to [-1, 1] (reuse the pattern); expose `act_min/max`.
- `state_shape` = the image shape tuple; expose `goal_emb_dim`, `proprio_dim`.
- RAM: ~500 demos × ~150 frames × 2 cams × ~10-14KB JPEG... LIBERO stores raw
  uint8 not JPEG — that's ~500×150×2×128×128×3 ≈ 7 GB. **Consider storing encoded
  or downsampling, or lazy per-__getitem__ HDF5 reads** (don't load all to RAM).
  Recommend: keep HDF5 handles open, read frames lazily in `__getitem__`.

### 3. Training wiring  (combinedv2_cpascounter_training.py)
- `load_dataset`: add `libero_goal_pixels` → `LiberoGoalPixelsDataset`.
- Model build: add branch mirroring `pushing_pixels` but pass `goal_dim` and feed
  `goal_emb` through the CP gen + Q est. `obs_normalizer=None` (encoder
  preprocesses). Thread proprio + goal into the forward calls (`q_score_candidates`).
- `norm_stats` save: persist `act_min/max`, `goal_embeddings`, `goal_task_names`,
  `libero_obs_keys` (proprio), `encoder_target_*`, `in_channels`, `proprio_dim`,
  `goal_emb_dim` so eval reconstructs the model + obs.

### 4. Pixel eval sim  (simulations/libero_goal_pixels_simulation.py)
Subclass/adapt `LiberoGoalSimulation` BUT:
- Env MUST render (need camera images): use `OffScreenRenderEnv(bddl, camera_heights=128,
  camera_widths=128)` (NOT the renderless ControlEnv).
- `select_action`: build image tensor from live `agentview_image`+`robot0_eye_in_hand_image`
  (note: live keys are `*_image`, demos are `*_rgb` — bridge in resolve), proprio
  via `resolve_live_obs`, goal emb per task → pixel CP-argmax (encode once, score CPs).
- **Group eval by task** (see gotchas) — render env create/destroy is the risk.

### 5. Config + batch
- `config_json/config.json`: add `libero_goal_pixels` block — `state_dim` as the
  image shape `[6, 128, 128]` (or 12 ch for 2 cams), `action_dim` 7, frame_stack 1-2,
  `encoder_target_height/width` 128, `goal_embeddings_path`, model widths.
- `hyperparam_search.py`: sim dispatch + pixel model build (mirror pushing_pixels;
  read shapes from norm_stats).
- `batches/liberoGoalStdA.txt`: first batch — vary `control_points`, encoder/value
  widths, frame_stack, learning rate. Pure CP-argmax inference (refinement wrappers
  don't thread goal yet). `uv run --extra libero` (no --managed-python), MUJOCO_GL=egl.

---

## GOTCHAS (learned the hard way on the state variant — heed these)

1. **Rendering DOES work on a GPU compute node** (`MUJOCO_GL=egl`), proven by the
   object-state extraction. It FAILS on login nodes (no libEGL). All pixel
   train+eval must run under `srun --gres=gpu:1 --mem=32G`.
2. **Render env create/destroy churn segfaults** (EGL teardown). The eval loop must
   **group episodes by task** so each task's render env is created ONCE
   (10 create/close total), not per-episode round-robin (50×, crashes). Either
   change the seed→task mapping to `task = seed // (N//n_tasks)` (consecutive seeds
   = same task) or pass `num_eval_seeds` to the sim so it groups.
3. **Do NOT cache all 10 render envs** — that OOM-killed the job. One env at a time.
4. **Live obs key names ≠ demo key names**: live = `agentview_image`,
   `robot0_eye_in_hand_image`, `robot0_eef_pos`, `robot0_gripper_qpos`,
   `robot0_joint_pos`; demos = `agentview_rgb`, `eye_in_hand_rgb`, `ee_pos`,
   `gripper_states`, `joint_states`. Add image aliases to `LIVE_KEY_ALIASES`.
   Image channel order / BGR-vs-RGB + uint8 scaling must match between demo frames
   and live render (`macros_image_convention` in the demo `data` attrs — check it;
   LIBERO images are often stored bottom-up, may need a vertical flip).
5. **torch.load + numpy**: `hyperparam_search.py` already monkeypatches
   `torch.load` to `weights_only=False` — keep it; norm_stats has numpy arrays.
6. **`uv run` prunes out-of-band installs**; everything is pinned in pyproject's
   `libero` extra. `uv sync --extra libero` once before submitting (submit script
   no longer pre-syncs).
7. **Compute**: pixel training is ~5–10× the state variant; eval renders every step.
   Budget accordingly; keep the first batch small (≤8 trials).

---

## Verification ladder (do in order, paste output each rung)
1. `LiberoGoalPixelsDataset` loads on a GPU node — check shapes, no OOM.
2. One training step runs (model fwd/bwd with image+proprio+goal).
3. Probe: live render obs keys + image shape match the dataset (extend
   `scripts/probe_libero_live_obs.py` for images).
4. Smoke trial: `--run ... training_steps=500` → eval prints `success_rate` with no
   segfault/OOM (grouped-by-task render).
5. Submit `batches/liberoGoalStdA.txt`.

---

## Definition of done
`./submit_experiments.sh batches/liberoGoalStdA.txt` runs N pixel trials to
completion, logging `success_rate` per trial in
`results/hyperparam_search/combinedv2_cpascounter_training/libero_goal_pixels/trials.jsonl`,
comparable (protocol, not encoder) to the published LIBERO-Goal table above.
