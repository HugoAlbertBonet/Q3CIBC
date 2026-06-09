# Handoff: Adding D4RL `door-human-v0` to Q3CIBC

This handoff is for a fresh agent session. Goal: add `door-human` as a new environment using the patterns we built for `pen-human`, then write the first batch of experiments (`doorA.txt`).

The Q3CIBC repo's pen-human support was built from scratch during a prior session — this document distills the protocol mistakes we made, the decisions that worked, and the exact files+lines to mirror for door.

---

## What Q3CIBC is

Hybrid IBC variant: a **Control Point Generator** outputs N candidate actions per state, jointly trained with a **Q Estimator** via MSE + InfoNCE. At inference time, pure CP-argmax picks the best CP — no MCMC/DFO refinement needed.

On `pen-human-v0` (5 seeds, 100 eval episodes each), Q3CIBC matches IBC paper EBM's raw return (2522 vs 2586) at **130× lower inference cost** (2.13 ms vs 276.8 ms per env-step on an RTX 3060). That headline is what we want to replicate on door.

Two training scripts wrap this:
- `combinedv2_cpascounter_training.py` — Q3CIBC main script (CP gen + Q est, joint training).
- `hyperparam_search.py` — trial runner with per-trial config isolation, env override, atomic JSONL logging.

Pure IBC paper reproduction is also implemented for sanity baselines:
- `hyperparam_search_dfo.py` — paper-faithful Langevin EBM, no CP cloud. Drives the same protocol on D4RL envs (`--active-env pen` or `--active-env particle` today).

---

## IBC paper's door-human target

From Florence et al. 2021 *Implicit Behavioral Cloning*:

**Table 2 (raw return, 3 seeds × 100 eval episodes each):**
| Method | door-human |
|---|---|
| BC (from CQL) | −41.7 |
| CQL | 234.3 |
| S4RL | 736.5 |
| Explicit BC (MSE) ours | 79 ± 15 |
| **Implicit BC (EBM) ours** | **361 ± 67** |
| Explicit BC (MSE) w/ RWR | 17.9 ± 13.8 |
| Implicit BC (EBM) w/ RWR | 399 ± 34 |

**Table 7 (env dimensions):**
- door-human: 25 demos, 39-D observations, **28-D actions** (1 dim less than pen's 24-D-... wait, pen has 24, door has 28 — door's hand has different joint count).

Actually re-check carefully: paper says pen=24, door=28. We confirmed pen=24 in our work. The door action_dim should be **28**.

**D4RL benchmark constants for normalization** (random=0, expert=100):
- `door-human-v0` random raw: **−56.62**
- `door-human-v0` expert raw: **2880.57**

Normalized score for IBC EBM raw=361: `100 × (361 − −56.62) / (2880.57 − −56.62) ≈ 14.2`.

That's the number to beat in the normalized-score table.

**Hyperparameters (App. D.1, same recipe for ALL D4RL human tasks — no per-task tuning):**
| Param | Value |
|---|---|
| EBM variant | Langevin |
| train iterations | 100,000 |
| batch size | 512 |
| learning rate | 5e-4 |
| LR decay | 0.99 every 100 steps |
| network size | 512 × 8 |
| dense layer type | **spectral norm** |
| activation | ReLU |
| train counter examples | 8 |
| gradient penalty | margin=1, hinge, final-step only |
| Langevin iterations (train) | 100 |
| Langevin step (init/final) | 0.5 / 1e-5 |
| Langevin noise scale | 0.5 |
| Langevin delta clip | 0.5 |
| Langevin decay power | 2.0 |
| Inference Langevin iterations | 100 |

Verified at: https://github.com/google-research/ibc/blob/master/ibc/configs/d4rl/mlp_ebm_langevin_best.gin (gin file lists `eval_episodes = 100`, confirming 3 seeds × 100 evals).

---

## Critical protocol gotchas (we hit ALL of these on pen)

### 1. Episode horizon: paper uses **100**, modern gymnasium-robotics defaults to **200**

D4RL `pen-human-v0` (legacy mujoco_py + Rajeswaran 2017) was registered with `max_episode_steps=100`. Modern `AdroitHandPen-v1` in gymnasium-robotics bumped to 200. Same dataset, different horizon. Paper numbers are at 100-step.

For door: assume same protocol — `AdroitHandDoor-v1` default is 200, **set `max_episode_steps=100`** in config to match paper.

Penalty for ignoring: raw rewards integrate over 2× more steps; numbers not comparable to paper.

### 2. Observation normalization: paper uses **standardize from dataset stats**, NOT JSON minmax

IBC paper App. B.1: *"All {x} and {y} (i.e. o and a for observations and actions), in the training dataset are normalized to per-dimension zero-mean, unit variance."*

We initially used JSON minmax bounds (`config_json/observation_bounds.json`) for pen. Wrong. After switching to dataset-derived `obs_mean`/`obs_std`, cross-seed variance tightened significantly.

For door: D4RLDataset already computes `obs_mean`/`obs_std` (we added this for pen). Code in `combinedv2_cpascounter_training.py` routes pen → standardize. Add door to the same branch:

```python
elif active_env in ("pushing", "pushing_multi", "pen"):
```
becomes
```python
elif active_env in ("pushing", "pushing_multi", "pen", "door"):
```

### 3. Action normalization: paper uses **per-dim min-max → [−1, 1] from dataset**

IBC App. B.3 for Langevin: *"all {y} (i.e. a for actions), in the training dataset are normalized per-dimension to span the range [ymin=−1, ymax=1]."*

`D4RLDataset(normalize_actions=True, action_norm_range=(-1, 1))` already does this. Just inherit the same setting.

### 4. Reward formula differs between mujoco_py and gymnasium-robotics ports

We measured a ~0.2/step reward offset between dataset-recorded rewards and `AdroitHandPen-v1`-emitted rewards. Same demonstrations, different reward formula → raw rewards are NOT directly comparable to paper. Use **D4RL normalized score** as the primary metric.

For door: this WILL also apply. Compute and report normalized scores.

### 5. Per-episode reward std is intrinsically high (~σ_ep≈1900 on pen)

Pen reward is dense, summed over 100 steps. Bimodal episode outcomes (success ~3000-9000, failure ~−500). Median always far below mean. `np.std(rewards)` over 100 episodes gives σ_ep ≈ 1900 — that's **intrinsic to the env**, not a bug.

Paper's reported "±65" is **cross-seed std of per-seed mean** (3 seeds × 100 eval episodes per seed). Different statistical object than per-episode std. Three reporting metrics matter:

| metric | definition | when to report |
|---|---|---|
| `σ_ep` (per-episode std) | spread of 100 individual episode rewards in one trial | per-trial; env-intrinsic |
| `cross_seed_std` | sample stdev of per-seed means over n training seeds | comparable to paper's "± std" label |
| `SEM` | `cross_seed_std / √n` | tightest alternative if paper meant SEM |

For door, expect similar bimodal reward structure (success = open door, failure = drop). Eval-noise floor on cross_seed_std will be `σ_ep / √100 ≈ ~σ_ep/10` regardless of policy quality.

### 6. Pure CP-argmax wins; Langevin/DFO inference refinement HURTS Q3C

We tried 30+ trials with `inference_langevin_iterations > 0` at paper hypers. All catastrophic (R = 60-200) regardless of training-time Q-coverage probes (uniform negatives, stronger gradient penalty, more Langevin train negatives).

Root cause: Q is sharp at trained points (CPs + Langevin negs), uncalibrated everywhere else. In 24/28-D action space, ~50 trained points per batch is a tiny fraction of the box. Inference refinement walks off-distribution within ~5 steps at paper-aggressive step sizes.

Gentle Langevin (lr_init=0.01, delta_clip=0.01, noise=0.1, 25 iters → walk envelope ~0.05) works (R≈2500 on pen) but doesn't beat pure CP-argmax (R≈2522 at 10 seeds).

DFO refinement (matched walk envelope) also works single-seed but seed-unstable; multi-seed mean below CP-argmax.

**Lesson for door:** start with `inference_langevin_iterations=0` from the very first batch. Skip all inference refinement experiments unless you have a specific reason.

---

## Files to add/modify for door support

### A. `config_json/config.json` — add door env block

Mirror the pen block. Look at lines for "pen" in current `config_json/config.json`:

```json
"door": {
    "dataset_name": "D4RL/door/human-v2",
    "env_id": "AdroitHandDoor-v1",
    "state_dim": 39,
    "action_dim": 28,
    "frame_stack": 1,
    "action_bounds": [-1, 1],
    "max_episode_steps": 100,
    "num_eval_seeds": 100,
    "training": { ... },
    "model": { ... }
}
```

Use pen's training + model blocks verbatim as a starting baseline. Same recipe across D4RL Adroit per paper.

### B. `config_json/observation_bounds.json` — add `AdroitHandDoor-v1` entry

This is FALLBACK only (the standardize path uses dataset stats, not this file). Still need a stub or the `ObservationNormalizer` minmax init crashes if it's ever invoked.

Easy stub: copy the pen entry's structure but mark `observation_dim: 39`. Per-dim bounds can be conservative (e.g., joint angles in [−π, π], positions in [−1, 1]). Won't be hit at training/eval if standardize routing is set up correctly.

### C. `simulations/door_human_v2_simulation.py` — new file, copy from pen

`simulations/pen_human_v2_simulation.py` is the template. Differences:
- `env_id = "AdroitHandDoor-v1"`
- `_register_envs` already calls `gym.register_envs(gymnasium_robotics)` — same for door
- Reward type: `"dense"` (same)
- Success flag: door env emits `info["is_success"]` per step while goal is met — sticky any-step success aggregation (same as pen)
- Action denormalization: same logic, per-dim from `norm_stats["act_min"]`/`act_max`

Copy-paste the file, sed `pen` → `door`, `Pen` → `Door`, `PEN` → `DOOR`. Verify env name.

### D. `simulations/__init__.py` — export

Add the new sim class export if there's an init that lists them.

### E. `hyperparam_search.py` — three sites

**Site 1: `evaluate_q3c` env dispatch (~line 622).** Add door branch:
```python
elif active_env == "door":
    from simulations.door_human_v2_simulation import DoorHumanV2Simulation
    SimulationCls = DoorHumanV2Simulation
```

**Site 2: `sim_kwargs` build (~line 1116).** Add door alongside pen:
```python
elif active_env in ("pen", "door"):
    pass  # no goal_dist_tolerance / n_dim args
```

**Site 3: metrics return (~line 1140).** Pen's return block (success_rate + reward triplet — mean/std/median) applies verbatim to door. Combine:
```python
if active_env in ("pen", "door"):
    return { ... pen's existing return dict ... }
```

**Site 4: `_results_dir` path map (~line 412).** Already supports `_ENV_PATH_MAP = {"pen": "d4rl/pen"}`. Add `"door": "d4rl/door"`:
```python
_ENV_PATH_MAP: dict[str, str] = {
    "pen": "d4rl/pen",
    "door": "d4rl/door",
}
```

Trials will land in `results/hyperparam_search/combinedv2_cpascounter_training/d4rl/door/trials.jsonl`.

### F. `combinedv2_cpascounter_training.py` — add door to standardize routing + norm_stats save

Two spots:

```python
elif active_env in ("pushing", "pushing_multi", "pen"):
    if not hasattr(dataset, "obs_mean") or not hasattr(dataset, "obs_std"):
        raise RuntimeError(...)
```
→ add `"door"`.

```python
if active_env in ("pushing", "pushing_multi", "pushing_pixels", "pen"):
    norm_stats = {...}
    torch.save(norm_stats, ...)
```
→ add `"door"`.

Also `load_dataset()` — pen uses `D4RLDataset`. Add door:
```python
elif active_env == "door":
    from utils.datasets import D4RLDataset
    return D4RLDataset(env_config["dataset_name"], download=True, frame_stack=frame_stack)
```

### G. `hyperparam_search_dfo.py` — optional, for IBC baseline reproduction

If you want to also run pure IBC paper-faithful baseline on door (mirror our `penAibc.txt`):
- Add `"door": "ibc_dfo_door"` to `_RESULTS_SLUG` at top
- Add door branch in `_make_env()` (eval loop env creation): use AdroitHandDoor-v1
- Add `door` to `--active-env` choices
- Add door to dataset loading branch (D4RLDataset)

Skip unless you have time. We already showed paper IBC reproduction is broken in our pipeline (R≈403 vs paper 2586 on pen — same code path on door will be similarly broken).

### H. Test pre-batch — DO THIS BEFORE LAUNCHING DOORA

1. `uv run --managed-python python -c "import minari; ds = minari.load_dataset('D4RL/door/human-v2', download=True); print(ds.total_episodes, ds.total_steps); ep = next(iter(ds.iterate_episodes())); print(ep.observations.shape, ep.actions.shape)"` — confirms dataset downloads, shows obs and action shapes (expect 39, 28).

2. `uv run --managed-python python -c "import gymnasium as gym; import gymnasium_robotics; gym.register_envs(gymnasium_robotics); env = gym.make('AdroitHandDoor-v1'); print(env.observation_space.shape, env.action_space.shape, env.spec.max_episode_steps)"` — confirms env shape and default horizon.

3. Run 1 trial at reduced steps to confirm config wiring:
   ```bash
   uv run --managed-python python hyperparam_search.py combinedv2_cpascounter_training.py --run --active-env door --reduced-steps 500 --fixed-params '{...minimal config...}'
   ```
   Verify trials.jsonl lands at `results/.../d4rl/door/trials.jsonl`.

---

## doorA.txt — first batch (8 trials, mirror penA structure)

Pattern from `batches/penA.txt`: 8 trials covering IBC paper-exact recipe + Q3CIBC's CP-cloud variants. All single-seed=0 to start.

Recipe: same Adroit-D4RL hypers from IBC paper App. D.1, with Q3CIBC modifications.

**Architecture (paper-faithful Q):**
- `q_network_kind=mlp, q_width=512, q_depth=8, q_use_spectral_norm=true`
- `cp_network_kind=mlp, cp_width=512, cp_depth=8`
- `learning_rate=5e-4, batch_size=512, training_steps=100000`
- `langevin_lr_init=0.5, langevin_lr_final=1e-5, langevin_decay_power=2.0, langevin_delta_clip=0.5, langevin_noise_scale=0.5`
- `gradient_penalty_weight=1.0, gradient_penalty_margin=1.0, gradient_penalty_form=hinge`

**Per-trial overrides (the 8 trials):**

| # | description | key params |
|---|---|---|
| A1 | paper-faithful baseline + CP cloud as extra negs | cp=20, top_k=8, lang_neg=8, lang_iter=100, inf_lit=100 |
| A2 | CP-only counter examples (no Langevin negs) | cp=20, top_k=8, lang_neg=0, lang_iter=0, inf_lit=100 |
| A3 | minimal CP (cp=1) paper-style — calibration anchor | cp=1, top_k=1, lang_neg=8, lang_iter=100, inf_lit=100 |
| A4 | A1 recipe + inference Langevin reduced 4× | cp=20, top_k=8, lang_neg=8, lang_iter=100, inf_lit=25 |
| **A5** | **zero inference Langevin — argmax over CPs ONLY** | cp=20, top_k=8, lang_neg=8, lang_iter=100, **inf_lit=0** |
| A6 | denser CP cloud | cp=50, top_k=20, lang_neg=8, lang_iter=100, inf_lit=100 |
| A7 | shorter training Langevin (3× cheaper train) | cp=20, top_k=8, lang_neg=8, lang_iter=25, inf_lit=100 |
| A8 | IBC mixture: CPs + Langevin + uniform negs | cp=20, top_k=8, lang_neg=8, unif_neg=16, lang_iter=100, inf_lit=100 |

**A5 is the predicted winner** — pen-A5 was Q3CIBC's headline (CP-argmax inference, no refinement, dominated all 6 other inference settings).

Each command should look exactly like pen batches:
```bash
uv run --managed-python python hyperparam_search.py combinedv2_cpascounter_training.py --run --active-env door --fixed-params '{...}'
```

Same as penA1 but s/pen/door/. Pen's penA.txt has the exact JSON structure to copy.

After all 8 trials run, analyze:
```bash
uv run --managed-python python hyperparam_search.py combinedv2_cpascounter_training.py --analyze --active-env door
```

The cross-seed aggregator and SEM/cross_std reporting already work for any env (env-agnostic).

---

## Pen-human results to anchor expectations

| metric | Q3C B4 (cp=100 multi-seed) | IBC paper EBM |
|---|---|---|
| Raw reward | 2522 ± 130 cross_std (10 seeds) | 2586 ± 65 (3 seeds) |
| D4RL normalized | 82.7 | 83.5 |
| Inference (RTX 3060) | 2.13 ms/env-step | 276.8 ms/env-step (130× slower) |
| Recipe | cp=100, top_k=30, **inf_lit=0** | 100 Langevin iters × 512 samples |

For door, paper EBM hit 361 ± 67 (normalized 14.2). Q3C target: match that (within statistical noise of the env's noise floor) with pure CP-argmax inference.

---

## Common failure modes we hit

1. **Spectral-norm checkpoint reload bug** — the `weight` keys get renamed to `weight_orig` under `torch.nn.utils.spectral_norm`. Naive `endswith(".weight")` filter sees zero matches → hidden_dims=[]. Fixed by inferring layer indices from `.bias` keys (unchanged under SN), and detecting SN via `weight_orig OR parametrizations.weight` keys. See `hyperparam_search_dfo.py:evaluate_checkpoint` (this fix already in place — door reuse should inherit it).

2. **Training duplicate trial-id silent grouping** — `--analyze` groups by params dict minus `trial_seed`. If 2 trials have identical params except seed, they get grouped — good. But if a hyperparam changes mid-batch (e.g., we changed protocol mid-experiment), trials look identical but aren't comparable. Use `--min-trial-id N` to scope to a recent batch.

3. **Reward-formula mismatch confused us early** — initially thought our pipeline was broken because our raw rewards (~2500) didn't match paper's exact 2586. They differ because env reward formulas differ. **Use D4RL normalized score** as the primary metric for door results. Footnote env port as the reason.

4. **Aggressive inference refinement is catastrophic** — every penA-E batch with `inference_langevin_iterations > 0` at paper hypers (lr=0.5, delta_clip=0.5, noise=0.5, 100 iters) gave R=60-200. Gentle hypers (lr=0.01, delta_clip=0.01, noise=0.1, 25 iters) work but don't beat pure CP-argmax.

   **For door, skip all `inf_lit>0` experiments unless explicitly debugging.**

5. **Single-seed wins don't multi-seed** — we found 3 single-seed configurations beating B4 base on pen (#70 = 2704, #59 = 2686, #43 = 2793). Multi-seed validation dropped all of them below B4 base. **Always multi-seed validate any single-seed winner before claiming.**

---

## Compute budget

- Per trial: ~12h SLURM at 100k steps + 100-episode eval (RTX 3060 or similar GPU).
- `submit_experiments.sh batches/doorA.txt doorA` creates 8 SLURM jobs from the batch file.
- 25h wall per job. All 8 trials fit comfortably.

After doorA: triseed the winner (likely the inf_lit=0 variant) in doorB to get 3-seed mean for paper-comparable SEM/cross_std.

---

## Critical lesson: the headline isn't "match paper" — it's "match paper at 130× cheaper inference"

The interesting Q3CIBC story is the **inference compute reduction**, not raw reward parity (which is within statistical noise of paper's number on pen, and likely similar on door). Lead the write-up and READMEs with the speedup. Reward is the control variable.

Already documented this framing in README.md's `### D4RL` section for pen. Door should follow the same pattern. After doorA + doorB completes, update README with door table mirroring the pen one.

---

## Final checklist for the receiving agent

- [ ] Read this entire doc + skim `README.md` D4RL section + skim `batches/penA.txt`
- [ ] Verify `D4RL/door/human-v2` is loadable via minari (run the smoke command above)
- [ ] Verify `AdroitHandDoor-v1` env in gymnasium-robotics (run the smoke command above)
- [ ] Add door to config.json (mirror pen entry, set action_dim=28 obs_dim=39 max_steps=100)
- [ ] Add door observation_bounds.json stub (39-D, conservative ranges)
- [ ] Copy pen_human_v2_simulation.py → door_human_v2_simulation.py + sed pen → door + verify env_id
- [ ] Patch hyperparam_search.py at the 4 sites (env dispatch, sim_kwargs, metrics, _ENV_PATH_MAP)
- [ ] Patch combinedv2_cpascounter_training.py at the 3 sites (load_dataset, obs normalizer branch, norm_stats save)
- [ ] Run reduced-steps smoke test (`--reduced-steps 500`) to verify wiring
- [ ] Write batches/doorA.txt (8 trials, mirror penA pattern, seed=0 only)
- [ ] Submit and wait for results
- [ ] After doorA: write doorB (multi-seed validate winner; expect inf_lit=0 to win)
- [ ] Update README with door D4RL table once doorB completes

You're set up to land Q3CIBC on door-human within ~2-3 batches if the pen pattern transfers directly. Good luck.
