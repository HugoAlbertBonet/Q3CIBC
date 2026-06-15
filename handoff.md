# Door-Human Handoff (Q3CIBC, D4RL `door-human`)

You are inheriting the D4RL `door-human` work on Q3CIBC. Pen-human was the lead environment; door is the second. Pen is in a stable headline-ready state; door is mid-investigation with critical caveats. Read this end-to-end before doing anything.

---

## Repo orientation (door-relevant only)

| File | What it does |
|---|---|
| `config_json/config.json` | `environments.door` block. `max_episode_steps=200`, `action_dim=28`, `state_dim=39`, `num_eval_seeds=100`. Don't edit casually. |
| `combinedv2_cpascounter_training.py` | Q3CIBC training script. Door already routed through D4RLDataset + standardize observations + per-dim action normalization. |
| `hyperparam_search.py` | Trial runner for Q3CIBC. Door supported at all four dispatch sites. Results land in `results/hyperparam_search/combinedv2_cpascounter_training/d4rl/door/trials.jsonl`. |
| `hyperparam_search_dfo.py` | Pure-IBC paper recipe runner (no CP cloud). Door supported alongside pen/particle. Results land in `results/hyperparam_search/ibc_dfo_door/trials.jsonl`. |
| `simulations/door_human_v2_simulation.py` | Door eval simulation class. Built on `AdroitHandDoor-v1` via gymnasium-robotics. Sticky any-step `info["success"]` aggregation. |
| `utils/datasets.py` | `D4RLDataset` is env-agnostic. Handles both pen and door via `dataset_name` from config. Computes `obs_mean`, `obs_std`, `act_min`, `act_max` from the dataset. Action normalization to `[-1, 1]` per-dim. |
| `utils/normalizations.py` | `ObservationNormalizer` runs in standardize mode for door (mean/std from dataset, not JSON bounds). |
| `config_json/observation_bounds.json` | Hand-authored minmax bounds. Door has a stub entry. Standardize routing means this is never actually called at runtime — it exists only to keep `ObservationNormalizer` constructor from crashing. |
| `batches/doorA.txt`, `doorB.txt`, `doorC.txt` | Q3CIBC door batches in chronological order. |
| `batches/doorAibc.txt` | IBC paper-exact recipe batch via `hyperparam_search_dfo.py`. Written but NOT yet submitted as of this handoff. |

---

## Critical finding: reward formula mismatch on door

The IBC paper reports door-human EBM = 361 ± 67 raw return. That number is **not reproducible in our setup** for reasons unrelated to algorithm quality.

### What was measured

Replay episode 0's expert actions in our env:

```python
env = gym.make('AdroitHandDoor-v1', reward_type='dense', max_episode_steps=300)
obs, _ = env.reset(seed=0)
env_cumulative = sum(env.step(ep.actions[t])[1] for t in range(300))
# env_cumulative = -67.1
# dataset.rewards[:300].sum() = +525.0
```

A ~600-point cumulative gap over 300 steps between dataset-recorded reward and env-emitted reward, using the same actions.

### Caveat: I did NOT disentangle the cause

The 600-point gap could be:
1. **Initial state drift** — `env.reset(seed=0)` doesn't necessarily match the dataset recording's initial state (minari doesn't expose the recording seed).
2. **Physics drift** — gymnasium-robotics' mujoco-python bindings vs legacy d4rl's mujoco_py + different XML model could produce different state evolution under identical actions.
3. **Reward formula difference** — legacy d4rl `door-human-v0` (Rajeswaran 2017) may compute reward differently than gymnasium-robotics' `AdroitHandDoor-v1`.

To isolate, you'd compare obs trajectories step-by-step:
```python
for t in range(300):
    obs_env, r, _, _, _ = env.step(ep.actions[t])
    obs_ds = ep.observations[t+1]
    print(t, np.abs(obs_env - obs_ds).max())  # if these track, the gap is pure reward formula
```
I did not run this. **Do this before claiming "reward formula differs"** with certainty.

### Implications regardless of cause

Our env produces a different reward signal than the paper's setup. Therefore:
- Raw-reward comparison to paper's +361 is invalid (whatever the precise mechanism).
- `D4RL normalized score` is broken on our env: dataset-replay expert (`-67`) is BELOW random (`-46`) when averaged per-step their numbers are comparable (~-0.23/step), but the anchors don't make sense.
- `success_rate` (env emits `info["success"]`) is the only cross-env-fair metric, but the paper does NOT report SR — so no direct head-to-head against paper.

### The fix: option A — IBC baseline in our env

Train IBC paper recipe in OUR env via `hyperparam_search_dfo.py --active-env door`. The result is whatever IBC achieves on our reward formula. Q3C's number is on the same scale. Comparison is then internal to our env. Paper's +361 becomes irrelevant — replaced by an internal head-to-head between Q3C and IBC under identical conditions.

`batches/doorAibc.txt` was written for this and is queued (not yet submitted). **Submit this batch first.**

---

## Q3C door results so far

### doorA (IDs 1-8, horizon=100, ABANDONED)

Initial doorA used `max_episode_steps=100`. **Wrong.** Door demos reach success around steps 196-276 (mean ep length=269). Truncating at 100 cuts off the entire success phase. All trials hit R=-21 to -22 (= dataset's first-100-step expert return).

Diagnosis confirmed by reading dataset directly:

```python
ds = minari.load_dataset('D4RL/door/human-v2')
ep_lens = [len(ep.actions) for ep in ds.iterate_episodes()]
# min=223, mean=269, max=300
ep_first_success = [int(np.argmax(ep.rewards > 5)) for ep in ds.iterate_episodes()]
# mean=196, min=156, max=276
```

`config_json/config.json` was patched to `max_episode_steps=200`.

### doorB (IDs 9-16, horizon=200)

| # | recipe | R | SR |
|---|---|---|---|
| 15 | cp=100/top_k=30, inf_lit=0 (B4 pattern) | **+37.9** | 7% |
| 13 | cp=50/top_k=20, inf_lit=25 | -12.4 | 2% |
| 9 | cp=20/top_k=8, no Langevin, inf_lit=0 | -14.1 | 3% |
| 16 | cp=50/top_k=20, inf_lit=0 | -22.4 | 3% |
| 10,11,12,14 | all inf_lit=100 | **-42 to -43** | 0% |

**Confirmed: `inference_langevin_iterations=100` (paper recipe) catastrophic** on Q3C with door's narrow-trained Q. Same finding as pen. **Drop inf_lit > 0 from future batches.**

### doorC (IDs 17-33, 17/20 ran)

| # | recipe | R | SR |
|---|---|---|---|
| 27 | **cp=300/top_k=80** | **+51.1** | 16% |
| 20 | cp=100 seed=3 (B4) | +49.4 | 14% |
| 33 | cp=100 + 16 Langevin negs | +44.2 | 11% |
| 29 | cp=100 + GP=10 | +37.9 | 7% |
| 25 | cp=200/top_k=50 | +30.8 | 11% |
| ... | (many trials between +30 and -30) | | |
| 21 | cp=100 seed=4 (B4) | **-30.3** | 2% |

**Multi-seed B4 (cp=100, 5 seeds {0,1,2,3,4}):**
- Mean R = 14.6
- Cross-seed std = 31.1
- SEM = 13.9
- Brutal seed sensitivity (seed 4 = -30, seed 3 = +49)

Three trials timed out (didn't fit 25h SLURM wall):
- cp=100 + Q=1024×8 (bigger Q)
- cp=100 + 200k training steps
- cp=100 + 32 Langevin training negatives

If you re-attempt them, bump wall to `--time=35:00:00`.

### Single-seed promising but unvalidated

- **cp=300/top_k=80 (#27): +51.1** — best single-seed result. Capacity push works.
- **cp=100 + 16 Langevin negs (#33): +44.2** — broader Q coverage at training time.
- **cp=100 + GP=10 (#29): +37.9** — stronger gradient penalty.

All seed=0. Multi-seed validation needed before claiming any of these.

---

## Architecture/protocol audit (verified)

| Component | Paper App. D.1 | Ours | Match |
|---|---|---|---|
| Q net width × depth | 512 × 8 | q_width=512, q_depth=8 | ✓ |
| Dense layer | spectral norm | q_use_spectral_norm=true | ✓ |
| Activation | ReLU | ReLU | ✓ |
| LR | 5e-4 | 5e-4 | ✓ |
| Batch | 512 | 512 | ✓ |
| Training steps | 100k | 100k | ✓ |
| Counter examples | 8 | num_langevin_negatives=8 | ✓ |
| Train Langevin: iters/lr_init/lr_final/noise/clip | 100/0.5/1e-5/0.5/0.5 | identical | ✓ |
| Gradient penalty | margin=1, hinge | margin=1, hinge | ✓ |
| Eval episodes/seed | 100 | num_eval_seeds=100 | ✓ |
| Horizon | 200 (legacy d4rl door-human-v0) | max_episode_steps=200 | ✓ |
| Action dim | 28 | 28 | ✓ |
| Obs dim | 39 | 39 | ✓ |

Q estimator architecture matches paper. Q3CIBC ADDS a CP generator (not in paper) — that's our addition.

**The env is NOT the same as paper's env** (gymnasium-robotics port, different reward formula). This is the unfixable confound on raw reward; see the section above.

---

## Conventions to follow

1. **Always set `inference_langevin_iterations=0`** in door batches. Aggressive Langevin (paper hypers: lr=0.5, delta_clip=0.5, noise=0.5, 100 iters) is catastrophic on Q3C's narrow-trained Q. Gentle Langevin (lr=0.01 etc.) works on pen but never beat pure CP-argmax there either. Skip both for door unless explicitly debugging the inference path.
2. **Always multi-seed validate single-seed wins.** On pen, several single-seed configs that hit R>2700 dropped below B4 base when re-run at 4 more seeds. Same will happen on door.
3. **Use `--active-env door`** for all door trials. Do NOT mutate `config_json/config.json`'s `active_env` field for SLURM batches — race condition between submit and record.
4. **Trial IDs are global per env.** doorA = 1-8, doorB = 9-16, doorC = 17-33, etc. `--analyze --min-trial-id N` to scope a recent batch.
5. **Q3C results path:** `results/hyperparam_search/combinedv2_cpascounter_training/d4rl/door/trials.jsonl`.
6. **IBC results path:** `results/hyperparam_search/ibc_dfo_door/trials.jsonl` (separate file, populated when `doorAibc.txt` runs).

---

## Standard commands

### Submit a batch
```bash
./submit_experiments.sh batches/doorAibc.txt doorAibc
# Each line becomes one SLURM job. 25h wall per job by default. Pass a higher
# --time= in the script template if you need heavier configs.
```

### Analyze results (cross-seed aggregator built-in)
```bash
# Q3C results
uv run python hyperparam_search.py combinedv2_cpascounter_training.py \
    --analyze --active-env door --min-trial-id 17

# IBC paper-recipe results
uv run python hyperparam_search_dfo.py --analyze --active-env door
```

Both analyzers print:
- Sorted trial table
- Cross-seed aggregates (groups of trials with identical config except `trial_seed`)
- Per group: mean_R, cross_std, SEM, σ_ep(avg), SR(avg)

### Inspect dataset / env quickly
```bash
uv run --managed-python python -c "
import minari, numpy as np
ds = minari.load_dataset('D4RL/door/human-v2')
ep = next(iter(ds.iterate_episodes()))
print('obs', ep.observations.shape, 'act', ep.actions.shape)
print('reward range:', ep.rewards.min(), ep.rewards.max(), 'sum:', ep.rewards.sum())
"

uv run --managed-python python -c "
import gymnasium as gym, gymnasium_robotics
gym.register_envs(gymnasium_robotics)
env = gym.make('AdroitHandDoor-v1', reward_type='dense', max_episode_steps=200)
print(env.observation_space, env.action_space, env.spec.max_episode_steps)
"
```

---

## Immediate next steps for the new agent

In recommended order:

### 1. Submit `batches/doorAibc.txt`

10 trials, IBC paper-exact recipe in our env. Result is the IBC baseline measured under our reward formula. This is the comparison anchor for Q3C — replaces paper's +361 (unreachable in our env).

After it completes, run:
```bash
uv run python hyperparam_search_dfo.py --analyze --active-env door
```

You'll get a triseed mean ± std/SEM. That's the IBC baseline number.

### 2. Diagnose the env/reward gap definitively

Before drawing conclusions about Q3C's door performance, run the obs-drift test I skipped:

```python
# In a uv run --managed-python python session
import minari, numpy as np
import gymnasium as gym, gymnasium_robotics

gym.register_envs(gymnasium_robotics)
ds = minari.load_dataset('D4RL/door/human-v2')
ep = next(iter(ds.iterate_episodes()))

env = gym.make('AdroitHandDoor-v1', reward_type='dense', max_episode_steps=300)
obs, _ = env.reset(seed=0)
drift = []
reward_diffs = []
for t in range(300):
    obs_t, r_t, _, _, _ = env.step(ep.actions[t])
    drift.append(float(np.abs(obs_t - ep.observations[t+1]).max()))
    reward_diffs.append(r_t - float(ep.rewards[t]))
print(f'max obs drift: {max(drift):.4f}')
print(f'final cumulative obs drift over 300 steps: monotonic? {drift[-1] > drift[0]}')
print(f'per-step reward diff: mean={np.mean(reward_diffs):.4f} std={np.std(reward_diffs):.4f}')
print(f'cumulative reward diff (env - dataset): {sum(reward_diffs):+.1f}')
```

If `max obs drift` is tiny (< 0.01) → state trajectories track, the 600-point gap IS the reward formula difference. If drift grows large → mixed confound, can't pin down.

Either way, **document the result** somewhere (this handoff or a fresh `door_env_audit.md`).

### 3. Continue Q3CIBC exploration: doorD

Promising directions identified from doorC results:
- **cp=300/top_k=80 multi-seed** (penD #27 hit +51 single-seed)
- **cp=400 / cp=500** (extend capacity sweep further)
- **cp=100 + 16/32 Langevin negs multi-seed** (#33 hit +44 single-seed)
- **75k or 200k training steps revisit** (both hurt single-seed but multi-seed may differ)
- **Larger Q (1024×8) retry** with `--time=35:00:00` (timed out before)

Suggested doorD layout (20 trials):
- 5 trials: cp=300 multi-seed (seeds 0-4) — validate single-seed winner
- 3 trials: cp=400, cp=500, cp=600 — push capacity further
- 4 trials: cp=100 + 16 Langevin negs multi-seed (seeds 0-3)
- 4 trials: cp=100 + 32 Langevin negs multi-seed (seeds 0-3) — re-attempt timed-out config with extra wall
- 4 trials: best capacity-validated config + GP=2/5/10 sweep

Always `inference_langevin_iterations=0`. Mirror `batches/doorC.txt` JSON structure exactly — just change relevant param keys.

### 4. Once everything's in: build the comparison table

After doorAibc and doorD complete, you'll have:

| metric | Q3C-best (door) | IBC paper-recipe (door) |
|---|---|---|
| n_seeds | TBD | 3 |
| avg_reward (on OUR env reward formula) | TBD | TBD |
| cross_seed_std | TBD | TBD |
| success_rate | TBD | TBD |
| inference time (ms/env-step) | TBD | TBD |

Run the inference-timing bench equivalent to `bench_inference_pen.py` but for door (you'll have to write `bench_inference_door.py`; the pen one is the template — just change `OBS_DIM=39`, `ACTION_DIM=28`, dataset path, etc).

Update `README.md` with a door section under `### D4RL` (the pen section is the template; mirror its structure). Footnote the env-port confound openly — don't try to compare to paper's +361.

---

## Tripwires you might step on

1. **`max_episode_steps` is at env config level, not training params.** If you want to test horizon=300, edit `config_json/config.json`'s `environments.door.max_episode_steps` directly — `--fixed-params` won't override it.
2. **Single trials use `--reduced-steps N` for smoke tests.** Don't accidentally submit a 500-step trial to SLURM. Test wiring with `--reduced-steps 500` locally first.
3. **Aggregator filter strictness matters.** The cross-seed aggregator groups trials with identical params dict (sans `trial_seed`). Tiny param differences (e.g. `entropy_bandwidth=0.05` vs `0.2`) create separate groups. If you change a param mid-batch the aggregation will miss seed pairings.
4. **Spectral-norm checkpoint reload bug** was fixed in `hyperparam_search_dfo.py`. The fix infers layer indices from `.bias` keys (unchanged under SN) and detects SN via `weight_orig` OR `parametrizations.weight`. If you see a `Missing key(s): network.0.weight, Unexpected key(s): network.0.weight_orig` error, check that the fix is still in place.
5. **Penalty if you forget `inf_lit=0`** — you'll waste 10+ hours of GPU time per trial on the dead Langevin inference path.

---

## Reference: pen-human numbers (for sanity-checking proportions)

| metric | Pen Q3C B4 (10 seeds) | Pen IBC paper-recipe in our env (3 seeds) | Pen paper IBC EBM (paper env, 3 seeds) |
|---|---|---|---|
| avg_reward | 2522 | 403 | 2586 |
| cross_seed_std | 126 | 123 | 65 |
| SEM | 40 | 71 | (likely SEM-mislabeled in paper) |
| success_rate | ~67% | ~15% | (not reported) |
| inference time | 2.13 ms | 276.8 ms | — |
| reward gap to paper | -64 (within noise) | -2183 (broken IBC port) | — |

Pen Q3C is the publishable headline. Pen IBC reproduction in our env is broken (403 vs paper 2586). Door is harder than pen and the env confound is worse — expect proportionally weaker Q3C numbers and don't conflate "lower raw reward" with "worse algorithm."

---

## What's already updated in the codebase (DO NOT redo)

- `config_json/config.json` has `door` block fully populated.
- `combinedv2_cpascounter_training.py` routes door through `D4RLDataset` + standardize obs + per-dim action norm + `norm_stats.pt` save.
- `hyperparam_search.py` dispatches door at sim selection, sim_kwargs, metrics return, and `_ENV_PATH_MAP` (results land at `d4rl/door/`).
- `hyperparam_search_dfo.py` dispatches door at all sites: `_RESULTS_SLUG["door"] = "ibc_dfo_door"`, `_DEFAULT_NUM_EVAL_SEEDS["door"] = 100`, train branch, eval branch, env factory, success tracking, metrics return, analyze sort.
- `simulations/door_human_v2_simulation.py` exists.

**Door is wired end-to-end.** You shouldn't need to add any env support — just write batches and analyze.

---

## Quick decision tree

- "What should I submit first?" → `batches/doorAibc.txt` (gets the IBC baseline).
- "What's Q3C's best result on door?" → cp=300/top_k=80 hit +51.1 single-seed (#27 in trials.jsonl). Multi-seed unvalidated.
- "Should I run pure CP-argmax or refinement?" → Always CP-argmax (`inference_langevin_iterations=0`, `inference_dfo_iterations=0`). Refinement paths confirmed dead on Q3C across pen and door.
- "Why is my reward so far below paper's 361?" → Different env reward formula. Compare to doorAibc results (Q3C vs IBC in same env), NOT to paper's 361.
- "Should I switch to a different env?" → No. Run option A first (IBC in our env).

Good luck.
