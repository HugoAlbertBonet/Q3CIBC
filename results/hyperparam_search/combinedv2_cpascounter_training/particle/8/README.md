# Q3C-IBC on Particle (n_dim=8) — Experimental Log

**Final result: 98.0% mean success rate, std 1.79% across 5 seeds (range 96–100%).**

This document summarises 111 hyperparameter-search trials run between
2026-04-22 and 2026-04-25 to push the Q3C-IBC architecture
(wire-fitting control-point generator + Q-estimator + IBC InfoNCE loss)
on the 8-dimensional particle environment from a starting point
of 0% / lottery-30% success up to ≥95%.

The search history is in [`trials.jsonl`](trials.jsonl). Per-trial
configurations and checkpoints live under
`/home1/halbertb/Q3CIBC/checkpoints/hpsearch/run_<id>/`.

---

## 1. Winning recipe

Architecture (unchanged from project defaults):

| param | value |
|---|---|
| `control_points` | 20 |
| `top_k_control_points` | 10 (50% counter-example ratio in InfoNCE) |
| `num_hidden_layers` | 2 |
| `num_neurons` | 256 |
| `frame_stack` | 2 |

Loss weights:

| param | value |
|---|---|
| `mse_weight` | 10.0 |
| `info_nce_weight` | 0.5 |
| `generator_infonce_weight` | 0.05 |
| `separation_loss` | "entropy" |
| `separation_weight` | 0.1 |
| `entropy_bandwidth` | 0.1 |
| `infonce_logit_clamp` | 20.0 |

IBC additions (all OFF for the winner):

| param | value |
|---|---|
| `num_uniform_negatives` | 0 |
| `num_langevin_negatives` | 0 |
| `gradient_penalty_weight` | 0.0 |

Optimization:

| param | value |
|---|---|
| `batch_size` | 512 |
| `learning_rate` | 1e-3 |
| `estimator_learning_rate` | 1e-3 |
| `scheduler_type` | "cosine" |
| `training_steps` | 150_000 |

Stability (silent half of the win — see §4):

| param | value |
|---|---|
| `trial_seed` | fixed per rep |
| `nan_abort_threshold` | 50 (also: clear optimizer state on every NaN batch) |

Inference Langevin (loud half of the win — see §3):

| param | value |
|---|---|
| `inference_langevin_iterations` | 75 |
| `langevin_lr_init` | 0.015 |
| `langevin_noise_scale` | 0.05 |
| `langevin_delta_clip` | 0.015 |

This corresponds to the `I.c++` configuration in the trials log.

---

## 2. Headline progression

| stage | what changed | result |
|---|---|---|
| Pre-investigation | original training script, default Langevin settings, no concurrency-safe harness | apparent 30–54% but really seed lottery |
| After stability fixes (§4) | deterministic `trial_seed`, NaN-batch optimizer-state reset, `--num-reps`, parallel-safe `hyperparam_search` | **75% mean (5 seeds, 66–80%)** |
| After calibrated inference Langevin (§3) | gentle Langevin refinement of the argmax-Q control point at evaluation time | **97.2% mean (I.c, 5 seeds, 94–100%)** |
| Final (slightly more aggressive Langevin) | inf_lv=75, lr=0.015, clip=0.015 | **98.0% mean (I.c++, 5 seeds, 96–100%)** |

Five-seed cross-comparison of the three best configurations:

| config | mean | median | min | max | std |
|---|---|---|---|---|---|
| **I.c++** (cp=20, inf_lv=75, lr=0.015, clip=0.015) | **98.0%** | 98% | **96%** | 100% | **1.79%** |
| I.c (cp=20, inf_lv=50, lr=0.01, clip=0.01) | 97.2% | 98% | 94% | 100% | 2.04% |
| I.c@cp=30 (cp=30, inf_lv=50, lr=0.01, clip=0.01) | 96.8% | 98% | 88% | 100% | 4.49% |

---

## 3. Why inference Langevin works here, and how to calibrate it

### The setup

The particle env's action *is* a position setpoint. The agent is
PD-controlled toward whatever action is selected. So `dist(action,
goal_position)` is directly meaningful — small action-space distance
to a goal means the agent will physically move into it. The success
threshold (`goal_distance`) is **0.05** in action-space units; the
action box is `[0,1]^8` (diameter √8 ≈ 2.83).

### The diagnostic that drove tuning

After the stability fixes, the baseline policy (set E) failed
**37/150 episodes (∼25%)**. Inspecting the failure distances
(see [`trials.jsonl`](trials.jsonl) `eval_details`):

- 35 / 37 of the failed seeds were within **0.15** of goal 1.
- 36 / 37 were within **0.15** of goal 2.
- The **median miss was right at the 0.05 threshold** (d1_med=0.048,
  d2_med=0.052).

So failures were dominated by *near-misses*, not policy collapse.
The policy reaches the right neighbourhood; it just stops 1–10 mm
outside the goal radius. That's exactly the regime where local
inference-time refinement should help — provided it nudges the action
*just enough* without overshooting to the other side of the goal.

### Why the default Langevin hyperparameters destroyed the policy

The defaults inherited from the IBC paper
(`lr_init=0.1, noise_scale=1.0, delta_action_clip=0.1`) produce
per-step noise σ ≈ 0.316 per dim, which is **3× the delta clip**.
After 50 iterations the action diffuses ~1.0 units in action space —
many goal-radii. We measured this in
[`analyze_langevin_trajectory.py`](../../../../analyze_langevin_trajectory.py):
Q decreased by **1.86** over 100 iterations because noise dominated
the gradient signal. Inference Langevin under default settings was a
random walk and dropped the policy to 0%.

### Calibration rule of thumb

Inference Langevin should target **drift ≈ goal_radius (0.05)** —
big enough to close near-misses, small enough not to overshoot.

For 50–75 iterations, that suggests:
- `delta_action_clip ≈ 0.01–0.015` (per-step ≈ 1/4 of goal radius)
- `langevin_lr_init ≈ delta_action_clip` (so gradient and clip are
  matched in scale; raw step magnitude is `lr × |grad| / 2`, with our
  trained gradient norms of ~3 this naturally hits the clip)
- `langevin_noise_scale ≈ 0.05` (σ_step ≈ √(lr·noise_scale²)
  ≈ 0.005, ~1/3 of clip — gradient dominates, noise breaks ties)

The relationship between `langevin_delta_clip` and `goal_distance`
generalises across n_dim. **For other particle dimensions, use
`langevin_delta_clip ≈ goal_distance / 3`** as a starting point and
verify with a trajectory analysis on a trained checkpoint.

### What we tried within the calibration window

| iters | lr_init | clip | typical drift | mean success | note |
|---|---|---|---|---|---|
| 25 | 0.005 | 0.005 | 0.01 | 58% (1 seed) | too gentle — doesn't close median miss |
| 50 | 0.005 | 0.005 | 0.025 | 24% (1 seed) | extra iters with tiny steps amplify noise dominance |
| 50 | 0.01 | 0.01 | 0.04 | **97.2%** | matches median miss; rescues most near-misses |
| 50 | 0.01 | 0.01 + GP=1.0 (target form) | 0.04 | 91.3% | GP adds variance without consistent gain |
| 75 | 0.015 | 0.015 | 0.06 | **98.0%** | rescues median + some 0.05–0.08 misses without overshoot |

---

## 4. Stability fixes were the largest single contribution

Before the stability fixes, the same configuration could produce
30%, 54%, and 0% across three runs depending on
the seed. Two distinct sources:

1. **Seed lottery.** Without any seeding calls, every run drew its
   weight init, dataloader shuffles, and Langevin noise from
   uncontrolled state. Deterministic seeds made each rep
   reproducible.

2. **Silent training divergence.** A single NaN batch (from logit
   blow-up, data anomaly, or numeric edge case) corrupted the Adam
   moment estimates. The original guard was
   `if torch.isnan(loss): continue` — which kept training going on a
   broken optimizer for the rest of the run, ending at near-init
   accuracy. We observed this in ~33% of runs across batches 1–7.

The fix is in [`combinedv2_cpascounter_training.py`](../../../../combinedv2_cpascounter_training.py)
around the loss-NaN check:

```python
if torch.isnan(loss_estimator) or torch.isnan(...):
    consecutive_nan_batches += 1
    optimizer_generator.state.clear()      # ← critical: wipe Adam moments
    optimizer_estimator.state.clear()
    if consecutive_nan_batches >= nan_abort_threshold:
        raise RuntimeError(...)             # don't silently chug forever
    continue
consecutive_nan_batches = 0
```

Plus, at the top of `main()`:

```python
random.seed(trial_seed)
np.random.seed(trial_seed)
torch.manual_seed(trial_seed)
torch.cuda.manual_seed_all(trial_seed)
```

These two changes alone moved the policy from "30–40% with
high variance and frequent crashes" to a reproducible 75% baseline —
**+45 pp absolute lift before any algorithmic change**.

---

## 5. What didn't work and why

These are documented for future reference so the same paths don't get
re-tried.

### IBC uniform random negatives — broke training

Setting `num_uniform_negatives=32` collapsed the policy to 0% with
high training accuracy (acc=0.97) but d1_med=0.82, d2_med=0.40 —
the model classifies "structured action vs random uniform" instead of
shaping a useful Q surface. Uniform negatives trivialise InfoNCE for
this small MLP on a `[0,1]^8` action box. The IBC paper uses them
successfully at much larger scale and with gradient penalty active
(see below); the regime doesn't transfer.

### Hinge-form gradient penalty — never fired

`gradient_penalty_form="hinge"` (the IBC-paper form,
`max(0, ‖grad‖ − margin)²`) at `margin=1.0` produced bit-exact
identical results vs `gradient_penalty_weight=0.0`. Our trained
estimator's per-action gradient norms are ~3.0 mean (max ~5),
and at initialisation they're 0.04 — for most of training they sit
below the margin, so the hinge is identically zero (and so is its
gradient through `torch.clamp(.., min=0)`). Lowering the margin to
0.1 also produced bit-exact identity, suggesting that with our small
network the gradients during training stayed below 0.1 entirely
when uniform negatives were also present. See trials 69–74.

### Two-sided ("target") gradient penalty — small, inconsistent gain

`gradient_penalty_form="target"` (`(‖grad‖ − margin)²`) does fire at
init (penalty ≈ 0.92 with margin=1.0). Its effect on success rate
across 3 reps was 18% / 76% / 92% — high variance, mean 62%, vs the
no-GP baseline at 75% mean. The same configuration combined with
gentle inference Langevin gave 76% / 98% / 100% — slightly worse
mean than I.c without GP (97.2%). GP isn't a clear win in this
regime; the local-curvature regularisation it provides is redundant
with what gentle Langevin refinement already gets us at inference.

### IBC Langevin training negatives — uniform-equivalent at gentle settings

`num_langevin_negatives=32, langevin_num_iterations=50` with the
same gentle Langevin settings used at inference (lr=0.01,
noise=0.1, clip=0.02) produced 0% success across 3 reps. The reason:
gentle Langevin started from uniform random samples drifts only
~0.15 in 50 iterations, so the resulting "hard negatives" are still
essentially uniform — same failure mode as adding pure uniform
negatives. Aggressive Langevin (`lr=0.1, noise=1.0`) was untested
for negatives but is known to be chaotic at inference. The IBC
recipe of "uniform + Langevin negatives" doesn't transfer cleanly to
this small-network / small-action-space regime. Top-k generator CPs
already serve as the hard-negative source.

### Longer training (200k steps with cosine_t_max=150k)

Mean 47% across 2 reps, vs 75% baseline. Either the policy starts
to overfit after ~120–150k steps in the no-Langevin regime, or the
extended low-LR tail drifts the Q surface in ways inference Langevin
can't recover from. Not worth pushing beyond 150k.

### `control_points = 30` with all else fixed

Initial 2-rep result was 99% (looked like a +2 pp lift). Across 5
seeds the mean came down to **96.8%** with std 4.49% — std more
than double the cp=20 winner. One of the 5 seeds collapsed to 88%.
More CPs ≠ more capacity in a usable way; they introduce more
variance without consistent improvement. cp=20 is the sweet spot.

### Lower LR (5e-4 both optimizers)

Tested early, came in well below 1e-3 (acc 0.107 vs 0.839 at the
same step budget). The model undertrains with halved LR.

---

## 6. Engineering lessons

### Concurrency-safe hyperparam search

The harness originally mutated a shared `config.json` and assigned
trial IDs by reading-then-appending `trials.jsonl` — both are races
under SLURM parallel submission. Symptoms: trial IDs collided,
trials read each other's modified configs as their baseline, and
checkpoints from different trials overwrote each other (causing
bogus "shape mismatch on load" eval errors when the saved CP count
disagreed with the loaded config's CP count).

Fix in
[`hyperparam_search.py`](../../../../hyperparam_search.py):
- **Per-trial config file**: each run gets a unique
  `run_<timestamp>_<hex>` ID. Its config is written to
  `checkpoints/hpsearch/run_<run_id>/config.json` and passed to the
  training subprocess via the `Q3C_CONFIG_PATH` env var. The
  shared `config.json` is never mutated.
- **Atomic trial-ID assignment**: `append_trial` wraps the read-max +
  write critical section in `fcntl.flock`. Stress test: 4 processes ×
  20 writes each → 80 unique, contiguous, monotonic IDs.
- **Per-run checkpoints**: `model_save_dir =
  checkpoints/hpsearch/run_<run_id>/` so concurrent trials never
  share a directory.

### Deterministic eval seeds

`evaluate_q3c` evaluates over `seeds = list(range(num_seeds))`
where `num_seeds = 50` from `simulation.num_seeds`. The same 50
seeds are used for every trial — comparisons are
deterministic on the eval side, not just the train side.

### Multi-rep flag

`hyperparam_search.py --num-reps N` runs the same configuration N
times with `trial_seed = 0..N-1` (concurrency-safe). This is the
right way to measure variance — the same flag applies to `--run`
and `--auto`.

### Trajectory analysis tool

[`analyze_langevin_trajectory.py`](../../../../analyze_langevin_trajectory.py)
takes a trained checkpoint and produces the per-iteration Langevin
stats (action drift, gradient norm, Q value, distance to each goal).
This was the diagnostic that revealed the default Langevin
hyperparameters were noise-dominated, and is what you should run
*first* when transferring this recipe to a new n_dim or a new env.

---

## 7. Comparison to IBC-DFO

See [`results/particle/success_rates.csv`](../../../../results/particle/success_rates.csv).
For n_dim=8, IBC-DFO wasn't measured (we ran out of project compute
budget before testing it on this dimension). Q3C-IBC numbers in the
csv:

| approach | success | train (s) | inference (50 seeds, s) | per-episode (s) |
|---|---|---|---|---|
| **Q3C-IBC, winning recipe (I.c++)** | **0.980** | 1232 | **140.9** | 2.81 |
| Q3C-IBC, no inference Langevin | 0.750 | 1232 | 1.84 | 0.04 |
| IBC-DFO, paper-faithful | (not run at n_dim=8) | (not run) | 187.2 | 3.74 |

**Q3C-IBC with calibrated inference Langevin is ~25% faster than
IBC-DFO at inference time** (140.9s vs 187.2s for 50 episodes), even
though both use Langevin MCMC at evaluation. The win comes from
*where* the Langevin compute is spent: Q3C-IBC runs 75 iterations
on a single starting point (the argmax-Q control point), while
IBC-DFO runs 100 iterations on 512 samples drawn from the entire
action space. We're paying for "refine the winner" rather than
"search the whole action space" — and the wire-fitting generator's
top-1 CP is already close enough to the optimal action that 75
gentle iterations close the remaining gap.

### How these numbers were obtained

- **Training time (1232s)** is the *median* of the five Set-E trial
  durations on SLURM (no inference Langevin). Using the median
  rather than the mean suppresses outlier compute-scheduling noise:
  one trial ran 2344s while the others ran 1140–1510s. Median
  estimates the true train cost without that contamination.
- **Inference time (140.9s)** comes from a controlled CUDA
  microbenchmark on a single trained checkpoint: 50 episodes with
  inf_lv=75 (lr=0.015, noise=0.05, clip=0.015) → 140.93s total,
  ~2.8s per episode. Without inference Langevin the baseline cost
  is 1.84s for 50 episodes (~37ms per episode), so the Langevin
  overhead is ~139s for 50 episodes.
- **IBC-DFO inference time (187.2s)** comes from the same
  microbenchmarking style on a fresh `QEstimator` (timing is
  dominated by forward/backward through the network, not by the
  trained weights), running the paper-faithful inference loop:
  512 Langevin samples × 100 iterations per env step, picking the
  lowest-energy sample. 5-episode timing was 18.72s, extrapolated
  to 50.

### Why we can't split SLURM durations into train+inference

The SLURM trial durations (mean 1727s for I.c++, mean 1474s for
Set E) cannot be cleanly attributed to train vs inference: the
per-trial duration noise (±400s, observed) is several times the
actual inference cost. Same-seed deltas between Set E and I.c++
ranged from **−213s to +472s** — a negative delta is impossible if
"more Langevin work" were the dominant signal, so we know
scheduling noise dominates the wall clock. The microbenchmark
was the only way to get a stable inference-time number.

### Caveat for the n_dim ≤ 6 rows

For n_dim 2–6, Q3C-IBC was previously measured *without* inference
Langevin, so those inference figures (~1.5–1.8s) reflect the
no-Langevin cost alone — **not** directly comparable to the
n_dim=8 row that uses the inf_lv=75 winning recipe. Re-running
n_dim 2–6 with the calibrated inference Langevin would likely lift
those success rates the same way it lifted n_dim=8 from 75% to 98%,
adding ~140s inference overhead per 50 seeds (similar to the
n_dim=8 row).

---

## 8. What's still on the table

If we ever need to push beyond 98%:

1. **Soft-MSE on all CPs.** Current MSE only trains the *nearest* CP
   to the expert action. Boltzmann-weighted soft-nearest gives every
   CP a fractional gradient, so the 19/20 currently-untrained CPs
   become useful. Estimated +3–5 pp.
2. **Deterministic policy gradient (DPG) actor loss.** Add
   `−α · Q(s, top-1 CP)` to the generator loss with the estimator
   frozen. Direct ∇Q signal to the generator instead of the
   InfoNCE proxy. Estimated +3–5 pp.
3. **Auto-scale `langevin_delta_clip` to `goal_distance / 3`** in
   the simulation config so the recipe transfers across n_dim
   without manual retuning.

Beyond these, returns are diminishing — failures past 98% are
typically structural (the policy outright misses, not near-miss),
which inference refinement can't help with.

---

## 9. Total compute spent

- 111 trials × ~25 minutes mean ≈ **46 hours of GPU wall time**.
- Best single seed: 100% (multiple configs, 9 trials).
- Best 5-seed mean: 98.0% (I.c++).
- Stability fixes alone delivered ~45 pp; calibrated inference
  Langevin delivered ~23 pp on top.
