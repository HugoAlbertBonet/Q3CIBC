# Q3CIBC

**Q3C for Imitation with Behavioral Cloning** – an offline imitation learning method that combines control-point generation with wire-fitting Q-value approximation and Implicit Behavioral Cloning (IBC).

---

## Overview

Q3CIBC learns a policy from offline expert demonstrations by:

1. **Control Point Generator** – produces multiple wire-fitting control points per state, bounded via `tanh` to the action space.
2. **Q-Estimator** – estimates the Q-values of each candidate action.
3. **Wire-Fitting Normalization** – smoothly interpolates Q-values using inverse-distance weighting, making the learned Q-function structurally maximizable.
4. **Implicit Behavioral Cloning** – instead of explicit models trained on MSE, we train Q3C as an implicit model, maximizing the probability of the expert's action.

---

## Environments

Configuration is driven by `config_json/config.json`. Set `"active_env"` to switch between environments. Trials run via `hyperparam_search.py` can override this with `--active-env <name>`, which is the recommended way to dispatch jobs (avoids races on the on-disk config).

The four environments form a difficulty curriculum, from a unit-test playground to a faithful re-implementation of the IBC paper's flagship robotics task:

| Env | Difficulty | Physics | State | Action | Dataset source |
|---|---|---|---|---|---|
| `dummy` | trivial | 2D kinematic | 4D | 1D angle | synthetic on-the-fly |
| `particle` | easy–hard (n-D) | 2D PD-controlled point mass | 4N D | N D position setpoint | TFRecords from Google IBC |
| `pen` | hard | MuJoCo, 24-DoF hand | 45D | 24D joint torque | D4RL human demos |
| `pushing` | hard | PyBullet xArm + block | 10D | 2D effector delta | TFRecords from Google IBC |

---

### `dummy` — 2D Grid Navigation (diagnostic)

The simplest env in the suite. Used to verify the full training/eval pipeline end-to-end and to visualize the model's decision surface.

**Task.** Drive a point agent from a random position to a random goal on a `[-1, 1]²` grid. The expert oracle uses `atan2` to compute the optimal angle and adds a small Gaussian noise for trajectory diversity.

| | |
|---|---|
| **Gym ID** | `Dummy-v0` |
| **State** | `[goal_x, goal_y, agent_x, agent_y]` (4D) |
| **Action** | scalar ∈ `[-1, 1]`, mapped to heading `θ = a · π` |
| **Dynamics** | each step, agent moves `step_size` in direction `θ`, clamped to grid |
| **Termination** | `distance(agent, goal) < goal_radius` (default 0.1) |
| **Reward** | `−distance(agent, goal)` (dense) |
| **Dataset** | `DummyDataset` — synthetic expert episodes generated at training time |

**Why it exists.** Because the action space is 1D and the optimal policy is closed-form, the dummy env is ideal for inspecting `ControlPointGenerator` outputs, `QEstimator` levels, and Langevin chain behavior. During evaluation, five diagnostic plots are saved at configurable mid-episode timesteps (`snapshot_steps`):

1. **CPs & Expert** — radial plot with control points colored by Q-value, the expert action on the same scale.
2. **Q-value Heatmap** — polar sweep of `Q(s, ·)` across all 360°.
3. **CP Probabilities** — softmax(Q) per CP; the chosen action is marked with ▲.
4. **Langevin Evolution** — uniform initialization → MCMC refinement → final samples.
5. **2D Map** — agent trajectory, current position, and goal.

---

### `particle` — n-Dimensional Goal-Reaching Particle

A reimplementation of the [Google IBC particle environment](https://github.com/google-research/ibc/blob/master/environments/particle/particle.py). The agent must visit two ordered goals (green → blue) with a PD-controlled point mass; the second goal is reached only after the first.

| | |
|---|---|
| **Gym ID** | `Particle-v0` |
| **State** | `[pos_agent, vel_agent, pos_first_goal, pos_second_goal]` — `4·n_dim` (8 in 2D, 64 in 16D) |
| **Action** | position setpoint in `[0, 1]ⁿ` |
| **Dynamics** | PD controller: `u = k_p · (action − pos) − k_v · vel`; Euler integration at 200 Hz (`dt=0.005`, `repeat_actions=10`) — control runs at 20 Hz |
| **Episode length** | 50 steps |
| **Termination** | always after `n_steps`; success measured at end |
| **Reward** | sparse: 1.0 if agent hit both goals and is at the second goal at the final step, else 0 |
| **Dataset** | `ParticleDataset` reads TFRecords from `datasets/particle/{n_dim}d_oracle_particle_*.tfrecord` (downloaded from [the IBC paper's data dump](https://storage.googleapis.com/brain-reach-public/ibc_data/particle.zip)) |

**Why it scales.** `n_dim` is configurable (1, 2, 3, 6, 8, 16, 32). The state grows linearly, but the policy difficulty grows much faster because the agent must commit to the right goal-visiting order across more dimensions. This makes particle the main hyperparameter-search testbed; trials in `results/hyperparam_search/.../particle/{n_dim}/trials.jsonl` are partitioned by `n_dim` so different difficulties don't mix.

---

### `pen` — D4RL Adroit Pen Manipulation

A 24-DoF dexterous robotic hand simulated in MuJoCo. The agent must rotate a pen held in the hand to match a target orientation.

| | |
|---|---|
| **Gym ID** | `AdroitHandPen-v1` (Gymnasium Robotics) |
| **State** | 45D (joint positions, joint velocities, pen pose, target pose) |
| **Action** | 24D continuous joint torques in `[-1, 1]²⁴` |
| **Episode length** | up to 200 steps |
| **Termination** | task-dependent; episode timeouts in the dataset |
| **Reward** | dense shaped reward (matching the D4RL spec) |
| **Dataset** | `D4RL/pen/human-v2` via Minari (human demonstrations) |

**Why it's the hardest.** Pen is a high-dimensional, contact-rich manipulation task with multi-modal expert behavior (humans rotate pens in idiosyncratic ways). It's our stress test for the implicit-BC objective against real human demos. Loading via Minari triggers an automatic download on first use.

---

### `pushing` — IBC Simulated Pushing (single target)

A near-faithful reimplementation of the IBC paper's "Simulated Pushing, single target, states" task (Florence et al., 2021, §5 / Table 3). A 6-DoF xArm with a cylindrical end-effector pushes a flat block to a target zone on a tabletop, simulated in PyBullet.

| | |
|---|---|
| **Gym ID** | `Pushing-v0` (Gymnasium wrapper around the vendored IBC `BlockPush(task=PUSH)` env in `simulations/ibc_block_pushing/`) |
| **State** | 10D: `[block_translation (2), block_orientation (1), effector_translation (2), effector_target_translation (2), target_translation (2), target_orientation (1)]` — IBC paper schema verbatim |
| **Action** | 2D effector position delta. Native range `≈ [−0.025, +0.043]` (mean ± 3σ from oracle); the model trains in normalized `[−1, 1]` and `PushingSimulation` denormalizes per-step before `env.step`. |
| **Episode length** | 100 steps (`max_episode_steps=100`, matches IBC's `BlockPush-v0` registration) |
| **Termination** | episode ends as soon as `‖block − target‖ < goal_dist_tolerance` (success) or at the step limit (failure) |
| **Reward** | `best_fraction_reduced_goal_distance` per step, monotonically growing; clamped to `1.0` and `done=True` on success |
| **Success metric** | `env.succeeded` at the last step (matches IBC's `AverageSuccessMetric`) |
| **Goal tolerance** | `0.02` (paper's `goal_tolerance` in `pushing_states/mlp_ebm_langevin.gin`; tighter than the env's built-in 0.01 default — we override) |
| **Dataset** | `PushingDataset` reads `datasets/block_push/block_push_states_location/oracle_push_*.tfrecord`, the [official IBC paper TFRecords](https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_states_location.zip) (1,400 episodes, 75k transitions after dropping `step_type=LAST` rows) |

**Faithful normalization.** To match `get_normalizers.py` + `stats.compute_dataset_statistics` in google-research/ibc, the pipeline standardizes observations and rescales actions:

- **Observations** are standardized per-dim: `(x − μ_data) / σ_data` (no clipping). Stats come from the full dataset on first load. The `ObservationNormalizer` automatically switches to this `standardize` mode when `obs_mean`/`obs_std` are provided.
- **Actions** are linearly mapped to `[−1, 1]` using per-dim `(act_min, act_max)` from the data (IBC's `compute_dataset_statistics.min_max_actions=True`). The training loop, InfoNCE negatives, Langevin chains, and the `ControlPointGenerator` output all live in `[−1, 1]`. Denormalization happens once per env.step.
- Both sets of stats persist in `<checkpoint_dir>/norm_stats.pt` and are reloaded at eval time by `PushingSimulation` so train-time and eval-time normalization are bit-identical.

**Network architecture mapping.** IBC's flagship `mlp_ebm_langevin.gin` uses `MLPEBM.layers='ResNetPreActivation'`, `width=128`, `depth=16`. IBC's `depth` counts **Linear layers in the body** (each block has 2 Linears), so 16 = 8 blocks. Q3CIBC's `q_depth` counts **blocks**, so the paper's setting is `q_network_kind=resnet, q_width=128, q_depth=8`.

**Setup.** Pushing requires extra dependencies (PyBullet + legacy `gym` for the vendored env) that the rest of the project doesn't need. They live in the `pushing` PEP 621 optional-extra:

```bash
uv sync --extra pushing                    # local dev
uv run --extra pushing python ...          # one-off command
```

SLURM batches submitting pushing trials should prefix their command with `uv run --extra pushing` (see `batches/pushingA.txt`).

**Vendored code.** The env source lives in `simulations/ibc_block_pushing/`, copied from [google-research/ibc](https://github.com/google-research/ibc/tree/master/environments/block_pushing) (Apache 2.0). Two small adaptations vs. upstream:

- `gin` is stubbed to no-ops (we set hyperparameters from Python, not `.gin` files).
- URDF asset paths are rewritten by `utils/utils_pybullet.py` to point at the vendored `assets/` directory; xArm URDFs come from PyBullet's bundled data.

---

---

## Project Structure

```
Q3CIBC/
├── combined_training.py            # Gen trained with MSE+separation; estimator InfoNCE; interpolates Q-values of expert action and counter samples from CPs
├── combinedv2_cpascounter_training.py  # Current main training script; CPs as counter examples + optional IBC Langevin / noisy-expert negatives + gradient penalty
├── mse_training.py                 # Gen MSE+separation; estimator InfoNCE; CPs as counter samples
├── uniform_training.py             # Gen InfoNCE+separation; estimator InfoNCE; counter samples + expert action interpolated
├── joint_training.py               # Gen InfoNCE+MSE+separation; estimator InfoNCE; CPs as negative samples
├── direct_training.py              # Gen MSE+separation; estimator InfoNCE; estimator decoupled from generator
├── hyperparam_search.py            # Trial runner: CLI for single/auto/analyze modes, --active-env override, per-trial config + checkpoint isolation
├── submit_experiments.sh           # Reads a batch file (one shell command per line) and submits each as a SLURM job
├── batches/                        # Hand-authored batch files for SLURM submission (V/T/U/pushingA)
├── config_json/
│   ├── config.json                 # All environment & training configuration
│   └── observation_bounds.json     # Observation min-max bounds for the legacy minmax normalizer
├── utils/
│   ├── models.py                   # ControlPointGenerator & QEstimator (MLP + ResNetPreActivation)
│   ├── loss.py                     # InfoNCE, MSE, separation losses
│   ├── normalizations.py           # ObservationNormalizer (minmax or standardize), wire-fitting Q-value norm
│   ├── datasets.py                 # D4RL, Particle, Dummy, Pushing dataset loaders
│   ├── sampling.py                 # Uniform & Langevin MCMC sampling
│   └── vis_dummy.py                # Dummy environment diagnostic plots
├── simulations/
│   ├── base_simulation.py          # Abstract simulation; declares no-op _denormalize_action default
│   ├── pen_human_v2_simulation.py  # Pen evaluation
│   ├── dummy_env.py                # 2D grid Gym env
│   ├── dummy_simulation.py         # Dummy evaluation with snapshot capture
│   ├── particle_env.py             # Particle Gym env
│   ├── particle_simulation.py      # Particle evaluation
│   ├── pushing_env.py              # Gymnasium wrapper around the vendored IBC BlockPush env
│   ├── pushing_simulation.py       # Pushing evaluation; loads norm_stats.pt and denormalizes actions
│   ├── ibc_block_pushing/          # Vendored Google IBC env (block_pushing.py, xarm_sim_robot.py, URDF/OBJ assets)
│   ├── run_simulation.py           # Multi-seed evaluation entry point
│   └── plots.py                    # Plots for evaluation
├── datasets/                       # External data (see Installation: particle.zip, block_push_states_location.zip)
├── tests/                          # Test suite
├── plots/                          # Generated evaluation plots
├── checkpoints/                    # Saved model weights (per-trial under checkpoints/hpsearch/run_<id>/)
├── results/hyperparam_search/      # JSONL trial logs partitioned by script/env/n_dim
├── pyproject.toml                  # Project metadata & dependencies (pushing extras = pybullet + gym)
└── README.md
```

---

## Installation

Requires **Python ≥ 3.12**.

```bash
git clone https://github.com/HugoAlbertBonet/Q3CIBC.git
cd Q3CIBC
uv sync                       # core deps: torch, tensorflow, minari, gymnasium-robotics
uv sync --extra pushing       # only if you'll run the pushing env (adds pybullet + legacy gym)
```

The pushing extras are gated behind a PEP 621 optional-dependency group because they require a PyBullet source build (no prebuilt cp312 wheel for `pybullet==3.1.6`), which fails on stricter SLURM toolchains. Particle / dummy / pen trials install cleanly without them.

### Datasets

| Env | Source | Destination | Size |
|---|---|---|---|
| `particle` | [particle.zip](https://storage.googleapis.com/brain-reach-public/ibc_data/particle.zip) (Google IBC) | `datasets/particle/` | ~80 MB |
| `pushing` | [block_push_states_location.zip](https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_states_location.zip) (Google IBC) | `datasets/block_push/` | ~5 MB |
| `pen` | downloaded automatically by Minari on first use | `~/.minari/datasets/` | — |
| `dummy` | generated synthetically at training time | — | — |

```bash
mkdir -p datasets/block_push && cd datasets/block_push
wget https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_states_location.zip
unzip block_push_states_location.zip && rm block_push_states_location.zip && cd ../..
```

---

## Usage

### Training (direct)

Training scripts read from `config_json/config.json`. Set `"active_env"` to choose the environment.

```bash
# Current main script (recommended)
uv run python combinedv2_cpascounter_training.py

# Older variants (kept for reproducing prior experiments)
uv run python combined_training.py     # MSE + InfoNCE, interpolated negatives
uv run python mse_training.py          # MSE-only generator
uv run python uniform_training.py      # Uniform counter examples
```

Pushing trials need the optional extra:

```bash
uv run --extra pushing python combinedv2_cpascounter_training.py
```

Models are saved to `checkpoints/` after training. For pushing, the corresponding `norm_stats.pt` (per-dim mean/std and action min/max) is saved next to the weights and is required at eval time.

### Hyperparameter search & batch trials

`hyperparam_search.py` wraps any training script with per-trial config isolation, atomic JSONL trial logging, and three modes:

```bash
# Single trial with explicit params; pinned env, won't touch config.json
uv run python hyperparam_search.py combinedv2_cpascounter_training.py \
    --run --active-env pushing \
    --fixed-params '{"trial_seed":0,"control_points":20,"q_network_kind":"resnet","q_width":128,"q_depth":8}'

# Auto-exploration: 5 adaptive trials
uv run python hyperparam_search.py combinedv2_cpascounter_training.py \
    --auto --max-trials 5 --reduced-steps 20000 --active-env particle

# Inspect past trials (env-partitioned table)
uv run python hyperparam_search.py combinedv2_cpascounter_training.py \
    --analyze --active-env pushing
```

Key flags:
- `--active-env <name>` — pin env for this invocation; overrides `config.json`'s `active_env` for training, evaluation, AND log placement (results land in `results/hyperparam_search/<script>/<env>/`). Recommended for all SLURM batches.
- `--fixed-params '{...}'` — locked hyperparameter dict; the rest come from config baseline.
- `--reduced-steps N` — short training for quick sanity checks (e.g. 500 steps).
- `--num-reps N` — replicate the same config with `trial_seed=0..N-1` to measure variance.

### SLURM batches

Batches live in `batches/<name>.txt` — one shell command per line. Submit a batch via:

```bash
./submit_experiments.sh batches/pushingA.txt
```

Each line becomes one SLURM job with a 25h GPU budget (template in `submit_experiments.sh`). Pushing batch lines prefix `uv run` with `--extra pushing` so SLURM compute nodes install PyBullet on first use; particle/pen/dummy batches don't need it.

### Evaluation (standalone, outside hyperparam_search)

```bash
uv run python -m simulations.run_simulation
```

Options:
- `--seeds 0 1 2` – Random seeds (default: 0, 35, 42)
- `--episodes N` – Episodes per seed (default: 1)
- `--checkpoint PATH` – Model checkpoint path
- `--output-dir DIR` – Where to save plots

---

### Configuration

All parameters are in `config_json/config.json`. Each environment has its own block under `environments.<name>`; `active_env` selects which one to use. `hyperparam_search.py --active-env <name>` overrides this for the duration of a trial without mutating the file.

| Parameter                | Scope              | Description                                                |
|--------------------------|--------------------|------------------------------------------------------------|
| `active_env`             | top-level          | Environment to use (`pen`, `dummy`, `particle`, `pushing`) |
| `state_dim`              | per-env            | Observation dimensionality                                 |
| `action_dim`             | per-env            | Action dimensionality                                      |
| `frame_stack`            | per-env            | Number of consecutive observations to stack                |
| `action_bounds`          | per-env            | Min/max for the `ControlPointGenerator` tanh range         |
| `max_episode_steps`      | per-env (optional) | Per-env override; falls back to `simulation.max_episode_steps` |
| `goal_dist_tolerance`    | per-env (pushing)  | Success threshold (m) for block-to-target distance         |
| `n_dim`                  | per-env (particle) | Workspace dimensionality (also partitions trial folders)   |
| `step_size`              | per-env (dummy)    | Agent movement per step                                    |
| `goal_radius`            | per-env (dummy)    | Success threshold distance                                 |
| `control_points`         | per-env model      | CP cloud size                                              |
| `top_k_control_points`   | per-env training   | Number of top CPs used as InfoNCE negatives                |
| `q_network_kind`         | per-env model      | `mlp` or `resnet` (ResNetPreActivation)                    |
| `q_width` / `q_depth`    | per-env model      | Q-net width / number of blocks (resnet) or layers (mlp)    |
| `snapshot_steps`         | per-env (dummy)    | Timesteps at which to save diagnostic plots                |

---

## Key Components

### `ControlPointGenerator`
Fully-connected network mapping states to `N` candidate action vectors, bounded via `tanh` to `[action_min, action_max]`.

### `QEstimator`
Fully-connected network mapping (state, action) pairs to scalar Q-values.

### `wireFittingNorm`
Vectorized wire-fitting normalization:

$$
Q(s,a) = \frac{\sum_i w_i(s,a)\,\hat{Q}_i(s)}{\sum_i w_i(s,a)}, \quad w_i(s,a) = \frac{1}{|a - \hat{a}_i(s)|^2 + c\,(\hat{Q}_{\max} - \hat{Q}_i(s))}
$$

### Losses
- **InfoNCE** – contrastive loss treating expert action as positive.
- **MSE** – distance from nearest control point to expert.
- **Separation** – encourages control-point diversity via inverse pairwise distances.

---

## Results

| Environment        | Metric        | Q3CIBC                                                                | IBC paper           |
|--------------------|---------------|-----------------------------------------------------------------------|---------------------|
| Dummy              | Extra steps   | 0                                                                     | 0.33                |
| Particle (2D)      | Success rate  | 100%                                                                  | 100%                |
| Particle (16D)     | Success rate  | best 66%, mean ~40% across seeds (`particle/16/trials.jsonl`)         | 99% (paper, Fig. 6) |
| Pushing (1 target) | Success rate  | running batch `pushingA` (24% in a 500-step smoke after gap-closure)  | 87% / 73% (Table 3) |

## Dependencies

- `torch >= 2.9`
- `minari[all] >= 0.5`
- `gymnasium-robotics >= 1.4`
- `numpy >= 2.4`
- `matplotlib >= 3.10`
