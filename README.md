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

Configuration is driven by `config_json/config.json`. Set `"active_env"` to switch between environments.

### Pen (AdroitHandPen-v1)
Dexterous hand manipulation task from D4RL. The agent must rotate a pen to match a target orientation.
- **State**: 45D (joint positions, velocities, pen pose)
- **Action**: 24D continuous (joint torques)
- **Dataset**: `D4RL/pen/human-v2` via Minari

### Dummy (2D Grid Navigation)
A diagnostic environment for verifying the training pipeline. An agent navigates a `[-1, 1]²` grid towards a randomly placed goal.
- **State**: `[goal_x, goal_y, agent_x, agent_y]` (4D, extensible via `frame_stack`)
- **Action**: scalar ∈ `[-1, 1]`, mapped to angle `θ = a × π`
- **Dynamics**: agent moves `step_size` per step, clamped to grid
- **Reward**: `-distance(agent, goal)`
- **Termination**: distance < `goal_radius`
- **Dataset**: synthetically generated expert trajectories using `atan2`

#### Dummy Diagnostic Plots
During simulation, 5 plots are saved at configurable mid-episode timesteps (`snapshot_steps`):
1. **CPs & Expert** – radial plot with CPs colored by Q-value, expert on same scale
2. **Q-value Heatmap** – polar sweep of Q across all 360°
3. **CP Probabilities** – softmax(Q) per CP, selected action marked as ▲
4. **Langevin Evolution** – uniform init → MCMC refinement → final samples
5. **2D Map** – agent trajectory, current position, and goal

### Particle
Particle-based environment for multi-dimensional control.
- **State/Action**: configurable via `n_dim`

---

## Project Structure

```
Q3CIBC/
├── combined_training.py    # Combined generator + estimator training
├── mse_training.py         # MSE-only training
├── uniform_training.py     # Uniform counter-example training
├── joint_training.py       # Joint training variant
├── config_json/
│   ├── config.json             # All environment & training configuration
│   └── observation_bounds.json # Observation normalization bounds
├── utils/
│   ├── models.py           # ControlPointGenerator & QEstimator
│   ├── loss.py             # InfoNCE, MSE, and separation losses
│   ├── normalizations.py   # Wire-fitting Q-value normalization
│   ├── datasets.py         # Dataset loaders (D4RL, Particle, Dummy)
│   ├── sampling.py         # Uniform & Langevin MCMC sampling
│   └── vis_dummy.py        # Dummy environment diagnostic plots
├── simulations/
│   ├── base_simulation.py          # Base simulation class
│   ├── pen_human_v2_simulation.py  # Pen environment simulation
│   ├── dummy_env.py                # 2D grid navigation Gymnasium env
│   ├── dummy_simulation.py         # Dummy evaluation with snapshot capture
│   ├── run_simulation.py           # Multi-seed evaluation entry point
│   └── plots.py                    # Plots for evaluation
├── tests/                  # Test suite
├── plots/                  # Generated evaluation plots
├── checkpoints/            # Saved model weights
├── pyproject.toml          # Project metadata & dependencies
└── README.md
```

---

## Installation

Requires **Python ≥ 3.12**.

```bash
git clone https://github.com/HugoAlbertBonet/Q3CIBC.git
cd Q3CIBC
uv sync          # or: pip install -e .
```

---

## Usage

### Training

All training scripts read from `config_json/config.json`. Set `"active_env"` to choose the environment.

```bash
# Combined training (generator MSE + estimator InfoNCE)
uv run combined_training.py

# MSE-only generator training
uv run mse_training.py

# Uniform counter-example training
uv run uniform_training.py
```

Models are saved to `checkpoints/` after training.

### Evaluation

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

All parameters are in `config_json/config.json`:

| Parameter          | Description                                    |
|--------------------|------------------------------------------------|
| `active_env`       | Environment to use (`pen`, `dummy`, `particle`) |
| `state_dim`        | Observation dimensionality                     |
| `action_dim`       | Action dimensionality                          |
| `frame_stack`      | Number of consecutive observations to stack    |
| `step_size`        | (dummy) Agent movement per step                |
| `goal_radius`      | (dummy) Success threshold distance             |
| `control_points`   | Candidate actions per state                    |
| `counter_examples` | Negative samples for InfoNCE                   |
| `snapshot_steps`   | (dummy) Timesteps at which to save diagnostic plots |

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

## Dependencies

- `torch >= 2.9`
- `minari[all] >= 0.5`
- `gymnasium-robotics >= 1.4`
- `numpy >= 2.4`
- `matplotlib >= 3.10`
