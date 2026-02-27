"""IBC testing with Autoregressive Derivative-Free Optimizer (DFO) on Dummy environment.

Runs inference using the trained energy model (QEstimator) with iterative DFO:
  1. Draw N uniform random action samples
  2. Evaluate energy E(obs, a) for each sample
  3. Compute softmax probabilities from negated energies
  4. Resample N actions proportionally (categorical with replacement)
  5. Add shrinking Gaussian noise
  6. Repeat for K iterations
  7. Select action with lowest energy

Reports:
  - Per-seed: path plot, steps taken vs optimal steps, reward
  - Global: mean ± std reward across all seeds

Following the IBC paper (Florence et al., 2021, arXiv:2109.00137), Section B.2.

Usage:
    python -m ibc.ibc_dfo_dummy_testing
"""

import os
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.models import QEstimator
from simulations.dummy_env import DummyEnv

# ─── Load config ─────────────────────────────────────────────────────────────
config_path = Path(__file__).parent.parent / "config_json" / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

env_config = config["environments"]["dummy"]
env_model = env_config.get("model", {})
sim_config = config.get("simulation", {})

# Model architecture (must match training)
num_hidden_layers = env_model.get("num_hidden_layers", 4)
num_neurons = env_model.get("num_neurons", 128)
state_dim = env_config["state_dim"]
action_dim = env_config["action_dim"]
frame_stack = env_config.get("frame_stack", 1)

# Environment parameters
step_size = env_config.get("step_size", 0.1)
goal_radius = env_config.get("goal_radius", 0.1)
max_episode_steps = sim_config.get("max_episode_steps", 200)
action_bounds = env_config.get("action_bounds", [-1, 1])

# Seeds from config
seeds = sim_config.get("default_seeds", [0, 35, 42, 10, 135, 142])

# DFO hyperparameters
DFO_NUM_SAMPLES = 512       # Number of action samples per iteration
DFO_NUM_ITERATIONS = 3      # Number of DFO refinement iterations
DFO_INITIAL_STD = 0.33      # Initial noise std for perturbation
DFO_TEMPERATURE = 1.0       # Softmax temperature

# Checkpoint and output paths
CHECKPOINT_DIR = os.path.join(
    config.get("training_shared", {}).get("model_save_dir", "checkpoints"), "ibc_dfo"
)
PLOT_DIR = "plots/ibc_dfo"


def dfo_optimize(
    energy_model: torch.nn.Module,
    state: torch.Tensor,
    device: torch.device,
    num_samples: int = DFO_NUM_SAMPLES,
    num_iterations: int = DFO_NUM_ITERATIONS,
    initial_std: float = DFO_INITIAL_STD,
    temperature: float = DFO_TEMPERATURE,
) -> np.ndarray:
    """Derivative-Free Optimizer for finding the minimum-energy action.

    Implements iterative sample-rank-resample from the IBC paper (Algorithm B.2):
      For each iteration:
        1. Evaluate energy for all action samples
        2. Compute softmax probabilities (negated energies → higher prob for lower energy)
        3. Resample actions proportionally
        4. Add Gaussian noise (shrinking by 0.5× each iteration)
      Finally, pick the single action with the lowest energy.

    Args:
        energy_model: Trained QEstimator E(obs, action).
        state: Observation tensor of shape (1, state_dim).
        device: Torch device.
        num_samples: Number of action samples N.
        num_iterations: Number of DFO refinement iterations K.
        initial_std: Initial std of perturbation noise.
        temperature: Softmax temperature.

    Returns:
        Best action as numpy array of shape (action_dim,).
    """
    action_min = action_bounds[0]
    action_max = action_bounds[1]

    # Step 1: Draw N uniform random action samples
    actions = (
        torch.rand(1, num_samples, action_dim, device=device)
        * (action_max - action_min)
        + action_min
    )  # (1, N, action_dim)

    noise_std = initial_std

    with torch.no_grad():
        for iteration in range(num_iterations):
            # Expand state to match action samples
            state_expanded = state.unsqueeze(1).expand(
                -1, num_samples, -1
            )  # (1, N, state_dim)

            # Step 2: Evaluate energy
            energies = energy_model(state_expanded, actions).squeeze(-1)  # (1, N)

            # Step 3: Softmax probabilities (negate: lower energy → higher prob)
            logits = -energies / temperature  # (1, N)
            probs = torch.softmax(logits, dim=-1)  # (1, N)

            # Step 4: Resample proportionally (categorical with replacement)
            indices = torch.multinomial(
                probs, num_samples=num_samples, replacement=True
            )  # (1, N)

            # Gather resampled actions
            actions = torch.gather(
                actions, 1, indices.unsqueeze(-1).expand(-1, -1, action_dim)
            )  # (1, N, action_dim)

            # Step 5: Add shrinking Gaussian noise
            noise = torch.randn_like(actions) * noise_std
            actions = actions + noise
            actions = torch.clamp(actions, action_min, action_max)

            noise_std *= 0.5  # Shrink noise each iteration

        # Final evaluation to pick the best action
        state_expanded = state.unsqueeze(1).expand(-1, num_samples, -1)
        energies = energy_model(state_expanded, actions).squeeze(-1)  # (1, N)
        best_idx = energies.argmin(dim=-1)  # (1,)
        best_action = actions[0, best_idx[0]].cpu().numpy()  # (action_dim,)

    return best_action


def compute_optimal_steps(agent_pos: np.ndarray, goal: np.ndarray) -> int:
    """Compute the minimum number of steps to reach the goal.

    The optimal path is a straight line. Each step covers step_size distance.
    We account for the goal_radius (agent reaches goal when within radius).

    Args:
        agent_pos: Starting position (2,).
        goal: Goal position (2,).

    Returns:
        Minimum number of steps (integer, at least 1).
    """
    dist = np.linalg.norm(goal - agent_pos)
    effective_dist = max(dist - goal_radius, 0.0)
    return max(int(math.ceil(effective_dist / step_size)), 1)


def plot_path(
    trajectory: list[np.ndarray],
    goal: np.ndarray,
    seed: int,
    steps: int,
    optimal_steps: int,
    reward: float,
    save_path: str,
) -> None:
    """Plot the agent's path, start position, and goal (like plot 5 in dummy_simulation.py).

    Args:
        trajectory: List of (2,) arrays representing agent positions.
        goal: Goal position (2,).
        seed: Random seed used.
        steps: Number of steps taken.
        optimal_steps: Optimal number of steps.
        reward: Total episode reward.
        save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    traj = np.array(trajectory)  # (T, 2)

    # Plot trajectory as a line with gradient color
    for i in range(len(traj) - 1):
        progress = i / max(len(traj) - 1, 1)
        color = plt.cm.viridis(progress)
        ax.plot(
            traj[i : i + 2, 0], traj[i : i + 2, 1],
            color=color, linewidth=1.5, alpha=0.8,
        )

    # Start and goal markers
    ax.plot(traj[0, 0], traj[0, 1], "s", color="blue", markersize=10, label="Start", zorder=5)
    ax.plot(goal[0], goal[1], "*", color="red", markersize=15, label="Goal", zorder=5)
    ax.plot(traj[-1, 0], traj[-1, 1], "D", color="green", markersize=8, label="End", zorder=5)

    # Goal radius circle
    circle = plt.Circle(goal, goal_radius, fill=False, color="red", linestyle="--", alpha=0.5)
    ax.add_patch(circle)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        f"IBC-DFO Path (Seed {seed})\n"
        f"Steps: {steps} (optimal: {optimal_steps}) | Reward: {reward:.2f}"
    )
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_energy_model(device: torch.device) -> torch.nn.Module:
    """Load the trained energy model from checkpoint."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "q_estimator.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'.\n"
            f"Train the model first with: python -m ibc.ibc_dfo_dummy_training"
        )

    model = QEstimator(
        state_dim=state_dim * frame_stack,
        action_dim=action_dim,
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)],
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"Loaded energy model from {checkpoint_path}")
    return model


def run_episode(
    energy_model: torch.nn.Module,
    device: torch.device,
    seed: int,
) -> dict:
    """Run a single episode using DFO inference.

    Args:
        energy_model: Trained energy model.
        device: Torch device.
        seed: Random seed for environment reset.

    Returns:
        Dictionary with trajectory, goal, steps, optimal_steps, reward, success.
    """
    env = DummyEnv(
        step_size=step_size,
        goal_radius=goal_radius,
        max_steps=max_episode_steps,
    )

    obs, info = env.reset(seed=seed)
    goal = obs[:2].copy()
    agent_pos = obs[2:4].copy()
    start_pos = agent_pos.copy()

    trajectory = [agent_pos.copy()]
    total_reward = 0.0
    done = False
    step_count = 0

    optimal_steps = compute_optimal_steps(start_pos, goal)

    while not done:
        step_count += 1

        # Build state tensor
        state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        # DFO action selection
        action = dfo_optimize(energy_model, state_tensor, device)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        agent_pos = obs[2:4].copy()
        trajectory.append(agent_pos.copy())

        done = terminated or truncated

    env.close()

    return {
        "trajectory": trajectory,
        "goal": goal,
        "start": start_pos,
        "steps": step_count,
        "optimal_steps": optimal_steps,
        "reward": total_reward,
        "success": terminated,
        "seed": seed,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running IBC-DFO inference on Dummy environment")
    print(f"DFO config: {DFO_NUM_SAMPLES} samples, {DFO_NUM_ITERATIONS} iterations, "
          f"initial_std={DFO_INITIAL_STD}, temperature={DFO_TEMPERATURE}")
    print(f"Seeds: {seeds}")
    print()

    # Load model
    energy_model = load_energy_model(device)

    # Create output directory
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ─── Run episodes for each seed ───────────────────────────────────────
    all_results = []

    print("=" * 75)
    print(f"{'Seed':>6} | {'Steps':>6} | {'Optimal':>7} | {'Reward':>10} | {'Success':>7}")
    print("-" * 75)

    for seed in seeds:
        result = run_episode(energy_model, device, seed)
        all_results.append(result)

        # Generate path plot
        plot_path(
            trajectory=result["trajectory"],
            goal=result["goal"],
            seed=seed,
            steps=result["steps"],
            optimal_steps=result["optimal_steps"],
            reward=result["reward"],
            save_path=os.path.join(PLOT_DIR, f"path_seed_{seed}.png"),
        )

        print(
            f"{seed:>6} | "
            f"{result['steps']:>6} | "
            f"{result['optimal_steps']:>7} | "
            f"{result['reward']:>10.2f} | "
            f"{'✓' if result['success'] else '✗':>7}"
        )

    print("=" * 75)

    # ─── Global summary ───────────────────────────────────────────────────
    rewards = [r["reward"] for r in all_results]
    steps_list = [r["steps"] for r in all_results]
    optimal_list = [r["optimal_steps"] for r in all_results]
    successes = [r["success"] for r in all_results]

    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
    steps_mean = np.mean(steps_list)
    optimal_mean = np.mean(optimal_list)
    success_rate = np.mean(successes)

    print(f"\n{'GLOBAL SUMMARY':^75}")
    print("=" * 75)
    print(f"  Seeds evaluated:      {len(seeds)}")
    print(f"  Avg reward:           {reward_mean:.4f} ± {reward_std:.4f}")
    print(f"  Avg steps:            {steps_mean:.1f} (optimal avg: {optimal_mean:.1f})")
    print(f"  Success rate:         {success_rate:.2%}")
    print(f"  Plots saved to:       {PLOT_DIR}/")
    print("=" * 75)


if __name__ == "__main__":
    main()
