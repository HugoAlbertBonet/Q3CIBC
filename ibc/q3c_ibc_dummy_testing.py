"""Q3C IBC testing on Dummy environment.

Runs inference using the pure Q3C approach:
  1. ControlPointGenerator produces N candidate actions (control points)
  2. All candidates are evaluated by the QEstimator
  3. The action with the highest Q is selected

Reports:
  - Per-seed: path plot, steps taken vs optimal steps, reward
  - Global: mean ± std reward across all seeds

Usage:
    python -m ibc.q3c_ibc_dummy_testing
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

from utils.models import ControlPointGenerator, QEstimator
from simulations.dummy_env import DummyEnv
from utils.normalizations import ObservationNormalizer

# ─── Load config ─────────────────────────────────────────────────────────────
config_path = Path(__file__).parent.parent / "config_json" / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

env_config = config["environments"]["dummy"]
env_id = env_config["env_id"]
env_model = env_config.get("model", {})
sim_config = config.get("simulation", {})
training_shared = config.get("training_shared", {})

# Model architecture (must match training)
num_hidden_layers = env_model.get("num_hidden_layers", 4)
num_neurons = env_model.get("num_neurons", 128)
control_points_n = env_model.get("control_points", 30)
state_dim = env_config["state_dim"]
action_dim = env_config["action_dim"]
frame_stack = env_config.get("frame_stack", 1)
action_bounds = env_config.get("action_bounds", [-1, 1])

# Environment parameters
step_size = env_config.get("step_size", 0.1)
goal_radius = env_config.get("goal_radius", 0.1)
max_episode_steps = sim_config.get("max_episode_steps", 200)

# Seeds from config
seeds = sim_config.get("default_seeds", [0, 35, 42, 10, 135, 142])

# Checkpoint and output paths
CHECKPOINT_DIR = os.path.join(
    training_shared.get("model_save_dir", "checkpoints"), "q3c_ibc"
)
PLOT_DIR = "plots/q3c_ibc"


def compute_optimal_steps(agent_pos: np.ndarray, goal: np.ndarray) -> int:
    """Compute the minimum number of steps to reach the goal."""
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
    """Plot the agent's path, start position, and goal."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    traj = np.array(trajectory)

    for i in range(len(traj) - 1):
        progress = i / max(len(traj) - 1, 1)
        color = plt.cm.viridis(progress)
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], color=color, linewidth=1.5, alpha=0.8)

    ax.plot(traj[0, 0], traj[0, 1], "s", color="blue", markersize=10, label="Start", zorder=5)
    ax.plot(goal[0], goal[1], "*", color="red", markersize=15, label="Goal", zorder=5)
    ax.plot(traj[-1, 0], traj[-1, 1], "D", color="green", markersize=8, label="End", zorder=5)

    circle = plt.Circle(goal, goal_radius, fill=False, color="red", linestyle="--", alpha=0.5)
    ax.add_patch(circle)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        f"Q3C IBC Path (Seed {seed})\n"
        f"Steps: {steps} (optimal: {optimal_steps}) | Reward: {reward:.2f}"
    )
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_models(device: torch.device):
    """Load the trained generator and estimator from checkpoint."""
    gen_path = os.path.join(CHECKPOINT_DIR, "control_point_generator.pt")
    est_path = os.path.join(CHECKPOINT_DIR, "q_estimator.pt")

    for path, name in [(gen_path, "ControlPointGenerator"), (est_path, "QEstimator")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} checkpoint not found at '{path}'.\n"
                f"Train first: python -m ibc.q3c_ibc_dummy_training"
            )

    generator = ControlPointGenerator(
        input_dim=state_dim * frame_stack,
        output_dim=action_dim,
        control_points=control_points_n,
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)],
        action_bounds=(action_bounds[0], action_bounds[1]),
    )
    generator.load_state_dict(torch.load(gen_path, map_location=device, weights_only=True))
    generator.to(device)
    generator.eval()

    estimator = QEstimator(
        state_dim=state_dim * frame_stack,
        action_dim=action_dim,
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)],
    )
    estimator.load_state_dict(torch.load(est_path, map_location=device, weights_only=True))
    estimator.to(device)
    estimator.eval()

    print(f"Loaded generator from {gen_path}")
    print(f"Loaded estimator from {est_path}")
    
    # Load normalizer
    obs_normalizer = ObservationNormalizer(env_id=env_id, device=device, frame_stack=frame_stack)
    return generator, estimator, obs_normalizer


def select_action(
    generator: torch.nn.Module,
    estimator: torch.nn.Module,
    state: torch.Tensor,
) -> np.ndarray:
    """Pure Q3C action selection: evaluate generator's control points and pick best.
    
    1. Generate control points from the generator
    2. Evaluate candidates with estimator
    3. Return best action (highest Q)
    """
    with torch.no_grad():
        # Step 1: Generator produces control points
        cps = generator(state)  # (1, CP, action_dim)
        
        # Step 2: Evaluate candidates with estimator
        state_exp = state.unsqueeze(1).expand(-1, cps.shape[1], -1)
        q_vals = estimator(state_exp, cps).squeeze(-1)  # (1, CP)

        # Step 3: Pick best action
        best_idx = q_vals.argmax(dim=1)
        best_action = cps[0, best_idx[0]].cpu().numpy()

    return best_action


def run_episode(
    generator: torch.nn.Module,
    estimator: torch.nn.Module,
    obs_normalizer: ObservationNormalizer,
    device: torch.device,
    seed: int,
) -> dict:
    """Run a single episode using pure Q3C inference."""
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
        
        # Build and normalize state tensor
        state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        state_tensor = obs_normalizer.normalize(state_tensor)
        
        action = select_action(generator, estimator, state_tensor)

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
    print(f"Running pure Q3C IBC inference on Dummy environment")
    print(f"Control points: {control_points_n}")
    print(f"Seeds: {seeds}")
    print()

    generator, estimator, obs_normalizer = load_models(device)
    os.makedirs(PLOT_DIR, exist_ok=True)

    all_results = []

    print("=" * 75)
    print(f"{'Seed':>6} | {'Steps':>6} | {'Optimal':>7} | {'Reward':>10} | {'Success':>7}")
    print("-" * 75)

    for seed in seeds:
        result = run_episode(generator, estimator, obs_normalizer, device, seed)
        all_results.append(result)

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
