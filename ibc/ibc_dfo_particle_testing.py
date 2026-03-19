"""IBC testing with Langevin MCMC on Particle environment.

Runs inference using the trained energy model (QEstimator) with Langevin MCMC.
Matches the paper-faithful training implementation:
  - Loads normalization stats saved during training
  - Normalizes observations before inference
  - Runs Langevin MCMC in normalized action space
  - Denormalizes actions before stepping in the environment

Reports:
  - Per-seed: path plot, steps taken, reward, success
  - Global: mean +/- std reward, success rate, goal distances

Following the IBC paper (Florence et al., 2021, arXiv:2109.00137).

Usage:
    python -m ibc.ibc_dfo_particle_testing
"""

import os
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.models import QEstimator
from utils.normalizations import ObservationNormalizer
from simulations.particle_env import ParticleEnv

# ─── Load config ─────────────────────────────────────────────────────────────
config_path = Path(__file__).parent.parent / "config_json" / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

env_config = config["environments"]["particle"]
sim_config = config.get("simulation", {})

# Environment parameters
state_dim = env_config["state_dim"]
action_dim = env_config["action_dim"]
frame_stack = env_config.get("frame_stack", 2)
n_dim = env_config.get("n_dim", 2)
n_steps = 50
action_bounds = env_config.get("action_bounds", [0, 1])

# Model architecture (must match training: 256x2)
HIDDEN_DIMS = [256, 256]

# Seeds from config
seeds = sim_config.get("default_seeds", [0, 35, 42, 10, 135, 142])

# Langevin MCMC inference hyperparameters
LANGEVIN_NUM_SAMPLES = 512
LANGEVIN_NUM_ITERATIONS = 100
LANGEVIN_LR_INIT = 0.1
LANGEVIN_LR_FINAL = 1e-5
LANGEVIN_DECAY_POWER = 2.0
LANGEVIN_DELTA_ACTION_CLIP = 0.1
LANGEVIN_NOISE_SCALE = 1.0
UNIFORM_BOUNDARY_BUFFER = 0.05

# Checkpoint and output paths
CHECKPOINT_DIR = os.path.join(
    config.get("training_shared", {}).get("model_save_dir", "checkpoints"),
    "ibc_dfo", "particle",
)
PLOT_DIR = "plots/ibc_dfo/particle"


def normalize_obs(obs_tensor, obs_normalizer):
    """Normalize observation tensor with shared bounds-based normalization."""
    return obs_normalizer.normalize(obs_tensor)


def denormalize_action(action_norm, norm_stats):
    """Denormalize action from [0, 1] back to original range."""
    act_min = norm_stats["act_min"]
    act_max = norm_stats["act_max"]
    rng = act_max - act_min
    rng = np.where(rng == 0, np.ones_like(rng), rng)
    return action_norm * rng + act_min


def langevin_optimize(
    energy_model,
    state_norm,
    device,
    num_samples=LANGEVIN_NUM_SAMPLES,
    num_iterations=LANGEVIN_NUM_ITERATIONS,
    lr_init=LANGEVIN_LR_INIT,
    lr_final=LANGEVIN_LR_FINAL,
    decay_power=LANGEVIN_DECAY_POWER,
    delta_action_clip=LANGEVIN_DELTA_ACTION_CLIP,
    noise_scale=LANGEVIN_NOISE_SCALE,
):
    """Langevin MCMC optimizer in normalized action space.

    Finds the minimum-energy action by running gradient-based Langevin dynamics.
    Operates entirely in normalized space [0, 1] (with buffer).

    Args:
        energy_model: Trained QEstimator E(obs, action).
        state_norm: Normalized observation tensor, shape (1, obs_dim).
        device: Torch device.

    Returns:
        Best action in NORMALIZED space as numpy array of shape (action_dim,).
    """
    act_min = 0.0 - UNIFORM_BOUNDARY_BUFFER
    act_max = 1.0 + UNIFORM_BOUNDARY_BUFFER

    # Scaled delta clip (paper: delta_action_clip * 0.5 * (max - min))
    delta_clip = delta_action_clip * 0.5 * (act_max - act_min)

    # Initialize with uniform random in normalized space
    actions = (
        torch.rand(1, num_samples, action_dim, device=device)
        * (act_max - act_min) + act_min
    )

    for k in range(num_iterations):
        # Polynomial decay learning rate
        fraction = 1.0 - k / max(num_iterations - 1, 1)
        lr_k = lr_final + (lr_init - lr_final) * (fraction ** decay_power)

        actions = actions.detach().requires_grad_(True)

        state_expanded = state_norm.unsqueeze(1).expand(-1, num_samples, -1)
        energies = energy_model(state_expanded, actions).squeeze(-1)
        grad = torch.autograd.grad(energies.sum(), actions)[0]

        # Langevin step: gradient descent on energy + noise
        noise = torch.randn_like(actions) * noise_scale
        delta = lr_k * (0.5 * grad + noise)
        delta = torch.clamp(delta, -delta_clip, delta_clip)

        actions = (actions.detach() - delta).detach()
        actions = torch.clamp(actions, act_min, act_max)

    # Pick the action with lowest energy
    with torch.no_grad():
        state_expanded = state_norm.unsqueeze(1).expand(-1, num_samples, -1)
        energies = energy_model(state_expanded, actions).squeeze(-1)
        best_idx = energies.argmin(dim=-1)
        best_action = actions[0, best_idx[0]].cpu().numpy()

    return best_action


def plot_path(
    trajectory,
    first_goal,
    second_goal,
    seed,
    steps,
    reward,
    success,
    min_dist_first,
    min_dist_second,
    save_path,
):
    """Plot the agent's path, start position, and both goals."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    traj = np.array(trajectory)

    for i in range(len(traj) - 1):
        progress = i / max(len(traj) - 1, 1)
        color = plt.cm.viridis(progress)
        ax.plot(
            traj[i : i + 2, 0], traj[i : i + 2, 1],
            color=color, linewidth=1.5, alpha=0.8,
        )

    ax.plot(traj[0, 0], traj[0, 1], "s", color="blue", markersize=10,
            label="Start", zorder=5)
    ax.plot(traj[-1, 0], traj[-1, 1], "D", color="orange", markersize=8,
            label="End", zorder=5)
    ax.plot(first_goal[0], first_goal[1], "*", color="green", markersize=15,
            label="Goal 1 (green)", zorder=5)
    ax.plot(second_goal[0], second_goal[1], "*", color="blue", markersize=15,
            label="Goal 2 (blue)", zorder=5)

    goal_distance = 0.05
    circle1 = plt.Circle(first_goal, goal_distance, fill=False, color="green",
                          linestyle="--", alpha=0.5)
    circle2 = plt.Circle(second_goal, goal_distance, fill=False, color="blue",
                          linestyle="--", alpha=0.5)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        f"IBC Langevin Particle (Seed {seed})\n"
        f"Steps: {steps} | Reward: {reward:.2f} | "
        f"{'Success' if success else 'Fail'}\n"
        f"Min dist G1: {min_dist_first:.4f}, G2: {min_dist_second:.4f}"
    )
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_energy_model(device):
    """Load the trained energy model and normalization stats from checkpoint."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "q_estimator.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'.\n"
            f"Train the model first with: python -m ibc.ibc_dfo_particle_training"
        )

    model = QEstimator(
        state_dim=state_dim * frame_stack,
        action_dim=action_dim,
        hidden_dims=HIDDEN_DIMS,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both old (plain state_dict) and new (dict with norm_stats) formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        norm_stats = checkpoint["norm_stats"]
        print(f"Loaded model (step {checkpoint.get('step', '?')}) with norm stats")
    else:
        model.load_state_dict(checkpoint)
        norm_stats = None
        print("WARNING: Old checkpoint format, no norm stats available.")
        print("  Inference will run WITHOUT normalization (may perform poorly).")

    model.to(device)
    model.eval()
    print(f"Loaded energy model from {checkpoint_path}")
    return model, norm_stats


def run_episode(energy_model, device, seed, norm_stats, obs_normalizer):
    """Run a single episode using Langevin MCMC inference.

    Args:
        energy_model: Trained energy model.
        device: Torch device.
        seed: Random seed for environment reset.
        norm_stats: Normalization statistics from training, or None.

    Returns:
        Dictionary with trajectory, goals, steps, reward, success, goal distances.
    """
    env = ParticleEnv(n_dim=n_dim, n_steps=n_steps)
    obs, info = env.reset(seed=seed)

    first_goal = obs[2 * n_dim : 3 * n_dim].copy()
    second_goal = obs[3 * n_dim : 4 * n_dim].copy()
    start_pos = obs[:n_dim].copy()

    # Frame stacking buffer
    frame_buffer = deque(maxlen=frame_stack)
    for _ in range(frame_stack):
        frame_buffer.append(obs.copy())

    def get_stacked_obs():
        if frame_stack <= 1:
            return frame_buffer[-1]
        return np.concatenate(list(frame_buffer))

    trajectory = [start_pos.copy()]
    total_reward = 0.0
    done = False
    step_count = 0

    while not done:
        step_count += 1

        stacked_obs = get_stacked_obs()
        state_tensor = torch.from_numpy(stacked_obs).float().unsqueeze(0).to(device)

        # Normalize observation with the same bounds-based pipeline as combined.
        state_norm = normalize_obs(state_tensor, obs_normalizer)

        # Langevin MCMC in normalized action space
        action_norm = langevin_optimize(energy_model, state_norm, device)

        # Denormalize action back to original space
        if norm_stats is not None:
            action = denormalize_action(action_norm, norm_stats)
        else:
            action = action_norm

        # Clip to env bounds
        action = np.clip(action, action_bounds[0], action_bounds[1])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame_buffer.append(obs.copy())
        agent_pos = obs[:n_dim].copy()
        trajectory.append(agent_pos.copy())

        done = terminated or truncated

    env.close()

    return {
        "trajectory": trajectory,
        "first_goal": first_goal,
        "second_goal": second_goal,
        "start": start_pos,
        "steps": step_count,
        "reward": total_reward,
        "success": info.get("success", False),
        "min_dist_first": info.get("min_dist_to_first_goal", np.inf),
        "min_dist_second": info.get("min_dist_to_second_goal", np.inf),
        "seed": seed,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running IBC Langevin MCMC inference on Particle environment")
    print(
        f"Langevin config: {LANGEVIN_NUM_SAMPLES} samples, "
        f"{LANGEVIN_NUM_ITERATIONS} iterations, "
        f"lr_init={LANGEVIN_LR_INIT}, lr_final={LANGEVIN_LR_FINAL}, "
        f"decay_power={LANGEVIN_DECAY_POWER}, noise_scale={LANGEVIN_NOISE_SCALE}"
    )
    print(f"Frame stack: {frame_stack}")
    print(f"Seeds: {seeds}")
    print()

    # Load model and norm stats
    energy_model, norm_stats = load_energy_model(device)
    obs_normalizer = ObservationNormalizer(
        env_id=env_config["env_id"],
        device=device,
        frame_stack=frame_stack,
    )

    os.makedirs(PLOT_DIR, exist_ok=True)

    # ─── Run episodes for each seed ───────────────────────────────────────
    all_results = []

    print("=" * 85)
    print(
        f"{'Seed':>6} | {'Steps':>6} | {'Reward':>8} | {'Success':>7} | "
        f"{'Dist G1':>8} | {'Dist G2':>8}"
    )
    print("-" * 85)

    for seed in seeds:
        result = run_episode(energy_model, device, seed, norm_stats, obs_normalizer)
        all_results.append(result)

        plot_path(
            trajectory=result["trajectory"],
            first_goal=result["first_goal"],
            second_goal=result["second_goal"],
            seed=seed,
            steps=result["steps"],
            reward=result["reward"],
            success=result["success"],
            min_dist_first=result["min_dist_first"],
            min_dist_second=result["min_dist_second"],
            save_path=os.path.join(PLOT_DIR, f"path_seed_{seed}.png"),
        )

        print(
            f"{seed:>6} | "
            f"{result['steps']:>6} | "
            f"{result['reward']:>8.2f} | "
            f"{'Y' if result['success'] else 'N':>7} | "
            f"{result['min_dist_first']:>8.4f} | "
            f"{result['min_dist_second']:>8.4f}"
        )

    print("=" * 85)

    # ─── Global summary ───────────────────────────────────────────────────
    rewards = [r["reward"] for r in all_results]
    steps_list = [r["steps"] for r in all_results]
    successes = [r["success"] for r in all_results]
    first_dists = [r["min_dist_first"] for r in all_results]
    second_dists = [r["min_dist_second"] for r in all_results]

    print(f"\n{'GLOBAL SUMMARY':^85}")
    print("=" * 85)
    print(f"  Seeds evaluated:         {len(seeds)}")
    print(f"  Avg reward:              {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")
    print(f"  Avg steps:               {np.mean(steps_list):.1f}")
    print(f"  Success rate:            {np.mean(successes):.2%}")
    print(f"  Avg min dist to Goal 1:  {np.mean(first_dists):.4f}")
    print(f"  Avg min dist to Goal 2:  {np.mean(second_dists):.4f}")
    print(f"  Plots saved to:          {PLOT_DIR}/")
    print("=" * 85)


if __name__ == "__main__":
    main()
