"""Run particle simulation with a deterministic oracle policy.

Instead of using the learned ControlPointGenerator + QEstimator,
the action at every step is simply the position of the first goal
extracted from the observation.

Usage:
    python -m simulations.run_deterministic_particle [--seeds 0 1 2] [--episodes 100]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulations.particle_simulation import ParticleSimulation
from simulations.plots import save_simulation_plots

# Load config
config_path = Path(__file__).parent.parent / "config_json" / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

env_config = config["environments"]["particle"]

# Simulation defaults from config
DEFAULT_SEEDS = config["simulation"]["default_seeds"]
DEFAULT_EPISODES = config["simulation"]["default_episodes"]


class DeterministicParticleSimulation(ParticleSimulation):
    """Particle simulation with a deterministic oracle policy.
    
    The action is always the position of the first goal, extracted
    directly from the (stacked) observation vector.
    """

    def __init__(self, **kwargs):
        # We don't need real models — pass dummy nn.Module instances
        # that won't be used since we override select_action.
        dummy_model = torch.nn.Linear(1, 1)
        kwargs.setdefault("control_point_generator", dummy_model)
        kwargs.setdefault("q_estimator", dummy_model)
        super().__init__(**kwargs)

    def select_action(self, observation: np.ndarray, return_q_range: bool = False):
        """Return the first goal position as the action.
        
        Observation layout per frame (n_dim=2, no hidden velocity):
            [pos_agent(2), vel_agent(2), pos_first_goal(2), pos_second_goal(2)]
        
        With frame_stack=K, the stacked observation is K concatenated frames.
        We extract the first goal from the LAST (most recent) frame.
        """
        single_frame_dim = 4 * self.n_dim  # 8 for n_dim=2
        
        # Get the last frame from the stacked observation
        last_frame = observation[-single_frame_dim:]
        
        # Extract pos_first_goal: after pos_agent(n_dim) + vel_agent(n_dim)
        goal_start = 2 * self.n_dim  # index 4 for n_dim=2
        goal_end = 3 * self.n_dim    # index 6 for n_dim=2
        action = last_frame[goal_start:goal_end].copy()
        
        # Clip to valid range [0, 1]
        action = np.clip(action, 0.0, 1.0)
        
        if return_q_range:
            return action, (0.0, 0.0)  # dummy q_range
        return action

    def run_simulation(self, num_episodes: int = 100, seed: int | None = None):
        """Run simulation without calling .eval() on models."""
        self.results = []
        for idx in range(num_episodes):
            episode_seed = (seed + idx) if seed is not None else None
            result = self.run_episode(seed=episode_seed)
            result["episode_index"] = idx
            self.results.append(result)
        return self.results


def set_seed(seed: int):
    """Set seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_multi_seed_evaluation(
    seeds: list[int],
    episodes_per_seed: int,
    render_mode: str | None = None,
) -> tuple[dict, list]:
    """Run evaluation across multiple seeds and aggregate results."""
    all_rewards = []
    all_episode_lengths = []
    all_results = []

    n_dim = env_config.get("n_dim", 2)
    max_steps = config["simulation"].get("max_episode_steps", 200)
    frame_stack = env_config.get("frame_stack", 1)

    for seed in seeds:
        print(f"\n  Seed {seed}...")
        set_seed(seed)

        simulation = DeterministicParticleSimulation(
            n_dim=n_dim,
            device="cpu",
            max_episode_steps=max_steps,
            render_mode=render_mode,
            frame_stack=frame_stack,
            save_gif_dir="plots/particle_deterministic/episodes" if render_mode else None,
        )
        results = simulation.run_simulation(num_episodes=episodes_per_seed, seed=seed)
        all_results.extend(results)
        simulation.close()

        # Collect metrics
        for r in results:
            all_rewards.append(r.get("total_reward", 0.0))
            all_episode_lengths.append(r.get("episode_length", 0))

        summary = simulation.get_summary()
        print(f"    Episodes: {summary['num_episodes']}, "
              f"Mean Reward: {summary['reward_mean']:.2f} ± {summary['reward_std']:.2f}")

        if "success_rate" in summary:
            print(f"    Success Rate: {summary['success_rate']:.2%}")

    # Aggregate statistics
    aggregated = {
        "num_seeds": len(seeds),
        "episodes_per_seed": episodes_per_seed,
        "total_episodes": len(all_rewards),
        "reward_mean": np.mean(all_rewards),
        "reward_std": np.std(all_rewards),
        "episode_length_mean": np.mean(all_episode_lengths),
        "episode_length_std": np.std(all_episode_lengths),
    }

    # Add success rate if available
    successes = [r.get("success", False) for r in all_results if "success" in r]
    if successes:
        aggregated["success_rate"] = np.mean(successes)

    return aggregated, all_results


def print_summary_table(aggregated: dict) -> str:
    """Print and return a markdown table of results."""
    success_col = ""
    if "success_rate" in aggregated:
        success_col = f" | {aggregated['success_rate']:.2%}"

    table = f"""
| Environment | Episodes | Reward (mean ± std) |{' Success Rate |' if success_col else ''}
|-------------|----------|---------------------|{'--------------|' if success_col else ''}
| particle (deterministic) | {aggregated['total_episodes']} | {aggregated['reward_mean']:.2f} ± {aggregated['reward_std']:.2f}{success_col} |
"""
    print(table)
    return table


def main():
    parser = argparse.ArgumentParser(
        description="Run particle simulation with deterministic oracle policy (action = first goal position)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Random seeds to use (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help=f"Number of episodes per seed (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots/particle_deterministic",
        help="Directory to save plots (default: plots/particle_deterministic)",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render the environment during simulation",
    )
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    render_from_config = config["simulation"].get("render", False)
    if render_from_config:
        render_mode = "human"

    # Clear previous plots before running
    if os.path.exists(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("DETERMINISTIC PARTICLE SIMULATION")
    print("Policy: action = position of first goal")
    print("=" * 70)
    print(f"  Seeds: {args.seeds}")
    print(f"  Episodes per seed: {args.episodes}")
    print(f"  Total episodes: {len(args.seeds) * args.episodes}")

    aggregated, all_results = run_multi_seed_evaluation(
        seeds=args.seeds,
        episodes_per_seed=args.episodes,
        render_mode=render_mode,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-SEED EVALUATION SUMMARY")
    print("=" * 70)
    print_summary_table(aggregated)
    print("=" * 70)

    # Save plots
    print(f"\nSaving plots to {args.output_dir}/...")
    save_simulation_plots(
        all_results,
        output_dir=args.output_dir,
        show_plots=False,
    )
    print("Done!")

    return aggregated


if __name__ == "__main__":
    main()
