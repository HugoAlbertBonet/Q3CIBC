"""Standalone script to run simulations on trained models.

Usage:
    python -m simulations.run_simulation [--checkpoint PATH] [--seeds 0 1 2] [--episodes 100]
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

from models import ControlPointGenerator
from simulations import PenHumanV2Simulation
from simulations.plots import save_simulation_plots

# Load config
config_path = Path(__file__).parent.parent / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

# Model architecture parameters from config
STATE_DIM = config["environment"]["state_dim"]
ACTION_DIM = config["environment"]["action_dim"]
CONTROL_POINTS = config["model"]["control_points"]
num_hidden_layers = config["model"]["num_hidden_layers"]
num_neurons = config["model"]["num_neurons"]

# Simulation defaults from config
DEFAULT_CHECKPOINT = config["simulation"]["default_checkpoint"]
DEFAULT_SEEDS = config["simulation"]["default_seeds"]
DEFAULT_EPISODES = config["simulation"]["default_episodes"]


def load_model(checkpoint_path: str, device: str = "cpu") -> ControlPointGenerator:
    """Load a trained control point generator from checkpoint."""
    model = ControlPointGenerator(
        input_dim=STATE_DIM,
        output_dim=ACTION_DIM,
        control_points=CONTROL_POINTS,
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)]
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_multi_seed_evaluation(
    model: ControlPointGenerator,
    seeds: list[int],
    episodes_per_seed: int,
    device: str = "cpu",
) -> tuple[dict, list]:
    """Run evaluation across multiple seeds and aggregate results."""
    all_rewards = []
    all_episode_lengths = []
    all_results = []
    
    for seed in seeds:
        print(f"\n  Seed {seed}...")
        set_seed(seed)
        
        simulation = PenHumanV2Simulation(
            control_point_generator=model,
            device=device,
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
    
    return aggregated, all_results


def print_summary_table(aggregated: dict, env_name: str) -> str:
    """Print and return a markdown table of results."""
    table = f"""
| Environment | Episodes | Reward (mean ± std) |
|-------------|----------|---------------------|
| {env_name} | {aggregated['total_episodes']} | {aggregated['reward_mean']:.2f} ± {aggregated['reward_std']:.2f} |
"""
    print(table)
    return table


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed simulation on D4RL/pen/human-v2 with a trained model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Random seeds to use (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Number of episodes per seed (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/D4RL/pen_human_v2",
        help="Directory to save plots (default: plots/D4RL/pen_human_v2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively",
    )
    args = parser.parse_args()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at '{args.checkpoint}'")
        print("Please train a model first using: python main.py")
        sys.exit(1)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)
    print("Model loaded successfully!")

    # Run multi-seed evaluation
    env_name = config["environment"]["dataset_name"]
    print(f"\nRunning evaluation on {config['environment']['env_id']} ({env_name})")
    print(f"  Seeds: {args.seeds}")
    print(f"  Episodes per seed: {args.episodes}")
    print(f"  Total episodes: {len(args.seeds) * args.episodes}")
    
    aggregated, all_results = run_multi_seed_evaluation(
        model=model,
        seeds=args.seeds,
        episodes_per_seed=args.episodes,
        device=args.device,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-SEED EVALUATION SUMMARY")
    print("=" * 70)
    table = print_summary_table(aggregated, env_name)
    print("=" * 70)

    # Save plots
    print(f"\nSaving plots to {args.output_dir}/...")
    save_simulation_plots(
        all_results,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
    )
    print("Done!")
    
    return aggregated


if __name__ == "__main__":
    main()
