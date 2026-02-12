"""Standalone script to run simulations on trained models.

Uses config.json to determine which environment to run.
Set "active_env" in config to switch between environments.

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
import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models import ControlPointGenerator, QEstimator
from simulations import PenHumanV2Simulation, ParticleSimulation
from simulations.plots import save_simulation_plots

# Load config
config_path = Path(__file__).parent.parent / "config_json" / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

# Get active environment
active_env = config.get("active_env", "pen")
env_config = config["environments"][active_env]
training_shared = config.get("training_shared", {})

# Model architecture parameters from active env config
STATE_DIM = env_config["state_dim"]
ACTION_DIM = env_config["action_dim"]
FRAME_STACK = env_config.get("frame_stack", 1)
CONTROL_POINTS = env_config["model"]["control_points"]
num_hidden_layers = env_config["model"]["num_hidden_layers"]
num_neurons = env_config["model"]["num_neurons"]
ACTION_BOUNDS = tuple(env_config.get("action_bounds", [-1, 1]))

# Simulation defaults from config
DEFAULT_CHECKPOINT = config["simulation"]["default_checkpoint"]
DEFAULT_SEEDS = config["simulation"]["default_seeds"]
DEFAULT_EPISODES = config["simulation"]["default_episodes"]


def load_model(checkpoint_path: str, device: str = "cpu") -> ControlPointGenerator:
    """Load a trained control point generator from checkpoint."""
    model = ControlPointGenerator(
        input_dim=STATE_DIM * FRAME_STACK,
        output_dim=ACTION_DIM,
        control_points=CONTROL_POINTS,
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)],
        action_bounds=ACTION_BOUNDS,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def load_q_estimator(checkpoint_path: str, device: str = "cpu") -> QEstimator:
    """Load a trained Q-estimator from checkpoint."""
    model = QEstimator(
        state_dim=STATE_DIM * FRAME_STACK,
        action_dim=ACTION_DIM,
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)]
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def load_smoothing_param(checkpoint_path: str, device: str = "cpu") -> torch.Tensor:
    """Load the trained smoothing parameter from checkpoint."""
    param = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return param


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_simulation(
    model: ControlPointGenerator,
    q_estimator: QEstimator,
    smoothing_param: torch.Tensor,
    device: str = "cpu",
    render_mode: str | None = None,
):
    """Create the appropriate simulation based on active_env."""
    max_steps = config["simulation"].get("max_episode_steps", 200)
    
    if active_env == "pen":
        return PenHumanV2Simulation(
            control_point_generator=model,
            q_estimator=q_estimator,
            smoothing_param=smoothing_param,
            device=device,
            max_episode_steps=max_steps,
            render_mode=render_mode,
            frame_stack=FRAME_STACK,
        )
    elif active_env == "particle":
        n_dim = env_config.get("n_dim", 2)
        return ParticleSimulation(
            control_point_generator=model,
            q_estimator=q_estimator,
            smoothing_param=smoothing_param,
            n_dim=n_dim,
            device=device,
            max_episode_steps=max_steps,
            render_mode=render_mode,
            frame_stack=FRAME_STACK,
            save_gif_dir="plots/particle/episodes" if render_mode else None,
        )
    elif active_env == "dummy":
        from simulations.dummy_simulation import DummySimulation
        langevin_config = env_config.get("model", {}).get("langevin_config", {})
        return DummySimulation(
            control_point_generator=model,
            q_estimator=q_estimator,
            device=device,
            render_mode=render_mode,
            langevin_config=langevin_config
        )
    else:
        raise ValueError(f"Unknown environment: {active_env}")


def run_multi_seed_evaluation(
    model: ControlPointGenerator,
    q_estimator: QEstimator,
    smoothing_param: torch.Tensor,
    seeds: list[int],
    episodes_per_seed: int,
    device: str = "cpu",
    render_mode: str | None = None,
) -> tuple[dict, list]:
    """Run evaluation across multiple seeds and aggregate results."""
    all_rewards = []
    all_episode_lengths = []
    all_results = []
    
    for seed in seeds:
        print(f"\n  Seed {seed}...")
        set_seed(seed)
        
        simulation = create_simulation(
            model=model,
            q_estimator=q_estimator,
            smoothing_param=smoothing_param,
            device=device,
            render_mode=render_mode,
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


def print_summary_table(aggregated: dict, env_name: str) -> str:
    """Print and return a markdown table of results."""
    success_col = ""
    if "success_rate" in aggregated:
        success_col = f" | {aggregated['success_rate']:.2%}"
    
    table = f"""
| Environment | Episodes | Reward (mean ± std) |{' Success Rate |' if success_col else ''}
|-------------|----------|---------------------|{'--------------|' if success_col else ''}
| {env_name} | {aggregated['total_episodes']} | {aggregated['reward_mean']:.2f} ± {aggregated['reward_std']:.2f}{success_col} |
"""
    print(table)
    return table


def get_render_mode(args_render: bool, config: dict) -> str | None:
    """Determine render mode based on args and config."""
    render_from_config = config["simulation"].get("render", False)
    if args_render or render_from_config:
        return "human"
    return None


def main():
    parser = argparse.ArgumentParser(
        description=f"Run simulation on {active_env} environment with a trained model"
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
        default=f"plots/{active_env}",
        help=f"Directory to save plots (default: plots/{active_env})",
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
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during simulation (overrides config)",
    )
    args = parser.parse_args()

    # Initialize W&B for simulation logging
    wandb.init(
        project="Q3CIBC",
        job_type="evaluation",
        config={
            "active_env": active_env,
            "checkpoint": args.checkpoint,
            "seeds": args.seeds,
            "episodes_per_seed": args.episodes,
        }
    )

    # Check checkpoint exists
    checkpoint_dir = os.path.dirname(args.checkpoint)
    q_estimator_path = os.path.join(checkpoint_dir, "q_estimator.pt")
    smoothing_param_path = os.path.join(checkpoint_dir, "smoothing_param.pt")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at '{args.checkpoint}'")
        print("Please train a model first using: python combined_training.py")
        sys.exit(1)
    
    if not os.path.exists(q_estimator_path):
        print(f"Error: Q-estimator checkpoint not found at '{q_estimator_path}'")
        print("Please train a model first using: python combined_training.py")
        sys.exit(1)

    # Load models
    print(f"Loading control point generator from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)
    
    print(f"Loading Q-estimator from {q_estimator_path}...")
    q_estimator = load_q_estimator(q_estimator_path, device=args.device)
    
    # Load smoothing param (use config default if not found)
    if os.path.exists(smoothing_param_path):
        print(f"Loading smoothing parameter from {smoothing_param_path}...")
        smoothing_param = load_smoothing_param(smoothing_param_path, device=args.device)
        print(f"  Learned smoothing_param: {smoothing_param.item():.6f}")
    else:
        smoothing_param = torch.tensor(training_shared.get("smoothing_param", 0.1))
        print(f"  Using config smoothing_param: {smoothing_param.item():.6f}")
    
    print("Models loaded successfully!")

    # Run multi-seed evaluation
    env_id = env_config["env_id"]
    print(f"\nRunning evaluation on {env_id} (active_env: {active_env})")
    print(f"  Seeds: {args.seeds}")
    print(f"  Episodes per seed: {args.episodes}")
    print(f"  Total episodes: {len(args.seeds) * args.episodes}")
    
    # Determine render mode
    render_mode = get_render_mode(args.render, config)
    if render_mode:
        print(f"Rendering enabled (mode: {render_mode})")

    aggregated, all_results = run_multi_seed_evaluation(
        model=model,
        q_estimator=q_estimator,
        smoothing_param=smoothing_param,
        seeds=args.seeds,
        episodes_per_seed=args.episodes,
        device=args.device,
        render_mode=render_mode,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-SEED EVALUATION SUMMARY")
    print("=" * 70)
    table = print_summary_table(aggregated, active_env)
    print("=" * 70)

    # Save plots
    print(f"\nSaving plots to {args.output_dir}/...")
    save_simulation_plots(
        all_results,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
    )
    print("Done!")
    
    # Log simulation results to W&B
    wandb.summary["reward_mean"] = aggregated["reward_mean"]
    wandb.summary["reward_std"] = aggregated["reward_std"]
    wandb.summary["total_episodes"] = aggregated["total_episodes"]
    if "success_rate" in aggregated:
        wandb.summary["success_rate"] = aggregated["success_rate"]
    
    # Log plots as W&B artifacts
    if os.path.exists(args.output_dir):
        artifact = wandb.Artifact("simulation-plots", type="plots")
        artifact.add_dir(args.output_dir)
        wandb.log_artifact(artifact)
    
    wandb.finish()
    
    return aggregated


if __name__ == "__main__":
    main()
