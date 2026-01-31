"""Plotting utilities for simulation results."""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from pathlib import Path


def save_simulation_plots(
    results: list[dict[str, Any]],
    output_dir: str = "plots/D4RL/pen_human_v2",
    show_plots: bool = False,
) -> None:
    """Save plots from simulation results.
    
    Args:
        results: List of episode result dictionaries from a simulation.
        output_dir: Directory to save the plots.
        show_plots: Whether to display plots interactively.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _plot_episode_rewards(results, output_path, show_plots)
    _plot_episode_lengths(results, output_path, show_plots)
    _plot_rewards_over_episodes(results, output_path, show_plots)

    print(f"Plots saved to: {output_path.absolute()}")


def _plot_episode_rewards(
    results: list[dict[str, Any]], 
    output_path: Path, 
    show_plots: bool
) -> None:
    """Plot distribution of episode rewards."""
    total_rewards = [r.get("total_reward", 0.0) for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(total_rewards, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel("Total Episode Reward", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Episode Rewards - D4RL/pen/human-v2", fontsize=14)
    ax.axvline(np.mean(total_rewards), color='red', linestyle='--', 
               label=f'Mean: {np.mean(total_rewards):.2f}')
    ax.axvline(np.mean(total_rewards) + np.std(total_rewards), color='orange', 
               linestyle=':', label=f'Std: {np.std(total_rewards):.2f}')
    ax.axvline(np.mean(total_rewards) - np.std(total_rewards), color='orange', linestyle=':')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "episode_rewards_distribution.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()


def _plot_episode_lengths(
    results: list[dict[str, Any]], 
    output_path: Path, 
    show_plots: bool
) -> None:
    """Plot distribution of episode lengths."""
    episode_lengths = [r.get("episode_length", 0) for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(episode_lengths, bins=30, edgecolor='black', alpha=0.7, color='seagreen')
    ax.set_xlabel("Episode Length", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Episode Lengths - D4RL/pen/human-v2", fontsize=14)
    ax.axvline(np.mean(episode_lengths), color='red', linestyle='--',
               label=f'Mean: {np.mean(episode_lengths):.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "episode_lengths_distribution.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()


def _plot_rewards_over_episodes(
    results: list[dict[str, Any]], 
    output_path: Path, 
    show_plots: bool
) -> None:
    """Plot rewards over episode index."""
    rewards = [r.get("total_reward", 0.0) for r in results]
    episodes = list(range(len(rewards)))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, rewards, marker='o', markersize=3, linestyle='-', 
            color='purple', alpha=0.7, linewidth=0.5)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("Episode Rewards Over Time - D4RL/pen/human-v2", fontsize=14)
    ax.axhline(np.mean(rewards), color='red', linestyle='--',
               label=f'Mean: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}')
    ax.fill_between(episodes, 
                    np.mean(rewards) - np.std(rewards), 
                    np.mean(rewards) + np.std(rewards),
                    alpha=0.2, color='red')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "rewards_over_episodes.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Run via: python -m simulations.run_simulation")
