"""Run particle simulation with full diagnostic logging.

Logs at each timestep:
  - Raw observation (stacked)
  - Normalized observation (model input)
  - Control points (model output)
  - Q-values for each control point
  - Selected action

Generates diagnostic plots showing all of the above over time.

Usage:
    python -m simulations.run_diagnostic_particle [--seed 42] [--output-dir plots/particle_diagnostic]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models import ControlPointGenerator, QEstimator
from utils.normalizations import ObservationNormalizer
from utils.sampling import sample_langevin
from simulations.particle_simulation import ParticleSimulation

# Load config
config_path = Path(__file__).parent.parent / "config_json" / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

env_config = config["environments"]["particle"]

STATE_DIM = env_config["state_dim"]
ACTION_DIM = env_config["action_dim"]
FRAME_STACK = env_config.get("frame_stack", 1)
CONTROL_POINTS = env_config["model"]["control_points"]
num_hidden_layers = env_config["model"]["num_hidden_layers"]
num_neurons = env_config["model"]["num_neurons"]
ACTION_BOUNDS = tuple(env_config.get("action_bounds", [0, 1]))


def _dim_names(n_dim: int) -> list[str]:
    """Return readable per-dimension labels for arbitrary dimensionality."""
    base = ["x", "y", "z", "w"]
    if n_dim <= len(base):
        return base[:n_dim]
    return [f"dim{d}" for d in range(n_dim)]


def _obs_component_names(n_dim: int) -> list[str]:
    """Observation component names for one unstacked frame."""
    dims = _dim_names(n_dim)
    names: list[str] = []
    names.extend([f"pos_agent_{d}" for d in dims])
    names.extend([f"vel_agent_{d}" for d in dims])
    names.extend([f"pos_goal1_{d}" for d in dims])
    names.extend([f"pos_goal2_{d}" for d in dims])
    return names


@dataclass
class StepLog:
    """Logged data from a single timestep."""
    step: int
    raw_obs: np.ndarray            # stacked raw observation
    normalized_obs: np.ndarray     # normalized observation (model input)
    control_points: np.ndarray     # (N, action_dim) control point outputs
    q_values: np.ndarray           # (N,) Q-value for each control point
    best_idx: int                  # index of selected control point
    action: np.ndarray             # final action taken


class DiagnosticParticleSimulation(ParticleSimulation):
    """Particle simulation with full diagnostic logging at each step."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_logs: list[StepLog] = []

    def select_action(self, observation: np.ndarray, return_q_range: bool = False):
        """Select action and log all intermediate values."""
        raw_obs = observation.copy()

        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        normalized_tensor = self.obs_normalizer.normalize(obs_tensor)
        normalized_obs = normalized_tensor.squeeze(0).cpu().numpy()

        with torch.no_grad():
            control_points = self.control_point_generator(normalized_tensor)  # (1, N, action_dim)
            obs_expanded = normalized_tensor.unsqueeze(1).expand(-1, control_points.shape[1], -1)
            q_values = self.q_estimator(obs_expanded, control_points).squeeze(-1)  # (1, N)

            best_idx = q_values.argmax(dim=1)  # (1,)
            action = control_points[0, best_idx[0], :].cpu().numpy()
            q_range = (q_values.min().item(), q_values.max().item())

        action_clipped = np.clip(action, 0.0, 1.0)

        log = StepLog(
            step=len(self.step_logs),
            raw_obs=raw_obs,
            normalized_obs=normalized_obs,
            control_points=control_points[0].cpu().numpy(),
            q_values=q_values[0].cpu().numpy(),
            best_idx=best_idx[0].item(),
            action=action_clipped.copy(),
        )
        self.step_logs.append(log)

        if return_q_range:
            return action_clipped, q_range
        return action_clipped


def generate_diagnostic_plots(
    step_logs: list[StepLog],
    output_dir: str,
    seed: int,
    q_estimator: QEstimator | None = None,
    action_bounds: tuple[float, float] = (0.0, 1.0),
    heatmap_grid_size: int = 81,
    n_dim: int = 2,
    langevin_config: dict | None = None,
    langevin_num_samples: int | None = None,
    show_langevin_trajectories: bool = True,
    show_langevin_final_positions: bool = True,
):
    """Generate comprehensive diagnostic plots from logged step data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    steps = [l.step for l in step_logs]
    n_steps = len(steps)

    # Extract arrays
    raw_obs_all = np.array([l.raw_obs for l in step_logs])           # (T, stacked_dim)
    norm_obs_all = np.array([l.normalized_obs for l in step_logs])   # (T, stacked_dim)
    actions_all = np.array([l.action for l in step_logs])            # (T, action_dim)
    q_values_all = np.array([l.q_values for l in step_logs])         # (T, N)
    cp_all = np.array([l.control_points for l in step_logs])         # (T, N, action_dim)
    best_idx_all = np.array([l.best_idx for l in step_logs])         # (T,)

    single_frame_dim = 4 * n_dim  # 8 for 2D
    obs_components = _obs_component_names(n_dim)

    if len(obs_components) != single_frame_dim:
        raise ValueError(
            f"Observation labeling mismatch: got {len(obs_components)} labels for single_frame_dim={single_frame_dim}."
        )

    # ─── PLOT 1: Raw observation components over time (current frame) ───
    cols = 2
    rows = int(np.ceil(single_frame_dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(3.0, 2.4 * rows)), sharex=True)
    axes = np.atleast_2d(axes)
    fig.suptitle(f"Raw Observation (Current Frame) — Seed {seed}", fontsize=14, fontweight="bold")

    for idx in range(single_frame_dim):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        # Use the last frame from stacked obs
        global_idx = raw_obs_all.shape[1] - single_frame_dim + idx
        ax.plot(steps, raw_obs_all[:, global_idx], ".-", markersize=3, linewidth=1)
        ax.set_ylabel(obs_components[idx], fontsize=10)
        ax.grid(alpha=0.3)
        if idx < n_dim or (n_dim * 2 <= idx < n_dim * 4):  # pos_agent or goal positions
            ax.set_ylim(-0.1, 1.1)

    for idx in range(single_frame_dim, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    for c in range(cols):
        axes[rows - 1][c].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(output_path / "1_raw_observation.png", dpi=150)
    plt.close()

    # ─── PLOT 2: Normalized observation components over time (current frame) ───
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(3.0, 2.4 * rows)), sharex=True)
    axes = np.atleast_2d(axes)
    fig.suptitle(f"Normalized Observation / Model Input (Current Frame) — Seed {seed}", fontsize=14, fontweight="bold")

    for idx in range(single_frame_dim):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        global_idx = norm_obs_all.shape[1] - single_frame_dim + idx
        ax.plot(steps, norm_obs_all[:, global_idx], ".-", markersize=3, linewidth=1, color="tab:orange")
        ax.set_ylabel(f"{obs_components[idx]}\n(norm)", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    for idx in range(single_frame_dim, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    for c in range(cols):
        axes[rows - 1][c].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(output_path / "2_normalized_observation.png", dpi=150)
    plt.close()

    # ─── PLOT 3: Raw vs Normalized side-by-side for each component ───
    fig, axes = plt.subplots(single_frame_dim, 1, figsize=(14, 2.5 * single_frame_dim), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f"Raw vs Normalized Observation — Seed {seed}", fontsize=14, fontweight="bold")

    for idx in range(single_frame_dim):
        ax = axes[idx]
        global_idx_raw = raw_obs_all.shape[1] - single_frame_dim + idx
        global_idx_norm = norm_obs_all.shape[1] - single_frame_dim + idx

        ax.plot(steps, raw_obs_all[:, global_idx_raw], ".-", markersize=3, linewidth=1, label="raw", color="tab:blue")
        ax_twin = ax.twinx()
        ax_twin.plot(steps, norm_obs_all[:, global_idx_norm], ".-", markersize=3, linewidth=1, label="normalized", color="tab:orange")
        ax_twin.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"{obs_components[idx]}\n(raw)", fontsize=9, color="tab:blue")
        ax_twin.set_ylabel("norm", fontsize=9, color="tab:orange")
        ax.grid(alpha=0.3)
        if idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(output_path / "3_raw_vs_normalized.png", dpi=150)
    plt.close()

    # ─── PLOT 4: Selected action over time + goal positions ───
    fig, axes = plt.subplots(n_dim, 1, figsize=(14, 4 * n_dim), sharex=True)
    if n_dim == 1:
        axes = [axes]
    fig.suptitle(f"Selected Action vs Goal Positions — Seed {seed}", fontsize=14, fontweight="bold")

    dim_names = _dim_names(n_dim)
    for d in range(n_dim):
        ax = axes[d]
        # Action
        ax.plot(steps, actions_all[:, d], ".-", markersize=4, linewidth=1.5, color="red", label=f"action_{dim_names[d]}", zorder=5)

        # Goal 1 position (from raw obs, last frame)
        goal1_idx = raw_obs_all.shape[1] - single_frame_dim + 2 * n_dim + d
        ax.plot(steps, raw_obs_all[:, goal1_idx], "--", linewidth=1.5, color="green", label=f"goal1_{dim_names[d]}")

        # Goal 2 position (from raw obs, last frame)
        goal2_idx = raw_obs_all.shape[1] - single_frame_dim + 3 * n_dim + d
        ax.plot(steps, raw_obs_all[:, goal2_idx], "--", linewidth=1.5, color="blue", label=f"goal2_{dim_names[d]}")

        # Agent position
        agent_idx = raw_obs_all.shape[1] - single_frame_dim + d
        ax.plot(steps, raw_obs_all[:, agent_idx], "-", linewidth=1, color="gray", alpha=0.6, label=f"agent_{dim_names[d]}")

        ax.set_ylabel(dim_names[d], fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(output_path / "4_action_vs_goals.png", dpi=150)
    plt.close()

    # ─── PLOT 5: Control points scatter at each step (2D) ───
    if n_dim == 2:
        # Show a grid of snapshots at evenly spaced steps
        snapshot_count = min(12, n_steps)
        snapshot_indices = np.linspace(0, n_steps - 1, snapshot_count, dtype=int)

        cols = 4
        rows = (snapshot_count + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        fig.suptitle(f"Control Points at Selected Steps — Seed {seed}", fontsize=14, fontweight="bold")
        axes_flat = axes.flatten() if snapshot_count > 1 else [axes]

        for i, step_idx in enumerate(snapshot_indices):
            ax = axes_flat[i]
            log = step_logs[step_idx]
            cp = log.control_points  # (N, 2)
            qv = log.q_values        # (N,)

            # Color control points by Q-value
            scatter = ax.scatter(cp[:, 0], cp[:, 1], c=qv, cmap="viridis", s=30, alpha=0.8, zorder=3)

            # Goal positions (from raw obs, last frame)
            raw = log.raw_obs
            g1x = raw[-single_frame_dim + 4]
            g1y = raw[-single_frame_dim + 5]
            g2x = raw[-single_frame_dim + 6]
            g2y = raw[-single_frame_dim + 7]
            ax.scatter(g1x, g1y, c="green", s=120, marker="*", zorder=5, label="goal1")
            ax.scatter(g2x, g2y, c="blue", s=120, marker="*", zorder=5, label="goal2")

            # Agent position
            ax.scatter(raw[-single_frame_dim + 0], raw[-single_frame_dim + 1],
                       c="gray", s=60, marker="o", edgecolors="black", zorder=4, label="agent")

            # Draw chosen action last so it stays on top of all other markers.
            ax.scatter(cp[log.best_idx, 0], cp[log.best_idx, 1],
                       c="red", s=120, marker="*", edgecolors="black", zorder=10, label="chosen action")

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
            ax.set_title(f"Step {log.step}", fontsize=10)
            ax.grid(alpha=0.2)
            if i == 0:
                ax.legend(fontsize=7, loc="upper left")

        # Hide unused axes
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path / "5_control_points_snapshots.png", dpi=150)
        plt.close()
    elif n_dim == 3:
        # 3D-compatible view: plot pairwise projections (xy, xz, yz) at selected steps.
        snapshot_count = min(8, n_steps)
        snapshot_indices = np.linspace(0, n_steps - 1, snapshot_count, dtype=int)
        pairs = [(0, 1), (0, 2), (1, 2)]

        rows = snapshot_count
        cols = len(pairs)
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
        axes = np.atleast_2d(axes)
        fig.suptitle(
            f"Control Points Pairwise Projections (3D) — Seed {seed}",
            fontsize=14,
            fontweight="bold",
        )

        for r, step_idx in enumerate(snapshot_indices):
            log = step_logs[step_idx]
            cp = log.control_points  # (N, 3)
            qv = log.q_values
            raw = log.raw_obs

            base_idx = raw.shape[0] - single_frame_dim
            agent = raw[base_idx: base_idx + n_dim]
            goal1 = raw[base_idx + 2 * n_dim: base_idx + 3 * n_dim]
            goal2 = raw[base_idx + 3 * n_dim: base_idx + 4 * n_dim]
            chosen = cp[log.best_idx]

            for c, (d0, d1) in enumerate(pairs):
                ax = axes[r][c]
                sc = ax.scatter(cp[:, d0], cp[:, d1], c=qv, cmap="viridis", s=28, alpha=0.85, zorder=3)
                ax.scatter(goal1[d0], goal1[d1], c="green", s=110, marker="*", zorder=5, label="goal1")
                ax.scatter(goal2[d0], goal2[d1], c="blue", s=110, marker="*", zorder=5, label="goal2")
                ax.scatter(agent[d0], agent[d1], c="gray", s=50, marker="o", edgecolors="black", zorder=4, label="agent")
                ax.scatter(
                    chosen[d0],
                    chosen[d1],
                    c="red",
                    s=120,
                    marker="*",
                    edgecolors="black",
                    zorder=10,
                    label="chosen action",
                )

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_aspect("equal")
                ax.grid(alpha=0.2)
                ax.set_xlabel(dim_names[d0])
                ax.set_ylabel(dim_names[d1])
                ax.set_title(f"Step {log.step} ({dim_names[d0]}-{dim_names[d1]})", fontsize=9)
                if r == 0 and c == 0:
                    ax.legend(fontsize=7, loc="upper left")

        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Q", fontsize=9)
        plt.tight_layout()
        plt.savefig(output_path / "5_control_points_snapshots.png", dpi=150)
        plt.close()
    else:
        print(f"Skipping Plot 5 (requires 2D or 3D) for n_dim={n_dim}.")

    # ─── PLOT 6: Q-value statistics over time ───
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Q-Value Statistics Over Time — Seed {seed}", fontsize=14, fontweight="bold")

    q_min = q_values_all.min(axis=1)
    q_max = q_values_all.max(axis=1)
    q_mean = q_values_all.mean(axis=1)
    q_std = q_values_all.std(axis=1)
    q_selected = np.array([l.q_values[l.best_idx] for l in step_logs])

    axes[0].fill_between(steps, q_min, q_max, alpha=0.3, color="tab:blue", label="min-max range")
    axes[0].plot(steps, q_mean, "-", color="tab:blue", linewidth=1.5, label="mean Q")
    axes[0].plot(steps, q_selected, ".-", color="red", linewidth=1.5, markersize=4, label="selected Q")
    axes[0].set_ylabel("Q-value")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, q_max - q_min, ".-", color="purple", markersize=3, linewidth=1)
    axes[1].set_ylabel("Q range\n(max - min)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, q_std, ".-", color="teal", markersize=3, linewidth=1)
    axes[2].set_ylabel("Q std")
    axes[2].set_xlabel("Step")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "6_q_value_stats.png", dpi=150)
    plt.close()

    # ─── PLOT 7: 2D trajectory with action arrows ───
    if n_dim == 2:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.suptitle(f"Agent Trajectory & Actions — Seed {seed}", fontsize=14, fontweight="bold")

        # Agent trajectory
        agent_x = raw_obs_all[:, raw_obs_all.shape[1] - single_frame_dim + 0]
        agent_y = raw_obs_all[:, raw_obs_all.shape[1] - single_frame_dim + 1]
        ax.plot(agent_x, agent_y, "-", color="gray", linewidth=1, alpha=0.5, zorder=2)
        scatter_agent = ax.scatter(agent_x, agent_y, c=steps, cmap="Reds", s=20, zorder=3)

        # Action arrows (from agent to action position)
        arrow_step = max(1, n_steps // 20)
        for t in range(0, n_steps, arrow_step):
            dx = actions_all[t, 0] - agent_x[t]
            dy = actions_all[t, 1] - agent_y[t]
            ax.annotate("", xy=(actions_all[t, 0], actions_all[t, 1]),
                        xytext=(agent_x[t], agent_y[t]),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.2, alpha=0.6))

        # Goals (constant, from first step)
        g1x = raw_obs_all[0, raw_obs_all.shape[1] - single_frame_dim + 4]
        g1y = raw_obs_all[0, raw_obs_all.shape[1] - single_frame_dim + 5]
        g2x = raw_obs_all[0, raw_obs_all.shape[1] - single_frame_dim + 6]
        g2y = raw_obs_all[0, raw_obs_all.shape[1] - single_frame_dim + 7]
        ax.scatter(g1x, g1y, c="green", s=200, marker="*", zorder=10, label="Goal 1")
        ax.scatter(g2x, g2y, c="blue", s=200, marker="*", zorder=10, label="Goal 2")

        # Start and end positions
        ax.scatter(agent_x[0], agent_y[0], c="black", marker="s", s=80, zorder=10, label="Start")
        ax.scatter(agent_x[-1], agent_y[-1], c="darkred", marker="D", s=80, zorder=10, label="End")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(scatter_agent, ax=ax, label="Step")
        plt.tight_layout()
        plt.savefig(output_path / "7_trajectory_with_actions.png", dpi=150)
        plt.close()
    elif n_dim == 3:
        # 3D-compatible view: trajectory/action projected to xy, xz, yz planes.
        pairs = [(0, 1), (0, 2), (1, 2)]
        fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5.2), sharex=False, sharey=False)
        axes = np.atleast_1d(axes)
        fig.suptitle(f"Agent Trajectory & Actions Pairwise Projections (3D) — Seed {seed}", fontsize=14, fontweight="bold")

        agent_xyz = raw_obs_all[:, raw_obs_all.shape[1] - single_frame_dim: raw_obs_all.shape[1] - single_frame_dim + n_dim]
        g1 = raw_obs_all[0, raw_obs_all.shape[1] - single_frame_dim + 2 * n_dim: raw_obs_all.shape[1] - single_frame_dim + 3 * n_dim]
        g2 = raw_obs_all[0, raw_obs_all.shape[1] - single_frame_dim + 3 * n_dim: raw_obs_all.shape[1] - single_frame_dim + 4 * n_dim]

        arrow_step = max(1, n_steps // 20)
        for i, (d0, d1) in enumerate(pairs):
            ax = axes[i]
            ax.plot(agent_xyz[:, d0], agent_xyz[:, d1], "-", color="gray", linewidth=1, alpha=0.5, zorder=2)
            scatter_agent = ax.scatter(agent_xyz[:, d0], agent_xyz[:, d1], c=steps, cmap="Reds", s=20, zorder=3)

            for t in range(0, n_steps, arrow_step):
                ax.annotate(
                    "",
                    xy=(actions_all[t, d0], actions_all[t, d1]),
                    xytext=(agent_xyz[t, d0], agent_xyz[t, d1]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.1, alpha=0.6),
                )

            ax.scatter(g1[d0], g1[d1], c="green", s=180, marker="*", zorder=10, label="Goal 1")
            ax.scatter(g2[d0], g2[d1], c="blue", s=180, marker="*", zorder=10, label="Goal 2")
            ax.scatter(agent_xyz[0, d0], agent_xyz[0, d1], c="black", marker="s", s=70, zorder=10, label="Start")
            ax.scatter(agent_xyz[-1, d0], agent_xyz[-1, d1], c="darkred", marker="D", s=70, zorder=10, label="End")

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_xlabel(dim_names[d0])
            ax.set_ylabel(dim_names[d1])
            ax.set_title(f"{dim_names[d0]}-{dim_names[d1]}")
            if i == 0:
                ax.legend(fontsize=9, loc="upper left")
            plt.colorbar(scatter_agent, ax=ax, fraction=0.046, pad=0.04, label="Step")

        plt.tight_layout()
        plt.savefig(output_path / "7_trajectory_with_actions.png", dpi=150)
        plt.close()
    else:
        print(f"Skipping Plot 7 (requires 2D or 3D) for n_dim={n_dim}.")

    # ─── PLOT 8: Action-space Q heatmaps at selected steps (2D) ───
    if n_dim == 2 and q_estimator is not None and ACTION_DIM == 2:
        snapshot_count = min(12, n_steps)
        snapshot_indices = np.linspace(0, n_steps - 1, snapshot_count, dtype=int)

        cfg = dict(env_config.get("model", {}).get("langevin_config", {}))
        if langevin_config is not None:
            cfg.update(langevin_config)
        num_langevin_samples = (
            int(langevin_num_samples)
            if langevin_num_samples is not None
            else int(env_config.get("training", {}).get("counter_examples", 32))
        )

        cols = 4
        rows = (snapshot_count + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.2 * rows))
        fig.suptitle(
            f"Q-Estimator Action-Space Heatmaps at Selected Steps — Seed {seed}",
            fontsize=14,
            fontweight="bold",
        )
        axes_flat = axes.flatten() if snapshot_count > 1 else [axes]

        low, high = float(action_bounds[0]), float(action_bounds[1])
        x_axis = np.linspace(low, high, heatmap_grid_size, dtype=np.float32)
        y_axis = np.linspace(low, high, heatmap_grid_size, dtype=np.float32)
        xx, yy = np.meshgrid(x_axis, y_axis)
        grid_actions_np = np.stack([xx.ravel(), yy.ravel()], axis=-1)

        q_model = q_estimator
        model_device = next(q_model.parameters()).device
        action_min_tensor = torch.full((ACTION_DIM,), low, dtype=torch.float32, device=model_device)
        action_max_tensor = torch.full((ACTION_DIM,), high, dtype=torch.float32, device=model_device)

        for i, step_idx in enumerate(snapshot_indices):
            ax = axes_flat[i]
            log = step_logs[step_idx]

            obs_t = torch.from_numpy(log.normalized_obs).float().unsqueeze(0).to(model_device)
            actions_t = torch.from_numpy(grid_actions_np).float().unsqueeze(0).to(model_device)
            obs_expanded = obs_t.unsqueeze(1).expand(-1, actions_t.shape[1], -1)

            with torch.no_grad():
                q_grid = q_model(obs_expanded, actions_t).squeeze(-1).squeeze(0).cpu().numpy()

            traj_np = None
            if show_langevin_trajectories or show_langevin_final_positions:
                # Recreate Langevin counter-example trajectories for this state and overlay them.
                obs_single = obs_t

                def energy_fn(obs_batch: torch.Tensor, act_batch: torch.Tensor) -> torch.Tensor:
                    return -q_model(obs_batch, act_batch).squeeze(-1)

                _, traj_steps = sample_langevin(
                    energy_function=energy_fn,
                    observations=obs_single,
                    num_samples=num_langevin_samples,
                    action_min=action_min_tensor,
                    action_max=action_max_tensor,
                    num_iterations=int(cfg.get("num_iterations", 100)),
                    lr_init=float(cfg.get("lr_init", 0.1)),
                    lr_final=float(cfg.get("lr_final", 1e-5)),
                    polynomial_decay_power=float(cfg.get("polynomial_decay_power", 2.0)),
                    delta_action_clip=float(cfg.get("delta_action_clip", 0.1)),
                    noise_scale=float(cfg.get("noise_scale", 1.0)),
                    return_trajectories=True,
                    device=model_device,
                )

                traj_np = torch.stack(traj_steps, dim=0).squeeze(1).cpu().numpy()  # (K, N, 2)

            q_map = q_grid.reshape(heatmap_grid_size, heatmap_grid_size)
            im = ax.imshow(
                q_map,
                origin="lower",
                extent=[low, high, low, high],
                cmap="viridis",
                aspect="equal",
            )

            if traj_np is not None:
                for traj_idx in range(traj_np.shape[1]):
                    path_xy = traj_np[:, traj_idx, :]
                    if show_langevin_trajectories:
                        ax.plot(
                            path_xy[:, 0],
                            path_xy[:, 1],
                            color="white",
                            alpha=0.35,
                            linewidth=0.8,
                            zorder=5,
                            label="Langevin counterexample traj" if (i == 0 and traj_idx == 0) else None,
                        )
                        if i == 0:
                            ax.scatter(path_xy[0, 0], path_xy[0, 1], c="white", s=12, alpha=0.45, zorder=6)

                    if show_langevin_final_positions:
                        # Highlight final Langevin positions with a larger high-contrast marker
                        # so they remain visible on top of the Q heatmap.
                        ax.scatter(
                            path_xy[-1, 0],
                            path_xy[-1, 1],
                            c="yellow",
                            s=46,
                            alpha=0.95,
                            marker="X",
                            edgecolors="black",
                            linewidths=0.65,
                            zorder=7,
                            label="Langevin final" if (i == 0 and traj_idx == 0) else None,
                        )

            raw = log.raw_obs
            ax_x = raw[-single_frame_dim + 0]
            ax_y = raw[-single_frame_dim + 1]
            g1x = raw[-single_frame_dim + 4]
            g1y = raw[-single_frame_dim + 5]
            g2x = raw[-single_frame_dim + 6]
            g2y = raw[-single_frame_dim + 7]

            ax.scatter(log.action[0], log.action[1], c="red", s=95, marker="*",
                       edgecolors="black", linewidths=0.8, zorder=6, label="chosen action")
            ax.scatter(ax_x, ax_y, c="white", s=70, marker="o", edgecolors="black",
                       linewidths=0.8, zorder=6, label="agent")
            ax.scatter(g1x, g1y, c="lime", s=110, marker="*", edgecolors="black",
                       linewidths=0.6, zorder=6, label="goal1")
            ax.scatter(g2x, g2y, c="cyan", s=110, marker="*", edgecolors="black",
                       linewidths=0.6, zorder=6, label="goal2")

            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_title(f"Step {log.step}", fontsize=10)
            ax.grid(alpha=0.15)
            if i == 0:
                ax.legend(fontsize=7, loc="upper left")

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label("Q", fontsize=8)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path / "8_q_heatmap_snapshots.png", dpi=150)
        plt.close()
    elif n_dim == 3 and q_estimator is not None and ACTION_DIM == 3:
        # 3D-compatible view: pairwise Q heatmaps with the third action dim fixed per snapshot.
        snapshot_count = min(6, n_steps)
        snapshot_indices = np.linspace(0, n_steps - 1, snapshot_count, dtype=int)
        pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]  # (x-axis dim, y-axis dim, fixed dim)

        cfg = dict(env_config.get("model", {}).get("langevin_config", {}))
        if langevin_config is not None:
            cfg.update(langevin_config)
        num_langevin_samples = (
            int(langevin_num_samples)
            if langevin_num_samples is not None
            else int(env_config.get("training", {}).get("counter_examples", 32))
        )

        rows = snapshot_count
        cols = len(pairs)
        fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.9 * rows))
        axes = np.atleast_2d(axes)
        fig.suptitle(
            f"Q-Estimator Pairwise Action-Space Slices (3D) — Seed {seed}",
            fontsize=14,
            fontweight="bold",
        )

        low, high = float(action_bounds[0]), float(action_bounds[1])
        axis_vals = np.linspace(low, high, heatmap_grid_size, dtype=np.float32)
        xx, yy = np.meshgrid(axis_vals, axis_vals)

        q_model = q_estimator
        model_device = next(q_model.parameters()).device
        action_min_tensor = torch.full((ACTION_DIM,), low, dtype=torch.float32, device=model_device)
        action_max_tensor = torch.full((ACTION_DIM,), high, dtype=torch.float32, device=model_device)

        for r, step_idx in enumerate(snapshot_indices):
            log = step_logs[step_idx]
            obs_t = torch.from_numpy(log.normalized_obs).float().unsqueeze(0).to(model_device)

            traj_np = None
            if show_langevin_trajectories or show_langevin_final_positions:
                def energy_fn(obs_batch: torch.Tensor, act_batch: torch.Tensor) -> torch.Tensor:
                    return -q_model(obs_batch, act_batch).squeeze(-1)

                _, traj_steps = sample_langevin(
                    energy_function=energy_fn,
                    observations=obs_t,
                    num_samples=num_langevin_samples,
                    action_min=action_min_tensor,
                    action_max=action_max_tensor,
                    num_iterations=int(cfg.get("num_iterations", 100)),
                    lr_init=float(cfg.get("lr_init", 0.1)),
                    lr_final=float(cfg.get("lr_final", 1e-5)),
                    polynomial_decay_power=float(cfg.get("polynomial_decay_power", 2.0)),
                    delta_action_clip=float(cfg.get("delta_action_clip", 0.1)),
                    noise_scale=float(cfg.get("noise_scale", 1.0)),
                    return_trajectories=True,
                    device=model_device,
                )
                traj_np = torch.stack(traj_steps, dim=0).squeeze(1).cpu().numpy()  # (K, N, 3)

            raw = log.raw_obs
            base_idx = raw.shape[0] - single_frame_dim
            agent = raw[base_idx: base_idx + n_dim]
            goal1 = raw[base_idx + 2 * n_dim: base_idx + 3 * n_dim]
            goal2 = raw[base_idx + 3 * n_dim: base_idx + 4 * n_dim]
            chosen = log.action

            for c, (dx, dy, dfixed) in enumerate(pairs):
                ax = axes[r][c]

                # Build 2D slice in (dx,dy), fixing the remaining dim to chosen action at this step.
                fixed_val = float(chosen[dfixed])
                grid_actions = np.zeros((heatmap_grid_size * heatmap_grid_size, ACTION_DIM), dtype=np.float32)
                grid_actions[:, dx] = xx.ravel()
                grid_actions[:, dy] = yy.ravel()
                grid_actions[:, dfixed] = fixed_val

                actions_t = torch.from_numpy(grid_actions).float().unsqueeze(0).to(model_device)
                obs_expanded = obs_t.unsqueeze(1).expand(-1, actions_t.shape[1], -1)
                with torch.no_grad():
                    q_grid = q_model(obs_expanded, actions_t).squeeze(-1).squeeze(0).cpu().numpy()

                q_map = q_grid.reshape(heatmap_grid_size, heatmap_grid_size)
                im = ax.imshow(
                    q_map,
                    origin="lower",
                    extent=[low, high, low, high],
                    cmap="viridis",
                    aspect="equal",
                )

                if traj_np is not None:
                    for traj_idx in range(traj_np.shape[1]):
                        path_xy = traj_np[:, traj_idx, [dx, dy]]
                        if show_langevin_trajectories:
                            ax.plot(
                                path_xy[:, 0],
                                path_xy[:, 1],
                                color="white",
                                alpha=0.35,
                                linewidth=0.8,
                                zorder=5,
                                label="Langevin traj" if (r == 0 and c == 0 and traj_idx == 0) else None,
                            )
                        if show_langevin_final_positions:
                            ax.scatter(
                                path_xy[-1, 0],
                                path_xy[-1, 1],
                                c="yellow",
                                s=42,
                                alpha=0.95,
                                marker="X",
                                edgecolors="black",
                                linewidths=0.6,
                                zorder=7,
                                label="Langevin final" if (r == 0 and c == 0 and traj_idx == 0) else None,
                            )

                ax.scatter(chosen[dx], chosen[dy], c="red", s=95, marker="*", edgecolors="black", linewidths=0.8, zorder=8, label="chosen action")
                ax.scatter(agent[dx], agent[dy], c="white", s=65, marker="o", edgecolors="black", linewidths=0.8, zorder=8, label="agent")
                ax.scatter(goal1[dx], goal1[dy], c="lime", s=100, marker="*", edgecolors="black", linewidths=0.6, zorder=8, label="goal1")
                ax.scatter(goal2[dx], goal2[dy], c="cyan", s=100, marker="*", edgecolors="black", linewidths=0.6, zorder=8, label="goal2")

                ax.set_xlim(low, high)
                ax.set_ylim(low, high)
                ax.set_title(
                    f"Step {log.step}: {dim_names[dx]}-{dim_names[dy]} | {dim_names[dfixed]}={fixed_val:.2f}",
                    fontsize=9,
                )
                ax.grid(alpha=0.15)
                if r == 0 and c == 0:
                    ax.legend(fontsize=7, loc="upper left")

                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
                cbar.set_label("Q", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path / "8_q_heatmap_snapshots.png", dpi=150)
        plt.close()
    else:
        if n_dim not in (2, 3):
            print(f"Skipping Plot 8 (implemented for 2D/3D) for n_dim={n_dim}.")
        elif ACTION_DIM != n_dim:
            print(f"Skipping Plot 8 because ACTION_DIM={ACTION_DIM} does not match n_dim={n_dim}.")
        elif q_estimator is None:
            print("Skipping Plot 8 because q_estimator is None.")

    print(f"Diagnostic plots saved to: {output_path.absolute()}")


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


def main():
    parser = argparse.ArgumentParser(
        description="Run particle simulation with diagnostic logging and plots"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=config["simulation"]["default_checkpoint"],
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Seeds for episodes (default: config simulation.default_seeds)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots/particle_diagnostic",
        help="Directory to save diagnostic plots",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render the environment during simulation",
    )
    parser.add_argument(
        "--show-langevin-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show Langevin trajectory lines in Plot 8 (use --no-show-langevin-trajectories to hide)",
    )
    parser.add_argument(
        "--show-langevin-final-positions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show Langevin final position markers in Plot 8 (use --no-show-langevin-final-positions to hide)",
    )
    parser.add_argument(
        "--hide-langevin-overlays",
        action="store_true",
        help="Hide both Langevin trajectories and final position markers in Plot 8",
    )
    args = parser.parse_args()
    seeds_to_run = args.seeds if args.seeds is not None else config["simulation"].get("default_seeds", [42])

    show_langevin_trajectories = args.show_langevin_trajectories
    show_langevin_final_positions = args.show_langevin_final_positions
    if args.hide_langevin_overlays:
        show_langevin_trajectories = False
        show_langevin_final_positions = False

    checkpoint_dir = os.path.dirname(args.checkpoint)
    q_estimator_path = os.path.join(checkpoint_dir, "q_estimator.pt")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at '{args.checkpoint}'")
        sys.exit(1)
    if not os.path.exists(q_estimator_path):
        print(f"Error: Q-estimator not found at '{q_estimator_path}'")
        sys.exit(1)

    # Load models
    print(f"Loading models from {checkpoint_dir}...")
    model = load_model(args.checkpoint)
    q_estimator = load_q_estimator(q_estimator_path)

    n_dim = env_config.get("n_dim", 2)
    max_steps = config["simulation"].get("max_episode_steps", 200)
    render_mode = "human" if args.render else None

    print(f"Running diagnostics for seeds: {seeds_to_run}")

    for seed in seeds_to_run:
        # Create a new simulation per seed so logs are isolated.
        sim = DiagnosticParticleSimulation(
            control_point_generator=model,
            q_estimator=q_estimator,
            n_dim=n_dim,
            device="cpu",
            max_episode_steps=max_steps,
            render_mode=render_mode,
            frame_stack=FRAME_STACK,
        )

        print(f"\nRunning episode with seed {seed}...")
        result = sim.run_episode(seed=seed)

        print(f"  Episode length: {result['episode_length']}")
        print(f"  Total reward:   {result['total_reward']}")
        print(f"  Success:        {result.get('success', False)}")
        print(f"  Min dist goal1: {result.get('min_dist_to_first_goal', 'N/A'):.6f}")
        print(f"  Min dist goal2: {result.get('min_dist_to_second_goal', 'N/A'):.6f}")
        print(f"  Steps logged:   {len(sim.step_logs)}")

        seed_output_dir = os.path.join(args.output_dir, str(n_dim), str(seed))
        if os.path.exists(seed_output_dir):
            import shutil
            shutil.rmtree(seed_output_dir)

        generate_diagnostic_plots(
            step_logs=sim.step_logs,
            output_dir=seed_output_dir,
            seed=seed,
            q_estimator=q_estimator,
            action_bounds=ACTION_BOUNDS,
            n_dim=n_dim,
            langevin_config=env_config.get("model", {}).get("langevin_config", {}),
            langevin_num_samples=env_config.get("training", {}).get("counter_examples", 32),
            show_langevin_trajectories=show_langevin_trajectories,
            show_langevin_final_positions=show_langevin_final_positions,
        )

        sim.close()

    print("Done!")


if __name__ == "__main__":
    main()
