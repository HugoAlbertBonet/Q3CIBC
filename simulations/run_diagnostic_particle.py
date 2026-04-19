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
from itertools import combinations
import json
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
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


def _pairwise_dims(n_dim: int) -> list[tuple[int, int]]:
    """Return pairwise dimension indices for projection plotting."""
    return list(combinations(range(n_dim), 2))


def _linear_weight_layers(state_dict: dict[str, torch.Tensor], prefix: str = "network") -> list[tuple[int, torch.Tensor]]:
    """Return sequential linear layer weights sorted by layer index.

    Keys are expected like `network.0.weight`, `network.2.weight`, etc.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    layers: list[tuple[int, torch.Tensor]] = []
    for key, tensor in state_dict.items():
        match = pattern.match(key)
        if match and tensor.ndim == 2:
            layers.append((int(match.group(1)), tensor))
    layers.sort(key=lambda x: x[0])
    if not layers:
        raise ValueError("No linear layer weights found in checkpoint state_dict.")
    return layers


def _infer_generator_arch(cp_sd: dict[str, torch.Tensor], action_dim: int) -> tuple[int, int, list[int]]:
    """Infer generator input dim, control points, and hidden dims from state dict."""
    layers = _linear_weight_layers(cp_sd, prefix="network")
    first_w = layers[0][1]
    last_w = layers[-1][1]

    input_dim = int(first_w.shape[1])
    output_features = int(last_w.shape[0])
    if action_dim <= 0 or output_features % action_dim != 0:
        raise ValueError(
            f"Cannot infer control_points: last layer out_features={output_features} not divisible by action_dim={action_dim}."
        )
    control_points = output_features // action_dim
    hidden_dims = [int(w.shape[0]) for _, w in layers[:-1]]
    return input_dim, control_points, hidden_dims


def _infer_q_arch(q_sd: dict[str, torch.Tensor], action_dim: int, fallback_state_input_dim: int) -> tuple[int, list[int]]:
    """Infer Q-estimator state dim and hidden dims from state dict."""
    layers = _linear_weight_layers(q_sd, prefix="network")
    first_w = layers[0][1]
    first_in = int(first_w.shape[1])

    state_dim = first_in - action_dim
    if state_dim <= 0:
        state_dim = fallback_state_input_dim

    hidden_dims = [int(w.shape[0]) for _, w in layers[:-1]]
    return state_dim, hidden_dims


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

            if self._act_min_t is not None:
                cp_for_q = (control_points - self._act_min_t) / self._act_rng_t
            else:
                cp_for_q = control_points

            q_values = self.q_estimator(obs_expanded, cp_for_q).squeeze(-1)  # (1, N)

            best_idx = q_values.argmax(dim=1)
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
    goal_distance_threshold: float = 0.05,
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

    # ─── PLOT 5: Control point snapshots for any action dimensionality ───
    action_dim = cp_all.shape[-1]
    snapshot_count = min(12 if action_dim <= 2 else 8, n_steps)
    snapshot_indices = np.linspace(0, n_steps - 1, snapshot_count, dtype=int)

    if action_dim == 1:
        fig, axes = plt.subplots(snapshot_count, 1, figsize=(11, max(3.0, 2.5 * snapshot_count)), sharex=True)
        axes = np.atleast_1d(axes)
        fig.suptitle(f"Control Points (1D Action Space) at Selected Steps — Seed {seed}", fontsize=14, fontweight="bold")

        for r, step_idx in enumerate(snapshot_indices):
            ax = axes[r]
            log = step_logs[step_idx]
            cp = log.control_points[:, 0]
            qv = log.q_values
            y = np.zeros_like(cp)

            sc = ax.scatter(cp, y, c=qv, cmap="viridis", s=34, alpha=0.85, zorder=3)
            ax.scatter(log.control_points[log.best_idx, 0], 0.0, c="red", s=120, marker="*", edgecolors="black", zorder=6, label="chosen action")
            ax.axvline(float(log.action[0]), color="red", linestyle="--", linewidth=1.0, alpha=0.8)

            ax.set_xlim(-0.05, 1.05)
            ax.set_yticks([])
            ax.set_ylabel(f"Step {log.step}")
            ax.grid(alpha=0.2)
            if r == 0:
                ax.legend(fontsize=8, loc="upper right")

        axes[-1].set_xlabel("action")
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Q", fontsize=9)
        plt.tight_layout()
        plt.savefig(output_path / "5_control_points_snapshots.png", dpi=150)
        plt.close()
    else:
        pairs = _pairwise_dims(action_dim)
        rows = snapshot_count
        cols = len(pairs)
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
        axes = np.array(axes, dtype=object).reshape(rows, cols)
        fig.suptitle(
            f"Control Points Pairwise Projections ({action_dim}D) — Seed {seed}",
            fontsize=14,
            fontweight="bold",
        )

        base_idx = raw_obs_all.shape[1] - single_frame_dim
        for r, step_idx in enumerate(snapshot_indices):
            log = step_logs[step_idx]
            cp = log.control_points
            qv = log.q_values
            raw = log.raw_obs
            agent = raw[base_idx: base_idx + n_dim]
            goal1 = raw[base_idx + 2 * n_dim: base_idx + 3 * n_dim]
            goal2 = raw[base_idx + 3 * n_dim: base_idx + 4 * n_dim]
            chosen = cp[log.best_idx]

            for c, (d0, d1) in enumerate(pairs):
                ax = axes[r][c]
                sc = ax.scatter(cp[:, d0], cp[:, d1], c=qv, cmap="viridis", s=28, alpha=0.85, zorder=3)

                if d0 < n_dim and d1 < n_dim:
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

    # ─── PLOT 7: Trajectory with action arrows for any dimensionality ───
    agent_xyz = raw_obs_all[:, raw_obs_all.shape[1] - single_frame_dim: raw_obs_all.shape[1] - single_frame_dim + n_dim]
    g1 = raw_obs_all[0, raw_obs_all.shape[1] - single_frame_dim + 2 * n_dim: raw_obs_all.shape[1] - single_frame_dim + 3 * n_dim]
    g2 = raw_obs_all[0, raw_obs_all.shape[1] - single_frame_dim + 3 * n_dim: raw_obs_all.shape[1] - single_frame_dim + 4 * n_dim]

    if n_dim == 1:
        fig, ax = plt.subplots(figsize=(12, 4.8))
        fig.suptitle(f"Agent Trajectory & Actions (1D) — Seed {seed}", fontsize=14, fontweight="bold")

        ax.plot(steps, agent_xyz[:, 0], "-", color="gray", linewidth=1.3, alpha=0.8, label="agent")
        scatter_agent = ax.scatter(steps, agent_xyz[:, 0], c=steps, cmap="Reds", s=20, zorder=3)
        ax.plot(steps, actions_all[:, 0], "-", color="red", linewidth=1.2, alpha=0.9, label="action")

        arrow_step = max(1, n_steps // 20)
        for t in range(0, n_steps, arrow_step):
            ax.annotate(
                "",
                xy=(steps[t], actions_all[t, 0]),
                xytext=(steps[t], agent_xyz[t, 0]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.0, alpha=0.6),
            )

        ax.axhline(g1[0], color="green", linestyle="--", linewidth=1.2, label="goal1")
        ax.axhline(g2[0], color="blue", linestyle="--", linewidth=1.2, label="goal2")
        ax.scatter(steps[0], agent_xyz[0, 0], c="black", marker="s", s=70, zorder=10, label="start")
        ax.scatter(steps[-1], agent_xyz[-1, 0], c="darkred", marker="D", s=70, zorder=10, label="end")

        ax.set_xlabel("Step")
        ax.set_ylabel(dim_names[0])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
        plt.colorbar(scatter_agent, ax=ax, label="Step")
        plt.tight_layout()
        plt.savefig(output_path / "7_trajectory_with_actions.png", dpi=150)
        plt.close()
    else:
        pairs = _pairwise_dims(n_dim)
        fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5.2), sharex=False, sharey=False)
        axes = np.atleast_1d(axes)
        fig.suptitle(f"Agent Trajectory & Actions Pairwise Projections ({n_dim}D) — Seed {seed}", fontsize=14, fontweight="bold")

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

    # ─── PLOT 8: Action-space Q slices for any action dimensionality ───
    if q_estimator is not None:
        q_action_dim = actions_all.shape[1]
        cfg = dict(env_config.get("model", {}).get("langevin_config", {}))
        if langevin_config is not None:
            cfg.update(langevin_config)
        num_langevin_samples = (
            int(langevin_num_samples)
            if langevin_num_samples is not None
            else int(env_config.get("training", {}).get("counter_examples", 32))
        )

        low, high = float(action_bounds[0]), float(action_bounds[1])
        q_model = q_estimator
        model_device = next(q_model.parameters()).device
        action_min_tensor = torch.full((q_action_dim,), low, dtype=torch.float32, device=model_device)
        action_max_tensor = torch.full((q_action_dim,), high, dtype=torch.float32, device=model_device)

        if q_action_dim == 1:
            snapshot_count = min(12, n_steps)
            snapshot_indices = np.linspace(0, n_steps - 1, snapshot_count, dtype=int)
            x_axis = np.linspace(low, high, heatmap_grid_size, dtype=np.float32)

            fig, axes = plt.subplots(snapshot_count, 1, figsize=(10.5, max(3.5, 2.8 * snapshot_count)), sharex=True)
            axes = np.atleast_1d(axes)
            fig.suptitle(f"Q-Estimator Action-Space Curves (1D) — Seed {seed}", fontsize=14, fontweight="bold")

            for r, step_idx in enumerate(snapshot_indices):
                ax = axes[r]
                log = step_logs[step_idx]
                obs_t = torch.from_numpy(log.normalized_obs).float().unsqueeze(0).to(model_device)

                grid_actions = x_axis[:, None]
                actions_t = torch.from_numpy(grid_actions).float().unsqueeze(0).to(model_device)
                obs_expanded = obs_t.unsqueeze(1).expand(-1, actions_t.shape[1], -1)

                with torch.no_grad():
                    q_curve = q_model(obs_expanded, actions_t).squeeze(-1).squeeze(0).cpu().numpy()

                ax.plot(x_axis, q_curve, color="tab:blue", linewidth=1.4)
                ax.axvline(float(log.action[0]), color="red", linestyle="--", linewidth=1.2, label="chosen action")
                ax.scatter([log.action[0]], [np.interp(log.action[0], x_axis, q_curve)], c="red", s=38, zorder=5)
                ax.set_ylabel(f"Q (step {log.step})")
                ax.grid(alpha=0.2)
                if r == 0:
                    ax.legend(fontsize=8, loc="upper left")

            axes[-1].set_xlabel("action")
            plt.tight_layout()
            plt.savefig(output_path / "8_q_heatmap_snapshots.png", dpi=150)
            plt.close()
        else:
            snapshot_count = min(6, n_steps)
            snapshot_indices = np.linspace(0, n_steps - 1, snapshot_count, dtype=int)
            pairs = _pairwise_dims(q_action_dim)

            rows = snapshot_count
            cols = len(pairs)
            fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.9 * rows))
            axes = np.array(axes, dtype=object).reshape(rows, cols)
            fig.suptitle(
                f"Q-Estimator Pairwise Action-Space Slices ({q_action_dim}D) — Seed {seed}",
                fontsize=14,
                fontweight="bold",
            )

            axis_vals = np.linspace(low, high, heatmap_grid_size, dtype=np.float32)
            xx, yy = np.meshgrid(axis_vals, axis_vals)

            for r, step_idx in enumerate(snapshot_indices):
                log = step_logs[step_idx]
                obs_t = torch.from_numpy(log.normalized_obs).float().unsqueeze(0).to(model_device)

                traj_np = None
                if show_langevin_trajectories or show_langevin_final_positions:
                    # Q-value convention: sample_langevin descends its energy
                    # argument, so negate Q to make Langevin ascend Q (find
                    # hard negatives — actions the model rates highly).
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
                    traj_np = torch.stack(traj_steps, dim=0).squeeze(1).cpu().numpy()

                raw = log.raw_obs
                base_idx = raw.shape[0] - single_frame_dim
                agent = raw[base_idx: base_idx + n_dim]
                goal1 = raw[base_idx + 2 * n_dim: base_idx + 3 * n_dim]
                goal2 = raw[base_idx + 3 * n_dim: base_idx + 4 * n_dim]
                chosen = log.action

                for c, (dx, dy) in enumerate(pairs):
                    ax = axes[r][c]

                    # Build a 2D slice in (dx, dy), fixing all remaining action dims to chosen action.
                    grid_actions = np.tile(chosen[None, :], (heatmap_grid_size * heatmap_grid_size, 1)).astype(np.float32)
                    grid_actions[:, dx] = xx.ravel()
                    grid_actions[:, dy] = yy.ravel()

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
                    if dx < n_dim and dy < n_dim:
                        ax.scatter(agent[dx], agent[dy], c="white", s=65, marker="o", edgecolors="black", linewidths=0.8, zorder=8, label="agent")
                        ax.scatter(goal1[dx], goal1[dy], c="lime", s=100, marker="*", edgecolors="black", linewidths=0.6, zorder=8, label="goal1")
                        ax.scatter(goal2[dx], goal2[dy], c="cyan", s=100, marker="*", edgecolors="black", linewidths=0.6, zorder=8, label="goal2")

                    fixed_dims = [k for k in range(q_action_dim) if k not in (dx, dy)]
                    fixed_desc = ", ".join(f"{dim_names[k]}={chosen[k]:.2f}" for k in fixed_dims) if fixed_dims else ""
                    title = f"Step {log.step}: {dim_names[dx]}-{dim_names[dy]}"
                    if fixed_desc:
                        title += f" | {fixed_desc}"
                    ax.set_title(title, fontsize=9)
                    ax.set_xlim(low, high)
                    ax.set_ylim(low, high)
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
        print("Skipping Plot 8 because q_estimator is None.")

    # ─── PLOT 9: Distance from closest control point to each goal over time ───
    cp_action_dim = cp_all.shape[-1]
    used_dims = min(cp_action_dim, n_dim)

    # Goals from current frame of stacked observation.
    goal1_all = raw_obs_all[:, raw_obs_all.shape[1] - single_frame_dim + 2 * n_dim: raw_obs_all.shape[1] - single_frame_dim + 3 * n_dim]
    goal2_all = raw_obs_all[:, raw_obs_all.shape[1] - single_frame_dim + 3 * n_dim: raw_obs_all.shape[1] - single_frame_dim + 4 * n_dim]

    cp_used = cp_all[:, :, :used_dims]                       # (T, N, used_dims)
    goal1_used = goal1_all[:, :used_dims][:, None, :]       # (T, 1, used_dims)
    goal2_used = goal2_all[:, :used_dims][:, None, :]       # (T, 1, used_dims)

    dist_cp_to_goal1 = np.linalg.norm(cp_used - goal1_used, axis=-1)  # (T, N)
    dist_cp_to_goal2 = np.linalg.norm(cp_used - goal2_used, axis=-1)  # (T, N)

    min_dist_goal1 = dist_cp_to_goal1.min(axis=1)  # (T,)
    min_dist_goal2 = dist_cp_to_goal2.min(axis=1)  # (T,)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Closest Control-Point Distance to Goals — Seed {seed}",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].plot(steps, min_dist_goal1, ".-", color="green", linewidth=1.4, markersize=4, label="min dist to goal1")
    axes[0].axhline(goal_distance_threshold, color="black", linestyle="--", linewidth=1.2, label=f"threshold={goal_distance_threshold:.3f}")
    axes[0].set_ylabel("distance")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper right")

    axes[1].plot(steps, min_dist_goal2, ".-", color="blue", linewidth=1.4, markersize=4, label="min dist to goal2")
    axes[1].axhline(goal_distance_threshold, color="black", linestyle="--", linewidth=1.2, label=f"threshold={goal_distance_threshold:.3f}")
    axes[1].set_ylabel("distance")
    axes[1].set_xlabel("Step")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9, loc="upper right")

    if used_dims < n_dim:
        fig.text(
            0.5,
            0.01,
            f"Note: distances use first {used_dims} dims because action_dim={cp_action_dim} and n_dim={n_dim}.",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path / "9_closest_control_point_goal_distance.png", dpi=150)
    plt.close()

    # ─── PLOT 10: Agent distance summary to control points + goals over time ───
    agent_all = raw_obs_all[:, raw_obs_all.shape[1] - single_frame_dim: raw_obs_all.shape[1] - single_frame_dim + n_dim]
    cp_action_dim = cp_all.shape[-1]
    used_dims = min(cp_action_dim, n_dim)

    agent_used = agent_all[:, :used_dims][:, None, :]  # (T, 1, used_dims)
    cp_used = cp_all[:, :, :used_dims]                 # (T, N, used_dims)
    dist_agent_to_cp = np.linalg.norm(cp_used - agent_used, axis=-1)  # (T, N)

    fig, ax_stats = plt.subplots(figsize=(14, 5.2))
    fig.suptitle(f"Agent Distances to Control Points and Goals — Seed {seed}", fontsize=14, fontweight="bold")

    dist_min = dist_agent_to_cp.min(axis=1)
    dist_mean = dist_agent_to_cp.mean(axis=1)
    dist_max = dist_agent_to_cp.max(axis=1)
    selected_cp_dist = np.array([
        dist_agent_to_cp[t, best_idx_all[t]] for t in range(n_steps)
    ])

    goal1_used_direct = goal1_all[:, :used_dims]
    goal2_used_direct = goal2_all[:, :used_dims]
    dist_agent_goal1 = np.linalg.norm(agent_all[:, :used_dims] - goal1_used_direct, axis=1)
    dist_agent_goal2 = np.linalg.norm(agent_all[:, :used_dims] - goal2_used_direct, axis=1)

    ax_stats.plot(steps, dist_min, ".-", color="tab:green", linewidth=1.2, markersize=3, label="min over CPs")
    ax_stats.plot(steps, selected_cp_dist, ".-", color="tab:red", linewidth=1.2, markersize=3, label="selected CP")
    ax_stats.plot(steps, dist_mean, "-", color="tab:blue", linewidth=1.2, alpha=0.9, label="mean over CPs")
    ax_stats.plot(steps, dist_max, "-", color="gray", linewidth=1.0, alpha=0.8, label="max over CPs")
    ax_stats.plot(steps, dist_agent_goal1, "--", color="green", linewidth=1.4, alpha=0.9, label="agent to goal1")
    ax_stats.plot(steps, dist_agent_goal2, "--", color="purple", linewidth=1.4, alpha=0.9, label="agent to goal2")
    ax_stats.set_xlabel("Step")
    ax_stats.set_ylabel("Distance")
    ax_stats.set_ylim(0.0, 0.06)
    ax_stats.grid(alpha=0.25)
    ax_stats.legend(fontsize=9, loc="upper right")

    if used_dims < n_dim:
        fig.text(
            0.5,
            0.01,
            f"Note: agent-to-control-point distances use first {used_dims} dims because action_dim={cp_action_dim} and n_dim={n_dim}.",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path / "10_agent_to_control_point_distance.png", dpi=150)
    plt.close()

    print(f"Diagnostic plots saved to: {output_path.absolute()}")


def load_model(checkpoint_path: str, device: str = "cpu") -> ControlPointGenerator:
    """Load a trained control point generator from checkpoint."""
    cp_sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
    input_dim, inferred_control_points, hidden_dims = _infer_generator_arch(cp_sd, ACTION_DIM)

    expected_input_dim = STATE_DIM * FRAME_STACK
    if input_dim != expected_input_dim:
        print(
            "Warning: checkpoint input_dim differs from config "
            f"({input_dim} vs {expected_input_dim}). Using checkpoint-inferred architecture."
        )

    model = ControlPointGenerator(
        input_dim=input_dim,
        output_dim=ACTION_DIM,
        control_points=inferred_control_points,
        hidden_dims=hidden_dims,
        action_bounds=ACTION_BOUNDS,
    )
    model.load_state_dict(cp_sd)
    model.to(device)
    model.eval()
    return model


def load_q_estimator(checkpoint_path: str, device: str = "cpu") -> QEstimator:
    """Load a trained Q-estimator from checkpoint."""
    q_sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
    fallback_state_input_dim = STATE_DIM * FRAME_STACK
    inferred_state_dim, hidden_dims = _infer_q_arch(q_sd, ACTION_DIM, fallback_state_input_dim)

    if inferred_state_dim != fallback_state_input_dim:
        print(
            "Warning: Q-estimator state_dim differs from config "
            f"({inferred_state_dim} vs {fallback_state_input_dim}). Using checkpoint-inferred architecture."
        )

    model = QEstimator(
        state_dim=inferred_state_dim,
        action_dim=ACTION_DIM,
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(q_sd)
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
    norm_stats_path = os.path.join(checkpoint_dir, "norm_stats.pt")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at '{args.checkpoint}'")
        sys.exit(1)
    if not os.path.exists(q_estimator_path):
        print(f"Error: Q-estimator not found at '{q_estimator_path}'")
        sys.exit(1)

    # Presence of norm_stats.pt = checkpoint was trained by ibc_with_cps_training.py
    # (actions normalized to [0,1] before the Q estimator sees them).
    norm_stats = None
    if os.path.exists(norm_stats_path):
        norm_stats = torch.load(norm_stats_path, weights_only=False)
        print(f"Detected norm_stats.pt -> applying action normalization for Q estimator")

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
            norm_stats=norm_stats,
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
            goal_distance_threshold=float(env_config.get("goal_distance", 0.05)),
        )

        sim.close()

    print("Done!")


if __name__ == "__main__":
    main()
