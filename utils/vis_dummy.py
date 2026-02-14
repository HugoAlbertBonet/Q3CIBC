
"""Visualization tools for Dummy 2D Grid Navigation Environment.

Generates 5 diagnostic plots per snapshot:
1. Radial CP + Expert (Q-value weighted)
2. Q-value heatmap (polar)
3. Probability polar (InfoNCE softmax)
4. Langevin evolution (polar)
5. 2D navigation map (trajectory + goal)
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import os


def plot_dummy_debug(
    model,
    estimator,
    device,
    save_path,
    state,
    trajectory,
    goal,
    agent_pos,
    step_idx,
    episode_idx,
    langevin_fn=None,
    title="Dummy Diagnostic",
):
    """
    Generates a 1x5 row (or 2-row layout) of diagnostic plots for one snapshot.

    Args:
        model: ControlPointGenerator (eval mode).
        estimator: QEstimator (eval mode).
        device: torch device.
        save_path: where to save the figure.
        state: state tensor (1, state_dim) — the observation fed to the model.
        trajectory: list of (x, y) positions up to current step.
        goal: (2,) numpy array — goal position.
        agent_pos: (2,) numpy array — current agent position.
        step_idx: current step within the episode.
        episode_idx: current episode index.
        langevin_fn: callable(model, estimator, device, state) -> (samples, trajs).
        title: figure title prefix.
    """
    model.eval()
    estimator.eval()

    # Ensure state is (1, D) tensor
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float().to(device)
    if state.ndim == 1:
        state = state.unsqueeze(0)

    with torch.no_grad():
        # Control points: (1, N, 1) -> angles in [-1, 1]
        control_points = model(state)
        cp_actions = control_points.squeeze(0).cpu().numpy()  # (N, 1)
        cp_angles_rad = cp_actions[:, 0] * np.pi  # Map to [-π, π]

        # Expert action: optimal angle towards goal
        diff = goal - agent_pos
        expert_angle_rad = np.arctan2(diff[1], diff[0])
        expert_action = np.array([[expert_angle_rad / np.pi]])  # [-1, 1]
        expert_action_t = torch.from_numpy(expert_action).float().to(device)

        # Q-values for CPs
        state_expanded = state.unsqueeze(1).expand(-1, control_points.shape[1], -1)
        q_cps = estimator(state_expanded, control_points).squeeze().cpu().numpy()  # (N,)
        if q_cps.ndim == 0:
            q_cps = np.array([q_cps.item()])

        # Q-value for expert
        q_expert = estimator(state, expert_action_t).item()

        # Softmax probabilities (including expert)
        q_all_for_softmax = np.concatenate([q_cps, [q_expert]])
        max_q = np.max(q_all_for_softmax)
        exps_all = np.exp(q_all_for_softmax - max_q)
        probs_all = exps_all / np.sum(exps_all)
        probs = probs_all[:-1]       # CP probabilities
        prob_expert = probs_all[-1]  # Expert probability

    # --- Figure: 2x3 grid (5 plots + 1 empty) ---
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"{title} | Episode {episode_idx}, Step {step_idx}", fontsize=14)

    # ========== Plot 1: Radial CP + Expert (Q-value weighted) ==========
    ax1 = fig.add_subplot(2, 3, 1, projection='polar')
    ax1.set_title("1. CPs & Expert\n(color = Q-value)", fontsize=10, pad=15)

    # Normalize Q for sizing
    q_all = np.concatenate([q_cps, [q_expert]])
    q_min, q_max = q_all.min(), q_all.max()
    q_range = q_max - q_min if q_max > q_min else 1.0

    cp_sizes = 50 + 150 * ((q_cps - q_min) / q_range)

    # Color CPs by Q-value
    norm1 = plt.Normalize(q_min, q_max)
    sc1 = ax1.scatter(
        cp_angles_rad, np.ones(len(cp_angles_rad)),
        c=q_cps, cmap='Blues', norm=norm1,
        s=cp_sizes, edgecolors='darkblue', linewidths=0.5, zorder=3
    )
    cb1 = plt.colorbar(sc1, ax=ax1, pad=0.12, shrink=0.6)
    cb1.set_label('Q-value', fontsize=8)

    # Expert (same color scale, star marker)
    ax1.scatter(
        [expert_angle_rad], [1.0],
        c=[q_expert], cmap='Blues', norm=norm1,
        marker='*', s=300, edgecolors='black', linewidths=0.8, zorder=5,
        label=f'Expert ★ Q={q_expert:.2f}'
    )

    ax1.set_yticks([])
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=7)

    # ========== Plot 2: Q-value Heatmap (polar) ==========
    ax2 = fig.add_subplot(2, 3, 2, projection='polar')
    ax2.set_title("2. Q-value Heatmap\n(across all angles)", fontsize=10, pad=15)

    n_sweep = 360
    sweep_actions_np = np.linspace(-1, 1, n_sweep).reshape(-1, 1).astype(np.float32)
    sweep_actions_t = torch.from_numpy(sweep_actions_np).to(device)
    state_sweep = state.repeat(n_sweep, 1)

    with torch.no_grad():
        q_sweep = estimator(state_sweep, sweep_actions_t).cpu().numpy().flatten()

    sweep_angles = sweep_actions_np[:, 0] * np.pi
    # Use bar segments for heatmap
    width = 2 * np.pi / n_sweep
    norm = plt.Normalize(q_sweep.min(), q_sweep.max())
    colors = cm.viridis(norm(q_sweep))
    ax2.bar(sweep_angles, np.ones(n_sweep), width=width, bottom=0.0,
            color=colors, alpha=0.8)
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax2, pad=0.1, shrink=0.6)
    cb.set_label('Q-value', fontsize=8)

    # Mark expert
    ax2.scatter([expert_angle_rad], [0.5], c='green', marker='*', s=200, zorder=5)
    ax2.set_yticks([])

    # ========== Plot 3: Probability Polar ==========
    ax3 = fig.add_subplot(2, 3, 3, projection='polar')
    ax3.set_title("3. CP Probabilities\n(color = softmax prob)", fontsize=10, pad=15)

    safe_probs_all = probs_all if (np.isfinite(probs_all).all() and probs_all.max() > 0) else np.ones_like(probs_all) / len(probs_all)
    safe_probs = safe_probs_all[:-1]
    safe_prob_expert = safe_probs_all[-1]
    prob_sizes = 50 + 250 * (safe_probs / safe_probs_all.max())

    # Color CPs by probability
    norm3 = plt.Normalize(safe_probs_all.min(), safe_probs_all.max())
    sc3 = ax3.scatter(
        cp_angles_rad, np.ones(len(cp_angles_rad)),
        c=safe_probs, cmap='Purples', norm=norm3,
        s=prob_sizes, edgecolors='darkviolet', linewidths=0.5, zorder=3
    )
    cb3 = plt.colorbar(sc3, ax=ax3, pad=0.12, shrink=0.6)
    cb3.set_label('Probability', fontsize=8)

    # Expert (distinct green star)
    ax3.scatter([expert_angle_rad], [1.0], c='green', marker='*', s=300,
                edgecolors='black', linewidths=0.8, zorder=5,
                label=f'Expert ★ p={safe_prob_expert:.3f}')
    # Selected CP (highest Q-value, matching action selection) as triangle
    sel_idx = np.argmax(q_cps)
    ax3.scatter([cp_angles_rad[sel_idx]], [1.0],
                c=[safe_probs[sel_idx]], cmap='Purples', norm=norm3,
                marker='^', s=200,
                edgecolors='black', linewidths=0.8, zorder=6,
                label=f'Selected ▲ p={safe_probs[sel_idx]:.3f}')
    ax3.set_yticks([])
    ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=7)

    # ========== Plot 4: Langevin Evolution ==========
    ax4 = fig.add_subplot(2, 3, 4, projection='polar')
    ax4.set_title("4. Langevin Evolution\n(radius = iteration progress)", fontsize=10, pad=15)

    if langevin_fn:
        samples, trajs = langevin_fn(model, estimator, device, state)
        # samples: (N_samples, 1), trajs: (N_samples, Steps, 1)

        if trajs is not None:
            starts = trajs[:, 0, 0] * np.pi
            ax4.scatter(starts, np.ones_like(starts) * 0.2, c='red', s=10,
                        alpha=0.5, label='Uniform Init')

            # Draw paths (up to 20)
            n_paths = min(20, len(samples))
            for i in range(n_paths):
                path = trajs[i, :, 0] * np.pi
                r_vals = np.linspace(0.2, 1.0, len(path))
                ax4.plot(path, r_vals, 'y-', alpha=0.3, linewidth=0.8)

        final_angles = samples.flatten() * np.pi
        ax4.scatter(final_angles, np.ones_like(final_angles), c='orange',
                    marker='x', s=30, label='Langevin Final')

    # Reference: CPs and expert
    ax4.scatter(cp_angles_rad, np.ones_like(cp_angles_rad) * 0.9, c='blue',
                s=20, alpha=0.6, label='CPs')
    ax4.scatter([expert_angle_rad], [1.0], c='green', marker='*', s=200,
                zorder=5, label='Expert')

    ax4.set_yticks([])
    ax4.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=6, ncol=2)

    # ========== Plot 5: 2D Navigation Map ==========
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("5. 2D Navigation Map", fontsize=10)
    ax5.set_xlim(-1.1, 1.1)
    ax5.set_ylim(-1.1, 1.1)
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)

    # Grid boundary
    rect = plt.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor='gray',
                          facecolor='lightyellow', alpha=0.3)
    ax5.add_patch(rect)

    # Trajectory
    traj_arr = np.array(trajectory)
    if len(traj_arr) > 1:
        ax5.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-', alpha=0.5, linewidth=1.5,
                 label='Path')
        ax5.scatter(traj_arr[:-1, 0], traj_arr[:-1, 1], c='lightblue', s=10,
                    edgecolors='blue', linewidths=0.3, zorder=3)

    # Current position
    ax5.scatter([agent_pos[0]], [agent_pos[1]], c='blue', s=100, marker='o',
                zorder=5, label=f'Agent ({agent_pos[0]:.2f}, {agent_pos[1]:.2f})')

    # Goal
    ax5.scatter([goal[0]], [goal[1]], c='green', s=200, marker='*',
                zorder=5, label=f'Goal ({goal[0]:.2f}, {goal[1]:.2f})')

    # Goal radius circle
    circle = plt.Circle(goal, 0.05, color='green', alpha=0.15, zorder=2)
    ax5.add_patch(circle)

    # Start position
    if len(traj_arr) > 0:
        ax5.scatter([traj_arr[0, 0]], [traj_arr[0, 1]], c='red', s=60,
                    marker='s', zorder=4, label='Start')

    ax5.legend(loc='upper left', fontsize=6)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')

    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.35, hspace=0.35)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
