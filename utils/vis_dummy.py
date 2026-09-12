
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


def _segment_dist_to_point(a, b, p):
    """Shortest distance from point p to segment a->b (numpy 2-vectors)."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        return float(np.linalg.norm(p - a))
    t = np.clip(float(np.dot(p - a, ab)) / denom, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


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
    obstacle_center=None,
    obstacle_radius=None,
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
        obstacle_center: optional (2,) numpy array — fixed obstacle position
            (dummy_bimodal only). When set, panel 5 draws the obstacle disk
            and, whenever the direct agent->goal line crosses it, panels 1-4
            plot BOTH valid detour headings (clockwise / counterclockwise) as
            twin experts instead of one — making the ground-truth bimodality
            explicit next to the model's control-point cloud.
        obstacle_radius: obstacle disk radius; required together with
            obstacle_center.
    """
    model.eval()
    estimator.eval()

    # Ensure state is (1, D) tensor
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float().to(device)
    if state.ndim == 1:
        state = state.unsqueeze(0)

    has_obstacle = obstacle_center is not None and obstacle_radius is not None
    blocked = False
    if has_obstacle:
        blocked = _segment_dist_to_point(agent_pos, goal, obstacle_center) < obstacle_radius

    with torch.no_grad():
        # Control points: (1, N, 1) -> angles in [-1, 1]
        control_points = model(state)
        cp_actions = control_points.squeeze(0).cpu().numpy()  # (N, 1)
        cp_angles_rad = cp_actions[:, 0] * np.pi  # Map to [-π, π]

        # Expert heading(s): direct-to-goal, UNLESS the direct line crosses
        # the obstacle, in which case there are two equally valid detours.
        if blocked:
            to_goal = goal - agent_pos
            norm = np.linalg.norm(to_goal) + 1e-8
            perp = np.array([-to_goal[1], to_goal[0]], dtype=np.float32) / norm
            margin = 0.15
            aim_points = [
                obstacle_center + side * perp * (obstacle_radius + margin)
                for side in (1.0, -1.0)
            ]
        else:
            aim_points = [goal]

        expert_diffs = [ap - agent_pos for ap in aim_points]
        expert_angles_rad = np.array(
            [np.arctan2(d[1], d[0]) for d in expert_diffs]
        )  # (K,), K=1 or 2
        expert_actions = (expert_angles_rad / np.pi).reshape(-1, 1).astype(np.float32)
        expert_actions_t = torch.from_numpy(expert_actions).float().to(device)

        # Q-values for CPs
        state_expanded = state.unsqueeze(1).expand(-1, control_points.shape[1], -1)
        q_cps = estimator(state_expanded, control_points).squeeze().cpu().numpy()  # (N,)
        if q_cps.ndim == 0:
            q_cps = np.array([q_cps.item()])

        # Q-values for expert(s), K=1 or 2
        state_for_experts = state.repeat(expert_actions_t.shape[0], 1)
        q_experts = estimator(state_for_experts, expert_actions_t).squeeze(-1).cpu().numpy()
        if q_experts.ndim == 0:
            q_experts = np.array([q_experts.item()])

        # Legacy single-expert aliases (used by panels that only show ONE
        # reference star, e.g. panel 4's "Expert" legend entry).
        expert_angle_rad = expert_angles_rad[0]
        q_expert = float(q_experts[0])

        # Softmax probabilities (including expert(s))
        q_all_for_softmax = np.concatenate([q_cps, q_experts])
        max_q = np.max(q_all_for_softmax)
        exps_all = np.exp(q_all_for_softmax - max_q)
        probs_all = exps_all / np.sum(exps_all)
        n_experts = len(q_experts)
        probs = probs_all[:-n_experts]        # CP probabilities
        probs_experts = probs_all[-n_experts:]  # Expert probabilities
        prob_expert = probs_experts[0]

    # --- Figure: 2x3 grid (5 plots + 1 empty) ---
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"{title} | Episode {episode_idx}, Step {step_idx}", fontsize=14)

    # ========== Plot 1: Radial CP + Expert (Q-value weighted) ==========
    ax1 = fig.add_subplot(2, 3, 1, projection='polar')
    ax1.set_title("1. CPs & Expert\n(color = Q-value)", fontsize=10, pad=15)

    # Normalize Q for sizing
    q_all = np.concatenate([q_cps, q_experts])
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

    # Expert(s) — one star if unambiguous, TWO if this state is blocked
    # (genuinely bimodal: both detour headings are equally valid experts).
    expert_labels = (
        [f'Expert ★ Q={q_experts[0]:.2f}']
        if n_experts == 1
        else [f'Expert {i+1} (side {"+" if i == 0 else "-"}) ★ Q={q_experts[i]:.2f}'
              for i in range(n_experts)]
    )
    for i in range(n_experts):
        ax1.scatter(
            [expert_angles_rad[i]], [1.0],
            c=[q_experts[i]], cmap='Blues', norm=norm1,
            marker='*', s=300, edgecolors='black', linewidths=0.8, zorder=5,
            label=expert_labels[i]
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

    # Mark expert(s)
    ax2.scatter(expert_angles_rad, np.full(n_experts, 0.5), c='green', marker='*',
                s=200, zorder=5)
    ax2.set_yticks([])

    # ========== Plot 3: Probability Polar ==========
    ax3 = fig.add_subplot(2, 3, 3, projection='polar')
    ax3.set_title("3. CP Probabilities\n(color = softmax prob)", fontsize=10, pad=15)

    safe_probs_all = probs_all if (np.isfinite(probs_all).all() and probs_all.max() > 0) else np.ones_like(probs_all) / len(probs_all)
    safe_probs = safe_probs_all[:-n_experts]
    safe_probs_experts = safe_probs_all[-n_experts:]
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

    # Expert(s) — distinct green star(s); TWO when this state is blocked
    for i in range(n_experts):
        lbl = (f'Expert ★ p={safe_probs_experts[i]:.3f}' if n_experts == 1
               else f'Expert {i+1} ★ p={safe_probs_experts[i]:.3f}')
        ax3.scatter([expert_angles_rad[i]], [1.0], c='green', marker='*', s=300,
                    edgecolors='black', linewidths=0.8, zorder=5, label=lbl)
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

    # Fixed obstacle (dummy_bimodal only) — highlighted red if the direct
    # agent->goal line is currently blocked by it.
    if has_obstacle:
        obstacle_patch = plt.Circle(
            obstacle_center, obstacle_radius,
            color='red' if blocked else 'gray', alpha=0.35, zorder=2,
        )
        ax5.add_patch(obstacle_patch)

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


def plot_dummy_dp_debug(
    sample_fn,
    save_path,
    state,
    trajectory,
    goal,
    agent_pos,
    step_idx,
    episode_idx,
    n_samples=64,
    title="Dummy Diagnostic (DP)",
    obstacle_center=None,
    obstacle_radius=None,
):
    """DP analog of `plot_dummy_debug`. A diffusion policy has no control-point
    cloud or Q function — the only thing directly comparable to Q3C/IBC's "CPs
    & Expert" panel is the SET of independently-drawn stochastic samples at
    one state, which is what the caller's `sample_fn` produces. Two panels
    only (no Q-heatmap/probabilities/Langevin-evolution equivalents to draw):

    1. Radial scatter of `n_samples` independently-sampled headings (each a
       fresh denoising chain from random noise), overlaid with the same
       twin-expert-star geometry as plot_dummy_debug — a mode-preserving
       sampler should visibly split into two angular clusters matching the
       two expert stars whenever the state is blocked.
    2. 2D navigation map — identical in content to plot_dummy_debug's panel 5.

    Args:
        sample_fn: callable(n_samples) -> np.ndarray of shape (n_samples,)
            action values in [-1, 1] (angle / pi), drawn independently at
            the CURRENT state (caller closes over state/denoiser/diffusion).
        Other args mirror plot_dummy_debug.
    """
    has_obstacle = obstacle_center is not None and obstacle_radius is not None
    blocked = False
    if has_obstacle:
        blocked = _segment_dist_to_point(agent_pos, goal, obstacle_center) < obstacle_radius

    if blocked:
        to_goal = goal - agent_pos
        norm = np.linalg.norm(to_goal) + 1e-8
        perp = np.array([-to_goal[1], to_goal[0]], dtype=np.float32) / norm
        margin = 0.15
        aim_points = [obstacle_center + side * perp * (obstacle_radius + margin)
                      for side in (1.0, -1.0)]
    else:
        aim_points = [goal]
    expert_diffs = [ap - agent_pos for ap in aim_points]
    expert_angles_rad = np.array([np.arctan2(d[1], d[0]) for d in expert_diffs])
    n_experts = len(expert_angles_rad)

    samples = sample_fn(n_samples)  # (n_samples,) in [-1, 1]
    sample_angles_rad = samples * np.pi

    fig = plt.figure(figsize=(12, 6.5))
    fig.suptitle(f"{title} | Episode {episode_idx}, Step {step_idx}", fontsize=14, y=0.98)

    # ========== Plot 1: Sampled headings (polar) ==========
    ax1 = fig.add_subplot(1, 2, 1, projection='polar')
    ax1.set_title(f"1. {n_samples} Independent DP Samples\n(fresh denoising chain each)",
                   fontsize=10, pad=25)
    ax1.scatter(sample_angles_rad, np.ones_like(sample_angles_rad), c='steelblue',
                s=40, alpha=0.5, edgecolors='darkblue', linewidths=0.3, zorder=3,
                label='Sampled action')
    expert_labels = (
        ['Expert ★'] if n_experts == 1
        else [f'Expert {i+1} (side {"+" if i == 0 else "-"}) ★' for i in range(n_experts)]
    )
    for i in range(n_experts):
        ax1.scatter([expert_angles_rad[i]], [1.0], c='green', marker='*', s=300,
                    edgecolors='black', linewidths=0.8, zorder=5, label=expert_labels[i])
    ax1.set_yticks([])
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=7)

    # ========== Plot 2: 2D Navigation Map ==========
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("2. 2D Navigation Map", fontsize=10)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    rect = plt.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor='gray',
                          facecolor='lightyellow', alpha=0.3)
    ax2.add_patch(rect)

    if has_obstacle:
        obstacle_patch = plt.Circle(
            obstacle_center, obstacle_radius,
            color='red' if blocked else 'gray', alpha=0.35, zorder=2,
        )
        ax2.add_patch(obstacle_patch)

    traj_arr = np.array(trajectory)
    if len(traj_arr) > 1:
        ax2.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-', alpha=0.5, linewidth=1.5, label='Path')
        ax2.scatter(traj_arr[:-1, 0], traj_arr[:-1, 1], c='lightblue', s=10,
                    edgecolors='blue', linewidths=0.3, zorder=3)

    ax2.scatter([agent_pos[0]], [agent_pos[1]], c='blue', s=100, marker='o',
                zorder=5, label=f'Agent ({agent_pos[0]:.2f}, {agent_pos[1]:.2f})')
    ax2.scatter([goal[0]], [goal[1]], c='green', s=200, marker='*',
                zorder=5, label=f'Goal ({goal[0]:.2f}, {goal[1]:.2f})')
    circle = plt.Circle(goal, 0.05, color='green', alpha=0.15, zorder=2)
    ax2.add_patch(circle)
    if len(traj_arr) > 0:
        ax2.scatter([traj_arr[0, 0]], [traj_arr[0, 1]], c='red', s=60,
                    marker='s', zorder=4, label='Start')

    ax2.legend(loc='upper left', fontsize=6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    fig.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.15, wspace=0.35)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
