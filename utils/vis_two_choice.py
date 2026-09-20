"""Visualization for TwoChoice-v0 — the simplest possible multimodality toy.

Trimmed to 4 panels (no 2D map / no obstacle — there's no spatial trajectory
here, every state IS the interesting state):
1. Radial CP + both true modes (Q-value weighted)
2. Q-value heatmap (polar sweep across all angles)
3. CP Probabilities (softmax over CPs + both modes)
4. Langevin evolution (uniform init -> MCMC refinement -> final samples)
"""

import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch


def plot_two_choice_debug(
    model,
    estimator,
    device,
    save_path,
    state,
    mode_a,
    mode_b,
    state_idx,
    langevin_fn=None,
    title="TwoChoice Diagnostic",
):
    """
    Args:
        model: ControlPointGenerator (eval mode).
        estimator: QEstimator (eval mode).
        device: torch device.
        save_path: where to save the figure.
        state: state tensor (1, 2) — [mode_a, mode_b], normalized.
        mode_a, mode_b: the two TRUE target angles in [-1, 1] (raw, unnormalized
            — these ARE the ground truth, no geometry/oracle computation needed).
        state_idx: index for the figure title.
        langevin_fn: callable(model, estimator, device, state) -> (samples, trajs).
        title: figure title prefix.
    """
    model.eval()
    estimator.eval()

    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float().to(device)
    if state.ndim == 1:
        state = state.unsqueeze(0)

    mode_angles_rad = np.array([mode_a, mode_b]) * np.pi

    with torch.no_grad():
        control_points = model(state)  # (1, N, 1)
        cp_actions = control_points.squeeze(0).cpu().numpy()
        cp_angles_rad = cp_actions[:, 0] * np.pi

        mode_actions = np.array([[mode_a], [mode_b]], dtype=np.float32)
        mode_actions_t = torch.from_numpy(mode_actions).float().to(device)

        state_expanded = state.unsqueeze(1).expand(-1, control_points.shape[1], -1)
        q_cps = estimator(state_expanded, control_points).squeeze().cpu().numpy()
        if q_cps.ndim == 0:
            q_cps = np.array([q_cps.item()])

        state_for_modes = state.repeat(2, 1)
        q_modes = estimator(state_for_modes, mode_actions_t).squeeze(-1).cpu().numpy()

        q_all_for_softmax = np.concatenate([q_cps, q_modes])
        max_q = np.max(q_all_for_softmax)
        exps_all = np.exp(q_all_for_softmax - max_q)
        probs_all = exps_all / np.sum(exps_all)
        probs = probs_all[:-2]
        probs_modes = probs_all[-2:]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{title} | State {state_idx} (modes={mode_a:.2f}/{mode_b:.2f})", fontsize=14)

    # ========== Plot 1: Radial CP + both true modes ==========
    ax1 = fig.add_subplot(2, 2, 1, projection='polar')
    ax1.set_title("1. CPs & True Modes\n(color = Q-value)", fontsize=10, pad=15)
    q_all = np.concatenate([q_cps, q_modes])
    q_min, q_max = q_all.min(), q_all.max()
    q_range = q_max - q_min if q_max > q_min else 1.0
    cp_sizes = 50 + 150 * ((q_cps - q_min) / q_range)
    norm1 = plt.Normalize(q_min, q_max)
    sc1 = ax1.scatter(cp_angles_rad, np.ones(len(cp_angles_rad)), c=q_cps, cmap='Blues',
                       norm=norm1, s=cp_sizes, edgecolors='darkblue', linewidths=0.5, zorder=3)
    cb1 = plt.colorbar(sc1, ax=ax1, pad=0.12, shrink=0.6)
    cb1.set_label('Q-value', fontsize=8)
    for i, lbl in enumerate(['Mode A', 'Mode B']):
        ax1.scatter([mode_angles_rad[i]], [1.0], c=[q_modes[i]], cmap='Blues', norm=norm1,
                    marker='*', s=350, edgecolors='black', linewidths=0.8, zorder=5,
                    label=f'{lbl} ★ Q={q_modes[i]:.2f}')
    ax1.set_yticks([])
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=8)

    # ========== Plot 2: Q-value heatmap ==========
    ax2 = fig.add_subplot(2, 2, 2, projection='polar')
    ax2.set_title("2. Q-value Heatmap\n(across all angles)", fontsize=10, pad=15)
    n_sweep = 360
    sweep_actions_np = np.linspace(-1, 1, n_sweep).reshape(-1, 1).astype(np.float32)
    sweep_actions_t = torch.from_numpy(sweep_actions_np).to(device)
    state_sweep = state.repeat(n_sweep, 1)
    with torch.no_grad():
        q_sweep = estimator(state_sweep, sweep_actions_t).cpu().numpy().flatten()
    sweep_angles = sweep_actions_np[:, 0] * np.pi
    width = 2 * np.pi / n_sweep
    norm2 = plt.Normalize(q_sweep.min(), q_sweep.max())
    colors = cm.viridis(norm2(q_sweep))
    ax2.bar(sweep_angles, np.ones(n_sweep), width=width, bottom=0.0, color=colors, alpha=0.8)
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm2)
    sm.set_array([])
    cb2 = plt.colorbar(sm, ax=ax2, pad=0.1, shrink=0.6)
    cb2.set_label('Q-value', fontsize=8)
    ax2.scatter(mode_angles_rad, [0.5, 0.5], c='green', marker='*', s=250, zorder=5)
    ax2.set_yticks([])

    # ========== Plot 3: CP Probabilities ==========
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')
    ax3.set_title("3. CP Probabilities\n(color = softmax prob)", fontsize=10, pad=15)
    safe_probs_all = probs_all if (np.isfinite(probs_all).all() and probs_all.max() > 0) else np.ones_like(probs_all) / len(probs_all)
    safe_probs = safe_probs_all[:-2]
    safe_probs_modes = safe_probs_all[-2:]
    prob_sizes = 50 + 250 * (safe_probs / safe_probs_all.max())
    norm3 = plt.Normalize(safe_probs_all.min(), safe_probs_all.max())
    sc3 = ax3.scatter(cp_angles_rad, np.ones(len(cp_angles_rad)), c=safe_probs, cmap='Purples',
                       norm=norm3, s=prob_sizes, edgecolors='darkviolet', linewidths=0.5, zorder=3)
    cb3 = plt.colorbar(sc3, ax=ax3, pad=0.12, shrink=0.6)
    cb3.set_label('Probability', fontsize=8)
    for i, lbl in enumerate(['Mode A', 'Mode B']):
        ax3.scatter([mode_angles_rad[i]], [1.0], c='green', marker='*', s=350,
                    edgecolors='black', linewidths=0.8, zorder=5,
                    label=f'{lbl} ★ p={safe_probs_modes[i]:.3f}')
    sel_idx = np.argmax(q_cps)
    ax3.scatter([cp_angles_rad[sel_idx]], [1.0], c=[safe_probs[sel_idx]], cmap='Purples',
                norm=norm3, marker='^', s=220, edgecolors='black', linewidths=0.8, zorder=6,
                label=f'Selected ▲ p={safe_probs[sel_idx]:.3f}')
    ax3.set_yticks([])
    ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=8)

    # ========== Plot 4: Langevin Evolution ==========
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')
    ax4.set_title("4. Langevin Evolution\n(radius = iteration progress)", fontsize=10, pad=15)
    if langevin_fn:
        samples, trajs = langevin_fn(model, estimator, device, state)
        if trajs is not None:
            starts = trajs[:, 0, 0] * np.pi
            ax4.scatter(starts, np.ones_like(starts) * 0.2, c='red', s=10, alpha=0.5, label='Uniform Init')
            n_paths = min(20, len(samples))
            for i in range(n_paths):
                path = trajs[i, :, 0] * np.pi
                r_vals = np.linspace(0.2, 1.0, len(path))
                ax4.plot(path, r_vals, 'y-', alpha=0.3, linewidth=0.8)
        final_angles = samples.flatten() * np.pi
        ax4.scatter(final_angles, np.ones_like(final_angles), c='orange', marker='x', s=30, label='Langevin Final')
    ax4.scatter(cp_angles_rad, np.ones_like(cp_angles_rad) * 0.9, c='blue', s=20, alpha=0.6, label='CPs')
    ax4.scatter(mode_angles_rad, [1.0, 1.0], c='green', marker='*', s=250, zorder=5, label='True Modes')
    ax4.set_yticks([])
    ax4.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=7, ncol=2)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.08, wspace=0.35, hspace=0.4)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
