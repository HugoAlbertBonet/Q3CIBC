
"""Visualization tools for Dummy 2D Directional Environment."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_dummy_debug(
    model, 
    estimator, 
    dataset, 
    device, 
    save_path, 
    langevin_fn=None,
    title="Dummy Verification"
):
    """
    Generates a 2x3 grid of diagnostic plots.
    
    1. Distance to Optimal: Histogram of |CP - 0|.
    2. CP Uniformity: Polar histogram of CP angles.
    3. Estimator Values: Bar chart of Q(s, CP) vs Q(s, Expert).
    4. Action Probabilities: Softmax distribution over CPs.
    5. InfoNCE Losss Context: Similarity scores.
    6. Sampling Locations: Polar plot of all sample types.
    """
    model.eval()
    estimator.eval()
    
    # 1. Prepare Data
    # ----------------
    # Fixed state
    state = torch.tensor([[1.0, 0.0]]).to(device) # (1, 2)
    
    with torch.no_grad():
        # Get Control Points
        control_points = model(state) # (1, N, 1) -> Angle in [-pi, pi]
        cp_angles = control_points.squeeze().cpu().numpy() # (N,)
        
        # Get Expert Action (approx 0)
        expert_action = torch.tensor([[0.0]]).to(device)
        
        # Q-values for CPs
        # expand state: (1, 2) -> (1, N, 2)
        state_expanded = state.unsqueeze(1).expand(-1, control_points.shape[1], -1)
        q_cps = estimator(state_expanded, control_points).squeeze().cpu().numpy() # (N,)
        
        # Q-value for Expert
        q_expert = estimator(state, expert_action).item()
        
        # Energy = -Q
        energy_cps = -q_cps
        # Softmax probs: exp(-E) / sum(exp(-E)) = exp(Q) / sum(exp(Q))
        # Numerical stability: shift by max Q
        max_q = np.max(q_cps)
        exps = np.exp(q_cps - max_q)
        probs = exps / np.sum(exps)

    # 2. Plotting
    # -----------
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=16)
    
    # --- Plot 1: Distance to Optimal ---
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title("1. Distance to Optimal (Goal=0)")
    # Wrap diff to [-pi, pi]
    diffs = np.arctan2(np.sin(cp_angles), np.cos(cp_angles)) # already in range, but safe
    abs_diffs = np.abs(diffs)
    ax1.hist(abs_diffs, bins=20, color='blue', alpha=0.7)
    ax1.set_xlabel("Absolute Angular Difference (rad)")
    ax1.set_ylabel("Count")
    ax1.axvline(np.min(abs_diffs), color='red', linestyle='--', label=f"Best: {np.min(abs_diffs):.3f}")
    ax1.legend()
    
    # --- Plot 2: CP Uniformity (Polar) ---
    ax2 = fig.add_subplot(2, 3, 2, projection='polar')
    ax2.set_title("2. CP Distribution (Uniformity)")
    # Histogram on circle
    ax2.hist(cp_angles, bins=30, color='blue', alpha=0.5, label='CPs')
    ax2.axvline(0, color='green', linewidth=2, label='Goal')
    ax2.legend()
    
    # --- Plot 3: Estimator Values ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title("3. Estimator Q-Values")
    # Plot Q(CPs) as bars, Q(Expert) as line
    indices = np.argsort(q_cps)
    ax3.bar(range(len(q_cps)), q_cps[indices], color='blue', alpha=0.5, label='CPs (Sorted)')
    ax3.axhline(q_expert, color='green', linestyle='-', linewidth=2, label=f'Expert Q: {q_expert:.2f}')
    ax3.set_xlabel("Control Point Index")
    ax3.set_ylabel("Q-Value")
    ax3.legend()
    
    # --- Plot 4: Action Probabilities ---
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title("4. Action Probabilities (Softmax)")
    ax4.bar(range(len(probs)), probs[indices], color='purple', alpha=0.7)
    ax4.set_xlabel("Control Point Index (Sorted by Q)")
    ax4.set_ylabel("Probability")
    
    # --- Plot 5: InfoNCE / Energy Landscape ---
    # Since we can't easily access the batch loss components here without the loss fn,
    # let's plot the Q-curve over the whole range [-pi, pi] to see the "Energy Landscape"
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("5. Energy Landscape (Q-Curve)")
    
    test_angles = np.linspace(-np.pi, np.pi, 100)
    test_tensor = torch.from_numpy(test_angles).float().unsqueeze(1).to(device) # (100, 1)
    # Expand state to (100, 2)
    state_repeated = state.repeat(100, 1)
    
    with torch.no_grad():
        q_curve = estimator(state_repeated, test_tensor).cpu().numpy().flatten()
        
    ax5.plot(test_angles, q_curve, 'k-', label='Q(a)')
    ax5.scatter(cp_angles, q_cps, c='blue', s=20, label='CPs')
    ax5.scatter([0], [q_expert], c='green', marker='*', s=100, label='Expert')
    ax5.set_xlim(-np.pi, np.pi)
    ax5.legend()
    
    # --- Plot 6: Sampling Locations (Polar) ---
    ax6 = fig.add_subplot(2, 3, 6, projection='polar')
    ax6.set_title("6. Sampling Locations")
    
    # Expert
    ax6.scatter([0], [1], c='green', marker='*', s=200, label='Expert')
    # CPs
    ax6.scatter(cp_angles, np.ones_like(cp_angles)*0.9, c='blue', s=20, label='Control Points')
    
    # Counter-examples (Langevin / Uniform)
    if langevin_fn:
        # Generate samples
        # samples: (N_samples, 1)
        samples, trajs = langevin_fn(model, estimator, device)
        
        # Plot Uniform/Initial (if available, usually start of traj)
        if trajs is not None:
            # Traj: (N_samples, Steps, 1)
            starts = trajs[:, 0, 0]
            ax6.scatter(starts, np.ones_like(starts)*0.8, c='red', s=10, alpha=0.5, label='Uniform Init')
            
            # Plot paths
            for i in range(min(20, len(samples))):
                path = trajs[i, :, 0]
                ax6.plot(path, np.linspace(0.8, 1.0, len(path)), 'y-', alpha=0.3)
                
        # Final Langevin
        final_samples = samples.flatten()
        ax6.scatter(final_samples, np.ones_like(final_samples)*1.0, c='orange', marker='x', s=30, label='Langevin Final')

    ax6.set_yticks([])
    ax6.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
