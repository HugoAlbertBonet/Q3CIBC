"""IBC training with Langevin MCMC for Particle environment.

Paper-faithful implementation of IBC (Florence et al., 2021, arXiv:2109.00137)
using InfoNCE loss + Langevin MCMC hard negatives + gradient penalty.

Key differences from naive implementation:
  - Counter-examples generated via Langevin MCMC (not uniform random)
  - Observation and action normalization from dataset statistics
  - Gradient penalty on ALL combined actions (expert + counter-examples)
  - L-inf gradient norm with margin
  - Expert action at LAST index in the candidate set

Usage:
    python -m ibc.ibc_dfo_particle_training
"""

import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from utils.models import QEstimator
from utils.datasets import ParticleDataset
from utils.normalizations import ObservationNormalizer

# ─── Load config ─────────────────────────────────────────────────────────────
config_path = Path(__file__).parent.parent / "config_json" / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

env_config = config["environments"]["particle"]
training_shared = config.get("training_shared", {})

# Environment parameters
state_dim = env_config["state_dim"]
action_dim = env_config["action_dim"]
action_bounds = env_config.get("action_bounds", [0, 1])
frame_stack = env_config.get("frame_stack", 2)
n_dim = env_config.get("n_dim", 2)

# ─── Paper hyperparameters (Table 7 / mlp_ebm_langevin.gin) ─────────────────
TRAINING_STEPS = 100_000
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
LR_DECAY_RATE = 0.99
LR_DECAY_STEPS = 100
NUM_COUNTER_EXAMPLES = 8
LANGEVIN_TRAIN_ITERATIONS = 100
LANGEVIN_STEPSIZE_INIT = 0.1
LANGEVIN_STEPSIZE_FINAL = 1e-5
LANGEVIN_STEPSIZE_POWER = 2.0
LANGEVIN_NOISE_SCALE = 1.0
LANGEVIN_DELTA_ACTION_CLIP = 0.1
GRADIENT_MARGIN = 1.0
SOFTMAX_TEMPERATURE = 1.0
UNIFORM_BOUNDARY_BUFFER = 0.05

# Model architecture (paper particle config: 256x2)
HIDDEN_DIMS = [256, 256]

LOG_INTERVAL = 1000
SAVE_INTERVAL = 5000

MODEL_SAVE_DIR = os.path.join(
    training_shared.get("model_save_dir", "checkpoints"), "ibc_dfo", "particle"
)


def compute_dataset_stats(dataset):
    """Compute min/max statistics from dataset for action normalization."""
    acts = dataset.actions
    return {
        "act_min": acts.min(axis=0).astype(np.float32),
        "act_max": acts.max(axis=0).astype(np.float32),
    }


def normalize_tensor(x, x_min, x_max, device):
    """Normalize tensor to [0, 1] using precomputed min/max."""
    rng = x_max - x_min
    rng = np.where(rng == 0, np.ones_like(rng), rng)
    x_min_t = torch.from_numpy(x_min).float().to(device)
    rng_t = torch.from_numpy(rng).float().to(device)
    return (x - x_min_t) / rng_t


def normalize_observations(x, obs_normalizer):
    """Normalize observations with shared bounds-based normalization."""
    return obs_normalizer.normalize(x)


def langevin_counter_examples(energy_model, obs_norm, device):
    """Generate hard negative actions via Langevin MCMC in normalized space.

    Matches the paper's langevin_step: gradient descent on energy + noise,
    with polynomial learning rate schedule and delta clipping.
    Operates entirely in normalized action space [0, 1] (with small buffer).

    Args:
        energy_model: The energy network E(obs, act).
        obs_norm: Normalized observations, shape (B, obs_dim).
        device: Torch device.

    Returns:
        Counter-example actions in normalized space, shape (B, N, action_dim).
    """
    B = obs_norm.shape[0]

    # Normalized action bounds with buffer
    act_min = 0.0 - UNIFORM_BOUNDARY_BUFFER
    act_max = 1.0 + UNIFORM_BOUNDARY_BUFFER

    # Initialize uniform in normalized space
    actions = (
        torch.rand(B, NUM_COUNTER_EXAMPLES, action_dim, device=device)
        * (act_max - act_min) + act_min
    )

    # Scaled delta clip (paper: delta_action_clip * 0.5 * (max - min))
    delta_clip = LANGEVIN_DELTA_ACTION_CLIP * 0.5 * (act_max - act_min)

    obs_expanded = obs_norm.unsqueeze(1).expand(-1, NUM_COUNTER_EXAMPLES, -1)

    # Disable param gradients during Langevin chain (stop_chain_grad)
    for p in energy_model.parameters():
        p.requires_grad_(False)

    for k in range(LANGEVIN_TRAIN_ITERATIONS):
        # Polynomial decay stepsize
        frac = 1.0 - k / max(LANGEVIN_TRAIN_ITERATIONS - 1, 1)
        stepsize = (
            LANGEVIN_STEPSIZE_FINAL
            + (LANGEVIN_STEPSIZE_INIT - LANGEVIN_STEPSIZE_FINAL)
            * (frac ** LANGEVIN_STEPSIZE_POWER)
        )

        actions = actions.detach().requires_grad_(True)
        energies = energy_model(obs_expanded, actions).squeeze(-1)
        grad = torch.autograd.grad(energies.sum(), actions)[0]
        grad = grad.detach()

        # Langevin step: gradient descent on energy + noise
        noise = torch.randn_like(actions) * LANGEVIN_NOISE_SCALE
        delta = stepsize * (0.5 * grad + noise)
        delta = torch.clamp(delta, -delta_clip, delta_clip)

        actions = (actions.detach() - delta).detach()
        actions = torch.clamp(actions, act_min, act_max)

    # Re-enable param gradients for training
    for p in energy_model.parameters():
        p.requires_grad_(True)

    return actions.detach()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training IBC (paper-faithful) on Particle environment")
    print(f"Steps: {TRAINING_STEPS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(
        f"Counter-examples: {NUM_COUNTER_EXAMPLES} "
        f"(Langevin, {LANGEVIN_TRAIN_ITERATIONS} iters)"
    )
    print(f"Architecture: MLP {HIDDEN_DIMS}")
    print(f"Grad penalty: L-inf, margin={GRADIENT_MARGIN}")

    # ─── Dataset ──────────────────────────────────────────────────────────
    data_dir = env_config["data_dir"]
    dataset = ParticleDataset(data_dir, n_dim=n_dim, frame_stack=frame_stack)
    print(f"Dataset size: {len(dataset)}")

    # Compute normalization stats from dataset
    norm_stats = compute_dataset_stats(dataset)
    print(
        f"Act range: [{norm_stats['act_min'].min():.3f}, "
        f"{norm_stats['act_max'].max():.3f}]"
    )

    obs_normalizer = ObservationNormalizer(
        env_id=env_config["env_id"],
        device=device,
        frame_stack=frame_stack,
        particle_n_dim=n_dim,
    )

    # ─── Energy model (no spectral norm — paper particle gin: 'regular') ──
    obs_dim = dataset.state_shape
    act_dim = dataset.action_shape

    energy_model = QEstimator(
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_dims=HIDDEN_DIMS,
    ).to(device)

    optimizer = torch.optim.Adam(energy_model.parameters(), lr=LEARNING_RATE)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )

    current_lr = LEARNING_RATE

    # ─── Training loop ────────────────────────────────────────────────────
    start_time = time.time()
    step = 0

    print("\nStarting training...")
    while step < TRAINING_STEPS:
        for batch in dataloader:
            if step >= TRAINING_STEPS:
                break

            states = batch["state"].float().to(device)
            actions = batch["action"].float().to(device)
            B = states.shape[0]

            # Normalize observations and actions
            states_norm = normalize_observations(states, obs_normalizer)
            actions_norm = normalize_tensor(
                actions, norm_stats["act_min"], norm_stats["act_max"], device
            )

            # Generate Langevin counter-examples (outside training gradient tape)
            counter_actions = langevin_counter_examples(
                energy_model, states_norm, device,
            )  # (B, N, action_dim) in normalized space

            # Combine: counter-examples first, expert LAST (paper convention)
            all_actions = torch.cat(
                [counter_actions, actions_norm.unsqueeze(1)], dim=1
            )  # (B, N+1, action_dim)

            states_expanded = states_norm.unsqueeze(1).expand(
                -1, NUM_COUNTER_EXAMPLES + 1, -1
            )

            # Compute energies for InfoNCE
            energies = energy_model(states_expanded, all_actions).squeeze(-1)

            # InfoNCE: expert at LAST index should have lowest energy
            logits = -energies / SOFTMAX_TEMPERATURE
            log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            loss_infonce = -log_probs[:, -1].mean()

            # Gradient penalty on ALL combined actions (L-inf norm, margin)
            gp_actions = all_actions.detach().reshape(
                B * (NUM_COUNTER_EXAMPLES + 1), -1
            )
            gp_actions = gp_actions.requires_grad_(True)
            gp_states = states_expanded.detach().reshape(
                B * (NUM_COUNTER_EXAMPLES + 1), -1
            )

            gp_energies = energy_model(gp_states, gp_actions)
            grad_gp = torch.autograd.grad(
                gp_energies.sum(), gp_actions, create_graph=True
            )[0]

            # L-inf norm
            grad_norms = grad_gp.abs().max(dim=-1).values
            grad_penalty = torch.clamp(
                grad_norms - GRADIENT_MARGIN, min=0
            ).pow(2).mean()

            loss = loss_infonce + grad_penalty

            if torch.isnan(loss):
                print(f"  Step {step}: NaN loss, skipping.")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # Exponential LR decay
            if step % LR_DECAY_STEPS == 0:
                current_lr *= LR_DECAY_RATE
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

            # Logging
            if step % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                with torch.no_grad():
                    best_idx = logits.argmax(dim=1)
                    accuracy = (
                        (best_idx == NUM_COUNTER_EXAMPLES).float().mean().item()
                    )
                print(
                    f"  Step {step}/{TRAINING_STEPS} | "
                    f"Loss: {loss.item():.4f} "
                    f"(NCE: {loss_infonce.item():.4f}, "
                    f"GP: {grad_penalty.item():.4f}) | "
                    f"Acc: {accuracy:.3f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

            # Save checkpoint
            if step % SAVE_INTERVAL == 0:
                os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": energy_model.state_dict(),
                        "norm_stats": norm_stats,
                        "step": step,
                    },
                    os.path.join(
                        MODEL_SAVE_DIR, f"q_estimator_step_{step}.pt"
                    ),
                )

    # ─── Save final model ─────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.2f} min)")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    final_path = os.path.join(MODEL_SAVE_DIR, "q_estimator.pt")
    torch.save(
        {
            "model_state_dict": energy_model.state_dict(),
            "norm_stats": norm_stats,
            "step": TRAINING_STEPS,
        },
        final_path,
    )
    print(f"Energy model saved to {final_path}")


if __name__ == "__main__":
    main()
