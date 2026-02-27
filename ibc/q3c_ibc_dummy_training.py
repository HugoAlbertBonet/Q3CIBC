"""Q3C IBC training for Dummy environment.

Based on combinedv2_training.py — trains both ControlPointGenerator and QEstimator.

Key design: the estimator is trained as a DIRECT energy model (like IBC-DFO) using
InfoNCE loss WITHOUT wire-fitting, so it produces reliable Q-values across the full
action space. The generator is trained with MSE + Separation loss to produce good
candidate actions near the expert.

At inference:
  - Generate control points from the generator
  - Evaluate each directly with the estimator (no interpolation)
  - Pick the action with the highest Q-value

This combines the best of both worlds: learned candidate generation + reliable
energy-based scoring.

Usage:
    python -m ibc.q3c_ibc_dummy_training
"""

import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn

from utils.models import ControlPointGenerator, QEstimator
from utils.loss import lossMSE, lossSeparation
from utils.normalizations import ObservationNormalizer
from utils.sampling import sample_uniform
from utils.datasets import DummyDataset

# ─── Load config ─────────────────────────────────────────────────────────────
config_path = Path(__file__).parent.parent / "config_json" / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

env_config = config["environments"]["dummy"]
training_shared = config.get("training_shared", {})
env_training = env_config.get("training", {})
env_model = env_config.get("model", {})

# Training parameters
training_steps = env_training.get("training_steps", training_shared.get("training_steps", 100000))
batch_size = env_training.get("batch_size", training_shared.get("batch_size", 128))
learning_rate = 1e-3
log_interval = 1000
save_interval = 10000

# Loss weights tuned for 1D dummy env
separation_weight = 0.1     # Small: just keep CPs diverse
mse_weight = 1.0            # Generator learns expert actions
info_nce_weight = 1.0       # Estimator learns energy landscape

# Counter-examples for the estimator's InfoNCE training
num_counter_examples = 64

# Model parameters
control_points = env_model.get("control_points", 30)
num_hidden_layers = env_model.get("num_hidden_layers", 4)
num_neurons = env_model.get("num_neurons", 128)

# Environment parameters
env_id = env_config["env_id"]
state_dim = env_config["state_dim"]
action_dim = env_config["action_dim"]
action_bounds = env_config.get("action_bounds", [-1, 1])
frame_stack = env_config.get("frame_stack", 1)

# Save directory
MODEL_SAVE_DIR = os.path.join(
    training_shared.get("model_save_dir", "checkpoints"), "q3c_ibc"
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training Q3C IBC on Dummy environment")
    print(f"Training steps: {training_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Control points: {control_points}")
    print(f"Loss weights — MSE: {mse_weight}, Sep: {separation_weight}, InfoNCE: {info_nce_weight}")
    print(f"Counter-examples (uniform): {num_counter_examples}")
    print(f"Estimator: DIRECT energy model (no wire-fitting)")

    # ─── Dataset ──────────────────────────────────────────────────────────
    dataset = DummyDataset(
        size=10000,
        step_size=env_config.get("step_size", 0.1),
        goal_radius=env_config.get("goal_radius", 0.1),
        n_dim=env_config.get("n_dim", 2),
        frame_stack=frame_stack,
    )
    print(f"Dataset size: {len(dataset)}")

    # ─── Models ───────────────────────────────────────────────────────────
    generator = ControlPointGenerator(
        input_dim=dataset.state_shape,
        output_dim=dataset.action_shape,
        control_points=control_points,
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)],
        action_bounds=(action_bounds[0], action_bounds[1]),
    ).to(device)

    estimator = QEstimator(
        state_dim=dataset.state_shape,
        action_dim=dataset.action_shape,
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)]
    ).to(device)

    optimizer_gen = torch.optim.AdamW(generator.parameters(), lr=learning_rate)
    optimizer_est = torch.optim.AdamW(estimator.parameters(), lr=learning_rate)

    # Cosine LR schedule
    scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_gen, T_max=training_steps, eta_min=1e-6
    )
    scheduler_est = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_est, T_max=training_steps, eta_min=1e-6
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    obs_normalizer = ObservationNormalizer(env_id=env_id, device=device, frame_stack=frame_stack)

    action_min = action_bounds[0]
    action_max = action_bounds[1]

    # ─── Training loop ────────────────────────────────────────────────────
    start_time = time.time()
    step = 0

    print("\nStarting training...")
    while step < training_steps:
        for batch in dataloader:
            if step >= training_steps:
                break

            states = batch['state'].float().to(device)
            states = obs_normalizer.normalize(states)
            actions = batch['action'].float().to(device)  # expert actions

            B = states.shape[0]

            # ========== Generator training: MSE + Separation ==========
            predicted_cps = generator(states)  # (B, CP, action_dim)
            loss_mse = mse_weight * lossMSE(predicted_cps, actions)
            loss_sep = separation_weight * lossSeparation(predicted_cps)
            loss_gen = loss_mse + loss_sep

            # ========== Estimator training: DIRECT InfoNCE (no wire-fitting) ==========
            # Positive: expert action
            # Negatives: uniform random actions
            counter_actions = (
                torch.rand(B, num_counter_examples, action_dim, device=device)
                * (action_max - action_min) + action_min
            )

            # Concatenate expert (index 0) with negatives
            all_actions = torch.cat([actions.unsqueeze(1), counter_actions], dim=1)
            states_exp = states.unsqueeze(1).expand(-1, 1 + num_counter_examples, -1)

            # Direct energy evaluation: E(obs, action)
            energies = estimator(states_exp, all_actions).squeeze(-1)  # (B, 1+N)

            # InfoNCE: expert should have HIGHEST Q-value
            # logits = Q-values directly (higher = better)
            logits = energies
            log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            loss_est = -log_probs[:, 0].mean()  # NLL of expert being ranked #1

            if torch.isnan(loss_est) or torch.isnan(loss_gen):
                print("NaN loss detected, skipping.")
                continue

            # ========== Update both models ==========
            optimizer_gen.zero_grad()
            optimizer_est.zero_grad()

            # Combined loss: generator loss + estimator loss
            total_loss = loss_gen + info_nce_weight * loss_est
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), 1.0)

            optimizer_gen.step()
            optimizer_est.step()
            scheduler_gen.step()
            scheduler_est.step()

            step += 1

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                current_lr = scheduler_gen.get_last_lr()[0]
                # Compute accuracy: is expert ranked #1?
                with torch.no_grad():
                    best_idx = logits.argmax(dim=1)
                    acc = (best_idx == 0).float().mean().item()
                print(f"  Step {step}/{training_steps} | "
                      f"Total: {total_loss.item():.4f} "
                      f"(MSE: {loss_mse.item():.4f}, "
                      f"Sep: {loss_sep.item():.4f}, "
                      f"EST: {loss_est.item():.4f}, "
                      f"Acc: {acc:.3f}) | "
                      f"LR: {current_lr:.2e} | {elapsed:.1f}s")

            if step % save_interval == 0:
                os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
                torch.save(generator.state_dict(),
                          os.path.join(MODEL_SAVE_DIR, f"control_point_generator_step_{step}.pt"))
                torch.save(estimator.state_dict(),
                          os.path.join(MODEL_SAVE_DIR, f"q_estimator_step_{step}.pt"))

    # ─── Save final models ────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.2f} min)")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(MODEL_SAVE_DIR, "control_point_generator.pt"))
    torch.save(estimator.state_dict(), os.path.join(MODEL_SAVE_DIR, "q_estimator.pt"))
    print(f"Models saved to {MODEL_SAVE_DIR}/")


if __name__ == "__main__":
    main()
