"""IBC training with Derivative-Free Optimizer (DFO) strategy for Dummy environment.

Trains a single energy-based model (QEstimator) using InfoNCE contrastive loss,
following the IBC paper (Florence et al., 2021, arXiv:2109.00137).

The energy model learns E(obs, action) such that expert actions have low energy
and random (uniform) actions have high energy. At inference time, actions are
found by iterative DFO optimization over the energy landscape.

Usage:
    python -m ibc.ibc_dfo_dummy_training
"""

import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from utils.models import QEstimator
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
training_steps = env_training.get("training_steps", 100000)
learning_rate = env_training.get("learning_rate", 0.001)
batch_size = env_training.get("batch_size", 128)
num_counter_examples = env_training.get("counter_examples", 64)
lr_decay = training_shared.get("lr_decay", 0.99)
lr_decay_interval = training_shared.get("lr_decay_interval", 100)
log_interval = training_shared.get("log_interval", 1000)
save_interval = training_shared.get("save_interval", 5000)

# Model parameters
num_hidden_layers = env_model.get("num_hidden_layers", 4)
num_neurons = env_model.get("num_neurons", 128)

# Environment parameters
state_dim = env_config["state_dim"]
action_dim = env_config["action_dim"]
action_bounds = env_config.get("action_bounds", [-1, 1])
frame_stack = env_config.get("frame_stack", 1)

# Save directory for IBC-DFO checkpoints
MODEL_SAVE_DIR = os.path.join(
    training_shared.get("model_save_dir", "checkpoints"), "ibc_dfo"
)


def main():
    global learning_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training IBC-DFO on Dummy environment")
    print(f"Training steps: {training_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Counter-examples (uniform): {num_counter_examples}")

    # ─── Dataset ──────────────────────────────────────────────────────────
    dataset = DummyDataset(
        size=10000,
        step_size=env_config.get("step_size", 0.1),
        goal_radius=env_config.get("goal_radius", 0.1),
        n_dim=env_config.get("n_dim", 2),
        frame_stack=frame_stack,
    )
    print(f"Dataset size: {len(dataset)}")

    # ─── Energy model (QEstimator) ────────────────────────────────────────
    energy_model = QEstimator(
        state_dim=dataset.state_shape,
        action_dim=dataset.action_shape,
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)],
    ).to(device)

    optimizer = torch.optim.AdamW(energy_model.parameters(), lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

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

            states = batch["state"].float().to(device)  # (B, state_dim)
            actions = batch["action"].float().to(device)  # (B, action_dim)

            B = states.shape[0]

            # Generate uniform counter-example actions
            counter_actions = (
                torch.rand(B, num_counter_examples, action_dim, device=device)
                * (action_max - action_min)
                + action_min
            )  # (B, N, action_dim)

            # Concatenate expert action (positive) with counter-examples (negatives)
            # Expert action is always at index 0
            all_actions = torch.cat(
                [actions.unsqueeze(1), counter_actions], dim=1
            )  # (B, 1+N, action_dim)

            # Expand states to match
            states_expanded = states.unsqueeze(1).expand(
                -1, 1 + num_counter_examples, -1
            )  # (B, 1+N, state_dim)

            # Evaluate energy for all candidates
            # Energy model outputs (B, 1+N, 1), we negate because lower energy = better
            energies = energy_model(states_expanded, all_actions).squeeze(
                -1
            )  # (B, 1+N)

            # InfoNCE loss: expert action (index 0) should have LOWEST energy
            # We negate energies so that lowest energy → highest logit → highest probability
            logits = -energies  # (B, 1+N) — higher logit for lower energy
            log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            loss = -log_probs[:, 0].mean()  # Negative log-probability of expert action

            if torch.isnan(loss):
                print(f"  Step {step}: NaN loss detected, skipping batch.")
                continue

            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # LR decay
            if step % lr_decay_interval == 0:
                learning_rate *= lr_decay
                optimizer.param_groups[0]["lr"] = learning_rate

            # Logging
            if step % log_interval == 0:
                elapsed = time.time() - start_time
                # Compute accuracy: is expert action ranked #1?
                with torch.no_grad():
                    best_idx = logits.argmax(dim=1)
                    accuracy = (best_idx == 0).float().mean().item()
                print(
                    f"  Step {step}/{training_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc: {accuracy:.3f} | "
                    f"LR: {learning_rate:.2e} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

            # Save checkpoint
            if step % save_interval == 0:
                os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
                torch.save(
                    energy_model.state_dict(),
                    os.path.join(MODEL_SAVE_DIR, f"q_estimator_step_{step}.pt"),
                )

    # ─── Save final model ─────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.2f} min)")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    final_path = os.path.join(MODEL_SAVE_DIR, "q_estimator.pt")
    torch.save(energy_model.state_dict(), final_path)
    print(f"Energy model saved to {final_path}")


if __name__ == "__main__":
    main()
