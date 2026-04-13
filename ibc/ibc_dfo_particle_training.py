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
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone

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
NUM_COUNTER_EXAMPLES = 16
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

LOG_INTERVAL = 200
SAVE_INTERVAL = 5000

MODEL_SAVE_DIR = os.path.join(
    training_shared.get("model_save_dir", "checkpoints"), "ibc_dfo", "particle"
)
RESULTS_DIR = os.path.join("results", "ibc_dfo", "particle")
TRAIN_LEDGER_PATH = os.path.join(RESULTS_DIR, "training_runs.jsonl")


# Strict experiment budget: max 10 runs.
# Select run with env var IBC_PARTICLE_RUN_ID in [1, 10].
RUN_ID_ENV = "IBC_PARTICLE_RUN_ID"
MAX_RUNS = 10


PAPER_BASELINE_EXPECTED = {
    "TRAINING_STEPS": 100_000,
    "BATCH_SIZE": 512,
    "LEARNING_RATE": 1e-3,
    "LR_DECAY_RATE": 0.99,
    "LR_DECAY_STEPS": 100,
    "NUM_COUNTER_EXAMPLES": 16,
    "LANGEVIN_TRAIN_ITERATIONS": 100,
    "LANGEVIN_STEPSIZE_INIT": 0.1,
    "LANGEVIN_STEPSIZE_FINAL": 1e-5,
    "LANGEVIN_STEPSIZE_POWER": 2.0,
    "LANGEVIN_NOISE_SCALE": 1.0,
    "LANGEVIN_DELTA_ACTION_CLIP": 0.1,
    "GRADIENT_MARGIN": 1.0,
    "SOFTMAX_TEMPERATURE": 1.0,
    "UNIFORM_BOUNDARY_BUFFER": 0.05,
    "HIDDEN_DIMS": [256, 256],
}


def get_run_schedule():
    """Return the run schedule (run 1 is paper-faithful baseline).

    Runs 1-4: original experiments (already trained).
    Runs 5-10: targeted improvements based on 50-seed eval analysis:
      - Baseline (run 1) = 80% success, best of existing checkpoints
      - Inference tuning ceiling = 82% (noise_scale=0.1)
      - Model quality is the bottleneck, not inference
    """
    return {
        1: {
            "name": "baseline_paper",
            "overrides": {},
        },
        2: {
            "name": "counter_examples_16",
            "overrides": {"NUM_COUNTER_EXAMPLES": 16},
        },
        3: {
            "name": "counter_examples_32",
            "overrides": {"NUM_COUNTER_EXAMPLES": 32},
        },
        4: {
            "name": "counter_examples_64",
            "overrides": {"NUM_COUNTER_EXAMPLES": 64},
        },
        5: {
            "name": "temp_0p1_quick",
            "overrides": {
                "SOFTMAX_TEMPERATURE": 0.1,
                "LANGEVIN_TRAIN_ITERATIONS": 50,
                "TRAINING_STEPS": 10_000,
            },
        },
        6: {
            "name": "temp_0p5_quick",
            "overrides": {
                "SOFTMAX_TEMPERATURE": 0.5,
                "LANGEVIN_TRAIN_ITERATIONS": 50,
                "TRAINING_STEPS": 10_000,
            },
        },
        7: {
            "name": "lr3e4_quick",
            "overrides": {
                "LEARNING_RATE": 3e-4,
                "LR_DECAY_RATE": 0.995,
                "LANGEVIN_TRAIN_ITERATIONS": 50,
                "TRAINING_STEPS": 10_000,
            },
        },
        8: {
            "name": "ce8_quick",
            "overrides": {
                "NUM_COUNTER_EXAMPLES": 8,
                "LANGEVIN_TRAIN_ITERATIONS": 50,
                "TRAINING_STEPS": 10_000,
            },
        },
        9: {
            "name": "baseline_langevin50_quick",
            "overrides": {
                "LANGEVIN_TRAIN_ITERATIONS": 50,
                "TRAINING_STEPS": 10_000,
            },
        },
        10: {
            "name": "best_combo_full",
            "overrides": {},
        },
    }


def resolve_run_id(run_id_override=None):
    """Resolve run id from CLI override or environment variable."""
    if run_id_override is not None:
        run_id = int(run_id_override)
    else:
        run_id_str = os.getenv(RUN_ID_ENV, "1").strip()
        if not run_id_str.isdigit():
            raise ValueError(
                f"{RUN_ID_ENV} must be an integer in [1, {MAX_RUNS}], got: {run_id_str!r}"
            )
        run_id = int(run_id_str)

    if run_id < 1 or run_id > MAX_RUNS:
        raise ValueError(
            f"{RUN_ID_ENV}={run_id} is outside allowed range [1, {MAX_RUNS}]"
        )
    return run_id


def build_hparams_from_run_id(run_id):
    """Build mutable hyperparameters from baseline + scheduled run overrides."""

    baseline = {
        "TRAINING_STEPS": TRAINING_STEPS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "LR_DECAY_RATE": LR_DECAY_RATE,
        "LR_DECAY_STEPS": LR_DECAY_STEPS,
        "NUM_COUNTER_EXAMPLES": NUM_COUNTER_EXAMPLES,
        "LANGEVIN_TRAIN_ITERATIONS": LANGEVIN_TRAIN_ITERATIONS,
        "LANGEVIN_STEPSIZE_INIT": LANGEVIN_STEPSIZE_INIT,
        "LANGEVIN_STEPSIZE_FINAL": LANGEVIN_STEPSIZE_FINAL,
        "LANGEVIN_STEPSIZE_POWER": LANGEVIN_STEPSIZE_POWER,
        "LANGEVIN_NOISE_SCALE": LANGEVIN_NOISE_SCALE,
        "LANGEVIN_DELTA_ACTION_CLIP": LANGEVIN_DELTA_ACTION_CLIP,
        "GRADIENT_MARGIN": GRADIENT_MARGIN,
        "SOFTMAX_TEMPERATURE": SOFTMAX_TEMPERATURE,
        "UNIFORM_BOUNDARY_BUFFER": UNIFORM_BOUNDARY_BUFFER,
        "HIDDEN_DIMS": deepcopy(HIDDEN_DIMS),
    }

    schedule = get_run_schedule()
    trial = schedule[run_id]
    hparams = deepcopy(baseline)
    hparams.update(trial["overrides"])
    hparams["RUN_ID"] = run_id
    hparams["RUN_NAME"] = trial["name"]
    return hparams


def check_paper_alignment(hparams):
    """Validate baseline constants against paper-faithful reference values."""
    mismatches = []
    for key, expected in PAPER_BASELINE_EXPECTED.items():
        got = hparams[key]
        if got != expected:
            mismatches.append((key, expected, got))

    if mismatches:
        mismatch_text = "\n".join(
            [f"  {k}: expected {exp}, got {got}" for k, exp, got in mismatches]
        )
        raise ValueError(
            "Paper-alignment gate failed. Baseline constants diverge from "
            f"expected values:\n{mismatch_text}"
        )


def append_jsonl(path, payload):
    """Append one JSON record as a single line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


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


def langevin_counter_examples(energy_model, obs_norm, device, hparams):
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
    act_min = 0.0 - hparams["UNIFORM_BOUNDARY_BUFFER"]
    act_max = 1.0 + hparams["UNIFORM_BOUNDARY_BUFFER"]

    # Scaled delta clip (paper: delta_action_clip * 0.5 * (max - min))
    n_counter = hparams["NUM_COUNTER_EXAMPLES"]

    actions = (
        torch.rand(B, n_counter, action_dim, device=device)
        * (act_max - act_min) + act_min
    )

    delta_clip = hparams["LANGEVIN_DELTA_ACTION_CLIP"] * 0.5 * (act_max - act_min)

    obs_expanded = obs_norm.unsqueeze(1).expand(-1, n_counter, -1)

    # Disable param gradients during Langevin chain (stop_chain_grad)
    for p in energy_model.parameters():
        p.requires_grad_(False)

    for k in range(hparams["LANGEVIN_TRAIN_ITERATIONS"]):
        # Polynomial decay stepsize
        frac = 1.0 - k / max(hparams["LANGEVIN_TRAIN_ITERATIONS"] - 1, 1)
        stepsize = (
            hparams["LANGEVIN_STEPSIZE_FINAL"]
            + (hparams["LANGEVIN_STEPSIZE_INIT"] - hparams["LANGEVIN_STEPSIZE_FINAL"])
            * (frac ** hparams["LANGEVIN_STEPSIZE_POWER"])
        )

        actions = actions.detach().requires_grad_(True)
        energies = energy_model(obs_expanded, actions).squeeze(-1)
        grad = torch.autograd.grad(energies.sum(), actions)[0]
        grad = grad.detach()

        # Langevin step: gradient descent on energy + noise
        noise = torch.randn_like(actions) * hparams["LANGEVIN_NOISE_SCALE"]
        delta = stepsize * (0.5 * grad + noise)
        delta = torch.clamp(delta, -delta_clip, delta_clip)

        actions = (actions.detach() - delta).detach()
        actions = torch.clamp(actions, act_min, act_max)

    # Re-enable param gradients for training
    for p in energy_model.parameters():
        p.requires_grad_(True)

    return actions.detach()


def run_single_experiment(hparams):
    check_paper_alignment({
        "TRAINING_STEPS": TRAINING_STEPS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "LR_DECAY_RATE": LR_DECAY_RATE,
        "LR_DECAY_STEPS": LR_DECAY_STEPS,
        "NUM_COUNTER_EXAMPLES": NUM_COUNTER_EXAMPLES,
        "LANGEVIN_TRAIN_ITERATIONS": LANGEVIN_TRAIN_ITERATIONS,
        "LANGEVIN_STEPSIZE_INIT": LANGEVIN_STEPSIZE_INIT,
        "LANGEVIN_STEPSIZE_FINAL": LANGEVIN_STEPSIZE_FINAL,
        "LANGEVIN_STEPSIZE_POWER": LANGEVIN_STEPSIZE_POWER,
        "LANGEVIN_NOISE_SCALE": LANGEVIN_NOISE_SCALE,
        "LANGEVIN_DELTA_ACTION_CLIP": LANGEVIN_DELTA_ACTION_CLIP,
        "GRADIENT_MARGIN": GRADIENT_MARGIN,
        "SOFTMAX_TEMPERATURE": SOFTMAX_TEMPERATURE,
        "UNIFORM_BOUNDARY_BUFFER": UNIFORM_BOUNDARY_BUFFER,
        "HIDDEN_DIMS": HIDDEN_DIMS,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training IBC (paper-faithful) on Particle environment")
    print(f"Run: {hparams['RUN_ID']}/{MAX_RUNS} ({hparams['RUN_NAME']})")
    print(
        f"Steps: {hparams['TRAINING_STEPS']}, "
        f"Batch: {hparams['BATCH_SIZE']}, "
        f"LR: {hparams['LEARNING_RATE']}"
    )
    print(
        f"Counter-examples: {hparams['NUM_COUNTER_EXAMPLES']} "
        f"(Langevin, {hparams['LANGEVIN_TRAIN_ITERATIONS']} iters)"
    )
    print(f"Architecture: MLP {hparams['HIDDEN_DIMS']}")
    print(f"Grad penalty: L-inf, margin={hparams['GRADIENT_MARGIN']}")

    run_model_save_dir = os.path.join(
        MODEL_SAVE_DIR, f"run_{hparams['RUN_ID']:02d}_{hparams['RUN_NAME']}"
    )
    run_start_time = time.time()

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
        hidden_dims=hparams["HIDDEN_DIMS"],
    ).to(device)

    optimizer = torch.optim.Adam(
        energy_model.parameters(), lr=hparams["LEARNING_RATE"]
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=hparams["BATCH_SIZE"], shuffle=True, drop_last=True,
    )

    current_lr = hparams["LEARNING_RATE"]

    # ─── Training loop ────────────────────────────────────────────────────
    start_time = time.time()
    step = 0
    last_loss = None
    last_loss_infonce = None
    last_grad_penalty = None
    last_accuracy = None

    print("\nStarting training...")
    while step < hparams["TRAINING_STEPS"]:
        for batch in dataloader:
            if step >= hparams["TRAINING_STEPS"]:
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
                energy_model, states_norm, device, hparams,
            )  # (B, N, action_dim) in normalized space

            # Combine: counter-examples first, expert LAST (paper convention)
            all_actions = torch.cat(
                [counter_actions, actions_norm.unsqueeze(1)], dim=1
            )  # (B, N+1, action_dim)

            n_counter = hparams["NUM_COUNTER_EXAMPLES"]
            states_expanded = states_norm.unsqueeze(1).expand(
                -1, n_counter + 1, -1
            )

            # Compute energies for InfoNCE
            energies = energy_model(states_expanded, all_actions).squeeze(-1)

            # InfoNCE: expert at LAST index should have lowest energy
            logits = -energies / hparams["SOFTMAX_TEMPERATURE"]
            log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            loss_infonce = -log_probs[:, -1].mean()

            # Gradient penalty on ALL combined actions (L-inf norm, margin)
            gp_actions = all_actions.detach().reshape(
                B * (n_counter + 1), -1
            )
            gp_actions = gp_actions.requires_grad_(True)
            gp_states = states_expanded.detach().reshape(
                B * (n_counter + 1), -1
            )

            gp_energies = energy_model(gp_states, gp_actions)
            grad_gp = torch.autograd.grad(
                gp_energies.sum(), gp_actions, create_graph=True
            )[0]

            # L-inf norm
            grad_norms = grad_gp.abs().max(dim=-1).values
            grad_penalty = torch.clamp(
                grad_norms - hparams["GRADIENT_MARGIN"], min=0
            ).pow(2).mean()

            loss = loss_infonce + grad_penalty

            if torch.isnan(loss):
                print(f"  Step {step}: NaN loss, skipping.")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                best_idx = logits.argmax(dim=1)
                accuracy = (best_idx == n_counter).float().mean().item()
            last_loss = float(loss.item())
            last_loss_infonce = float(loss_infonce.item())
            last_grad_penalty = float(grad_penalty.item())
            last_accuracy = float(accuracy)

            step += 1

            # Exponential LR decay
            if step % hparams["LR_DECAY_STEPS"] == 0:
                current_lr *= hparams["LR_DECAY_RATE"]
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

            # Logging
            if step % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Step {step}/{hparams['TRAINING_STEPS']} | "
                    f"Loss: {loss.item():.4f} "
                    f"(NCE: {loss_infonce.item():.4f}, "
                    f"GP: {grad_penalty.item():.4f}) | "
                    f"Acc: {accuracy:.3f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Elapsed: {elapsed:.1f}s",
                    flush=True,
                )

            # Save checkpoint
            if step % SAVE_INTERVAL == 0:
                os.makedirs(run_model_save_dir, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": energy_model.state_dict(),
                        "norm_stats": norm_stats,
                        "step": step,
                        "run_id": hparams["RUN_ID"],
                        "run_name": hparams["RUN_NAME"],
                        "hparams": hparams,
                    },
                    os.path.join(
                        run_model_save_dir, f"q_estimator_step_{step}.pt"
                    ),
                )

    # ─── Save final model ─────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.2f} min)")

    os.makedirs(run_model_save_dir, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    final_path = os.path.join(run_model_save_dir, "q_estimator.pt")
    legacy_final_path = os.path.join(MODEL_SAVE_DIR, "q_estimator.pt")
    final_payload = {
        "model_state_dict": energy_model.state_dict(),
        "norm_stats": norm_stats,
        "step": hparams["TRAINING_STEPS"],
        "run_id": hparams["RUN_ID"],
        "run_name": hparams["RUN_NAME"],
        "hparams": hparams,
    }
    torch.save(
        final_payload,
        final_path,
    )
    torch.save(final_payload, legacy_final_path)
    print(f"Energy model saved to {final_path}")
    print(f"Legacy-compatible model saved to {legacy_final_path}")

    train_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": hparams["RUN_ID"],
        "run_name": hparams["RUN_NAME"],
        "checkpoint_dir": run_model_save_dir,
        "final_checkpoint": final_path,
        "legacy_checkpoint": legacy_final_path,
        "training_steps": hparams["TRAINING_STEPS"],
        "duration_seconds": total_time,
        "hparams": hparams,
        "final_train_metrics": {
            "loss": last_loss,
            "infonce_loss": last_loss_infonce,
            "gradient_penalty": last_grad_penalty,
            "accuracy": last_accuracy,
            "final_lr": current_lr,
        },
    }
    summary_path = os.path.join(run_model_save_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)
    append_jsonl(TRAIN_LEDGER_PATH, train_summary)
    print(f"Training summary saved to {summary_path}")
    print(f"Training ledger appended to {TRAIN_LEDGER_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Train IBC-DFO on Particle with optional hyperparameter exploration sweep."
    )
    parser.add_argument(
        "--hyperparameter_exploration",
        action="store_true",
        help="Run the full scheduled sweep across all configured runs (1-10).",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help=(
            "Run id in [1, 10] for single-run mode. "
            "If omitted, reads IBC_PARTICLE_RUN_ID (default 1)."
        ),
    )
    args = parser.parse_args()

    if args.hyperparameter_exploration:
        if args.run_id is not None:
            raise ValueError("--run_id cannot be combined with --hyperparameter_exploration")

        print(f"Running hyperparameter exploration sweep over {MAX_RUNS} runs")
        for run_id in range(1, MAX_RUNS + 1):
            print("\n" + "=" * 85)
            print(f"Starting exploration run {run_id}/{MAX_RUNS}")
            print("=" * 85)
            hparams = build_hparams_from_run_id(run_id)
            run_single_experiment(hparams)
        return

    run_id = resolve_run_id(args.run_id)
    hparams = build_hparams_from_run_id(run_id)
    run_single_experiment(hparams)


if __name__ == "__main__":
    main()
