"""IBC + independent Control Point Generator training.

Variation of combinedv2_cpascounter_training.py where:
  - The QEstimator is trained exactly as in ibc/ibc_dfo_particle_training.py
    (InfoNCE + gradient penalty, Langevin MCMC negatives, exponential LR decay,
     action normalization, paper-faithful hyperparameters).
  - The ControlPointGenerator is trained independently with only MSE + separation
    losses (no InfoNCE/adversarial coupling to the estimator).

Uses config.json to determine which environment to train on.
"""

import os
import time
import json
import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import wandb

from utils.models import ControlPointGenerator, QEstimator
from utils.loss import lossInfoNCE, lossMSE, lossSeparation, lossEntropyKDE
from utils.normalizations import ObservationNormalizer

# ─── Load config ─────────────────────────────────────────────────────────────
config_path = Path(__file__).parent / "config_json" / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

# Get active environment
active_env = config.get("active_env", "pen")
env_config = config["environments"][active_env]
training_shared = config.get("training_shared", {})
env_training = env_config.get("training", {})
env_model = env_config.get("model", {})

# ─── Generator training parameters (from combined config) ───────────────────
training_steps = env_training.get("training_steps", training_shared.get("training_steps", 100000))
batch_size = env_training.get("batch_size", training_shared.get("batch_size", 128))
generator_learning_rate = env_training.get("learning_rate", training_shared.get("learning_rate", 1e-3))

separation_weight = training_shared.get("separation_weight", 0.1)
mse_weight = training_shared.get("mse_weight", 1.0)

MODEL_SAVE_DIR = training_shared.get("model_save_dir", "checkpoints")
log_interval = training_shared.get("log_interval", 1000)
save_interval = training_shared.get("save_interval", 10000)

separation_epsilon = env_training.get("separation_epsilon", training_shared.get("separation_epsilon", 1.0))
separation_loss_type = env_training.get("separation_loss", training_shared.get("separation_loss", "separation"))
entropy_bandwidth = env_training.get("entropy_bandwidth", training_shared.get("entropy_bandwidth", 0.1))

# Generator scheduler
scheduler_type = env_training.get(
    "scheduler_type",
    training_shared.get("scheduler_type", "cosine"),
)
cosine_t0 = env_training.get(
    "cosine_t0",
    training_shared.get("cosine_t0", 50000),
)
cosine_t_max = env_training.get(
    "cosine_t_max",
    training_shared.get("cosine_t_max", None),
)

# Model parameters
control_points = env_model.get("control_points", 50)
num_hidden_layers = env_model.get("num_hidden_layers", 8)
num_neurons = env_model.get("num_neurons", 512)

# ─── IBC estimator hyperparameters (paper-faithful defaults) ────────────────
ESTIMATOR_LEARNING_RATE = env_training.get(
    "estimator_learning_rate",
    training_shared.get("estimator_learning_rate", 1e-3),
)
LR_DECAY_RATE = 0.995
LR_DECAY_STEPS = 100
NUM_COUNTER_EXAMPLES = env_training.get(
    "counter_examples",
    training_shared.get("counter_examples", 16),
)
LANGEVIN_TRAIN_ITERATIONS = 100
LANGEVIN_STEPSIZE_INIT = 0.1
LANGEVIN_STEPSIZE_FINAL = 1e-5
LANGEVIN_STEPSIZE_POWER = 2.0
LANGEVIN_NOISE_SCALE = 1.0
LANGEVIN_DELTA_ACTION_CLIP = 0.1
GRADIENT_MARGIN = 1.0
SOFTMAX_TEMPERATURE = 0.5
UNIFORM_BOUNDARY_BUFFER = 0.05
ESTIMATOR_HIDDEN_DIMS = [num_neurons for _ in range(num_hidden_layers)]

# Environment parameters
env_id = env_config["env_id"]
state_dim = env_config["state_dim"]
action_dim = env_config["action_dim"]
action_bounds = env_config.get("action_bounds", [-1, 1])
frame_stack = env_config.get("frame_stack", 1)


def load_dataset():
    """Load the appropriate dataset based on active_env."""
    if active_env == "pen":
        from utils.datasets import D4RLDataset
        dataset_name = env_config["dataset_name"]
        return D4RLDataset(dataset_name, download=True, frame_stack=frame_stack)
    elif active_env == "particle":
        from utils.datasets import ParticleDataset
        data_dir = env_config["data_dir"]
        n_dim = env_config.get("n_dim", 2)
        return ParticleDataset(data_dir, n_dim=n_dim, frame_stack=frame_stack)
    elif active_env == "dummy":
        from utils.datasets import DummyDataset
        return DummyDataset(
            size=10000,
            step_size=env_config.get("step_size", 0.1),
            goal_radius=env_config.get("goal_radius", 0.05),
            n_dim=env_config.get("n_dim", 2),
            frame_stack=frame_stack,
        )
    else:
        raise ValueError(f"Unknown environment: {active_env}")


def compute_dataset_stats(dataset):
    """Compute min/max statistics from dataset for action normalization."""
    acts = dataset.actions
    return {
        "act_min": acts.min(axis=0).astype(np.float32),
        "act_max": acts.max(axis=0).astype(np.float32),
    }


def normalize_actions(actions, act_min, act_max, device):
    """Normalize actions to [0, 1] using precomputed min/max."""
    rng = act_max - act_min
    rng = np.where(rng == 0, np.ones_like(rng), rng)
    act_min_t = torch.from_numpy(act_min).float().to(device)
    rng_t = torch.from_numpy(rng).float().to(device)
    return (actions - act_min_t) / rng_t


def langevin_counter_examples(q_model, obs_norm, device):
    """Generate hard negative actions via Langevin MCMC in normalized space.

    Q-value convention (high = expert): Langevin ASCENDS the Q function to
    find actions the model currently rates highly that are NOT the expert —
    exactly the hard negatives InfoNCE needs to push down.
    Operates entirely in normalized action space [0, 1] (with small buffer).
    """
    B = obs_norm.shape[0]

    act_min = 0.0 - UNIFORM_BOUNDARY_BUFFER
    act_max = 1.0 + UNIFORM_BOUNDARY_BUFFER

    actions = (
        torch.rand(B, NUM_COUNTER_EXAMPLES, action_dim, device=device)
        * (act_max - act_min) + act_min
    )

    delta_clip = LANGEVIN_DELTA_ACTION_CLIP * 0.5 * (act_max - act_min)

    obs_expanded = obs_norm.unsqueeze(1).expand(-1, NUM_COUNTER_EXAMPLES, -1)

    for p in q_model.parameters():
        p.requires_grad_(False)

    for k in range(LANGEVIN_TRAIN_ITERATIONS):
        frac = 1.0 - k / max(LANGEVIN_TRAIN_ITERATIONS - 1, 1)
        stepsize = (
            LANGEVIN_STEPSIZE_FINAL
            + (LANGEVIN_STEPSIZE_INIT - LANGEVIN_STEPSIZE_FINAL)
            * (frac ** LANGEVIN_STEPSIZE_POWER)
        )

        actions = actions.detach().requires_grad_(True)
        q_vals = q_model(obs_expanded, actions).squeeze(-1)
        grad = torch.autograd.grad(q_vals.sum(), actions)[0]
        grad = grad.detach()

        noise = torch.randn_like(actions) * LANGEVIN_NOISE_SCALE
        delta = stepsize * (0.5 * grad + noise)
        delta = torch.clamp(delta, -delta_clip, delta_clip)

        # Ascend Q to find hard negatives (regions model thinks are expert-like).
        actions = (actions.detach() + delta).detach()
        actions = torch.clamp(actions, act_min, act_max)

    for p in q_model.parameters():
        p.requires_grad_(True)

    return actions.detach()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Active environment: {active_env}")
    print(f"Training steps: {training_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Generator LR: {generator_learning_rate} (scheduler: {scheduler_type})")
    print(f"Estimator LR: {ESTIMATOR_LEARNING_RATE} (exponential decay: rate={LR_DECAY_RATE}, every {LR_DECAY_STEPS} steps)")
    print(f"Estimator architecture: MLP {ESTIMATOR_HIDDEN_DIMS}")
    print(f"Langevin counter-examples: {NUM_COUNTER_EXAMPLES} ({LANGEVIN_TRAIN_ITERATIONS} iters)")
    print(f"Gradient penalty margin: {GRADIENT_MARGIN}")
    print(f"Frame stack: {frame_stack}")

    wandb.init(
        project="Q3CIBC",
        config={
            "active_env": active_env,
            "env_config": env_config,
            "training_shared": training_shared,
            "estimator_lr": ESTIMATOR_LEARNING_RATE,
            "lr_decay_rate": LR_DECAY_RATE,
            "lr_decay_steps": LR_DECAY_STEPS,
            "num_counter_examples": NUM_COUNTER_EXAMPLES,
            "langevin_iterations": LANGEVIN_TRAIN_ITERATIONS,
            "gradient_margin": GRADIENT_MARGIN,
            "softmax_temperature": SOFTMAX_TEMPERATURE,
        },
        name=f"{active_env}_ibc_with_cps_cp{control_points}_lr{generator_learning_rate}",
    )

    # Load dataset
    print(f"Loading {active_env} dataset...")
    dataset = load_dataset()
    print(f"Dataset size: {len(dataset)}")

    if active_env == "particle" and hasattr(dataset, "_episode_starts"):
        episode_starts = int(dataset._episode_starts.sum())
        tfrecord_count = len(getattr(dataset, "tfrecord_files", []))
        avg_episode_length = len(dataset) / max(episode_starts, 1)
        print(
            f"Particle dataset episodes: {episode_starts} | "
            f"Avg samples/episode: {avg_episode_length:.2f}"
        )
        if tfrecord_count and episode_starts <= tfrecord_count:
            raise RuntimeError(
                "Particle dataset episode boundary parsing looks wrong: "
                f"detected {episode_starts} episode starts across {tfrecord_count} TFRecord files. "
                "This usually means step_type decoding failed and frame stacking would mix episodes."
            )

    # Compute action normalization stats for the estimator
    norm_stats = compute_dataset_stats(dataset)
    print(
        f"Act range: [{norm_stats['act_min'].min():.3f}, "
        f"{norm_stats['act_max'].max():.3f}]"
    )

    # Pre-compute normalization tensors for generator CP normalization
    act_min_t = torch.from_numpy(norm_stats["act_min"]).float().to(device)
    act_range_np = norm_stats["act_max"] - norm_stats["act_min"]
    act_range_np = np.where(act_range_np == 0, np.ones_like(act_range_np), act_range_np)
    act_rng_t = torch.from_numpy(act_range_np.astype(np.float32)).to(device)

    # ─── Create models ────────────────────────────────────────────────────
    control_point_generator = ControlPointGenerator(
        input_dim=dataset.state_shape,
        output_dim=dataset.action_shape,
        control_points=control_points,
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)],
        action_bounds=(action_bounds[0], action_bounds[1]),
    ).to(device)

    # Estimator uses paper-faithful architecture (no spectral norm)
    estimator = QEstimator(
        state_dim=dataset.state_shape,
        action_dim=dataset.action_shape,
        hidden_dims=ESTIMATOR_HIDDEN_DIMS,
    ).to(device)

    # ─── Optimizers ───────────────────────────────────────────────────────
    optimizer_generator = torch.optim.AdamW(
        control_point_generator.parameters(), lr=generator_learning_rate
    )
    # Estimator uses Adam (paper-faithful)
    optimizer_estimator = torch.optim.Adam(
        estimator.parameters(), lr=ESTIMATOR_LEARNING_RATE
    )

    # Generator LR schedule (cosine, from combined config)
    effective_t_max = cosine_t_max if cosine_t_max is not None else training_steps
    if scheduler_type == "cosine_warm_restarts":
        scheduler_generator = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_generator, T_0=cosine_t0, eta_min=1e-6
        )
    else:
        scheduler_generator = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_generator, T_max=effective_t_max, eta_min=1e-6
        )

    # Estimator LR: exponential decay (paper-faithful), managed manually
    estimator_current_lr = ESTIMATOR_LEARNING_RATE

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    # Observation normalizer
    particle_n_dim = env_config.get("n_dim") if active_env == "particle" else None
    obs_normalizer = ObservationNormalizer(
        env_id=env_id,
        device=device,
        frame_stack=frame_stack,
        particle_n_dim=particle_n_dim,
    )

    # Training timing
    start_time = time.time()
    step = 0

    while step < training_steps:
        for batch in dataloader:
            if step >= training_steps:
                break

            states = batch["state"].float().to(device)
            states_norm = obs_normalizer.normalize(states)
            actions = batch["action"].float().to(device)
            B = states.shape[0]

            # Normalize actions for the estimator (into [0,1])
            actions_norm = normalize_actions(
                actions, norm_stats["act_min"], norm_stats["act_max"], device
            )

            # ============================================================
            # Generator loss: MSE + Separation (independent, no estimator)
            # ============================================================
            predicted_actions = control_point_generator(states_norm)

            loss_mse = mse_weight * (lossMSE(predicted_actions, actions) / B)
            if separation_loss_type == "entropy":
                loss_sep = separation_weight * lossEntropyKDE(
                    predicted_actions, bandwidth=entropy_bandwidth
                )
            elif separation_loss_type == "separation":
                loss_sep = separation_weight * (
                    lossSeparation(predicted_actions, epsilon=separation_epsilon) / B
                )
            else:
                raise ValueError(
                    f"Unknown separation_loss '{separation_loss_type}'. "
                    "Expected 'separation' or 'entropy'."
                )
            loss_generator = loss_mse + loss_sep

            # ============================================================
            # Estimator loss: InfoNCE (Q-value convention) + gradient penalty
            # Counter-examples: generator control points (detached) + Langevin
            # hard-negatives. Expert at index 0 (matches lossInfoNCE convention).
            # ============================================================
            # Normalize generator control points into estimator's training space
            cp_norm = (predicted_actions.detach() - act_min_t) / act_rng_t

            langevin_counter = langevin_counter_examples(
                estimator, states_norm, device
            )  # (B, N_lang, action_dim) in normalized space

            counter_actions = torch.cat([cp_norm, langevin_counter], dim=1)
            n_counter = counter_actions.shape[1]

            # Expert at index 0 (Q-value convention; matches utils.loss.lossInfoNCE)
            all_actions = torch.cat(
                [actions_norm.unsqueeze(1), counter_actions], dim=1
            )  # (B, 1+N, action_dim)

            states_expanded = states_norm.unsqueeze(1).expand(-1, 1 + n_counter, -1)

            q_values = estimator(states_expanded, all_actions).squeeze(-1)

            # InfoNCE: expert (index 0) should have the HIGHEST Q-value
            logits = q_values / SOFTMAX_TEMPERATURE
            log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            loss_infonce = -log_probs[:, 0].mean()

            # Gradient penalty on expert + Langevin hard-negatives only
            # (generator CPs are detached anyway; skipping them keeps the GP
            # paper-faithful and avoids OOM from the 2nd-order graph.)
            gp_all = torch.cat(
                [actions_norm.unsqueeze(1), langevin_counter], dim=1
            )
            n_gp = gp_all.shape[1]
            gp_actions = gp_all.detach().reshape(B * n_gp, -1).requires_grad_(True)
            gp_states = (
                states_norm.unsqueeze(1).expand(-1, n_gp, -1).detach().reshape(B * n_gp, -1)
            )

            gp_q = estimator(gp_states, gp_actions)
            grad_gp = torch.autograd.grad(
                gp_q.sum(), gp_actions, create_graph=True
            )[0]

            grad_norms = grad_gp.abs().max(dim=-1).values
            grad_penalty = (
                torch.clamp(grad_norms - GRADIENT_MARGIN, min=0).pow(2).mean()
            )

            loss_estimator = loss_infonce + grad_penalty

            # ============================================================
            # NaN guard
            # ============================================================
            if torch.isnan(loss_generator) or torch.isnan(loss_estimator):
                print("NaN loss detected, skipping this batch.")
                continue

            # ============================================================
            # Update models independently
            # ============================================================
            optimizer_generator.zero_grad()
            loss_generator.backward()
            torch.nn.utils.clip_grad_norm_(control_point_generator.parameters(), 1.0)
            optimizer_generator.step()
            scheduler_generator.step()

            optimizer_estimator.zero_grad()
            loss_estimator.backward()
            optimizer_estimator.step()

            # Estimator exponential LR decay (paper-faithful)
            step += 1
            if step % LR_DECAY_STEPS == 0:
                estimator_current_lr *= LR_DECAY_RATE
                for pg in optimizer_estimator.param_groups:
                    pg["lr"] = estimator_current_lr

            # ─── Logging ──────────────────────────────────────────────
            if step % log_interval == 0:
                gen_lr = scheduler_generator.get_last_lr()[0]

                with torch.no_grad():
                    best_idx = q_values.argmax(dim=1)
                    accuracy = (best_idx == 0).float().mean().item()

                elapsed = time.time() - start_time
                print(
                    f"Step {step}/{training_steps} | "
                    f"Gen: {loss_generator.item():.4f} "
                    f"(MSE: {loss_mse.item():.4f}, Sep: {loss_sep.item():.4f}) | "
                    f"Est: {loss_estimator.item():.4f} "
                    f"(NCE: {loss_infonce.item():.4f}, GP: {grad_penalty.item():.4f}) | "
                    f"Acc: {accuracy:.3f} | "
                    f"GenLR: {gen_lr:.2e} EstLR: {estimator_current_lr:.2e} | "
                    f"{elapsed:.1f}s"
                )

                wandb.log({
                    "step": step,
                    "loss/generator": loss_generator.item(),
                    "loss/mse": loss_mse.item(),
                    "loss/separation": loss_sep.item(),
                    "loss/estimator": loss_estimator.item(),
                    "loss/infonce": loss_infonce.item(),
                    "loss/gradient_penalty": grad_penalty.item(),
                    "metric/accuracy": accuracy,
                    "learning_rate/generator": gen_lr,
                    "learning_rate/estimator": estimator_current_lr,
                })

            # ─── Checkpoints ─────────────────────────────────────────
            if step % save_interval == 0:
                os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
                torch.save(
                    control_point_generator.state_dict(),
                    os.path.join(MODEL_SAVE_DIR, f"control_point_generator_step_{step}.pt"),
                )
                torch.save(
                    estimator.state_dict(),
                    os.path.join(MODEL_SAVE_DIR, f"q_estimator_step_{step}.pt"),
                )

    # ─── Training complete ────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.2f} min)")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save(
        control_point_generator.state_dict(),
        os.path.join(MODEL_SAVE_DIR, "control_point_generator.pt"),
    )
    torch.save(
        estimator.state_dict(),
        os.path.join(MODEL_SAVE_DIR, "q_estimator.pt"),
    )
    # Save norm_stats separately for inference scripts that need it
    torch.save(norm_stats, os.path.join(MODEL_SAVE_DIR, "norm_stats.pt"))

    # Remove stale smoothing param if exists
    smoothing_param_path = os.path.join(MODEL_SAVE_DIR, "smoothing_param.pt")
    if os.path.exists(smoothing_param_path):
        os.remove(smoothing_param_path)
        print(f"Removed stale {smoothing_param_path}")

    print(f"Models saved to {MODEL_SAVE_DIR}/")

    artifact = wandb.Artifact("model-checkpoints", type="model")
    artifact.add_file(os.path.join(MODEL_SAVE_DIR, "control_point_generator.pt"))
    artifact.add_file(os.path.join(MODEL_SAVE_DIR, "q_estimator.pt"))
    wandb.log_artifact(artifact)

    wandb.summary["total_training_time_min"] = total_time / 60
    wandb.finish()


if __name__ == "__main__":
    main()
