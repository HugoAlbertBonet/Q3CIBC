"""Combined training script for generator (MSE) and estimator (Uniform/Langevin).

Uses config.json to determine which environment to train on.
Set "active_env" in config to switch between environments.
"""

import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import wandb

from utils.models import ControlPointGenerator, QEstimator
from utils.loss import lossInfoNCE, lossSeparation, lossEntropyKDE
from utils.normalizations import ObservationNormalizer
from utils.sampling import sample_langevin

# Load config
config_path = Path(__file__).parent / "config_json" / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

# Get active environment
active_env = config.get("active_env", "pen")
env_config = config["environments"][active_env]
training_shared = config.get("training_shared", {})
env_training = env_config.get("training", {})
env_model = env_config.get("model", {})

# Training parameters (merge env-specific with shared, env-specific takes priority)
training_steps = env_training.get("training_steps", training_shared.get("training_steps", 100000))
batch_size = env_training.get("batch_size", training_shared.get("batch_size", 128))
learning_rate = env_training.get("learning_rate", training_shared.get("learning_rate", 1e-3))

# Q3C IBC Loss parameters
separation_weight = training_shared.get("separation_weight", 0.1)
mse_weight = training_shared.get("mse_weight", 1.0)
info_nce_weight = training_shared.get("info_nce_weight", 1.0)
generator_infonce_weight = env_training.get(
    "generator_infonce_weight",
    training_shared.get("generator_infonce_weight", 0.05),
)

MODEL_SAVE_DIR = training_shared.get("model_save_dir", "checkpoints")
log_interval = training_shared.get("log_interval", 1000)
save_interval = training_shared.get("save_interval", 10000)

num_counter_examples = env_training.get("counter_examples", training_shared.get("counter_examples", 64))
sampling_method = env_training.get("sampling_method", training_shared.get("sampling_method", "uniform"))
top_k_control_points = env_training.get(
    "top_k_control_points",
    training_shared.get("top_k_control_points", num_counter_examples),
)

langevin_config = env_model.get("langevin_config", {})
langevin_num_iterations = langevin_config.get("num_iterations", 50)
langevin_lr_init = langevin_config.get("lr_init", 0.1)
langevin_lr_final = langevin_config.get("lr_final", 1e-5)
langevin_decay_power = langevin_config.get("polynomial_decay_power", 2.0)
langevin_delta_clip = langevin_config.get("delta_action_clip", 0.1)
langevin_noise_scale = langevin_config.get("noise_scale", 1.0)
# Separation loss epsilon: must be << action-space diameter so overlapping control
# points are strongly repelled.  Default 1.0 is too large for particle's [0,1]^2.
separation_epsilon = env_training.get("separation_epsilon", training_shared.get("separation_epsilon", 1.0))
separation_loss_type = env_training.get("separation_loss", training_shared.get("separation_loss", "separation"))
entropy_bandwidth = env_training.get("entropy_bandwidth", training_shared.get("entropy_bandwidth", 0.1))

# Model parameters
control_points = env_model.get("control_points", 50)
num_hidden_layers = env_model.get("num_hidden_layers", 8)
num_neurons = env_model.get("num_neurons", 512)

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


def main():
    global learning_rate
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Active environment: {active_env}")
    print(f"Training steps: {training_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Top-k control points as counter examples: {top_k_control_points}")
    print(f"Langevin counter examples (from counter_examples): {num_counter_examples}")
    print(f"Frame stack: {frame_stack}")
    
    # Initialize Weights & Biases
    wandb.init(
        project="Q3CIBC",
        config={
            "active_env": active_env,
            "env_config": env_config,
            "training_shared": training_shared,
        },
        name=f"{active_env}_combined_cp{control_points}_lr{learning_rate}"
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
    
    # Create models
    control_point_generator = ControlPointGenerator(
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
    
    optimizer_generator = torch.optim.AdamW(control_point_generator.parameters(), lr=learning_rate)
    optimizer_estimator = torch.optim.AdamW(estimator.parameters(), lr=learning_rate)
    
    # Cosine Annealing Learning Rate Schedules
    scheduler_generator = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_generator, T_max=training_steps, eta_min=1e-6
    )
    scheduler_estimator = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_estimator, T_max=training_steps, eta_min=1e-6
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Observation normalizer
    particle_n_dim = env_config.get("n_dim") if active_env == "particle" else None
    obs_normalizer = ObservationNormalizer(
        env_id=env_id,
        device=device,
        frame_stack=frame_stack,
        particle_n_dim=particle_n_dim,
    )
    
    # Number of generated control points used as counter examples.
    k_cp_counter_examples = max(1, min(top_k_control_points, control_points))
    
    # Training timing
    start_time = time.time()
    step = 0
    
    # Cycle through dataloader indefinitely until steps are reached
    while step < training_steps:
        for batch in dataloader:
            if step >= training_steps:
                break
            
            step_start = time.time()
            states = batch['state'].float().to(device)
            states = obs_normalizer.normalize(states)
            actions = batch['action'].float().to(device)
            B = states.shape[0]
            
            # ==================== Generator Loss (Best-Q MSE + Separation) ====================
            predicted_actions = control_point_generator(states)

            # Select the control point with highest estimated Q-value for each sample,
            # then compute MSE against the expert action.
            with torch.no_grad():
                states_for_cp = states.unsqueeze(1).expand(-1, predicted_actions.shape[1], -1)
                cp_q_values_for_mse = estimator(states_for_cp, predicted_actions.detach()).squeeze(-1)
                best_q_idx = cp_q_values_for_mse.argmax(dim=1)  # (B,)

            best_q_actions = predicted_actions[torch.arange(B, device=device), best_q_idx]  # (B, action_dim)
            # Match previous scale: mean over batch of squared L2 distance in action space.
            mse_per_sample = ((best_q_actions - actions) ** 2).sum(dim=1)
            loss_mse = mse_weight * mse_per_sample.mean()

            if separation_loss_type == "entropy":
                loss_sep = separation_weight * lossEntropyKDE(predicted_actions, bandwidth=entropy_bandwidth)
            elif separation_loss_type == "separation":
                loss_sep = separation_weight * (lossSeparation(predicted_actions, epsilon=separation_epsilon) / B)
            else:
                raise ValueError(f"Unknown separation_loss '{separation_loss_type}'. Expected 'separation' or 'entropy'.")
            loss_generator = loss_mse + loss_sep
            
            # ==================== Estimator Training (Direct InfoNCE) ====================
            # Use top-k generated control points directly as counter examples.
            # Detach control points so InfoNCE loss gradients only flow to estimator, not generator.
            predicted_actions_detached = predicted_actions.detach()
            with torch.no_grad():
                states_for_cp = states.unsqueeze(1).expand(-1, predicted_actions_detached.shape[1], -1)
                cp_q_values = estimator(states_for_cp, predicted_actions_detached).squeeze(-1)
                topk_idx = torch.topk(cp_q_values, k=k_cp_counter_examples, dim=1).indices

            gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, predicted_actions_detached.shape[2])
            cp_counter_samples = torch.gather(predicted_actions_detached, dim=1, index=gather_idx)

            # Optional Langevin counter-examples generated directly from counter_examples.
            if num_counter_examples > 0:
                action_min = torch.full((dataset.action_shape,), action_bounds[0], device=device)
                action_max = torch.full((dataset.action_shape,), action_bounds[1], device=device)

                # Freeze estimator params during Langevin MCMC; gradients are only for actions.
                for p in estimator.parameters():
                    p.requires_grad_(False)

                def _energy_fn(obs_expanded, actions_batch):
                    return estimator(obs_expanded, actions_batch).squeeze(-1)

                langevin_samples = sample_langevin(
                    energy_function=_energy_fn,
                    observations=states,
                    num_samples=num_counter_examples,
                    action_min=action_min,
                    action_max=action_max,
                    num_iterations=langevin_num_iterations,
                    lr_init=langevin_lr_init,
                    lr_final=langevin_lr_final,
                    polynomial_decay_power=langevin_decay_power,
                    delta_action_clip=langevin_delta_clip,
                    noise_scale=langevin_noise_scale,
                    device=device,
                )

                for p in estimator.parameters():
                    p.requires_grad_(True)

                counter_samples = torch.cat([cp_counter_samples, langevin_samples.detach()], dim=1)
            else:
                counter_samples = cp_counter_samples

            total_counter_examples = counter_samples.shape[1]
            
            # Concatenate expert action (index 0) with counter-examples
            all_actions = torch.cat([actions.unsqueeze(1), counter_samples], dim=1)
            states_expanded = states.unsqueeze(1).expand(-1, 1 + total_counter_examples, -1)
            
            # Direct energy evaluation for InfoNCE
            energies = estimator(states_expanded, all_actions).squeeze(-1)
            
            # InfoNCE loss: expert action should have the highest Q value (lowest energy equivalent)
            loss_estimator = lossInfoNCE(energies)

            # Generator receives InfoNCE with opposite sign.
            # Rebuild counter samples from non-detached control points so gradients reach generator,
            # while freezing estimator parameters so this branch updates only the generator.
            cp_counter_samples_for_generator = torch.gather(predicted_actions, dim=1, index=gather_idx)
            if num_counter_examples > 0:
                counter_samples_for_generator = torch.cat(
                    [cp_counter_samples_for_generator, langevin_samples.detach()],
                    dim=1,
                )
            else:
                counter_samples_for_generator = cp_counter_samples_for_generator

            all_actions_for_generator = torch.cat([actions.unsqueeze(1), counter_samples_for_generator], dim=1)
            for param in estimator.parameters():
                param.requires_grad_(False)
            energies_for_generator = estimator(states_expanded, all_actions_for_generator).squeeze(-1)
            loss_infonce_generator = lossInfoNCE(energies_for_generator)
            for param in estimator.parameters():
                param.requires_grad_(True)
            
            if (
                torch.isnan(loss_estimator)
                or torch.isnan(loss_infonce_generator)
                or torch.isnan(loss_generator)
            ):
                print("NaN loss detected, skipping this batch.")
                continue
            
            # ==================== Update Models ====================
            optimizer_estimator.zero_grad()
            optimizer_generator.zero_grad()
            
            loss_estimator_total = info_nce_weight * loss_estimator
            loss_generator_total = loss_generator - generator_infonce_weight * loss_infonce_generator
            total_loss = loss_generator_total + loss_estimator_total
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(control_point_generator.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), 1.0)
            
            optimizer_estimator.step()
            optimizer_generator.step()
            scheduler_generator.step()
            scheduler_estimator.step()
            
            step += 1
            
            # Logging
            if step % log_interval == 0:
                current_lr = scheduler_generator.get_last_lr()[0]
                
                # Compute accuracy of estimator
                with torch.no_grad():
                    best_idx = energies.argmax(dim=1)
                    accuracy = (best_idx == 0).float().mean().item()
                    
                elapsed = time.time() - start_time
                print(f"Step {step}/{training_steps} | Total: {total_loss.item():.4f} "
                      f"(MSE: {loss_mse.item():.4f}, "
                      f"Sep: {loss_sep.item():.4f}, "
                      f"EST: {loss_estimator.item():.4f}, "
                      f"GEN_INF_ONCE(-): {loss_infonce_generator.item():.4f}, "
                      f"Acc: {accuracy:.3f}) | LR: {current_lr:.2e} | {elapsed:.1f}s")
                
                log_dict = {
                    "step": step,
                    "loss/total": total_loss.item(),
                    "loss/generator": loss_generator_total.item(),
                    "loss/estimator": loss_estimator.item(),
                    "loss/infonce_generator_opposite": loss_infonce_generator.item(),
                    "loss/mse": loss_mse.item(),
                    "loss/separation": loss_sep.item(),
                    "metric/accuracy": accuracy,
                    "learning_rate": current_lr,
                }
                wandb.log(log_dict)
            
            # Save checkpoint
            if step % save_interval == 0:
                os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
                torch.save(control_point_generator.state_dict(), 
                          os.path.join(MODEL_SAVE_DIR, f"control_point_generator_step_{step}.pt"))
                torch.save(estimator.state_dict(), 
                          os.path.join(MODEL_SAVE_DIR, f"q_estimator_step_{step}.pt"))
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.2f} min)")
    
    # Save trained models
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save(control_point_generator.state_dict(), os.path.join(MODEL_SAVE_DIR, "control_point_generator.pt"))
    torch.save(estimator.state_dict(), os.path.join(MODEL_SAVE_DIR, "q_estimator.pt"))
    
    # Remove stale smoothing param if exists
    smoothing_param_path = os.path.join(MODEL_SAVE_DIR, "smoothing_param.pt")
    if os.path.exists(smoothing_param_path):
        os.remove(smoothing_param_path)
        print(f"Removed stale {smoothing_param_path}")
    
    print(f"Models saved to {MODEL_SAVE_DIR}/")
    
    # Log model artifacts to W&B
    artifact = wandb.Artifact("model-checkpoints", type="model")
    artifact.add_file(os.path.join(MODEL_SAVE_DIR, "control_point_generator.pt"))
    artifact.add_file(os.path.join(MODEL_SAVE_DIR, "q_estimator.pt"))
    wandb.log_artifact(artifact)
    
    # Log final metrics
    wandb.summary["total_training_time_min"] = total_time / 60
    
    wandb.finish()


if __name__ == "__main__":
    main()
