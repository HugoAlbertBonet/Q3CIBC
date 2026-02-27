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
from utils.loss import lossInfoNCE, lossMSE, lossSeparation
from utils.normalizations import ObservationNormalizer
from utils.sampling import sample_uniform, sample_langevin

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

MODEL_SAVE_DIR = training_shared.get("model_save_dir", "checkpoints")
log_interval = training_shared.get("log_interval", 1000)
save_interval = training_shared.get("save_interval", 10000)

num_counter_examples = env_training.get("counter_examples", training_shared.get("counter_examples", 64))
sampling_method = env_training.get("sampling_method", training_shared.get("sampling_method", "uniform"))

# Langevin config
langevin_config = env_model.get("langevin_config", {})
langevin_num_iterations = langevin_config.get("num_iterations", 50)
langevin_lr_init = langevin_config.get("lr_init", 0.1)
langevin_lr_final = langevin_config.get("lr_final", 1e-5)
langevin_decay_power = langevin_config.get("polynomial_decay_power", 2.0)
langevin_delta_clip = langevin_config.get("delta_action_clip", 0.1)
langevin_noise_scale = langevin_config.get("noise_scale", 1.0)

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
    print(f"Sampling method: {sampling_method}")
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
    obs_normalizer = ObservationNormalizer(env_id=env_id, device=device, frame_stack=frame_stack)
    
    # Action bounds for counter-example sampling
    action_min = action_bounds[0]
    action_max = action_bounds[1]
    
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
            
            # ==================== Generator Loss (MSE + Separation) ====================
            predicted_actions = control_point_generator(states)
            # Both lossMSE and lossSeparation return SUMS over the batch.
            # We divide by B to make them MEANs over the batch, matching InfoNCE.
            loss_mse = mse_weight * (lossMSE(predicted_actions, actions) / B)
            loss_sep = separation_weight * (lossSeparation(predicted_actions) / B)
            loss_generator = loss_mse + loss_sep
            
            # ==================== Estimator Training (Direct InfoNCE) ====================
            # Generate counter-example samples
            if sampling_method == "langevin":
                action_min_tensor = torch.full((actions.shape[1],), action_min, device=device)
                action_max_tensor = torch.full((actions.shape[1],), action_max, device=device)
                
                def energy_fn(obs, act):
                    return -estimator(obs, act).squeeze(-1)
                
                counter_samples = sample_langevin(
                    energy_function=energy_fn,
                    observations=states,
                    num_samples=num_counter_examples,
                    action_min=action_min_tensor,
                    action_max=action_max_tensor,
                    num_iterations=langevin_num_iterations,
                    lr_init=langevin_lr_init,
                    lr_final=langevin_lr_final,
                    polynomial_decay_power=langevin_decay_power,
                    delta_action_clip=langevin_delta_clip,
                    noise_scale=langevin_noise_scale,
                    device=device
                )
            else:
                # Uniform sampling
                counter_samples = (
                    torch.rand(states.shape[0], num_counter_examples, actions.shape[1], device=device)
                    * (action_max - action_min) + action_min
                )
            
            # Concatenate expert action (index 0) with counter-examples
            all_actions = torch.cat([actions.unsqueeze(1), counter_samples], dim=1)
            states_expanded = states.unsqueeze(1).expand(-1, 1 + num_counter_examples, -1)
            
            # Direct energy evaluation for InfoNCE
            energies = estimator(states_expanded, all_actions).squeeze(-1)
            
            # InfoNCE loss: expert action should have the highest Q value (lowest energy equivalent)
            loss_estimator = lossInfoNCE(energies)
            
            if torch.isnan(loss_estimator) or torch.isnan(loss_generator):
                print("NaN loss detected, skipping this batch.")
                continue
            
            # ==================== Update Models ====================
            optimizer_estimator.zero_grad()
            optimizer_generator.zero_grad()
            
            total_loss = loss_generator + info_nce_weight * loss_estimator
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
                      f"Acc: {accuracy:.3f}) | LR: {current_lr:.2e} | {elapsed:.1f}s")
                
                log_dict = {
                    "step": step,
                    "loss/total": total_loss.item(),
                    "loss/generator": loss_generator.item(),
                    "loss/estimator": loss_estimator.item(),
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
