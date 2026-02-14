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
from utils.normalizations import wireFittingInterpolation, ObservationNormalizer
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
training_steps = env_training.get("training_steps", training_shared.get("training_steps", 50000))
learning_rate = env_training.get("learning_rate", training_shared.get("learning_rate", 0.0005))
batch_size = env_training.get("batch_size", training_shared.get("batch_size", 128))
smoothing_param = training_shared.get("smoothing_param", 0.1)
smoothing_param_trainable = training_shared.get("smoothing_param_trainable", False)
use_wire_fitting = training_shared.get("use_wire_fitting", True)
separation_weight = training_shared.get("separation_weight", 10)
lr_decay = training_shared.get("lr_decay", 0.99)
MODEL_SAVE_DIR = training_shared.get("model_save_dir", "checkpoints")
log_interval = training_shared.get("log_interval", 1000)
save_interval = training_shared.get("save_interval", 10000)
lr_decay_interval = training_shared.get("lr_decay_interval", 100)
num_counter_examples = env_training.get("counter_examples", training_shared.get("counter_examples", 8))
top_k_control_points = min(
    env_training.get("top_k_control_points", training_shared.get("top_k_control_points", 10)),
    env_model.get("control_points", 50)
)
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
    
    # Smoothing parameter
    if smoothing_param_trainable:
        smoothing_param_tensor = nn.Parameter(torch.tensor(smoothing_param, device=device))
        optimizer_estimator = torch.optim.AdamW(
            list(estimator.parameters()) + [smoothing_param_tensor], lr=learning_rate
        )
    else:
        smoothing_param_tensor = torch.tensor(smoothing_param, device=device)
        optimizer_estimator = torch.optim.AdamW(estimator.parameters(), lr=learning_rate)
    
    optimizer_generator = torch.optim.AdamW(control_point_generator.parameters(), lr=learning_rate)
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
            
            # ==================== Generator Training (MSE) ====================
            predicted_actions = control_point_generator(states)
            loss_mse = lossMSE(predicted_actions, actions)
            loss_sep = separation_weight * lossSeparation(predicted_actions)
            loss_generator = loss_mse + loss_sep
            
            optimizer_generator.zero_grad()
            loss_generator.backward()
            optimizer_generator.step()
            
            # ==================== Estimator Training (Uniform/Langevin Sampling) ====================
            with torch.no_grad():
                predicted_actions_for_est = control_point_generator(states).detach()
            
            # Expand states to match control points
            states_expanded = states.unsqueeze(1).expand(-1, predicted_actions_for_est.shape[1], -1)
            estimations = estimator(states_expanded, predicted_actions_for_est).squeeze(-1)
            
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
                counter_samples = sample_uniform(
                    num_counter_examples,
                    states.shape[0],
                    [action_min for _ in range(actions.shape[1])],
                    [action_max for _ in range(actions.shape[1])]
                )
                counter_samples = torch.from_numpy(counter_samples).float().to(device)
            
            interpolated_points = torch.cat([actions.unsqueeze(1), counter_samples], dim=1)
            interpolated_estimations = wireFittingInterpolation(
                control_points=predicted_actions_for_est,
                interpolated_points=interpolated_points,
                control_point_values=estimations,
                c=smoothing_param_tensor.expand(states.shape[0], predicted_actions_for_est.shape[1]),
                k=top_k_control_points,
            )
            
            loss_estimator = lossInfoNCE(interpolated_estimations)
            
            if torch.isnan(loss_estimator):
                print("NaN loss detected, skipping this batch.")
                continue
            
            optimizer_estimator.zero_grad()
            loss_estimator.backward()
            optimizer_estimator.step()
            
            step += 1
            
            # Decay learning rate
            if step % lr_decay_interval == 0:
                learning_rate *= lr_decay
                optimizer_generator.param_groups[0]['lr'] = learning_rate
                optimizer_estimator.param_groups[0]['lr'] = learning_rate
            
            # Logging
            if step % log_interval == 0:
                batch_size_actual = states.shape[0]
                loss_est_per_sample = loss_estimator.item() / batch_size_actual
                elapsed = time.time() - start_time
                print(f"Step {step}/{training_steps} | Loss Gen: {loss_generator.item():.4f}, "
                      f"Loss Est: {loss_est_per_sample:.4f} | LR: {learning_rate:.2e} | Elapsed: {elapsed:.1f}s")
                
                log_dict = {
                    "step": step,
                    "loss/generator": loss_generator.item() / batch_size_actual,
                    "loss/estimator": loss_est_per_sample,
                    "loss/mse": loss_mse.item() / batch_size_actual,
                    "loss/separation": loss_sep.item() / batch_size_actual,
                    "learning_rate": learning_rate,
                }
                if smoothing_param_trainable:
                    log_dict["smoothing_param"] = smoothing_param_tensor.item()
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
    
    if smoothing_param_trainable:
        torch.save(smoothing_param_tensor, os.path.join(MODEL_SAVE_DIR, "smoothing_param.pt"))
        print(f"Learned smoothing_param: {smoothing_param_tensor.item():.6f}")
    else:
        smoothing_param_path = os.path.join(MODEL_SAVE_DIR, "smoothing_param.pt")
        if os.path.exists(smoothing_param_path):
            os.remove(smoothing_param_path)
            print(f"Removed stale {smoothing_param_path}")
    
    print(f"Models saved to {MODEL_SAVE_DIR}/")
    
    # Log model artifacts to W&B
    artifact = wandb.Artifact("model-checkpoints", type="model")
    artifact.add_file(os.path.join(MODEL_SAVE_DIR, "control_point_generator.pt"))
    artifact.add_file(os.path.join(MODEL_SAVE_DIR, "q_estimator.pt"))
    if smoothing_param_trainable:
        artifact.add_file(os.path.join(MODEL_SAVE_DIR, "smoothing_param.pt"))
    wandb.log_artifact(artifact)
    
    # Log final metrics
    wandb.summary["total_training_time_min"] = total_time / 60
    if smoothing_param_trainable:
        wandb.summary["final_smoothing_param"] = smoothing_param_tensor.item()
    
    wandb.finish()


if __name__ == "__main__":
    main()
