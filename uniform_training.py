import os
import time
import json
from utils.datasets import D4RLDataset
from utils.models import ControlPointGenerator, QEstimator
import torch
import torch.nn as nn
import wandb
from utils.loss import lossInfoNCE, lossMSE, lossSeparation
from utils.normalizations import wireFittingInterpolation, ObservationNormalizer
from utils.sampling import sample_uniform

# Load config
with open("config_json/config.json", "r") as f:
    config = json.load(f)

# Training parameters
training_steps = config["training"]["training_steps"]
learning_rate = config["training"]["learning_rate"]
batch_size = config["training"]["batch_size"]
smoothing_param = config["training"]["smoothing_param"]
smoothing_param_trainable = config["training"]["smoothing_param_trainable"]
use_wire_fitting = config["training"]["use_wire_fitting"]
separation_weight = config["training"]["separation_weight"]
lr_decay = config["training"]["lr_decay"]
MODEL_SAVE_DIR = config["training"]["model_save_dir"]
log_interval = config["training"].get("log_interval", 1000)
save_interval = config["training"].get("save_interval", 10000)
lr_decay_interval = config["training"].get("lr_decay_interval", 100)
num_counter_examples = config["training"]["counter_examples"]
top_k_control_points = min(config["training"]["top_k_control_points"], config["model"]["control_points"])
    
# Model parameters
control_points = config["model"]["control_points"]
num_hidden_layers = config["model"]["num_hidden_layers"]
num_neurons = config["model"]["num_neurons"]
    
# Environment parameters
dataset_name = config["environment"]["dataset_name"]
env_id = config["environment"]["env_id"]
    
def main():
    global learning_rate
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Weights & Biases
    wandb.init(
        project="Q3CIBC",
        config=config,
        name=f"uniform_sampling_cp{control_points}_lr{learning_rate}_sep{separation_weight}"
    )
        
    dataset = D4RLDataset(dataset_name, download=True)
    control_point_generator = ControlPointGenerator(
        input_dim=dataset.state_shape, 
        output_dim=dataset.action_shape, 
        control_points=control_points, 
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)]
    ).to(device)
    estimator = QEstimator(
        state_dim=dataset.state_shape,
        action_dim=dataset.action_shape, 
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)]
    ).to(device)
        
    # Smoothing parameter (environment-level, not state-dependent)
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
        
    # Observation normalizer (uses official bounds from JSON file)
    obs_normalizer = ObservationNormalizer(env_id=env_id, device=device)
        
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
            states = obs_normalizer.normalize(states)  # Normalize to [0, 1]
            actions = batch['action'].float().to(device)
    
            predicted_actions = control_point_generator(states)
            loss_sep = separation_weight * lossSeparation(predicted_actions)
    
            # Expand states to match control points: (B, state_dim) -> (B, N, state_dim)
            states_expanded = states.unsqueeze(1).expand(-1, predicted_actions.shape[1], -1)
            estimations = estimator(states_expanded, predicted_actions).squeeze(-1)  # (B, N)
            counter_samples = sample_uniform(num_counter_examples, states.shape[0], [-1 for _ in range(actions.shape[1])], [1 for _ in range(actions.shape[1])]) # (B, M, D)
            counter_samples = torch.from_numpy(counter_samples).float().to(device)
            interpolated_points = torch.cat([actions.unsqueeze(1), counter_samples], dim=1) # (B, M+1, D)
            interpolated_estimations = wireFittingInterpolation(
                control_points=predicted_actions,
                interpolated_points=interpolated_points,
                control_point_values=estimations,
                c=smoothing_param_tensor.expand(states.shape[0], predicted_actions.shape[1]),
                k=top_k_control_points,
            ) # (B, M+1)
            
            # Normalize interpolated estimations to prevent numerical instability
            # This keeps relative ordering while preventing extreme values from causing underflow
            interp_mean = interpolated_estimations.mean(dim=1, keepdim=True)
            interp_std = interpolated_estimations.std(dim=1, keepdim=True) + 1e-8
            interpolated_estimations_normalized = (interpolated_estimations - interp_mean) / interp_std
            
            # InfoNCE loss on normalized values
            loss_infonce = lossInfoNCE(interpolated_estimations_normalized)
            
            """# Direct supervision: expert action should have higher Q-value than random samples
            # Target: first column (expert) = 1, rest (counter-examples) = 0
            targets = torch.zeros_like(interpolated_estimations_normalized)
            targets[:, 0] = 1.0
            loss_supervision = torch.nn.functional.mse_loss(interpolated_estimations_normalized, targets)
            
            # Combined estimator loss"""
            loss_estimator = loss_infonce #+ loss_supervision
            
            if torch.isnan(loss_estimator):
                print("NaN loss detected, skipping this batch.")
                exit()
            loss_generator = loss_sep + loss_estimator
    
            optimizer_generator.zero_grad()
            optimizer_estimator.zero_grad()
            loss_generator.backward()
            optimizer_generator.step()
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
                print(f"Step {step}/{training_steps} | Loss Gen: {loss_generator.item():.4f}, Loss Est (per sample): {loss_est_per_sample:.4f} | LR: {learning_rate:.2e} | Elapsed: {elapsed:.1f}s")
                    
                log_dict = {
                    "step": step,
                    "loss/generator": loss_generator.item() / batch_size_actual,
                    "loss/estimator": loss_est_per_sample,
                    "loss/separation": loss_sep.item() / batch_size_actual,
                    "learning_rate": learning_rate,
                }
                if smoothing_param_trainable:
                    log_dict["smoothing_param"] = smoothing_param_tensor.item()
                wandb.log(log_dict)
                
            # Save checkpoint
            if step % save_interval == 0:
                os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
                torch.save(control_point_generator.state_dict(), os.path.join(MODEL_SAVE_DIR, f"control_point_generator_step_{step}.pt"))
                torch.save(estimator.state_dict(), os.path.join(MODEL_SAVE_DIR, f"q_estimator_step_{step}.pt"))
    
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
        # Remove any existing smoothing param checkpoint if not trainable
        smoothing_param_path = os.path.join(MODEL_SAVE_DIR, "smoothing_param.pt")
        if os.path.exists(smoothing_param_path):
            os.remove(smoothing_param_path)
            print(f"Removed stale {smoothing_param_path} (smoothing_param_trainable=False)")
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