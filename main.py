import os
import time
import json
from utils.datasets import D4RLDataset
from utils.models import ControlPointGenerator, QEstimator
import torch
import torch.nn as nn
import wandb
from utils.loss import lossInfoNCE, lossMSE, lossSeparation
from utils.normalizations import wireFittingNorm, ObservationNormalizer

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
    
# Model parameters
control_points = config["model"]["control_points"]
num_hidden_layers = config["model"]["num_hidden_layers"]
num_neurons = config["model"]["num_neurons"]
    
# Environment parameters
dataset_name = config["environment"]["dataset_name"]
env_id = config["environment"]["env_id"]
    
def main():
    global learning_rate
    # Initialize Weights & Biases
    wandb.init(
        project="Q3CIBC",
        config=config,
        name=f"cp{control_points}_lr{learning_rate}_sep{separation_weight}"
    )
        
    dataset = D4RLDataset(dataset_name, download=True)
    control_point_generator = ControlPointGenerator(
        input_dim=dataset.state_shape, 
        output_dim=dataset.action_shape, 
        control_points=control_points, 
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)]
    )
    estimator = QEstimator(
        state_dim=dataset.state_shape,
        action_dim=dataset.action_shape, 
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)]
    )
        
    # Smoothing parameter (environment-level, not state-dependent)
    if smoothing_param_trainable:
        smoothing_param_tensor = nn.Parameter(torch.tensor(smoothing_param))
        optimizer_estimator = torch.optim.AdamW(
            list(estimator.parameters()) + [smoothing_param_tensor], lr=learning_rate
        )
    else:
        smoothing_param_tensor = torch.tensor(smoothing_param)
        optimizer_estimator = torch.optim.AdamW(estimator.parameters(), lr=learning_rate)
        
    optimizer_generator = torch.optim.AdamW(control_point_generator.parameters(), lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    # Observation normalizer (uses official bounds from JSON file)
    obs_normalizer = ObservationNormalizer(env_id=env_id)
        
    # Training timing
    start_time = time.time()
    step = 0
        
    # Cycle through dataloader indefinitely until steps are reached
    while step < training_steps:
        for batch in dataloader:
            if step >= training_steps:
                break
                
            step_start = time.time()
            states = batch['state'].float()
            states = obs_normalizer.normalize(states)  # Normalize to [0, 1]
            actions = batch['action'].float()
    
            predicted_actions = control_point_generator(states)
            loss_mse = lossMSE(predicted_actions, actions)  
            loss_sep = separation_weight*lossSeparation(predicted_actions)
            loss_generator = loss_mse + loss_sep
            optimizer_generator.zero_grad()
            loss_generator.backward()
            optimizer_generator.step()
            # Detach predicted actions to reconstruct the computational graph for the estimator
            with torch.no_grad(): # No gradient for generator here
                predicted_actions_for_est = predicted_actions.detach()
    
            # Expand states to match control points: (B, state_dim) -> (B, N, state_dim)
            states_expanded = states.unsqueeze(1).expand(-1, predicted_actions_for_est.shape[1], -1)
            estimations = estimator(states_expanded, predicted_actions_for_est).squeeze(-1)  # (B, N)
            estimations_target = estimator(states, actions).squeeze(-1)  
            if use_wire_fitting:                    # (B,)
                estimations = wireFittingNorm(
                    control_points=predicted_actions_for_est,
                    expert_action=actions,
                    control_point_values=estimations,
                    expert_action_value=estimations_target,
                    c=smoothing_param_tensor.expand(states.shape[0], predicted_actions_for_est.shape[1]+1)
                )
            else:
                estimations = torch.cat([estimations_target.unsqueeze(1), estimations], dim=1)
                
            loss_estimator = lossInfoNCE(estimations)
            if torch.isnan(loss_estimator):
                print("NaN loss detected, skipping this batch.")
                exit()
    
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
                print(f"Step {step}/{training_steps} | Loss Gen: {loss_generator.item():.4f}, Loss Est (per sample): {loss_est_per_sample:.4f} | LR: {learning_rate:.2e} | Elapsed: {elapsed:.1f}s")
                    
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
