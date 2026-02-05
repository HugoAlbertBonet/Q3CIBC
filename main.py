import os
import time
import json
from datasets import D4RLDataset
from models import ControlPointGenerator, QEstimator
import torch
import torch.nn as nn
import wandb
from loss import lossInfoNCE, lossMSE, lossSeparation
from normalizations import wireFittingNorm, ObservationNormalizer

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Training parameters
epochs = config["training"]["epochs"]
learning_rate = config["training"]["learning_rate"]
batch_size = config["training"]["batch_size"]
smoothing_param = config["training"]["smoothing_param"]
smoothing_param_trainable = config["training"]["smoothing_param_trainable"]
use_wire_fitting = config["training"]["use_wire_fitting"]
separation_weight = config["training"]["separation_weight"]
MODEL_SAVE_DIR = config["training"]["model_save_dir"]

# Model parameters
control_points = config["model"]["control_points"]
num_hidden_layers = config["model"]["num_hidden_layers"]
num_neurons = config["model"]["num_neurons"]

# Environment parameters
dataset_name = config["environment"]["dataset_name"]
env_id = config["environment"]["env_id"]

def main():
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
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss_gen = 0.0
        epoch_loss_est = 0.0
        epoch_loss_mse = 0.0
        epoch_loss_sep = 0.0
        num_batches = 0
        
        for batch in dataloader:
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
                estimations = torch.cat([estimations, estimations_target.unsqueeze(1)], dim=1)
            
            loss_estimator = lossInfoNCE(estimations)
            if torch.isnan(loss_estimator):
                print("NaN loss detected, skipping this batch.")
                exit()

            optimizer_estimator.zero_grad()
            loss_estimator.backward()
            optimizer_estimator.step()
            
            # Accumulate losses
            epoch_loss_gen += loss_generator.item()
            epoch_loss_est += loss_estimator.item()
            epoch_loss_mse += loss_mse.item()
            epoch_loss_sep += loss_sep.item()
            num_batches += 1

        # Compute average losses for the epoch
        avg_loss_gen = epoch_loss_gen / num_batches
        avg_loss_est = epoch_loss_est / num_batches
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | Avg Loss Gen: {avg_loss_gen/num_batches:.4f}, Avg Loss Est: {avg_loss_est/num_batches:.4f} | Elapsed: {elapsed:.1f}s")
        print(f"Decomposition of Generator Loss: {epoch_loss_mse/num_batches:.2f} + {epoch_loss_sep/num_batches:.2f}")
        
        # Log to W&B
        log_dict = {
            "epoch": epoch + 1,
            "loss/generator": avg_loss_gen / num_batches,
            "loss/estimator": avg_loss_est / num_batches,
            "loss/mse": epoch_loss_mse / num_batches,
            "loss/separation": epoch_loss_sep / num_batches,
            "time/epoch_seconds": epoch_time,
        }
        if smoothing_param_trainable:
            log_dict["smoothing_param"] = smoothing_param_tensor.item()
        wandb.log(log_dict)

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
