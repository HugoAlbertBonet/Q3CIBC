import os
import time
import json
from datasets import D4RLDataset
from models import ControlPointGenerator, QEstimator
import torch
from loss import lossInfoNCE, lossMSE, lossSeparation
from normalizations import wireFittingNorm

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Training parameters
epochs = config["training"]["epochs"]
learning_rate = config["training"]["learning_rate"]
batch_size = config["training"]["batch_size"]
SMOOTHING_PARAM = config["training"]["smoothing_param"]
MODEL_SAVE_DIR = config["training"]["model_save_dir"]

# Model parameters
control_points = config["model"]["control_points"]
num_hidden_layers = config["model"]["num_hidden_layers"]
num_neurons = config["model"]["num_neurons"]

# Environment parameters
dataset_name = config["environment"]["dataset_name"]

def main():
    dataset = D4RLDataset(dataset_name, download=True)
    control_point_generator = ControlPointGenerator(
        input_dim=dataset.state_shape, 
        output_dim=dataset.action_shape, 
        control_points=control_points, 
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)]
    )
    estimator = QEstimator(
        input_dim=dataset.action_shape, 
        output_dim=1,
        hidden_dims=[num_neurons for _ in range(num_hidden_layers)]
    )

    optimizer_estimator = torch.optim.AdamW(estimator.parameters(), lr=learning_rate)
    optimizer_generator = torch.optim.AdamW(control_point_generator.parameters(), lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training timing
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss_gen = 0.0
        epoch_loss_est = 0.0
        num_batches = 0
        
        for batch in dataloader:
            states = batch['state'].float()
            actions = batch['action'].float()

            predicted_actions = control_point_generator(states)
            loss_generator = lossMSE(predicted_actions, actions) + lossSeparation(predicted_actions)
            optimizer_generator.zero_grad()
            loss_generator.backward()
            optimizer_generator.step()
            # Detach predicted actions to reconstruct the computational graph for the estimator
            with torch.no_grad(): # No gradient for generator here
                predicted_actions_for_est = predicted_actions.detach()

            estimations = estimator(predicted_actions_for_est).squeeze(-1)  # (B, N)
            estimations_target = estimator(actions).squeeze(-1)             # (B,)
            estimations = wireFittingNorm(
                control_points=predicted_actions_for_est,
                expert_action=actions,
                control_point_values=estimations,
                expert_action_value=estimations_target,
                c=torch.ones(states.shape[0], predicted_actions_for_est.shape[1]+1) * SMOOTHING_PARAM
            )
            
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
            num_batches += 1

        # Compute average losses for the epoch
        avg_loss_gen = epoch_loss_gen / num_batches
        avg_loss_est = epoch_loss_est / num_batches
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | Avg Loss Gen: {avg_loss_gen/num_batches:.4f}, Avg Loss Est: {avg_loss_est/num_batches:.4f} | Elapsed: {elapsed:.1f}s")

    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.2f} min)")

    # Save trained models
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save(control_point_generator.state_dict(), os.path.join(MODEL_SAVE_DIR, "control_point_generator.pt"))
    torch.save(estimator.state_dict(), os.path.join(MODEL_SAVE_DIR, "q_estimator.pt"))
    print(f"Models saved to {MODEL_SAVE_DIR}/")


if __name__ == "__main__":
    main()
