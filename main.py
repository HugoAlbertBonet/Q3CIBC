import os
from datasets import D4RLDataset
from models import ControlPointGenerator, QEstimator
import torch
from loss import lossInfoNCE, lossMSE, lossSeparation
from normalizations import wireFittingNorm

epochs = 100
learning_rate = 0.00001
batch_size = 64
SMOOTHING_PARAM = 0.1
MODEL_SAVE_DIR = "checkpoints"

def main():
    dataset = D4RLDataset('D4RL/pen/human-v2', download=True)
    control_point_generator = ControlPointGenerator(input_dim =dataset.state_shape, output_dim=dataset.action_shape, control_points=30)
    estimator = QEstimator(input_dim =dataset.action_shape, output_dim=1)

    optimizer_estimator = torch.optim.AdamW(estimator.parameters(), lr=learning_rate)
    optimizer_generator = torch.optim.AdamW(control_point_generator.parameters(), lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
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
                c=torch.ones(states.shape[0], predicted_actions_for_est.shape[1]+1) * SMOOTHING_PARAM  # Example smoothing parameters
            )
            
            loss_estimator = lossInfoNCE(estimations)
            #check if ther is nan in loss   
            if torch.isnan(loss_estimator):
                print("NaN loss detected, skipping this batch.")
                exit()

            optimizer_estimator.zero_grad()
            loss_estimator.backward()
            optimizer_estimator.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss Generator: {loss_generator.item()}, Loss Estimator: {loss_estimator.item()}")

    # Save trained models
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save(control_point_generator.state_dict(), os.path.join(MODEL_SAVE_DIR, "control_point_generator.pt"))
    torch.save(estimator.state_dict(), os.path.join(MODEL_SAVE_DIR, "q_estimator.pt"))
    print(f"Models saved to {MODEL_SAVE_DIR}/")


if __name__ == "__main__":
    main()

