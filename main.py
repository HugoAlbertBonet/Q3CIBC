from datasets import D4RLDataset
from models import ControlPointGenerator, QEstimator
import torch
from loss import lossInfoNCE, lossMSE

epochs = 100
learning_rate = 0.00001
batch_size = 64

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
            # WE ALSO NEED TO ADD NORMALIZATION TO THE OUTPUTS OF THE ESTIMATOR
            loss_generator = lossMSE(predicted_actions, actions)
            optimizer_generator.zero_grad()
            loss_generator.backward()
            optimizer_generator.step()
            # Detach predicted actions to reconstruct the computational graph for the estimator
            with torch.no_grad(): # No gradient for generator here
                predicted_actions_for_est = predicted_actions.detach()

            estimations = estimator(predicted_actions_for_est)
            estimations_target = estimator(actions)
            estimations_target = estimations_target.unsqueeze(1)  # Match dimensions
            
            loss_estimator = lossInfoNCE(estimations, estimations_target)
            #check if ther is nan in loss   
            if torch.isnan(loss_estimator):
                print("NaN loss detected, skipping this batch.")
                exit()

            optimizer_estimator.zero_grad()
            loss_estimator.backward()
            optimizer_estimator.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss Generator: {loss_generator.item()}, Loss Estimator: {loss_estimator.item()}")



if __name__ == "__main__":
    main()
