from datasets import D4RLDataset
from models import FeedForwardNN
import torch
from loss import lossInfoNCE

epochs = 100
learning_rate = 0.001
batch_size = 64

def main():
    dataset = D4RLDataset('D4RL/pen/human-v2', download=True)
    control_point_generator = FeedForwardNN(input_dim =dataset.state_shape, output_dim=dataset.action_shape)
    estimator = FeedForwardNN(input_dim =dataset.action_shape, output_dim=1)

    optimizer = torch.optim.AdamW(list(control_point_generator.parameters()) + list(estimator.parameters()), lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = lossInfoNCE
    for epoch in range(epochs):
        for batch in dataloader:
            states = batch['state'].float()
            actions = batch['action'].float()

            predicted_actions = control_point_generator(states)
            # NOW CONTROL POINT GENERATOR IS GIVING JUST ONE ACTION PER STATE
            # WE NEED TO CREATE MULTIPLE CONTROL POINT VALUES PER STATE
            # WE ALSO NEED TO ADD NORMALIZATION TO THE OUTPUTS OF THE ESTIMATOR
            estimations = estimator(predicted_actions)
            estimations_target = estimator(actions)
            
            loss = loss_fn(estimations, estimations_target)
            #check if ther is nan in loss   
            if torch.isnan(loss):
                print("NaN loss detected, skipping this batch.")
                exit()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")





if __name__ == "__main__":
    main()
