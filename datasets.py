from torch.utils.data import Dataset
import minari 
import numpy as np

class D4RLDataset(Dataset):
    
    def __init__(self, root:str, download:bool=True):
        self.dataset = minari.load_dataset(root, download=download)
        self.observations = np.concatenate([ep.observations[:-1] for ep in self.dataset.iterate_episodes()])
        self.actions = np.concatenate([ep.actions for ep in self.dataset.iterate_episodes()])
        self.state_shape = self.observations.shape[1]
        self.action_shape = self.actions.shape[1]

    def __getitem__(self, index):
        return {'state': self.observations[index], 'action': self.actions[index]}


    def __len__(self):
        return len(self.observations)


if __name__ == "__main__":
    dataset = D4RLDataset('D4RL/pen/human-v2', download=True)
    print(f"Dataset length: {len(dataset), len(dataset.observations), len(dataset.actions)}")
    sample = dataset[0]
    print(f"Sample state shape: {sample['state'].shape}, action shape: {sample['action'].shape}")