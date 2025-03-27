import torch
from base import BaseDataLoader
import numpy as np


class FunctionDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=100, function='linear'):
        self.n_samples = n_samples
        self.function = function

        # Generate random x values from 0 to 2pi
        self.x = torch.rand(n_samples) * 2 * np.pi

        # Generate y values based on the function type
        if function == 'linear':
            self.y = 1.5 * self.x + 0.3 + torch.empty(n_samples).uniform_(-1, 1)
        elif function == 'quadratic':
            self.y = 2 * self.x**2 + 0.5 * self.x + 0.3 + torch.empty(n_samples).uniform_(-1, 1)
        elif function == 'harmonic':
            self.y = 0.5 * self.x**2 + 5 * torch.sin(self.x) + 3 * torch.cos(3 * self.x) + 2 + torch.empty(n_samples).uniform_(-1, 1)
        else:
            raise ValueError(f"Unknown function type: {function}")

        # Normalize the data
        self.x = (self.x - self.x.mean()) / self.x.std()
        self.y = (self.y - self.y.mean()) / self.y.std()

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx].unsqueeze(0)


class FunctionDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, function='linear', n_samples=100):
        self.dataset = FunctionDataset(n_samples, function)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)