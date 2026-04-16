"""
Dataloader of Coupled Josephson Junction Trajectory Data.
"""

from typing import Iterable, Tuple

from sklearn.model_selection import train_test_split
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

class JosephsonJunctionDataset(Dataset):
    """
    Dataset for Josephson Junction Trajectory Data of the following form
    
    - X: Trajectory list, each element shape [n_all, n_steps, n_states]
    - Y: Trajectory list, each element shape [n_all, n_params]
    
    in which n_all = n_settings * n_trajs.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X: np.ndarray = X
        self.X = self.X.reshape(-1, *self.X.shape[2:])
        self.Y: np.ndarray = y
        self.Y = self.Y.reshape(-1, *self.Y.shape[2:])
        
        assert self.X.ndim == 3, self.X.ndim
        assert self.Y.ndim == 2, self.Y.ndim
        assert self.X.shape[0] == self.Y.shape[0]
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]

def create_dataloaders(
        batchsize: int = 64, test_ratio: float = 0.2
    ) -> Tuple[DataLoader|Iterable[Tuple[Tensor, Tensor]]]:
    
    data = np.load('trajectory_data.npz')
    X, y = data['X'], data['Y']
    
    print(f"Load Successful. {X.shape=}, y shape: {y.shape=}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    
    train_dataset = JosephsonJunctionDataset(X_train, y_train)
    test_dataset = JosephsonJunctionDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batchsize,  shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    # test the classes and functions.
    
    train_l, test_l = create_dataloaders()
    
    tr_x, tr_y = next(iter(train_l))
    print(tr_x.shape, tr_y.shape)
    
    te_x, te_y = next(iter(test_l))
    print(te_x.shape, te_y.shape)
