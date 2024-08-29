import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainDataset(Dataset):
    def __init__(self, fnameIn, fnameOut):
        # data loading
        x = np.loadtxt(f"Train_Data/{fnameIn}", delimiter=";", dtype=np.float32)
        y = np.loadtxt(f"Train_Data/{fnameOut}", delimiter=";", dtype=np.float32)
        self.x = torch.from_numpy(x).type(torch.float32)
        self.y = torch.from_numpy(y).type(torch.float32)
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index, :], self.y[index, :]

    def __len__(self):
        return self.n_samples


