import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainDataset(Dataset):
    def __init__(self):
        # data loading
        x = np.loadtxt("Train_Data/data_input.dat", delimiter=";", dtype=np.float32)
        y = np.loadtxt("Train_Data/data_output.dat", delimiter=";", dtype=np.float32)
        self.x = torch.from_numpy(x).type(torch.float32)
        self.y = torch.from_numpy(y).type(torch.float32)
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index, :], self.y[index, :]

    def __len__(self):
        return self.n_samples


