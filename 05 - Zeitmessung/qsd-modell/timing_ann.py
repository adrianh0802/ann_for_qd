import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mpcpu
import numpy as np
from tqdm import tqdm
from time import time
from System import QSDSystem
# Fürs Arbeiten auf mehreren GPU's

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16


class Net(nn.Module):
    def __init__(self, N_input, N_hidden, N_hidden_layers, N_output):
        super(Net, self).__init__()
        self.relu = nn.ReLU
        self.inp = nn.Linear(N_input, N_hidden)
        self.hidden_layers = nn.Sequential( *[nn.Sequential( *[nn.Linear(N_hidden, N_hidden), self.relu()] ) for _ in range(N_hidden_layers-1)], nn.Linear(N_hidden, N_hidden))
        #self.layer_norm1 = nn.LayerNorm(N_hidden)
        #self.layer_norm2 = nn.LayerNorm(N_hidden)
        #self.layer_norm3 = nn.LayerNorm(N_hidden)
        #self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(N_hidden, N_output)
        #self.Lrelu = nn.LeakyReLU()
        #self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.inp(x)
        x = self.hidden_layers(x)
        x = self.out(x)
        return x


def calc_norm(psi_t):
    return torch.sum(psi_t ** 2)

N_x_arr = np.array([10, 25, 35, 50, 60, 75, 100, 150, 200, 450, 600, 750, 900, 1000])
N_hidden_layers_arr = np.array([4, 4, 4, 4, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16])
N_hidden_arr = np.array([1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 2048, 4096, 4096, 4096])

#N_x = 1000
t_steps = 500
#model = Net(N_input=3 * N_x + 1, N_hidden=4096, N_hidden_layers=32, N_output=2 * N_x)
#model = model.to(device)
system = QSDSystem(3, 2, 0, 1, 1)
dt = system.dt
N = 10
times = np.zeros(N)
times_model = np.zeros(len(N_x_arr))

for i in tqdm(range(len(N_x_arr))):
    N_x = N_x_arr[i]
    model = Net(N_input=4*N_x, N_hidden=N_hidden_arr[i], N_hidden_layers=N_hidden_layers_arr[i], N_output=2 * N_x)
    model = model.to(device)
    model.eval()
    for j in range(N):
        system = QSDSystem(N_x, 2, 0, 1, 1)
        start = time()
        with torch.no_grad():
            for step in range(t_steps):
                if step == 0:
                    psi_pred = system.get_neuron_input()
                    psi_pred = torch.from_numpy(psi_pred).to(device).type(torch.float32)
                    psi_pred = torch.reshape(psi_pred, (1,-1))
                else: 
                    system.prop_z()
                    psi_pred = psi_pred.detach().cpu().numpy()
                    psi_pred = psi_pred.reshape(-1)
                    temp_z = system.get_z_net()
                    psi_pred = np.concatenate((psi_pred, temp_z), axis=0)
                    psi_pred = torch.from_numpy(psi_pred).to(device).type(torch.float32)
                    psi_pred = torch.reshape(psi_pred, (1,-1))
                psi_pred = model(psi_pred)
                norm = calc_norm(psi_pred[0,:].detach())
                psi_pred[0,:] = psi_pred[0, :].detach() / torch.sqrt(norm)
                psi_pred = psi_pred.detach()
        end = time()
        times[j] = (end - start)
    times_model[i] = np.mean(times)
    times = np.zeros(N)

time_model_save = np.concatenate((N_x_arr.reshape(-1,1), N_hidden_layers_arr.reshape(-1,1), N_hidden_arr.reshape(-1,1), times_model.reshape(-1,1)), axis=1)

np.savetxt("timing_ann.dat", time_model_save, delimiter=";")