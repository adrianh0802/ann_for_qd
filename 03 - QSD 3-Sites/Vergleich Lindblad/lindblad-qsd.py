import torch
import System
import SQDModel
#import SQDModel3
import matplotlib.pyplot as plt
#import pandas as pd
from tqdm import tqdm
import scienceplots
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
plt.style.use(["science"])
#eps = 0.01
N_x = 3
t_steps = 200
dt = 10**-3

def calc_norm(psi_t):
    return torch.sum(psi_t**2)
#print(model)

def to_psi(psi_pred):
    psi = torch.zeros(N_x).type(torch.complex64)
    for i in range(0, 2*N_x, 2):
        psi[int(i/2)] = psi_pred[i]+1j*psi_pred[i+1]
    return psi

n_trajectories = 10000
trajectories_ann = np.zeros((n_trajectories, N_x)).astype(np.complex64)
with torch.no_grad():

    for i in tqdm(range(n_trajectories)):
        #sys_nr = 0
        system = System.SQDSystem(N_x, 2, 0, 1, 1)
        system.psi = np.zeros(N_x).astype(np.complex64)
        system.psi[0] = 1

        #psi_t = np.loadtxt(f"Train_Data/psi_sys_{sys_nr}.dat", delimiter=";", dtype=np.complex64)
        for step in range(0, t_steps):
            if step == 0:
                system.propagate_psi(no_init=False)
            else:
                system.propagate_psi(no_init=True)
            if step == t_steps - 1:
                trajectories_ann[i, :] = system.get_psi()

rho_t = np.zeros((N_x, N_x), dtype=np.complex64)
for trajectorie in trajectories_ann:
    rho_t += np.outer(trajectorie, np.conjugate(trajectorie))

rho_t /= n_trajectories

fig, ax = plt.subplots(1,2, figsize=(12,5))
axes = ax[0]
im = axes.matshow(np.real(rho_t), cmap="Wistia")
for i in range(N_x):
    for j in range(N_x):
        c = np.round(np.real(rho_t)[j,i], decimals=4)
        axes.text(i,j,str(c), va="center", ha="center", fontsize=11)

axes.set_title("Im($\\rho$)", fontsize=15)
axes.tick_params(labelsize=12)
axes.set_title("Re($\\rho$)")
fig.colorbar(im, ax=axes)

axes = ax[1]
im = axes.matshow(np.imag(rho_t), cmap="Wistia")
for i in range(N_x):
    for j in range(N_x):
        c = np.round(np.imag(rho_t)[j,i], decimals=4)
        axes.text(i,j,str(c), va="center", ha="center", fontsize=11)

axes.set_title("Im($\\rho$)", fontsize=15)
axes.tick_params(labelsize=12)
axes.set_title("Im($\\rho$)")
fig.colorbar(im, ax=axes)
fig.savefig("lindblad-qsd200.png", dpi=300, format="png")

plt.show()
