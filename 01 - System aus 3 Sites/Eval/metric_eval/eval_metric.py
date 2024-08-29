import torch
import SystemModel3
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import preprocess
from tqdm import tqdm
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
plt.style.use(["science", "grid"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_sys_params = pd.read_csv('Train_Data/sys_params.dat', sep=';')
N_x = 3
t_steps = 200
model = SystemModel3.Net(3*N_x+1, 256, 2*N_x)
model.load_state_dict(torch.load("model_epoch400.pth", map_location=device))
dt = df_sys_params.values[0,5]
model.to(device)
overlap_abs = np.zeros(t_steps)
overlap_abs[0] = 1
overlap_phase = np.zeros(t_steps)
overlap_phase[0] = 0

mse_psi = np.zeros(t_steps)
mse_psi[0] = 0
def calc_norm(psi_t):
    return torch.sum(psi_t**2)
#print(model)

def to_psi(psi_t):
    psi = torch.zeros(3).type(torch.complex64)
    for i in range(0, psi_t.shape[1], 2):
        psi[int(i/2)] = psi_t[0,i]+1j*psi_t[0,i+1]
    return psi
model.eval()
N_systems = 100
EnCounts = [0]
En_t = np.zeros(t_steps)

with torch.no_grad():
    for sys_nr in tqdm(range(N_systems)):
        psi_theo_t = np.zeros((t_steps, 3)).astype(np.complex64)
        psi_pred_t = np.zeros((t_steps, 3)).astype(np.complex64)
        temp_overlap_abs = np.zeros(t_steps)
        temp_overlap_abs[0] = 1
        temp_overlap_phase = np.zeros(t_steps)
        temp_overlap_phase[0] = 0
        for step in range(t_steps-1):
            if step == 0:
                psi_pred = SystemModel3.create_batch(1, step, dt, sys_nr, complex_bool=True).to(device)
            else:
                psi_pred = SystemModel3.create_batch(1, step, dt, sys_nr, complex_bool=False, psi=psi_pred).to(device)
            psi_pred = model(psi_pred)
            EnT = preprocess.get_H_neuron(sys_nr, dt*step)[0, 1] # Energie zur jeweiligen Zeit
            norm = calc_norm(psi_pred[0,:].detach())
            psi_pred[0,:] = psi_pred[0, :].detach() / torch.sqrt(norm)
            psi_theo = SystemModel3.create_batch(1, step+1, dt, sys_nr, complex_bool=True, theo=True)
            psi_predicted = to_psi(psi_pred)
            psi_pred_t[step+1,:] = psi_predicted.cpu().numpy()
            psi_theory = to_psi(psi_theo)
            overlap = np.dot(np.conjugate(psi_predicted), psi_theory)
            psi_theo_t[step+1,:] = psi_theory.cpu().numpy()
            temp_overlap_abs[step+1] = np.abs(overlap)
            temp_overlap_phase[step+1] = np.tan(np.imag(overlap)/np.real(overlap))
            #mse_psi[step+1] = test_losses.MSEpsi(psi_pred, psi_theo)
            psi_pred = psi_pred.detach()
        overlap_abs += temp_overlap_abs
        overlap_phase += temp_overlap_phase


overlap_abs = overlap_abs / N_systems
overlap_phase = overlap_phase / N_systems
np.savetxt("s_abs.dat", overlap_abs, delimiter=";")
np.savetxt("s_phase.dat", overlap_phase, delimiter=";")
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,5))
ax1.plot(np.arange(0,t_steps), overlap_abs, label="abs")
ax2.plot(np.arange(0,t_steps), overlap_phase, label="phase")
fig.show()