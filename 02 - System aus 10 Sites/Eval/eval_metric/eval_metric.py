import torch
import SystemModel
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import preprocess
from tqdm import tqdm
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
plt.style.use(["science", "grid"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_sys_params = pd.read_csv('metrictest/sys_params.dat', sep=';')
N_x = 10
t_steps = 200
model = SystemModel.Net(3*N_x+1, 1024, 2*N_x)
model.load_state_dict(torch.load("model_checkpoint4250.pth", map_location=device))
dt = df_sys_params.values[0,5]
model.to(device)
overlap_abs = np.zeros(t_steps)
overlap_phase = np.zeros(t_steps)

mse_psi = np.zeros(t_steps)
mse_psi[0] = 0
def calc_norm(psi_t):
    return torch.sum(psi_t**2)
#print(model)

def to_psi(psi_pred):
    psi = torch.zeros(N_x).type(torch.complex64)
    #print(psi_pred)
    for i in range(0, 2*N_x, 2):
        psi[int(i/2)] = psi_pred[i]+1j*psi_pred[i+1]
    return psi
model.eval()
N_systems = 100
EnCounts = [0]
En_t = np.zeros(t_steps)

with torch.no_grad():
    for sys_nr in tqdm(range(1, N_systems+1)):
        psi_theo_t = np.zeros((t_steps, N_x)).astype(np.complex64)
        psi_pred_t = np.zeros((t_steps, N_x)).astype(np.complex64)
        psi_t = np.loadtxt(f"metrictest/psi_sys_{sys_nr}.dat", delimiter=";", dtype=np.complex64)
        temp_overlap_abs = np.zeros(t_steps)
        temp_overlap_abs[0] = 1
        temp_overlap_phase = np.zeros(t_steps)
        temp_overlap_phase[0] = 0
        for step in range(t_steps-1):
            if step == 0:
                psi0 = psi_t[step,:]
                psi_pred = preprocess.getInput(psi=psi0, sys_nr=sys_nr, t=step*dt, complex=True)
                psi_pred = torch.from_numpy(psi_pred).to(device).type(torch.float32)
                psi_pred = torch.reshape(psi_pred, (1,-1))
            else:
                psi_pred = psi_pred.detach().cpu().numpy()
                psi_pred = psi_pred.reshape(-1)
                psi_pred = preprocess.getInput(psi_pred, sys_nr, step*dt, complex=False)
                psi_pred = torch.from_numpy(psi_pred).to(device).type(torch.float32)
                psi_pred = torch.reshape(psi_pred, (1,-1))
            psi_pred = model(psi_pred)
            norm = calc_norm(psi_pred[0,:].detach())
            psi_pred[0,:] = psi_pred[0, :].detach() / torch.sqrt(norm)
            psi_theo = preprocess.preprocess_psi_t(psi_t[step,:])
            psi_theo = torch.from_numpy(psi_theo)
            psi_predicted = torch.zeros((1,N_x), dtype=torch.complex64)
            psi_predicted[0,:] = to_psi(psi_pred[0,:])
            psi_pred_t[step+1,:] = psi_predicted[0,:].cpu().numpy()
            psi_theory = to_psi(psi_theo)
            overlap = np.dot(np.conjugate(psi_predicted), psi_theory)
            psi_theo_t[step+1,:] = psi_theory.cpu().numpy()
            temp_overlap_abs[step+1] = np.abs(overlap)
            temp_overlap_phase[step+1] = np.tan(np.imag(overlap)/np.real(overlap))
            psi_pred = psi_pred.detach()
        overlap_abs += temp_overlap_abs
        overlap_phase += temp_overlap_phase


overlap_abs = overlap_abs / N_systems
overlap_phase = overlap_phase / N_systems
np.savetxt("s_abs_10.dat", overlap_abs, delimiter=";")
np.savetxt("s_phase_10.dat", overlap_phase, delimiter=";")
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,5))
ax1.plot(np.arange(0,t_steps), overlap_abs, label="abs")
ax2.plot(np.arange(0,t_steps), overlap_phase, label="phase")
plt.show()