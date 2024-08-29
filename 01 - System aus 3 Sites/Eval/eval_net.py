import torch
from torch.nn import MSELoss
import SystemModel3
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import preprocess
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
plt.style.use(["science", "grid"])
eps = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_sys_params = pd.read_csv('data/sys_params.dat', sep=';')
N_x = 3
t_steps = 100
model = SystemModel3.Net(3*N_x+1, 256, 2*N_x)
model.load_state_dict(torch.load("model_epoch400.pth", map_location=device))
dt = df_sys_params.values[0,5]
model.to(device)
overlap_abs = np.zeros(t_steps)
overlap_abs[0] = 1
overlap_phase = np.zeros(t_steps)
overlap_phase[0] = 0

rmsd_psi = np.zeros(t_steps)
rmsd_psi[0] = 0
def calc_norm(psi_t):
    return torch.sum(psi_t**2)
#print(model)

def to_psi(psi_t):
    psi = torch.zeros(3).type(torch.complex64)
    for i in range(0, psi_t.shape[1], 2):
        psi[int(i/2)] = psi_t[0,i]+1j*psi_t[0,i+1]
    return psi
model.eval()

criterion = MSELoss()
psi_pred_t = np.zeros((t_steps, 3)).astype(np.complex64)
EnCounts = [0]
En_t = np.zeros(t_steps)
psi_theo_t = np.zeros((t_steps, 3)).astype(np.complex64)
with torch.no_grad():
    sys_nr = 9
    for step in range(0, t_steps-1):
        if step == 0:
            psi_pred = SystemModel3.create_batch(1, step, dt, sys_nr, complex_bool=True).to(device)
        else:
            psi_pred = SystemModel3.create_batch(1, step, dt, sys_nr, complex_bool=False, psi=psi_pred).to(device)
        psi_pred = model(psi_pred)
        # normieren
        norm = calc_norm(psi_pred[0,:].detach())
        psi_pred[0,:] = psi_pred[0, :].detach() / torch.sqrt(norm)
        # theoriewert einlesen
        psi_theo = SystemModel3.create_batch(1, step+1, dt, sys_nr, complex_bool=True, theo=True)
        # ausgabe des netzes in eine komplexe wellenfunktion schreiben + das gleiche mit dem theoriewert
        psi_predicted = to_psi(psi_pred)
        psi_pred_t[step+1,:] = psi_predicted.cpu().numpy()
        psi_theory = to_psi(psi_theo)
        # metrik berechnen
        overlap = np.dot(np.conjugate(psi_predicted), psi_theory)
        psi_theo_t[step+1,:] = psi_theory.cpu().numpy()
        overlap_abs[step+1] = np.abs(overlap)
        overlap_phase[step+1] = np.tan(np.imag(overlap)/np.real(overlap))
        # rmsd berechnen
        rmsd_psi[step+1] = np.sqrt(np.mean(np.abs(psi_pred_t[step+1,:] - psi_theo_t[step+1,:])**2))
        psi_pred = psi_pred.detach()
        #print(psi_theo_t[step+1,:])

EnCounts = np.array(EnCounts)
fig, ax = plt.subplots(1,2, figsize=(12,5))
ax1 = ax[0] # |c_n|^2 (i_1)
ax2 = ax[1] # |c_n|^2 (i_2 > i_1)
i = 10
width=0.025
distance = 0.015
ax1.set_xlabel("$n$", fontsize=13)
ax1.set_ylabel("$|c_n|^2$", fontsize=13)
#ax1.set_xlim(0.5, 3)
ax1.set_ylim(0,1.1)
#ax1.set_title(f"t={np.round(i*dt, 3)}; |S| = {np.round(overlap_abs[i], 5)}; $\\theta$ = {np.round(overlap_phase[i], 3)}")
ax1.grid(visible=True)
ax1.bar(np.arange(1,4) - (width + distance)/2, np.abs(psi_pred_t[i, :])**2, width=width, color="blue", label="MLP Lösung")
ax1.bar(np.arange(1,4) + (width + distance)/2, np.abs(psi_theo_t[i, :])**2, width=width, color="red", label="numerische Lösung")
ax1.set_xticks(np.arange(1,4))
ax1.tick_params(labelsize=12)
ax1.legend()

i = 99
#ax6.set_xlim(1,3)
ax2.set_xlabel("$n$", fontsize=13)
#ax6.set_ylabel("$|c_n|^2$", fontsize=13)
#ax6.set_title(f"t={np.round(i*dt, 3)}; |S| = {np.round(overlap_abs[i], 5)}; $\\theta$ = {np.round(overlap_phase[i], 3)}")
ax2.grid(visible=True)
ax2.set_ylim(0,1.1)
ax2.bar(np.arange(1,4) - (width + distance)/2, np.abs(psi_pred_t[i, :])**2, width=width, color="blue", label="MLP Lösung")
ax2.bar(np.arange(1,4) + (width + distance)/2, np.abs(psi_theo_t[i, :])**2, width=width, color="red", label="numerische Lösung")
ax2.set_xticks(np.arange(1,4))
ax2.tick_params(labelsize=12)
ax2.legend()
#fig.savefig(f"plots/i=10-{i}.png", dpi=200, format="png")
plt.show()

plt.figure(figsize=(9,5))
plt.plot(np.arange(100), rmsd_psi, color="blue", label="RMSD")
plt.ylabel("RMSD", fontsize=12)
plt.xlabel("Anzahl der Iterationen", fontsize=12)
plt.legend(fontsize=11)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("rmsd_3site.png", dpi=300, format="png")
plt.show()