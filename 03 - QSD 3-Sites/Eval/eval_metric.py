import torch
import SQDModel
from System import SQDSystem
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from tqdm import tqdm
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
plt.style.use(["science", "grid"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#df_sys_params = pd.read_csv('metrictest/sys_params.dat', sep=';')
N_x = 3
t_steps = 200
model = SQDModel.Net(4*N_x, 1024, 2*N_x)
model.load_state_dict(torch.load("model_checkpoint5000.pth", map_location=device))
dt = 10**-3
model.to(device)
overlap_abs = np.zeros(t_steps)
overlap_abs[0] = 0
overlap_phase = np.zeros(t_steps)
overlap_phase[0] = 0

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

with torch.no_grad():
    for sys_nr in tqdm(range(1, N_systems+1)):
        psi_theo_t = np.zeros((t_steps, N_x)).astype(np.complex64)
        psi_pred_t = np.zeros((t_steps, N_x)).astype(np.complex64)
        system = SQDSystem(N_x, 2, 0, 1, 1, seed = sys_nr)
        system.psi = np.zeros(N_x).astype(np.complex64)
        system.psi[0] = 1
        #psi_t = np.loadtxt(f"metrictest/psi_sys_{sys_nr}.dat", delimiter=";", dtype=np.complex64)
        temp_overlap_abs = np.zeros(t_steps)
        temp_overlap_abs[0] = 1
        temp_overlap_phase = np.zeros(t_steps)
        temp_overlap_phase[0] = 0
        for step in range(t_steps-1):
            if step == 0:
                psi_pred = system.get_neuron_input()
                psi_pred = torch.from_numpy(psi_pred).to(device).type(torch.float32)
                psi_pred = torch.reshape(psi_pred, (1,-1))
                system.propagate_psi(no_init=False)
            else:
                system.propagate_psi(no_init=True)
                psi_pred = psi_pred.detach().cpu().numpy()
                psi_pred = psi_pred.reshape(-1)
                temp_z = system.get_z_net()
                psi_pred = np.concatenate((psi_pred, temp_z), axis=0)
                psi_pred = torch.from_numpy(psi_pred).to(device).type(torch.float32)
                psi_pred = torch.reshape(psi_pred, (1,-1))

            psi_pred = model(psi_pred)
            norm = calc_norm(psi_pred[0,:].detach())
            psi_pred[0,:] = psi_pred[0, :].detach() / torch.sqrt(norm)
            #print(psi_pred)
            

            psi_theo = system.get_psi_net()
            #print(system.get_psi())
            psi_theo = torch.from_numpy(psi_theo).to(device).type(torch.float32)
            #psi_predicted = torch.zeros((1,N_x), dtype=torch.complex64)
            psi_predicted = to_psi(psi_pred[0,:])
            psi_pred_t[step+1,:] = psi_predicted.cpu().numpy()
            psi_theory = to_psi(psi_theo)
            psi_theo_t[step + 1, :] = system.get_psi()
            #print(calc_norm(psi_pred[0,:]))
            overlap = np.dot(np.conjugate(psi_predicted), psi_theory)
            temp_overlap_abs[step+1] = np.abs(overlap)
            temp_overlap_phase[step+1] = np.tan(np.imag(overlap)/np.real(overlap))
            #mse_psi[step+1] = test_losses.MSEpsi(psi_pred, psi_theo)
            psi_pred = psi_pred.detach()

        overlap_abs += temp_overlap_abs
        overlap_phase += temp_overlap_phase

overlap_abs = overlap_abs / N_systems
overlap_phase = overlap_phase / N_systems
np.savetxt("s_abs_qsd_3.dat", overlap_abs, delimiter=";")
np.savetxt("s_phase_qsd_3.dat", overlap_phase, delimiter=";")
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,5))
ax1.plot(np.arange(0,t_steps), overlap_abs, label="abs")
ax2.plot(np.arange(0,t_steps), overlap_phase, label="phase")
plt.show()