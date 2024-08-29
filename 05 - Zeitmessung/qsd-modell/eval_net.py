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
plt.style.use(["science", "grid"])
#eps = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#df_sys_params = pd.read_csv('Train_Data/sys_params.dat', sep=';')
N_x = 3
t_steps = 1001
model = SQDModel.Net(4*N_x, 1024, 2*N_x)
model.load_state_dict(torch.load("model_checkpoint4050.pth", map_location=device))
dt = 10**-3
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

def to_psi(psi_pred):
    psi = torch.zeros(N_x).type(torch.complex64)
    for i in range(0, 2*N_x, 2):
        psi[int(i/2)] = psi_pred[i]+1j*psi_pred[i+1]
    return psi
model.eval()

psi_pred_t = np.zeros((t_steps, N_x)).astype(np.complex64)
#EnCounts = [0]
#En_t = np.zeros((t_steps, N_x)).astype(np.float32)
psi_theo_t = np.zeros((t_steps, N_x)).astype(np.complex64)
with torch.no_grad():
    #sys_nr = 0
    system = System.SQDSystem(N_x, 2, 0, 1, 1, seed=0)
    system.psi = np.zeros(N_x).astype(np.complex64)
    system.psi[0] = 1

    #psi_t = np.loadtxt(f"Train_Data/psi_sys_{sys_nr}.dat", delimiter=";", dtype=np.complex64)
    for step in tqdm(range(0, t_steps-1)):
        if step == 0:
            psi_pred_t[step] = system.get_psi()
            psi_theo_t[step] = system.get_psi()
            psi_pred = system.get_neuron_input()
            psi_pred = torch.from_numpy(psi_pred).to(device).type(torch.float32)
            psi_pred = torch.reshape(psi_pred, (1,-1))
            # propagation of psi_theo - if step = 0 -> t = 0 ->+
            # first propagation -> no_init = True (no updating of stochastic variable z)
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
        overlap_abs[step+1] = np.abs(overlap)
        overlap_phase[step+1] = np.tan(np.imag(overlap)/np.real(overlap))
        #mse_psi[step+1] = test_losses.MSEpsi(psi_pred, psi_theo)
        psi_pred = psi_pred.detach()
        #print(psi_theo_t[step+1,:])
    
#print(psi_pred_t)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
#ax2 = ax[1]
#ax2 = ax[0,1]
#ax3 = ax[1,0]
#ax4 = ax[1,1]

ax1.plot([], [])
#ax1.set_xlim(0, 9)
ax1.set_ylim(0,1.1)
ax1.set_xlabel("n")
ax1.set_ylabel("$|\psi|^2$")
ax1.grid(visible=True)
'''
ax2.plot([], [])
#ax2.set_xlim(0,(t_steps+1)*dt)
#ax2.set_ylim(0,2*np.pi)
ax2.set_xlabel("t")
ax2.set_ylabel("#En")
ax2.grid(visible=True)
#t = np.arange(0,(t_steps+1)*dt, dt)

ax3.plot([], [])
ax3.set_xlim(0, 2)
ax3.set_ylim(-1,1)
ax3.set_xlabel("n")
ax3.set_ylabel("Re$(\psi)$")
ax3.grid(visible=True)

ax4.plot([], [])
ax4.set_xlim(0, 2)
ax4.set_ylim(-1,1)
ax4.set_xlabel("n")
ax4.set_ylabel("Im$(\psi)$")
ax4.grid(visible=True)


ax5.plot([], [])
ax5.set_xlim(0, 2)
ax5.set_ylim(-1,1)
ax5.set_xlabel("n")
ax5.set_ylabel("Im$(\psi)$")
ax5.grid(visible=True)
'''
def animate(i):
    ax1.cla()
    width=0.025
    distance = 0.015
    ax1.set_xlabel("n")
    ax1.set_ylabel("$|c_n|^2$")
    #ax1.set_xlim(0, N_x-1)
    ax1.set_ylim(0,1.1)
    ax1.set_title(f"t={np.round(i*dt, 3)}; |S| = {np.round(overlap_abs[i], 5)}; $\\theta$ = {np.round(overlap_phase[i], 3)}")
    #ax1.set_title(f"Norm={np.round(norm, 7)}; time= {np.round(i*sys.dt*K, 3):.2f}; mu={np.round(sys.mu, 3)}; sx={np.round(sys.sx, 3)}; k = {np.round(sys.k, 3)}")
    ax1.grid(visible=True)
    ax1.bar(np.arange(1,4) - (width + distance)/2, np.abs(psi_pred_t[i, :])**2, width=width, color="blue", label="MLP Lösung")
    ax1.bar(np.arange(1,4) + (width + distance)/2, np.abs(psi_theo_t[i, :])**2, width=width, color="red", label="numerische Lösung")
    ax1.legend()
    '''
    ax2.cla()
    #ax2.set_xlim(0,(t_steps+1)*dt)
    ax2.set_ylim(np.min(En_t),np.max(En_t))
    ax2.set_xlabel("n")
    ax2.set_ylabel("$\epsilon_n$")
    ax2.grid(visible=True)
    #print(np.arange(0,i*dt, dt))
    #print(overlap_abs[:i])
    #ax2.plot(t[:i+1], EnCounts[:i+1], label="Counts of Energys")
    #print(overlap[:i+1])
    ax2.plot(np.arange(0, N_x), En_t[i, :], label="$\epsilon_n$")
    ax2.legend()
    
    ax3.cla()
    ax3.set_xlim(0, N_x-1)
    ax3.set_ylim(-1,1)
    ax3.set_xlabel("n")
    ax3.set_ylabel("Re$(\psi)$")
    ax3.grid(visible=True)
    ax3.plot(np.arange(0, N_x), np.real(psi_pred_t[i,:]), label="predicted")
    ax3.plot(np.arange(0, N_x), np.real(psi_theo_t[i,:]), label="theory")

    ax4.cla()
    ax4.set_xlim(0, N_x-1)
    ax4.set_ylim(-1,1)
    ax4.set_xlabel("n")
    ax4.set_ylabel("Im$(\psi)$")
    ax4.grid(visible=True)
    ax4.plot(np.arange(0, N_x), np.imag(psi_pred_t[i,:]), label="predicted")
    ax4.plot(np.arange(0, N_x), np.imag(psi_theo_t[i,:]), label="theory")

   
    ax5.cla()
    ax5.set_xlim(0, (t_steps+1)*dt)
    #ax5.set_ylim(-1,1)
    ax5.set_xlabel("t")
    ax5.set_ylabel("Anzahl Energien in TrainData")
    ax5.grid(visible=True)
    ax5.plot(t[:i+1], EnCounts[:i+1])
    '''

animation = False
if animation:
    ani = FuncAnimation(fig, animate, frames=int(t_steps), interval=100)
    #ani.save("Animation.gif", writer=PillowWriter(fps=40))
else:
    width=0.025
    distance = 0.015
    i1 = 10
    ax1.set_xlabel("n", fontsize=12)
    ax1.set_ylabel("$|c_n|^2$", fontsize=12)
    ax1.set_ylim(0,1.1)
    #ax1.set_title(f"t={np.round(i1*dt, 3)}; |S| = {np.round(overlap_abs[i1], 5)}; $\\theta$ = {np.round(overlap_phase[i1], 4)}")
    #ax1.set_title(f"Norm={np.round(norm, 7)}; time= {np.round(i*sys.dt*K, 3):.2f}; mu={np.round(sys.mu, 3)}; sx={np.round(sys.sx, 3)}; k = {np.round(sys.k, 3)}")
    ax1.grid(visible=True)
    ax1.bar(np.arange(1,N_x+1) - (width + distance)/2, np.abs(psi_pred_t[i1, :])**2, width=width, color="blue", label="MLP Lösung")
    ax1.bar(np.arange(1,N_x+1) + (width + distance)/2, np.abs(psi_theo_t[i1, :])**2, width=width, color="red", label="numerische Lösung")
    ax1.set_xticks(np.arange(1, N_x+1))
    ax1.tick_params(labelsize=12)
    ax1.legend(fontsize=11)

    i2 = 150
    ax2.set_xlabel("n", fontsize=12)
    #ax2.set_ylabel("$|c_n|^2$", fontsize=12)
    ax2.set_ylim(0,1.1)
    #ax2.set_title(f"t={np.round(i2*dt, 3)}; |S| = {np.round(overlap_abs[i2], 5)}; $\\theta$ = {np.round(overlap_phase[i2], 4)}")
    #ax2.set_title(f"Norm={np.round(norm, 7)}; time= {np.round(i*sys.dt*K, 3):.2f}; mu={np.round(sys.mu, 3)}; sx={np.round(sys.sx, 3)}; k = {np.round(sys.k, 3)}")
    ax2.grid(visible=True)
    ax2.bar(np.arange(1,N_x+1) - (width + distance)/2, np.abs(psi_pred_t[i2, :])**2, width=width, color="blue", label="MLP Lösung")
    ax2.bar(np.arange(1,N_x+1) + (width + distance)/2, np.abs(psi_theo_t[i2, :])**2, width=width, color="red", label="numerische Lösung")
    ax2.set_xticks(np.arange(1, N_x+1))
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=11)
    fig.savefig(f"plots/i={i1}-{i2}.png", dpi=200, format="png")

plt.show()
