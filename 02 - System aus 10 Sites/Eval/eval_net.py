import torch
import SystemModel
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import scienceplots
import preprocess
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
plt.style.use(["science", "grid"])
#eps = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_sys_params = pd.read_csv('Data/sys_params.dat', sep=';')
N_x = 10
t_steps = 201
model = SystemModel.Net(3*N_x+1, 1024, 2*N_x)
model.load_state_dict(torch.load("model_checkpoint4250.pth", map_location=device))
dt = 0.01#df_sys_params.values[0,5]
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

def to_psi(psi_pred):
    psi = torch.zeros(N_x).type(torch.complex64)
    #print(psi_pred)
    for i in range(0, 2*N_x, 2):
        psi[int(i/2)] = psi_pred[i]+1j*psi_pred[i+1]
    return psi
model.eval()

psi_pred_t = np.zeros((t_steps, N_x)).astype(np.complex64)
#EnCounts = [0]
En_t = np.zeros((t_steps, N_x)).astype(np.float32)
psi_theo_t = np.zeros((t_steps, N_x)).astype(np.complex64)
with torch.no_grad():
    sys_nr = 0
    psi_t = np.loadtxt(f"Data/psi_sys_{sys_nr}.dat", delimiter=";", dtype=np.complex64)
    for step in tqdm(range(0, t_steps-1)):
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
        #print(psi_pred[0,:])
        EnT = preprocess.get_H_neuron(sys_nr, dt*step)[:-1] # Energie zur jeweiligen Zeit
        En_t[step,:] = EnT
        #print(EnT.item())
        #DeltE = train_En - EnT.item()
        #count = (np.abs(DeltE) <= eps).sum() # Zählen, wie viele Energien in den Trainingsdaten im Bereich von eps um
                                             # den Energiewert der Simulation sind
        #print(count)
        #EnCounts.append(count)
        norm = calc_norm(psi_pred[0,:].detach())
        psi_pred[0,:] = psi_pred[0, :].detach() / torch.sqrt(norm)
        #print(calc_norm(psi_pred[0,:]))
        #print(psi_pred[0,:])
        #psi_theo = SystemModel3.create_batch(1, step+1, dt, sys_nr, complex_bool=True, theo=True)
        psi_theo = preprocess.preprocess_psi_t(psi_t[step,:])
        psi_theo = torch.from_numpy(psi_theo)
        #print(psi_pred)
        psi_predicted = torch.zeros((1,N_x), dtype=torch.complex64)
        psi_predicted[0,:] = to_psi(psi_pred[0,:])
        #print(psi_predicted)
        psi_pred_t[step+1,:] = psi_predicted[0,:].cpu().numpy()
        psi_theory = to_psi(psi_theo)
        #print(calc_norm(psi_pred[0,:]))
        overlap = np.dot(np.conjugate(psi_predicted), psi_theory)
        psi_theo_t[step+1,:] = psi_theory.cpu().numpy()
        overlap_abs[step+1] = np.abs(overlap)
        overlap_phase[step+1] = np.tan(np.imag(overlap)/np.real(overlap))
        rmsd_psi[step+1] = np.sqrt(np.mean(np.abs(psi_pred_t[step+1,:] - psi_theo_t[step+1,:])**2))
        psi_pred = psi_pred.detach()
        #print(psi_theo_t[step+1,:])
    '''plt.figure(figsize=(9,5))
    plt.plot((np.abs(psi_pred_t[t_steps-1,:])**2), label="predicted")
    plt.plot((np.abs(psi_theo_t[t_steps-1,:])**2), label="theory")
    plt.legend()
    plt.show()'''

#print(overlap_abs[:5])
#EnCounts = np.array(EnCounts)
fig, ax = plt.subplots(1,2, figsize=(12,5))
ax1 = ax[0]
ax2 = ax[1]


width=0.1
distance = 0.05
def animate(i):
    ax1.cla()
    ax1.set_xlabel("n")
    ax1.set_ylabel("$|\psi|^2$")
    #ax1.set_xlim(0, N_x-1)
    ax1.set_ylim(0,1.1)
    ax1.set_title(f"t={np.round(i*dt, 3)}; |S| = {np.round(overlap_abs[i], 5)};$\\theta$ = {np.round(overlap_phase[i], 4)}")
    #ax1.set_title(f"Norm={np.round(norm, 7)}; time= {np.round(i*sys.dt*K, 3):.2f}; mu={np.round(sys.mu, 3)}; sx={np.round(sys.sx, 3)}; k = {np.round(sys.k, 3)}")
    ax1.grid(visible=True)
    ax1.bar(np.arange(1, N_x+1)-(width + distance)/2, np.abs(psi_pred_t[i, :])**2, width=width, color="blue", label="MLP Lösung")
    ax1.bar(np.arange(1, N_x+1)+(width + distance)/2, np.abs(psi_theo_t[i, :])**2, width=width, color="red", label="numerische Lösung")
    ax1.set_xticks(np.arange(1, N_x+1))
    ax1.legend()

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
    ax2.plot(np.arange(1, N_x+1), En_t[i, :], label="$\epsilon_n$")
    ax2.legend()
    '''
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
animation=False

if animation:
    ani = FuncAnimation(fig, animate, frames=t_steps, interval=100)
    #ani.save("Animation.gif", writer=PillowWriter(fps=40))
else:
    i = 100
    width=0.1
    distance = 0.05
    ax1.set_xlabel("n", fontsize=12)
    ax1.set_ylabel("$|c_n|^2$", fontsize=12)
    #ax1.set_xlim(0, N_x-1)
    ax1.set_ylim(0,1.1)
    #print(overlap_phase[i])
    #ax1.set_title(f"t={np.round(i*dt, 3)}; $|S|$ = {np.round(overlap_abs[i], 5)};$\\theta$ = {np.round(overlap_phase[i], 4)}")
    #ax1.set_title(f"Norm={np.round(norm, 7)}; time= {np.round(i*sys.dt*K, 3):.2f}; mu={np.round(sys.mu, 3)}; sx={np.round(sys.sx, 3)}; k = {np.round(sys.k, 3)}")
    ax1.grid(visible=True)
    ax1.bar(np.arange(1, N_x+1)-(width + distance)/2, np.abs(psi_pred_t[i, :])**2, width=width, color="blue", label="MLP Lösung")
    ax1.bar(np.arange(1, N_x+1) + (width + distance)/2, np.abs(psi_theo_t[i, :])**2, width=width, color="red", label="numerische Lösung")
    ax1.set_xticks(np.arange(1,N_x+1))
    ax1.tick_params(labelsize=12)
    ax1.legend(fontsize=11)

    i = 200
    ax2.set_xlabel("n", fontsize=12)
    #ax2.set_ylabel("$|\psi|^2$")
    #ax1.set_xlim(0, N_x-1)
    ax2.set_ylim(0,1.1)
    #print(overlap_phase[i])
    #ax2.set_title(f"t={np.round(i*dt, 3)}; $|S|$ = {np.round(overlap_abs[i], 5)}; $\\theta$ = {np.round(overlap_phase[i], 4)}")
    #ax1.set_title(f"Norm={np.round(norm, 7)}; time= {np.round(i*sys.dt*K, 3):.2f}; mu={np.round(sys.mu, 3)}; sx={np.round(sys.sx, 3)}; k = {np.round(sys.k, 3)}")
    ax2.grid(visible=True)
    ax2.bar(np.arange(1, N_x+1)-(width + distance)/2, np.abs(psi_pred_t[i, :])**2, width=width, color="blue", label="MLP Lösung")
    ax2.bar(np.arange(1, N_x+1) + (width + distance)/2, np.abs(psi_theo_t[i, :])**2, width=width, color="red", label="numerische Lösung")
    ax2.set_xticks(np.arange(1,N_x+1))
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=11)
    #ax2.set_ylim(np.min(En_t),np.max(En_t))
    #ax2.set_xlabel("n")
    #ax2.set_ylabel("$\epsilon_n$")
    #ax2.grid(visible=True)
    #print(np.arange(0,i*dt, dt))
    #print(overlap_abs[:i])
    #ax2.plot(t[:i+1], EnCounts[:i+1], label="Counts of Energys")
    #print(overlap[:i+1])
    #ax2.plot(np.arange(1, N_x+1), En_t[i, :], color="blue", label="$\epsilon_n$")
    #ax2.set_xticks(np.arange(1,N_x+1))
    #ax2.legend()

    #fig.savefig(f"plots/i=10-200.png", dpi=200, format="png")

plt.show()
plt.figure(figsize=(9,5))
plt.plot(np.arange(201), rmsd_psi, color="blue", label="RMSD")
plt.ylabel("RMSD", fontsize=12)
plt.xlabel("Anzahl der Iterationen", fontsize=12)
plt.legend(fontsize=11)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("rmsd_10site.png", dpi=300, format="png")
plt.show()