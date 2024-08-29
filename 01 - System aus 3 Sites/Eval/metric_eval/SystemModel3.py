import torch
import torch.nn as nn
import torch.nn.functional as F
import preprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
#import test_losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_systems = 1000
df_sys_params = pd.read_csv('Train_Data/sys_params.dat', sep=';')
N_x = 3
#print(N_x)
dt = df_sys_params.values[0,5]
t_steps = 100

class Net(nn.Module):
    def __init__(self, N_input, N_hidden, N_output):
        super(Net, self).__init__()
        self.inp = nn.Linear(N_input, N_hidden)
        self.hidden1 = nn.Linear(N_hidden, N_hidden)
        self.hidden2 = nn.Linear(N_hidden, N_hidden)
        self.hidden3 = nn.Linear(N_hidden, N_hidden)
        self.hidden4 = nn.Linear(N_hidden, N_hidden)
        #self.layer_norm1 = nn.LayerNorm(N_hidden)
        #self.layer_norm2 = nn.LayerNorm(N_hidden)
        #self.layer_norm3 = nn.LayerNorm(N_hidden)
        #self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(N_hidden, N_output)
        self.relu = nn.ReLU()
        #self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.inp(x)
        #x = self.layer_norm1(x)
        #x = self.prelu1(x)
        x = self.hidden1(x)
        #x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.hidden2(x)
        #x = self.drop(x)
        #x = self.layer_norm3(x)
        x = self.relu(x)
        x = self.hidden3(x)
        #x = self.drop(x)
        #x = self.relu(x)
        #x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.hidden4(x)
        #x = self.relu(x)
        x = self.out(x)
        return x


model = Net(N_input=3*N_x + 1, N_hidden=256, N_output=2*N_x)
'''
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
'''
#model.load_state_dict(torch.load("model_epoch99_sys100_tstep100.pth"))
#model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()
losses = []
loss_temp = 0
#scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=10)

def create_batch(batch_size, ind, dt, low_sys_nr, complex_bool, psi = None, theo = False ):

    if psi is None and theo is False:
        batch = torch.zeros((batch_size, 3 * N_x + 1), dtype=torch.float32,
                            device=device)  # Batch der größe batch_size*(3*N+1) - (N: Gridpoints)
        for i in range(batch_size):
            psi_sys_t = pd.read_csv(f"metrictest/psi_sys_{i+low_sys_nr}.dat", sep=';', header=None).values.astype(np.complex64)[ind, :]
            psi_sys_t = torch.from_numpy(psi_sys_t).to(device).type(torch.complex64)
            psi_sys_t = torch.reshape(psi_sys_t, (1, -1)) # aus daten einen Tensor definieren
            batch[i,:] = preprocess.getInput(psi_sys_t,low_sys_nr+i, ind*dt, complex=complex_bool) # auf Input der KI anpassen
    elif theo is False:
        batch = torch.zeros((batch_size, 3 * N_x + 1), dtype=torch.float32,
                            device=device)  # Batch der größe batch_size*(3*N+1) - (N: Gridpoints)
        for i in range(batch_size):
            batch[i,:] = preprocess.getInput(torch.reshape(psi[i, :], (1,-1)), low_sys_nr+i, ind*dt, complex=False)

    elif theo:
        batch = torch.zeros((batch_size, 2*N_x), dtype=torch.float32,
                            device=device)  # Batch der größe batch_size*(3*N+1) - (N: Gridpoints)
        for i in range(batch_size):
            psi_sys_t = pd.read_csv(f"Train_Data/psi_sys_{i+low_sys_nr}.dat", sep=';', header=None).values.astype(np.complex64)[ind, :]
            psi_sys_t = torch.from_numpy(psi_sys_t).to(device).type(torch.complex64)
            psi_sys_t = torch.reshape(psi_sys_t, (1, -1)) # aus daten einen Tensor definieren
            batch[i, :] = preprocess.preprocess_psi_t(psi_sys_t) # auf Input der KI anpassen

    return batch

# create_batch für zufällige ordnung der systeme (mit bspw. np.random.shuffle(num_systems))
def create_batch_shuffle(batch_size, ind, dt, systems_arr, complex_bool,  psi = None, theo = False ):
    if psi is None and theo is False:
        batch = torch.zeros((batch_size, 3 * N_x + 1), dtype=torch.float32,
                            device=device)  # Batch der größe batch_size*(3*N+1) - (N: Gridpoints)
        for i, system in enumerate(systems_arr):
            psi_sys_t = pd.read_csv(f"Train_Data/psi_sys_{system}.dat", sep=';', header=None).values.astype(np.complex64)[ind, :]
            psi_sys_t = torch.from_numpy(psi_sys_t).to(device).type(torch.complex64)
            psi_sys_t = torch.reshape(psi_sys_t, (1, -1)) # aus daten einen Tensor definieren
            batch[i,:] = preprocess.getInput(psi_sys_t, system, ind*dt, complex=complex_bool) # auf Input der KI anpassen
    elif theo is False:
        batch = torch.zeros((batch_size, 3 * N_x + 1), dtype=torch.float32,
                            device=device)  # Batch der größe batch_size*(3*N+1) - (N: Gridpoints)
        for i, system in enumerate(systems_arr):
            batch[i,:] = preprocess.getInput(torch.reshape(psi[i, :], (1,-1)), system, ind*dt, complex=False)

    elif theo:
        batch = torch.zeros((batch_size, 2*N_x), dtype=torch.float32,
                            device=device)  # Batch der größe batch_size*(3*N+1) - (N: Gridpoints)
        for i, system in enumerate(systems_arr):
            psi_sys_t = pd.read_csv(f"Train_Data/psi_sys_{system}.dat", sep=';', header=None).values.astype(np.complex64)[ind, :]
            psi_sys_t = torch.from_numpy(psi_sys_t).to(device).type(torch.complex64)
            psi_sys_t = torch.reshape(psi_sys_t, (1, -1)) # aus daten einen Tensor definieren
            batch[i, :] = preprocess.preprocess_psi_t(psi_sys_t) # auf Input der KI anpassen

    return batch





def calc_norm(psi_t):
    return torch.sum(psi_t**2)
#print(model)

def evaluate():
# testing on one test_set
    with torch.no_grad():
        model.eval() # set model to evaluation-mode
        sys_nr = 0 # np.random.randint(0,100)
        mse_rho = 0
        mse_psi = 0
        overlap_abs = 0
        overlap_phase = 0
        for step in range(0, t_steps-1):
            if step == 0:
                psi_pred = create_batch(1, step, dt, sys_nr, complex_bool=True)
            else:
                psi_pred = create_batch(1, step, dt, sys_nr, complex_bool=False, psi=psi_pred)
            psi_pred = model(psi_pred)
            norm = calc_norm(psi_pred[0,:].detach())
            psi_pred[0,:] = psi_pred[0, :].detach() / torch.sqrt(norm)
            psi_theo = create_batch(1, step+1, dt, sys_nr, complex_bool=True, theo=True)
            mse_rho += test_losses.MSErho(psi_pred, psi_theo)
            mse_psi += test_losses.MSEpsi(psi_pred, psi_theo)
            if step == t_steps-2:
                (overlap_abs, overlap_phase) = test_losses.overlap(psi_pred, psi_theo)
            psi_pred = psi_pred.detach()
        mse_rho /= len(range(0,t_steps-1))
        mse_psi /= len(range(0,t_steps-1))
        with open("test_losses.csv", "a") as file:
            file.write(f"{np.round(overlap_abs.cpu().numpy(),5)};{np.round(overlap_phase.cpu().numpy(), 5)};{np.round(mse_psi.cpu().numpy(), 5)};{np.round(mse_rho.cpu().numpy(), 5)} \n")
        model.train() # set model to training-mode

#torch.autograd.set_detect_anomaly(True)
def train(n_epochs):
    loss_temp = 0
    losses = []
    batch_size = 200
    time_batch = 1
    for i in range(n_epochs):
        print(f"Epoch {i+1}: \n")
        systems = np.arange(N_systems)
        np.random.shuffle(systems)
        loss_epoch = 0
        #print(systems)
        for step in tqdm(range(0, t_steps-1)):
        # Die Batch-Size ist die Anzahl der Systeme, welche gleichzeitig durch das Netz geschickt werden. Nach jedem Zeitschritt werden die Gewichte aktualisiert
            for batch in range(int(N_systems/batch_size)):
                system_batch = systems[batch*batch_size:(batch+1)*batch_size]
                optimizer.zero_grad()
                psi_pred = create_batch_shuffle(batch_size, step, dt, system_batch, complex_bool=True)
                psi_pred = model(psi_pred)
                for j in range(batch_size):
                    norm = calc_norm(psi_pred[j, :].detach())
                    psi_pred[j, :] = psi_pred[j, :] / torch.sqrt(norm)
                psi_theo = create_batch_shuffle(batch_size, step+1, dt, system_batch, complex_bool=True, theo=True)
                #print(psi_theo.shape)
                loss = criterion(psi_pred, psi_theo)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_epoch += loss.detach().item()
                #loss.backward()
                #optimizer.step()
                #optimizer.zero_grad()
                #psi_pred = psi_pred.detach()
        print(f"\nMean-Loss (per Epoch): {loss_epoch/(N_systems/(batch_size*time_batch))}")
        losses.append(float(loss_epoch/(N_systems/(batch_size*time_batch))))
        with open("train_losses.csv", "a") as file:
            file.write(f"{float(loss_epoch/(N_systems/(batch_size*time_batch)))} \n")
        
        model.eval()
        evaluate()
        model.train()
        with torch.no_grad():
            torch.save(model.state_dict(), f"Models/model_epoch{i+1}_sys100_tstep100.pth")  # (10, 128, 6) 
        #scheduler.step()
    return losses

if __name__ == '__main__':
    epochs = 100
    losses = train(epochs)
    
    plt.figure(figsize=(9, 5))
    plt.plot(losses)
    plt.grid()
    plt.xlabel("N")
    plt.ylabel("loss")
    plt.savefig("Models/losses.png", dpi=200, format="png")
