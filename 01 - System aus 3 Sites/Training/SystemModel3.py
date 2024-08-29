import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import multiprocessing as mp
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import trainLoader
import matplotlib.pyplot as plt
num_workers = mp.cpu_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_x = 3
dt = 10**-2


class Net(nn.Module):
    def __init__(self, N_input, N_hidden, N_output):
        super(Net, self).__init__()
        self.inp = nn.Linear(N_input, N_hidden)
        self.hidden1 = nn.Linear(N_hidden, N_hidden)
        self.hidden2 = nn.Linear(N_hidden, N_hidden)
        self.hidden3 = nn.Linear(N_hidden, N_hidden)
        self.hidden4 = nn.Linear(N_hidden, N_hidden)
        self.out = nn.Linear(N_hidden, N_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.inp(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.hidden4(x)
        x = self.out(x)
        return x


model = Net(N_input=3*N_x + 1, N_hidden=256, N_output=2*N_x)

'''
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
'''

#model.load_state_dict(torch.load("model_epoch500.pth"))
model = model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

def calc_norm(psi_t):
    return torch.sum(psi_t**2)


# Model testen mit Testdaten - Hier sogesagt MSE von der Wellenfunktion, was hier dem MSE wiederspiegelt
def evaluate(test_dataloader, n):
    with torch.no_grad():
        loss_wp = 0
        for it, (psi_in, psi_out) in enumerate(test_dataloader):
            psi_in = psi_in.to(device)
            psi_out = psi_out.to(device)
            psi_pred = model(psi_in)
            '''
            for j in range(psi_out.shape[0]):
                norm = calc_norm(psi_pred[j, :].detach())
                psi_pred[j, :] = psi_pred[j, :] / torch.sqrt(norm)
            '''
            loss = criterion(psi_pred, psi_out)
            loss_wp += loss.item()
        return loss_wp

def train(start_epoch, end_epoch, train_dataset, test_dataset, n):
    loss_temp = 0
    losses = []
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for i in range(start_epoch, end_epoch):
        if i % 50 == 0:
            batch_size *= 2
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        loss_epoch = 0
        print(f"Epoch {i} - Batch-Size {batch_size}: \n")
        #print(systems)
        for it, (psi_in, psi_out) in enumerate(tqdm(train_dataloader)):
            psi_in = psi_in.to(device)
            psi_out = psi_out.to(device)
            psi_pred = model(psi_in)
            '''
            for j in range(psi_out.shape[0]):
                norm = calc_norm(psi_pred[j, :].detach())
                psi_pred[j, :] = psi_pred[j, :] / torch.sqrt(norm)
            '''
            loss = criterion(psi_pred, psi_out)
            loss_temp += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if (it+1) % 625 == 0:
            #    print(f"Epoch {i+1}/{n_epochs}; Step {it+1}/{n/batch_size}: Loss {loss_temp/(it+1/batch_size):.10f}")
        model.eval()
        test_loss = evaluate(test_dataloader,n)
        model.train()
        with open("train_losses.csv", "a") as f:
            f.write(f"{loss_temp/(math.ceil(0.9*n/batch_size)):.10f} \n")
        with open("test_losses.csv", "a") as f:
            f.write(f"{test_loss/(math.ceil(0.1*n/batch_size)):.10f} \n")
        
        losses.append(loss_temp/(math.ceil(0.9*n/batch_size)))
        print(f"\nMean training loss after epoch {i}: {loss_temp/(math.ceil(0.9*n/batch_size)):.10f}")
        print(f"Mean test loss after epoch {i}: {test_loss/(math.ceil(0.1*n/batch_size)):.10f} \n")
        loss_temp = 0
        if i % 10 == 0:
          with torch.no_grad():
              torch.save(model.state_dict(), f"Models/model_epoch{i}_testloss_{test_loss/(math.ceil(0.1*n/batch_size)):.10f}.pth")  # (10, 256, 6)
        #scheduler.step()
    return losses


if __name__ == '__main__':
    epochs = 400
    start_epoch = 1
    dataset = trainLoader.TrainDataset()
    num_data = len(dataset)
    train_size = int(0.9*num_data)
    test_size = num_data - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loss = train(start_epoch, start_epoch+epochs, train_dataset, test_dataset, num_data)
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, epochs+1), loss)
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("train_loss")
    plt.savefig("Models/losses.png", dpi=200, format="png")
