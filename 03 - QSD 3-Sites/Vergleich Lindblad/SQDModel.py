import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mpcpu
import numpy as np
import math
from tqdm import tqdm
#import trainLoader
import os
from time import time
# Fürs Arbeiten auf mehreren GPU's
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank, world_size):
    """
    :param rank: Unique identifier of each process
    :param world_size: Total number of distributed processes
    :return:
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '27182'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


#import test_losses
num_workers = mpcpu.cpu_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#N_systems = 1000
#df_sys_params = pd.read_csv('Train_Data/sys_params.dat', sep=';')
N_x = 10
#print(N_x)
dt = 10 **-3


#batch_size = 16


class Net(nn.Module):
    def __init__(self, N_input, N_hidden, N_output):
        super(Net, self).__init__()
        self.inp = nn.Linear(N_input, N_hidden)
        '''
        self.hidden = nn.ModuleList()
        self.batch = nn.ModuleList()
        for i in range(N_hidden_layers):
            self.hidden.append(nn.Linear(N_hidden, N_hidden))
            if i % 2 == 0 and i != 0:
                self.batch.append(nn.BatchNorm1d(N_hidden))
        '''
        #self.N_hidden_layers = N_hidden_layers
        self.hidden1 = nn.Linear(N_hidden, N_hidden)
        self.hidden2 = nn.Linear(N_hidden, N_hidden)
        self.hidden3 = nn.Linear(N_hidden, N_hidden)
        self.hidden4 = nn.Linear(N_hidden, N_hidden)
        self.hidden5 = nn.Linear(N_hidden, N_hidden)
        #self.hidden6 = nn.Linear(N_hidden, N_hidden)
        #self.hidden7 = nn.Linear(N_hidden, N_hidden)
        #self.hidden8 = nn.Linear(N_hidden, N_hidden) 
        #self.batch_norm1 = nn.BatchNorm1d(N_hidden)
        #self.batch_norm2 = nn.BatchNorm1d(N_hidden)
        #self.batch_norm3 = nn.BatchNorm1d(N_hidden)
        self.out = nn.Linear(N_hidden, N_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.inp(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.hidden4(x)
        x = self.relu(x)
        x = self.hidden5(x)
        #x = self.relu(x)
        #x = self.hidden6(x)
        x = self.out(x)
     
        return x


#model = Net(N_input=3 * N_x + 1, N_hidden=1024, N_output=2 * N_x)

#model.load_state_dict(torch.load("model_epoch2000.pth"))
#model = model.to(device)
#model.train()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#criterion = nn.MSELoss()


def calc_norm(psi_t):
    return torch.sum(psi_t ** 2)

'''
def evaluate(test_dataloader, n):
    with torch.no_grad():
        loss_wp = 0
        for it, (psi_in, psi_out) in enumerate(test_dataloader):
            psi_in = psi_in.to(device)
            psi_out = psi_out.to(device)
            psi_pred = model(psi_in)
        
            for j in range(psi_out.shape[0]):
                norm = calc_norm(psi_pred[j, :].detach())
                psi_pred[j, :] = psi_pred[j, :] / torch.sqrt(norm)
            
            loss = criterion(psi_pred, psi_out)
            loss_wp += loss.item()
        return loss_wp
'''
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )

def load_train_objs(train_input, train_output):
    train_set = trainLoader.TrainDataset(train_input, train_output)
    inp_neur = len(train_set.x[0, :])
    output_neur = len(train_set.y[0, :])
    model = Net(N_input=inp_neur, N_hidden_layers=8, N_hidden=1024, N_output=output_neur)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    return train_set, model, optimizer

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 gpu_id: int,
                 save_every: int) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[gpu_id])
        #self.test_data = test_data

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        psi_pred = self.model(source)
        loss = torch.nn.MSELoss()(psi_pred, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0]) # Batch-Size of Training-Data
        epoch_loss = 0
        #print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize {b_sz} | Steps: {len(self.train_data)}")
        for source, target in self.train_data:
            source = source.to(device=self.gpu_id)
            target = target.to(device=self.gpu_id)
            loss = self._run_batch(source, target)
            epoch_loss += loss.item()
        return epoch_loss/len(self.train_data)
            
    def _eval(self):
        self.model.module.eval()
        with torch.no_grad():
            loss_epoch = 0
            for source, target in self.test_data:
                source = source.to(device=self.gpu_id)
                target = target.to(device=self.gpu_id)
                loss = self._run_batch(source, target)
                loss_epoch += loss.item()
        return loss_epoch/len(self.test_data)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        torch.save(ckp, f"Models/model_checkpoint{epoch}.pth")

    def train(self, start_epoch, end_epoch):
        for epoch in tqdm(range(start_epoch, end_epoch)):
            loss = self._run_epoch(epoch)
            '''
            if self.gpu_id == 0:
                mseloss = self._eval()
                self.model.train()
                with open("test_losses.csv", "a") as f:
                    f.write(f"{mseloss:.15f}\n")
                if (epoch+1) % self.save_every == 0:
                    print(f"Loss on Test-Data at Epoch {epoch+1}: {mseloss:.15f}")
            '''
            if self.gpu_id == 0 and (epoch+1) % self.save_every == 0:
            # self.gpu_id == 0 ist als bedingung vorgegeben, da alle GPU's die gleichen Netze besitzen (redundanz vermeiden)
                print(f"Training-Loss in Epoch {epoch+1}: {loss/2:.20f}")
                self._save_checkpoint(epoch+1)
                


def main(rank, world_size, save_every,  train_dataset, model, optimizer, start_epoch, end_epoch, batch_size):
    ddp_setup(rank, world_size)
    train_data = prepare_dataloader(train_dataset, batch_size=batch_size)
    #test_dataloader = prepare_dataloader(test_dataset, batch_size=batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(start_epoch, end_epoch)
    destroy_process_group()



'''
def train(start_epoch, end_epoch, train_dataset, test_dataset, n):
    loss_temp = 0
    losses = []
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    start = time()
    for i in tqdm(range(start_epoch, end_epoch)):
        if i % 200 == 0 and batch_size < 8192:
            batch_size *= 2
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            print(f"New Batch-Size: {batch_size}")
        loss_epoch = 0
        #print(f"Epoch {i} - Batch-Size {batch_size}: \n")
        #print(systems)
        for it, (psi_in, psi_out) in enumerate(train_dataloader):
            psi_in = psi_in.to(device)
            psi_out = psi_out.to(device)
            psi_pred = model(psi_in)
            
            for j in range(psi_out.shape[0]):
                norm = calc_norm(psi_pred[j, :].detach())
                psi_pred[j, :] = psi_pred[j, :] / torch.sqrt(norm)
            
            loss = criterion(psi_pred, psi_out)
            loss_temp += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if (it+1) % 625 == 0:
            #    print(f"Epoch {i+1}/{n_epochs}; Step {it+1}/{n/batch_size}: Loss {loss_temp/(it+1/batch_size):.10f}")
        model.eval()
        test_loss = evaluate(test_dataloader, n)
        model.train()
        with open("train_losses.csv", "a") as f:
            f.write(f"{loss_temp / (math.ceil(0.9 * n / batch_size)):.10f} \n")
        with open("test_losses.csv", "a") as f:
            f.write(f"{test_loss / (math.ceil(0.1 * n / batch_size)):.10f} \n")

        losses.append(loss_temp / (math.ceil(0.9 * n / batch_size)))
        loss_temp = 0
        if i % 100 == 0:
            print(f"\nMean training loss after epoch {i}: {loss_temp / (math.ceil(0.9 * n / batch_size)):.15f}")
            print(f"Mean test loss after epoch {i}: {test_loss / (math.ceil(0.1 * n / batch_size)):.15f} \n")
            with torch.no_grad():
                torch.save(model.state_dict(),
                           f"Models/model_epoch{i}_testloss_{test_loss / (math.ceil(0.1 * n / batch_size)):.15f}.pth")  # (10, 256, 6)
        #scheduler.step()
    end = time()

    return losses, end - start
'''

if __name__ == '__main__':
    import sys
    epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    train_input = sys.argv[3]
    train_output = sys.argv[4]
    batch_size = 256
    world_size = torch.cuda.device_count()
    dataset, model, optimizer = load_train_objs(train_input, train_output)
    #test_dataset = trainLoader.TrainDataset("test_input.dat", "test_output.dat")
    #epochs = 500
    for i in range(10):
        if i != 0:
            model.load_state_dict(torch.load(f"Models/model_checkpoint{i*epochs}.pth"))
        if batch_size < 32768 and i != 0:
            batch_size *= 2
        print(f"Batch-Size: {int(batch_size)}")
        mp.spawn(main, args=(world_size, save_every, dataset, model, optimizer, i*epochs, (i+1)*epochs, batch_size),
                 nprocs=world_size)

    '''
    epochs = 2000
    start_epoch = 2001
    dataset = trainLoader.TrainDataset("data_input300.dat", "data_output300.dat")
    num_data = len(dataset)
    train_size = int(0.9 * num_data)
    test_size = num_data - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loss, duration = train(start_epoch, start_epoch + epochs, train_dataset, test_dataset, num_data)
    with open("train_time.txt", "w") as file:
        file.write(f"Training-Duration: {np.round(duration / 3600, 2)}")
    '''
