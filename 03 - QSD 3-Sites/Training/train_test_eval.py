import torch
import torch.nn as nn
from math import ceil
from torch.utils.data import DataLoader, Dataset
from trainLoader import TrainDataset
from QSDModel import Net
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
N_x = 3
batch_size = 64
# Training-Data
train_data = TrainDataset("data_input.dat", "data_output.dat")
train_dataload = DataLoader(train_data, batch_size=batch_size, num_workers=4)
test_data = TrainDataset("test_input.dat", "test_output.dat")
test_dataload = DataLoader(test_data, batch_size=batch_size, num_workers=4)
# Training - Evaluation
for i in tqdm(range(1,101)):
    model = Net(N_input=4*N_x, N_hidden_neur=1024, N_output=2 * N_x)
    model.load_state_dict(torch.load(f"Models/model_checkpoint{int(i*50)}.pth", map_location=device))
    model = model.to(device)
    model.eval()
    loss = 0
    with torch.no_grad():
        for _, (source, target) in enumerate(train_dataload):
            source = source.to(device)
            target = target.to(device)
            pred = model(source)
            temp_loss = criterion(pred, target)
            loss += temp_loss.item()
        loss /= ceil((len(train_data)/batch_size))
        with open("train_losses.csv", "a") as file:
            file.write(f"{int(i*50)};{loss} \n")

print("Train-Evaluation done")
# Test - Evaluation
for i in tqdm(range(1,101)):
    model = Net(N_input=4*N_x, N_hidden_neur=1024, N_output=2 * N_x)
    model.load_state_dict(torch.load(f"Models/model_checkpoint{int(i*50)}.pth", map_location=device))
    model = model.to(device)
    model.eval()
    loss = 0
    with torch.no_grad():
        for _, (source, target) in enumerate(test_dataload):
            source = source.to(device)
            target = target.to(device)
            pred = model(source)
            temp_loss = criterion(pred, target)
            loss += temp_loss.item()
        loss /= ceil((len(test_data)/batch_size))
        with open("test_losses.csv", "a") as file:
            file.write(f"{int(i*50)};{loss} \n")
        
print("Test-Evaluation done")