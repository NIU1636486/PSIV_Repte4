import os
import numpy as np
import platform

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import Adam
import torchvision.transforms as T

from Models.EpilepsyLSTM import EpilepsyLSTM
from Models.ModelWeightsInit import init_weights_xavier_normal
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Models.EpilepsyLSTM import *
from loadDataFirstTrainProvesLSTM import loadData

if platform.system() == 'Linux':
    DATA_PATH = "/fhome/maed/EpilepsyDataSet"
else:
    DATA_PATH = "input_reduit"



### DEFINE VARIABLES
DEVICE = 'cpu'       # options: 'cpu', 'cuda:0', 'cuda:1'
N_CLASSES = 2        # number of classes. This case 2={seizure ,non-seizure}

# Default hyper parameters
def get_default_hyperparameters():
   
    # initialize dictionaries
    inputmodule_params={}
    net_params={}
    outmodule_params={}
    
    # network input parameters
    inputmodule_params['n_nodes'] = 21
    
    # LSTM unit  parameters
    net_params['Lstacks'] = 1  # stacked layers (num_layers)
    net_params['dropout'] = 0.0
    net_params['hidden_size']= 256  #h
   
    # network output parameters
    outmodule_params['n_classes']=2
    outmodule_params['hd']=128
    
    return inputmodule_params, net_params, outmodule_params

### LOAD DATASET
# IMPLEMENT YOUR OWN CODE FOR LOADING ndarray X with EEG WINDOW SIGNAL
# and array y with label for each window
# X should be of size [NSamp,21,128]
# y should be a binary vector of size NSamp


class Standard_Dataset(Dataset):
    def __init__(self, X, Y=None, transformation=None):
        super().__init__()
        X = np.array(X, dtype=np.float32)
        self.X = X
        self.y = Y
        self.transformation = transformation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_item = torch.from_numpy(self.X[idx]).float()
        if self.transformation:
            X_item = self.transformation(X_item)

        if self.y is not None:
            y_item = torch.tensor(self.y[idx], dtype=torch.long)
            return X_item, y_item
        else:
            return X_item
            

print("Loading data...")
X, y, groups = loadData(DATA_PATH)
dataset = Standard_Dataset(X, y, transformation=T.Compose([]))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
print("Data loaded")


# Create EpilepsyLSTM model and initialize weights
inputmodule_params, net_params, outmodule_params = get_default_hyperparameters()
model = EpilepsyLSTM(inputmodule_params, net_params, outmodule_params)
model.init_weights()
model.to(device)
print("Model Creat")

#Execute lstm unit of shape [batch, sequence_length, features]
# x = x.permute(0, 2, 1).to(DEVICE)                   # permute and send to the same device as the model
# out, (hn, cn) = model.lstm(x)
# Delete a model
# del model

### TRAIN MODEL ###
optimizer = Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
epochs = 10
print("Starting train")
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        print(X_batch.shape)
        
        # Reorganitzar les dimensions per la LSTM: [batch, sequence_length, features]
        # com les linies comentades anteriorment de execute a lstm unit
        # NS SI ESTA Bé
        # X_batch = X_batch.permute(0, 2, 1).to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        print(outputs)
        loss = criterion(outputs, y_batch)
        print("LOSS", loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(total_loss)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        print(total)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {correct / total:.4f}")

# Guardar el modelo
torch.save(model.state_dict(), "epilepsy_lstm.pth")
