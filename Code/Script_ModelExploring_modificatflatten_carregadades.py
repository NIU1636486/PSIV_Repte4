import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import Adam
from Models.EpilepsyLSTM import EpilepsyLSTM
from Models.ModelWeightsInit import init_weights_xavier_normal
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Models.EpilepsyLSTM import *
from loadDataFirstTrain import loadData


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


# LOAD DATA: EL MATEIX QUE LOADDATAFIRSTTRAIN.PY PERO AMB EL FLATTEN !!!
def loadData(pathDir):
    files = os.listdir(pathDir)
    
    # Filtrar fitxers .parquet i .npz
    parquets = sorted([file for file in files if file.endswith('.parquet')])
    npzs = sorted([file for file in files if file.endswith('.npz')])

    labels = []
    windows = []

    for parquet, npz in zip(parquets, npzs):
        # Verificar que los archivos coinciden
        if parquet.split('_')[0] != npz.split('_')[0]:
            print(f"Error: Archivos no coinciden -> {parquet}, {npz}")
            continue
        
        # Leer archivo parquet
        parquet_path = os.path.join(pathDir, parquet)
        meta = pd.read_parquet(parquet_path, engine='fastparquet')
        print(f"Archivo parquet cargado: {parquet}")
        label_list = meta.iloc[:, 0].to_numpy()
        
        # Cargar archivo npz
        npz_path = os.path.join(pathDir, npz)
        data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
        print(f"Archivo npz cargado: {npz}")
        EEG_win = data["EEG_win"]
        
        ###FER UN FLATTEN DE LES FINESTRES: on numfeatures= 128 *21
        EEG_win_flatten = EEG_win.reshape(EEG_win.shape[0], -1)
        
        # Almacenar directamente los resultados sin listas intermedias
        labels.extend(label_list.tolist())
        windows.append(EEG_win_flatten)
        
        # Liberar memoria intermedia
        del label_list, data, EEG_win
        gc.collect()

    # FER UN FLATTEN A LES WINDOWS EN CANVI D?UN CONCATENATE??
    return windows, labels

class Standard_Dataset(Dataset):
    def __init__(self, X, Y=None, transformation=None):
        super().__init__()
        X = np.array(X, dtype=np.float32)
        self.X = X
        self.y = Y
        self.transformation = transformation
        print(type(self.X))
        print(self.X[0])

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
X, y = loadData(DATA_PATH)
dataset = Standard_Dataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print("Data loaded")


# Create EpilepsyLSTM model and initialize weights
inputmodule_params, net_params, outmodule_params = get_default_hyperparameters()
model = EpilepsyLSTM(inputmodule_params, net_params, outmodule_params)
model.init_weights()
model.to(device)
print("Model Creat")

#Execute lstm unit of shape [batch, sequence_length, features]
x = torch.from_numpy(np.array(X[0:2,:,:])).float()  # convert the numpy to tensor
x = x.permute(0, 2, 1).to(DEVICE)                   # permute and send to the same device as the model
out, (hn, cn) = model.lstm(x)
# Delete a model
del model

### TRAIN MODEL ###
#optimizer = Adam(model.parameters(), lr=3e-4)
#criterion = nn.CrossEntropyLoss()
#epochs = 10
#for epoch in range(epochs):
    #model.train()
    #total_loss = 0.0
    #correct = 0
    #total = 0

    #for X_batch, y_batch in dataloader:
        
        #Reorganitzar les dimensions per la LSTM: [batch, sequence_length, features]
        #com les linies comentades anteriorment de execute a lstm unit
        #NS SI ESTA Bé
        #X_batch = X_batch.permute(0, 2, 1).to(device)
        #y_batch = y_batch.to(device)

        #optimizer.zero_grad()
        #outputs = model(X_batch)
        #loss = criterion(outputs, y_batch)
        #loss.backward()
        #optimizer.step()

        #total_loss += loss.item()
        #_, predicted = torch.max(outputs, 1)
        #correct += (predicted == y_batch).sum().item()
        #total += y_batch.size(0)

    #print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {correct / total:.4f}")

# Guardar el modelo
#torch.save(model.state_dict(), "epilepsy_lstm.pth")
