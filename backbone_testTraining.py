import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from model_Backbone import CNNModel
from torch import nn
from loadDataFirstTrain import loadData
import wandb



wandb.login(key="8e9b2ed0a8b812e7888f16b3aa28491ba440d81a")
epochs = 20
batch_size = 128
wandb.init(project="PSIV_Repte4", config={"epochs": epochs, "batch_size": batch_size}, dir="wandb")
if platform.system() == 'Linux':
    DATA_PATH = "/fhome/maed/EpilepsyDataSet"
else:
    DATA_PATH = "input_reduit"



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

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

print("Loading data...")
X, y = loadData(DATA_PATH)
dataset = Standard_Dataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("Data loaded")

model = CNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
initialize_weights(model)
optimizer = Adam(model.parameters(), lr=3e-4)
criterion = CrossEntropyLoss()
print("Model Creat")

### TRAIN MODEL ###
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        wandb.log({"loss": loss.item()})
    wandb.log({"loss": total_loss / len(dataloader), "accuracy": correct / total, "epoch": epoch})
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {correct / total:.4f}")

model = model.to("cpu")
torch.save(model.state_dict(), "model_backbone.pth")

torch.cuda.empty_cache()
gc.collect()

wandb.finish()