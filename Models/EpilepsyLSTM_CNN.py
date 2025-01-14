
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.ModelWeightsInit import *


class EpilepsyLSTMCNN(nn.Module):
    """
    Implementation:
        A channel independent generalized seizure detection method for pediatric epileptic seizures
        batch_size 600
        epochs 1000
        lr = 1e-4
        optmizer Adam
    """
    def __init__(self, pretrained_cnn, inputmodule_params, net_params, outmodule_params):
        super().__init__()

        print('Running class: ', self.__class__.__name__)
        
        ### Pretrained CNN (Feature Extractor)
        self.cnn = pretrained_cnn
        for param in self.cnn.parameters():
            param.requires_grad = False

        ### NETWORK PARAMETERS
        n_nodes = self.cnn.output_channels[-1]  # Dimensionality of input (input_size)


        Lstacks = net_params['Lstacks']  # Number of stacked LSTM layers
        dropout = net_params['dropout']
        hidden_size = net_params['hidden_size']
       
        n_classes = outmodule_params['n_classes']
        hd = outmodule_params['hd']
        
        self.inputmodule_params = inputmodule_params
        self.net_params = net_params
        self.outmodule_params = outmodule_params
        
        ### NETWORK ARCHITECTURE
        self.lstm = nn.LSTM(
            input_size=n_nodes,  # Input size matches the input vector's dimensionality
            hidden_size=hidden_size,  # Number of features in hidden state
            num_layers=Lstacks,  # Number of stacked LSTM layers
            batch_first=True,  # Input shape is [batch, sequence_length, features]
            bidirectional=False,  # Single-directional LSTM
            dropout=dropout  # Dropout probability
        )
        
        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hd),
            nn.ReLU(),
            nn.Linear(hd, n_classes)
        ) 

    def init_weights(self):
        init_weights_xavier_normal(self)
        
    def forward(self, x):
        ## LSTM Processing
        # Reshape input to [batch, sequence_length=1, features=n]
        with torch.no_grad():
            x = self.cnn(x)
        print(x.shape)
        x = x.unsqueeze(1)  # Add a sequence_length dimension (set to 1)
        out, (hn, cn) = self.lstm(x)
        # out is [batch, sequence_length=1, hidden_size] for the last stack output
        out = out[:, -1, :]  # Extract the last time step's hidden state

        ## Output Classification (Class Probabilities)
        x = self.fc(out)

        return x




