import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.ModelWeightsInit import *

#handles time-series data: EEG signals
#LSTM layer followed by fully connected layers for classification


class EpilepsyLSTM(nn.Module):
    """
    Implementation:
        A channel independent generalized seizure detection method for pediatric epileptic seizures
        batch_size 600
        epochs 1000
        lr = 1e-4
        optmizer Adam
    """
    def __init__(self, inputmodule_params,net_params,outmodule_params):
        super().__init__()

        print('Running class: ', self.__class__.__name__)
        
        ### NETWORK PARAMETERS
        n_nodes=inputmodule_params['n_nodes'] #input features: EEG channels
    
        Lstacks=net_params['Lstacks'] #number of LSTM layers
        dropout=net_params['dropout'] 
        hidden_size=net_params['hidden_size'] #number of features in the LSTM hidden state
       
        n_classes=outmodule_params['n_classes'] #2 classes seizure vs non-seizure
        hd=outmodule_params['hd'] #hidden layer for the FC layer
        
        self.inputmodule_params=inputmodule_params
        self.net_params=net_params
        self.outmodule_params=outmodule_params
        
        ### NETWORK ARCHITECTURE
        # IF batch_first THEN (batch, timesteps, features), ELSE (timesteps, batch, features)
        #input: batch size, sequence_length, features
        #output: batch size, sequence_length, hidden_size
        self.lstm = nn.LSTM(input_size=n_nodes, # the number of expected features (out of convs)
                                       hidden_size= hidden_size, # the number of features in the hidden state h
                                       num_layers= Lstacks, # number of stacked lstms 
                                       batch_first = True,
                                       bidirectional = False,
                                       dropout=dropout)
        
        #maps hidden_size to hd and then to n_classes
        self.fc = nn.Sequential(nn.Linear(hidden_size, hd),
                                nn.ReLU(),
                                nn.Linear(hd, n_classes)
                                ) 

    
    def init_weights(self):
         init_weights_xavier_normal(self)
        
    def forward(self, x):
        
        ## Reshape input
        # input [batch, features (=n_nodes), sequence_length (T)] ([N, 21, 640])
        x = x.permute(0, 2, 1) # lstm  [batch, sequence_length, features]
        
        ## LSTM Processing
        out, (hn, cn) = self.lstm(x)
        # out is [batch, sequence_length, hidden_size] for last stack output
        # hn and cn are [1, batch, hidden_size]
        out = out[:, -1, :] # hT state of lenght hidden_size
        
        #uses the last time-step output for classification
        ## Output Classification (Class Probabilities)
        x = self.fc(out)

        return x
