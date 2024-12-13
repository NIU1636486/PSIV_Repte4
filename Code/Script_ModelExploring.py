
###MAIN ENTRY POINT and how to use the EpilepsyLSTM model

#Main project directory
Main_dir = r''


from Models.EpilepsyLSTM import *

### DEFINE VARIABLES
DEVICE = 'cpu'       # options: 'cpu', 'cuda:0', 'cuda:1'
N_CLASSES = 2        # number of classes. This case 2={seizure ,non-seizure}

# Default hyper parameters: initializes dictionaries for model parameters
def get_default_hyperparameters():
   
    # initialize dictionaries
    inputmodule_params={}
    net_params={}
    outmodule_params={}
    
    # network input parameters
    ##contains the number of nodes n_nodes: 21
    inputmodule_params['n_nodes'] = 21
    
    # LSTM unit  parameters: hidden_size, Lstacks and dropout
    net_params['Lstacks'] = 1  # stacked layers (num_layers)
    net_params['dropout'] = 0.0
    net_params['hidden_size']= 256  #h
   
    # network output parameters: n_classes and hd
    outmodule_params['n_classes']=2
    outmodule_params['hd']=128
    
    return inputmodule_params, net_params, outmodule_params


### LOAD DATASET
# IMPLEMENT YOUR OWN CODE FOR LOADING ndarray X with EEG WINDOW SIGNAL
# and array y with label for each window
# X should be of size [NSamp,21,128]
# y should be a binary vector of size NSamp

# Create EpilepsyLSTM model and initialize weights
inputmodule_params, net_params, outmodule_params = get_default_hyperparameters()
model = EpilepsyLSTM(inputmodule_params, net_params, outmodule_params)
model.init_weights() #initializes model weights

model.to(DEVICE)

#Execute lstm unit of shape [batch, sequence_length, features]
x = torch.from_numpy(np.array(X[0:2,:,:])).float()  # convert the numpy to tensor
x = x.permute(0, 2, 1).to(DEVICE)                   # permute and send to the same device as the model
out, (hn, cn) = model.lstm(x)

# Delete a model
del model


















