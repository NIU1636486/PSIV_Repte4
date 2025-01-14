import torch
import torch.nn as nn
import torch.nn.functional as F





class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.n_channels = 21
        self.n_classes = 2

        self.kernel_sizes = [5, 3, 3] 
        self.pool_sizes = [2, 2, 2]  

        self.output_channels = [16, 32, 64]

        self.features = 64

        self.blocks = nn.ModuleList()
        in_channels = 1 

        for i in range(3):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=self.output_channels[i],
                              kernel_size=(1, self.kernel_sizes[i])),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(1, self.pool_sizes[i]), stride=(1, self.pool_sizes[i]))
                )
            )
            in_channels = self.output_channels[i] 

        self.fusion = nn.Flatten(start_dim=2)

        self.fc = nn.Linear(self.output_channels[-1], self.features)
        
        self.output_unit = nn.Linear(self.features, self.n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        for block in self.blocks:
            x = block(x) 

        x = self.fusion(x)

        x = x.mean(dim=2)
        x = self.fc(x)

        print("x.shape after fc", x.shape)
        
        x = self.output_unit(x)

        return x
