import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, cnn_model, lstm_params, n_classes):
        super(CNNLSTMModel, self).__init__()

        # Pretrained CNN Model (freeze its parameters)
        self.cnn = cnn_model
        for param in self.cnn.parameters():
            param.requires_grad = False  # Freeze the CNN parameters

        # LSTM Parameters
        self.hidden_size = lstm_params['hidden_size']
        self.num_layers = lstm_params['num_layers']
        self.dropout = lstm_params['dropout']
        self.bidirectional = lstm_params.get('bidirectional', False)

        # Input size to LSTM is the number of output channels of the CNN
        self.lstm = nn.LSTM(
            input_size=self.cnn.output_channels[-1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        # Fully connected layer for classification
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, n_classes)

    def forward(self, x):
        # Pass input through the CNN
        with torch.no_grad():  # Ensure the CNN is not updated
            x = self.cnn(x, feature_extraction=True)

        # Reshape CNN output for LSTM (batch, seq_length, features)
        x = x.unsqueeze(1)  # Add a sequence dimension for LSTM

        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last output of the sequence

        # Final classification
        out = self.fc(lstm_out)
        return out
