from torch import nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,dropout_rate):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional =True
        self.num_directions = 2 if self.bidirectional else 1
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            num_layers=num_layers,
                            bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(self.hidden_size*self.num_directions, self.output_size)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[2]
        x_in = x.permute(1, 0, 2)
        device = next(self.lstm.parameters()).device
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(x_in, (h_0, c_0))
        out=self.dropout(output.squeeze())
        out = self.linear(out)
        return out




