import torch


class LSTMEncoder(torch.nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=1, hidden_size=256, num_layers=2
        )
        self.linear = torch.nn.Linear(256, 160)

    def forward(self, x):
        return self.linear(self.lstm(x.permute(2, 0, 1))[0][-1])
