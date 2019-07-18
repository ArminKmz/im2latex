import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, device):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size,
                            batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        # x -> (batch, time, encoder_input_size)
        outputs, (hn, cn) = self.lstm(x, self.init_hidden(x.shape[0]))
        # outputs: ecnoder hiddens -> (batch, time, directions x encoder_hidden_size)
        # hn, cn -> (#layers, batch, directions x encoder_hidden_size)
        return outputs, (self.cat_directions(hn), self.cat_directions(cn))

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (torch.zeros(2, batch_size, self.hidden_size, device=self.device),
                    torch.zeros(2, batch_size, self.hidden_size, device=self.device))
        else:
            return (torch.zeros(1, batch_size, self.hidden_size, device=self.device),
                    torch.zeros(1, batch_size, self.hidden_size, device=self.device))

    def cat_directions(self, h):
        if self.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
