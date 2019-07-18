import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, input_size, embedded_size, padding_idx):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(input_size, embedded_size, padding_idx=padding_idx)

    def forward(self, x):
        # x: symbols id -> (batch, time)
        embedded = self.embedding(x)
        # embedded -> (batch, time, embedded_size)
        return embedded
