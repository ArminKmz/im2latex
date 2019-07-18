import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, query_size, memory_size, hidden_size):
        '''
            query_size = decoder_hidden_size
            memory_size = encoder_hidden_size
        '''
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.memory_layer = nn.Linear(memory_size, hidden_size, bias=False)
        self.alignment_layer = nn.Linear(hidden_size, 1, bias=False)

    def score(self, query, keys):
        query = self.query_layer(query) # (batch, 1, hidden_size)
        keys = self.memory_layer(keys) # (batch, time, hidden_size)
        alignments = self.alignment_layer(F.tanh(query + keys)) # (batch, time, 1)
        return alignments

    def forward(self, query, keys):
        '''
            query: last hidden of top layer decoder -> (batch, 1, query_size)
            keys: encoder outputs -> (batch, time, memory_size)
            return context -> (batch, 1, memory_size),
                   weights -> (batch, 1, time)
        '''
        scores = self.score(query, keys) # (batch, time, 1)
        weights = F.softmax(scores, dim=1).transpose(1, 2) # (batch, 1, time)
        context = torch.bmm(weights, keys) # (batch, 1, memory_size)
        return context, weights

class AttnDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, encoder_hidden_size, output_size, device):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.lstm = nn.LSTM(input_size+hidden_size, hidden_size, batch_first=True)
        self.attn = BahdanauAttention(hidden_size, encoder_hidden_size, hidden_size)
        self.o = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x, hp, cp, op, encoder_outputs):
        # x -> (batch, 1, input_size)
        # op -> (batch, hidden_size)
        # hp, cp -> (#layers, batch, hidden_size)
        # encoder_outputs -> (batch, time, #directions*encoder_hidden_size)
        lstm_input = torch.cat([x, op.unsqueeze(dim=1)], dim=2) # (batch, 1, input_size + hidden_size)
        hidden, (hn, cn) = self.lstm(lstm_input, (hp, cp))
        # hidden -> (batch, 1, hidden_size)
        # (hn, cn) -> (#layers, batch, hidden_size)
        context, attn_weights = self.attn(hidden, encoder_outputs)
        # context -> (batch, 1, hidden_size)
        # attn_weights -> (batch, 1, time)
        o_input = torch.cat([hidden.squeeze(1), context.squeeze(1)], dim=1) # (batch, 2*hidden_size)
        on = F.tanh(self.o(o_input))
        # next_o -> (batch, hidden_size)
        output = F.log_softmax(self.out(on), dim=1)
        # output: log softmax of symbol scores -> (batch, output_size)
        return output, (hn, cn), on, attn_weights
