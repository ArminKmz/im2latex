import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from .generator import GreedyDecoder, BeamSearchDecoder
import numpy as np

class Model(nn.Module):
    def __init__(self, cnn, encoder, embedding, decoder, device):
        super(Model, self).__init__()
        self.cnn = cnn
        self.encoder = encoder
        self.embedding = embedding
        self.decoder = decoder
        self.device = device

        directions = 2 if self.encoder.bidirectional else 1

        self.init_h = nn.Linear(directions*self.encoder.hidden_size, self.decoder.hidden_size).to(self.device)
        self.init_o = nn.Linear(directions*self.encoder.hidden_size, self.decoder.hidden_size).to(self.device)

    def get_cnn_feature(self, x):
        '''
            input:
                > x: image (batch, 1, H, W)
            return:
                > cnn feature map (batch, C, H', W')
        '''
        x = self.cnn(x)
        return x

    def get_encoder_anotation_grid(self, x):
        '''
            input:
                > x: cnn feature map (batch, C, H', W')

            return:
                > encoder_outputs: top layer hidden states of encoder applied on H' row of x
                                   (batch, H', W', directions * encoder_hidden_size)
                > hn, cn: last hidden states of encoder applied on H' row of x
                          (#layers, batch, H', decoder_hidden_size)

        '''
        B, C, H_prime, W_prime = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B*H_prime, W_prime, C) # (batch * H', W', C)
        encoder_outputs, (hn, cn) = self.encoder(x)
        # encoder_outputs -> (batch * H', W', directions * encoder_hidden_size)
        # hn, cn -> (#layers, batch * H', directions * encoder_hidden_size)
        encoder_outputs = encoder_outputs.view(B, H_prime, W_prime, -1).contiguous()
        hn = hn.view(hn.shape[0], B, H_prime, -1).contiguous()
        cn = hn.view(hn.shape[0], B, H_prime, -1).contiguous()
        return encoder_outputs, (hn, cn)

    def add_positional_embedding(self, x):
        '''
            input:
                > x: (batch, W'*H', directions * encoder_hidden_size)
            return:
                > x + positional_embedding
        '''
        n_position = x.shape[1]
        emb_dim = x.shape[2]
        position_enc = np.array([
                        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
                         for pos in range(n_position)])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
        return x + torch.from_numpy(position_enc).type(torch.FloatTensor).to(self.device)

    def decoder_step(self, input_token, hp, cp, op, encoder_outputs):
        '''
            input:
                > input_token: one hot input token (batch)
                > hp, cp: previous states (#layers, batch, decoder_hidden_size)
                > op: previous o (batch, decoder_hidden_size)
                > encoder_outputs: top layer encoder states (batch, H'*W', directions * encoder_hidden_size)
            return:
                > output: log probability of next token (batch, vocab_size)
                > hn, cn: next hidden states (#layers, batch, decoder_hidden_size)
                > on: next (batch, decoder_hidden_size)
                > attn_weight: attention alignments (batch, 1, H'*W')
        '''
        token_embedding = self.embedding(input_token).unsqueeze(1) # (batch, 1, embedding_size)
        output, (hn, cn), on, attn_weight = self.decoder(token_embedding, hp, cp, op, encoder_outputs)
        return output, (hn, cn), on, attn_weight

    def forward(self, x, y, teacher_forcing_ratio):
        '''
            input:
                > x: image (batch, 1, H, W)
                > y: one hot formulas (batch, len)
                > teacher_forcing_ratio
            return:
                > logits: log probability of tokens (batch, len-1, vocab_size) [excluding start token]
        '''
        x = self.get_cnn_feature(x)

        encoder_outputs, (hn, cn) = self.get_encoder_anotation_grid(x)

        b, h, w, _ = encoder_outputs.shape
        encoder_outputs = encoder_outputs.view(b, h*w, -1).contiguous() # (batch, H'*W', directions * encoder_hidden_size)
        encoder_outputs = self.add_positional_embedding(encoder_outputs) # (batch, H'*W', directions * encoder_hidden_size)

        hn = F.tanh(self.init_h(torch.mean(encoder_outputs, dim=1))) # (batch, decoder_hidden_size)
        hn = hn.view(1, hn.shape[0], hn.shape[1]).contiguous() # (1, batch, decoder_hidden_size)
        cn = torch.zeros(1, b, self.decoder.hidden_size, device=self.device) # (1, batch, decoder_hidden_size)
        on = F.tanh(self.init_o(torch.mean(encoder_outputs, dim=1))) # (batch, decoder_hidden_size)

        next_token = y[:, 0] # (batch)
        logits = torch.zeros(y.shape[0], y.shape[1], self.decoder.output_size, device=self.device) # (batch, len, vocab_size)
        logits[:, 0, y[0, 0]] = 1
        for t in range(1, y.shape[1]):
            output, (hn, cn), on, attn_weight = self.decoder_step(next_token, hn, cn, on, encoder_outputs)
            logits[:, t, :] = output
            if teacher_forcing_ratio < torch.rand(1).item():
                next_token = output.exp().multinomial(1).squeeze(1) # (batch)
            else:
                next_token = y[:, t] # (batch)
        return logits

    def generate(self, x, start_token, max_len, method):
        '''
            input:
                > x: image (batch, 1, H, W)
                > start_token: start token index
                > max_len: length of all generated formulas would be eless than max_len
                > method: 'greedy' or 'beam-seach'
            return:
                > preds: generated sequences (batch, max_len)
                > logits: log probability of tokens (batch, max_len-1, vocab_size)
                > attn_weights: attention alignments (batch, max_len-1, H'*W')
        '''
        x = self.get_cnn_feature(x)
        encoder_outputs, (hn, cn) = self.get_encoder_anotation_grid(x)
        b, h, w, _ = encoder_outputs.shape
        encoder_outputs = encoder_outputs.view(b, h*w, -1)
        encoder_outputs = self.add_positional_embedding(encoder_outputs) # (batch, H'*W', directions * encoder_hidden_size)

        hn = F.tanh(self.init_h(torch.mean(encoder_outputs, dim=1))) # (batch, decoder_hidden_size)
        hn = hn.view(1, hn.shape[0], hn.shape[1]).contiguous() # (1, batch, decoder_hidden_size)
        cn = torch.zeros(1, b, self.decoder.hidden_size, device=self.device) # (1, batch, decoder_hidden_size)
        on = F.tanh(self.init_o(torch.mean(encoder_outputs, dim=1))) # (batch, decoder_hidden_size)

        if method == 'greedy':
            generator = GreedyDecoder(self, max_len, hn, cn, on, encoder_outputs, self.device)

        elif method == 'beam-search':
            raise NotImplementedError('beam-search method is not implemented yet.')
            generator = BeamSearchDecoder()

        preds, logits, attn_weights = generator.generate(start_token)
        return preds, logits, attn_weights
