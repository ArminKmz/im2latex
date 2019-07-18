import torch

class GreedyDecoder:
    def __init__(self, model, max_len, hn, cn, on, encoder_outputs, device):
        self.model = model
        self.max_len = max_len
        self.hn = hn
        self.cn = cn
        self.on = on
        self.encoder_outputs = encoder_outputs
        self.device = device

    def generate(self, start_token):
        next_token = torch.Tensor([start_token]).long().to(self.device) # (1)
        b = self.encoder_outputs.shape[0]
        next_token = next_token.repeat(b) # (batch)
        logits = []
        preds = [next_token]
        attn_weights = []
        hn = self.hn
        cn = self.cn
        on = self.on
        for t in range(1, self.max_len):
            output, (hn, cn), on, attn_weight = self.model.decoder_step(next_token, hn, cn, on, self.encoder_outputs)
            attn_weights.append(attn_weight.squeeze(1))
            logits.append(output)
            next_token = output.argmax(dim=1) # (batch)
            preds.append(next_token)
        logits = [torch.zeros(logits[1].shape[0], logits[1].shape[1], device=self.device)] + logits
        logits[0][:, start_token] = 1
        logits = torch.stack(logits, dim=1)
        preds = torch.stack(preds, dim=1)
        attn_weights = torch.stack(attn_weights, dim=1)
        return preds, logits, attn_weights

class BeamSearchDecoder:
    def __init__(self):
        pass
    def generate(self):
        pass
