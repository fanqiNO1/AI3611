import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, dropout, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, num_embeddings)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_embeddings, 2).float() * (-math.log(10000.0) / num_embeddings))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)


    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, num_tokens, num_embeddings, num_heads, num_hidden, num_layers, dropout, is_tie_weights):
        super(Transformer, self).__init__()
        self.source_mask = None
        self.position_embedding = PositionalEmbedding(num_embeddings, dropout)
        encoder_layer = nn.TransformerEncoderLayer(num_embeddings, num_heads, num_hidden, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.encoder = nn.Embedding(num_tokens, num_embeddings)
        self.decoder = nn.Linear(num_embeddings, num_tokens)
        self.num_tokens = num_tokens
        self.num_embeddings = num_embeddings

        self._init_weights()
        
        if is_tie_weights:
            if num_hidden != num_embeddings:
                raise ValueError("When using the tied flag, num_hidden must be equal to num_embeddings")
            self.decoder.weight = self.encoder.weight

    def _init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def _generate_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, has_mask=True):
        if has_mask:
            if self.source_mask is None or self.source_mask.size(0) != len(x):
                self.source_mask = self._generate_mask(len(x)).to(x.device)
        else:
            self.source_mask = None

        x = self.encoder(x) * math.sqrt(self.num_embeddings)
        x = self.position_embedding(x)
        x = self.transformer(x, self.source_mask)
        x = self.decoder(x)
        return F.log_softmax(x, dim=-1).reshape(-1, self.num_tokens)


if __name__ == "__main__":
    from torchinfo import summary
    model = Transformer(num_tokens=100, num_embeddings=10, num_heads=2, num_hidden=10, num_layers=2, dropout=0.1)
    x = torch.randint(0, 10, (35, 32)).long()
    summary(model, input_data=[x])
