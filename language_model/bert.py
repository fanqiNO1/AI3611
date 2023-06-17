import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, max_len=512):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, num_embeddings).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_embeddings, 2).float() * (-math.log(10000.0) / num_embeddings))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BertEmbedding(nn.Module):
    def __init__(self, num_tokens, num_embeddings, dropout):
        super(BertEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(num_tokens, num_embeddings, padding_idx=0)
        self.position_embedding = PositionalEmbedding(num_embeddings)
        self.dropout = nn.Dropout(dropout)
        self.num_embeddings = num_embeddings

    def forward(self, x):
        x = self.token_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class Attention(nn.Module):
    def forward(self, q, k, v, dropout, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_embeddings, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.head_dim = num_embeddings // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(num_embeddings, num_embeddings)
        self.linear_k = nn.Linear(num_embeddings, num_embeddings)
        self.linear_v = nn.Linear(num_embeddings, num_embeddings)
        self.linear_o = nn.Linear(num_embeddings, num_embeddings)

        self.attention = Attention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        b = q.size(0)
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        q, k, v = [x.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2) for x in [q, k, v]]
        x, attn = self.attention(q, k, v, self.dropout, mask=mask)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.num_heads * self.head_dim)
        return self.linear_o(x)


class NormResConnect(nn.Module):
    def __init__(self, size, dropout):
        super(NormResConnect, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Lambda(nn.Module):
    def __init__(self, attention):
        super(Lambda, self).__init__()
        self.attention = attention
        self.mask = torch.zeros((4))

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        return self.attention(x, x, x, self.mask)


class FeedForward(nn.Module):
    def __init__(self, num_embeddings, num_hidden, dropout):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(num_embeddings, num_hidden)
        self.linear_2 = nn.Linear(num_hidden, num_embeddings)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.linear_2(self.dropout(self.act(self.linear_1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, num_embeddings, num_heads, num_hidden, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(num_embeddings, num_heads, dropout)
        self.lambda_attention = Lambda(self.attention)
        self.feed_forward = FeedForward(num_embeddings, num_hidden, dropout)
        self.input_res = NormResConnect(num_embeddings, dropout)
        self.output_res = NormResConnect(num_embeddings, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        self.lambda_attention.set_mask(mask)
        x = self.input_res(x, self.lambda_attention)
        x = self.output_res(x, self.feed_forward)
        return self.dropout(x)

    
class BertModel(nn.Module):
    def __init__(self, num_tokens, num_embeddings, num_heads, num_hidden, num_layers, dropout, is_tie_weights):
        super(BertModel, self).__init__()
        self.embedding = BertEmbedding(num_tokens, num_embeddings, dropout)
        self.layers = nn.ModuleList([EncoderBlock(num_embeddings, num_heads, num_hidden, dropout) for _ in range(num_layers)])
        self.decoder = nn.Linear(num_embeddings, num_tokens)
        self.num_tokens = num_tokens

        self._init_weights()

        if is_tie_weights:
            if num_hidden != num_embeddings:
                raise ValueError("When using the tied flag, num_hidden must be equal to num_embeddings")
            self.decoder.weight = self.embedding.token_embedding.weight

    def _init_weights(self):
        init_range = 0.1
        self.embedding.token_embedding.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
        
    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)

        x = self.decoder(x)

        return F.log_softmax(x, dim=-1).reshape(-1, self.num_tokens)


if __name__ == "__main__":
    from torchinfo import summary
    model = BertModel(num_tokens=100, num_embeddings=10, num_heads=2, num_hidden=10, num_layers=2, dropout=0.1, is_tie_weights=False)
    x = torch.randint(0, 10, (35, 32)).long()
    summary(model, input_data=[x])