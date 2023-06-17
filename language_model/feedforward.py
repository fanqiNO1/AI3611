import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, num_tokens, num_embeddings, num_hidden, num_layers, dropout, is_tie_weights):
        super(FeedForward, self).__init__()
        self.num_tokens = num_tokens
        self.num_embeddings = num_embeddings
        self.num_hidden = num_hidden

        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_tokens, num_embeddings)
        self.feedforward = [nn.Linear(num_embeddings, num_hidden)]
        for i in range(num_layers - 1):
            self.feedforward.append(nn.Linear(num_hidden, num_hidden))
        self.feedforward = nn.Sequential(*self.feedforward)
        self.decoder = nn.Linear(num_hidden, num_tokens)

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

    def forward(self, x):
        embedding = self.dropout(self.encoder(x))
        output = self.feedforward(embedding)
        output = self.dropout(output)
        decoded = self.decoder(output)
        return F.log_softmax(decoded, dim=-1).reshape(-1, self.num_tokens)


if __name__ == "__main__":
    from torchinfo import summary
    model = FeedForward(num_tokens=100, num_embeddings=10, num_hidden=10, num_layers=2, dropout=0.1)
    x = torch.randint(0, 10, (35, 32)).long()
    summary(model, input_data=[x])
