import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, rnn_type, num_tokens, num_embeddings, num_hidden, num_layers, dropout, is_tie_weights):
        super(RNN, self).__init__()
        self.rnn_type = rnn_type
        self.num_tokens = num_tokens
        self.num_embeddings = num_embeddings
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.is_tie_weights = is_tie_weights

        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_tokens, num_embeddings)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(num_embeddings, num_hidden, num_layers, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(num_embeddings, num_hidden, num_layers, dropout=dropout)
        elif rnn_type == "RNN_TANH":
            self.rnn = nn.RNN(num_embeddings, num_hidden, num_layers, nonlinearity="tanh", dropout=dropout)
        elif rnn_type == "RNN_RELU":
            self.rnn = nn.RNN(num_embeddings, num_hidden, num_layers, nonlinearity="relu", dropout=dropout)
        else:
            raise ValueError("Invalid RNN type: {}".format(rnn_type))
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

    def forward(self, x, hidden):
        embedding = self.dropout(self.encoder(x))
        output, hidden = self.rnn(embedding, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.num_tokens)
        return F.log_softmax(decoded, dim=1), self.repackage_hidden(hidden)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (weight.new_zeros(self.num_layers, batch_size, self.num_hidden).detach(),
                    weight.new_zeros(self.num_layers, batch_size, self.num_hidden).detach())
        else:
            return weight.new_zeros(self.num_layers, batch_size, self.num_hidden).detach()

    def repackage_hidden(self, hidden):
        if self.rnn_type == "LSTM":
            return (hidden[0].detach(), hidden[1].detach())
        else:
            return hidden.detach()


if __name__ == "__main__":
    from torchinfo import summary
    model = RNN("LSTM", num_tokens=100, num_embeddings=10, num_hidden=10, num_layers=2, dropout=0.1, is_tie_weights=False)
    hidden = model.init_hidden(32)
    x = torch.randint(0, 10, (35, 32)).long()
    summary(model, input_data=[x, hidden])
