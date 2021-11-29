"""
lstm.py
Author: Michael Probst
Purpose: Implements a lstm rnn
refernece: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, n_layers, drop_prob=0.5):
        super(LSTM, self).__init__()
        self.dim_input = dim_input # num features
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.n_layers = n_layers

        # note: skipping the embedding layer because notes are already embedded
        self.embedding = nn.Embedding(dim_input, dim_output)
        self.lstm = nn.LSTM(dim_output, dim_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(dim_hidden, dim_output)
        self.sigmoid = nn.Sigmoid()

        if T.cuda.is_available():
            self.device = T.device("cuda")
        else:
            self.device = T.device("cpu")

    def forward(self, data, hidden):
        batch_size = data.size(0)
        data = T.tensor(data).to(self.device).long()

        embeds = self.embedding(data)
        lstm_out, hidden = self.lstm(embeds, hidden)
        #print(f'lstm out {lstm_out.size()}')
        lstm_out = lstm_out.contiguous().view(-1, self.dim_hidden)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        print(f'before view: {out.size()}')
        out = out.view(batch_size, self.dim_output, -1)
        print(f'after view: {out.size()}')
        out = out[0][:,-1]
        print(f'after trunc: {out.size()}')
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.dim_hidden).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.dim_hidden).zero_().to(self.device))
        return hidden
