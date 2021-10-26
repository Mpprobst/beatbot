"""
lstm.py
Author: Michael Probst
Purpose: Implements a lstm rnn
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(dim_input, dim_hidden)
        self.linear = nn.Linear(dim_hidden, dim_input)  # input size = output size

    def forward(self, data):
        data = T.tensor(data)
        out, _ = self.lstm(data)
        lin_out = self.linear(out)
        scores = F.log_softmax(lin_out, dim=1)
        return scores
