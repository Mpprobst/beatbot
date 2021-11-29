"""
lstm_agent.py
Author: Michael Probst
Purpose: Implements an agent which trains using an lstm
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from agents.models import lstm

class LSTMAgent():
    def __init__(self, input_size, output_size, hidden_size):
        # create the lstm
        self.hidden_size = hidden_size
        self.model = lstm.LSTM(input_size, output_size, hidden_size, 2)
        self.loss_function = nn.BCELoss()
        self.optimizer = T.optim.Adam(self.model.parameters(), lr=0.15)
        self.running_loss = 0

    def train(self, data, target, h):
        data = T.tensor(data).float().to(self.model.device)
        target = T.tensor(target).float().to(self.model.device)
        h = tuple([e.data for e in h])

        self.model.zero_grad()
        out, h = self.model(data, h)
        loss = self.loss_function(out, target)
        loss.backward()
        self.optimizer.step()

        self.running_loss += loss

    def test(self, data, h):
        h = tuple([e.data for e in h])
        data = T.tensor(data).float().to(self.model.device)
        #out, h = self.model(data, h)
        return self.model(data, h)

    def save(self, path):
        T.save(self.model.state_dict(), f'{path}/trained_lstm.pth')

    def load(self, path):
        print("TODO: Implement load")
