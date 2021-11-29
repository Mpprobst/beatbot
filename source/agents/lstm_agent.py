"""
lstm_agent.py
Author: Michael Probst
Purpose: Implements an agent which trains using an lstm
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from agents.models import lstm
import numpy as np

class LSTMAgent():
    def __init__(self, input_size, output_size, hidden_size):
        # create the lstm
        self.hidden_size = hidden_size
        self.chunk_size = 4     # number of notes to feed at once
        self.output_size = output_size
        self.model = lstm.LSTM(input_size, self.chunk_size, hidden_size, 2)
        self.loss_function = nn.BCELoss()
        self.optimizer = T.optim.Adam(self.model.parameters(), lr=0.15)
        self.running_loss = 0

    def train(self, data, target, h):
        # feed a few notes at a time
        T.autograd.set_detect_anomaly(True)
        h = tuple([e.data for e in h])
        #print(h)
        #print(f'size of data, tar: {data[0]}, {target.size()}')
        for i in range(0, len(data[0]), self.chunk_size):
            #print(f'[{data[0][i]}, {data[0][i+1]}, {data[0][i+2]}, {data[0][i+3]}], {data[0][i:i+4]}')
            d_chunk = np.zeros((1, self.chunk_size))
            d_chunk[0] = data[0][i:i+self.chunk_size]
            d_chunk = T.tensor(d_chunk).float().to(self.model.device)
            t_chunk = np.zeros((1, self.chunk_size))
            t_chunk = target[0][i:i+self.chunk_size]
            t_chunk = T.tensor(t_chunk).float().to(self.model.device)

            self.optimizer.zero_grad()
            #print(f'd_chunk = {d_chunk.size()} h = {h[0].size()}, {h[1].size()}')
            out, h = self.model(d_chunk, h)
            #print(f't_chunk = {t_chunk.size()} out size = {out.size()}')
            loss = self.loss_function(out, t_chunk)
            loss.backward()
            self.optimizer.step()

            self.running_loss += loss

    def test(self, data, hidden):
        hidden = tuple([e.data for e in hidden])
        data = T.tensor(data).float().to(self.model.device)
        o = []
        h = []
        for i in range(0, len(data), self.chunk_size):
            out, hid = self.model(chunk, hidden)
            o.append(out)
            h.append(hid)
        #out, h = self.model(data, h)
        return o, h

    def save(self, path):
        T.save(self.model.state_dict(), f'{path}/trained_lstm.pth')

    def load(self, path):
        print("TODO: Implement load")
