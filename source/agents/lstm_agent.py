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
        self.chunk_size = 8     # number of notes to feed at once
        self.output_size = output_size
        self.model = lstm.LSTM(input_size, self.chunk_size, hidden_size, 3)
        self.loss_function = nn.BCELoss()
        self.optimizer = T.optim.Adam(self.model.parameters(), lr=0.05)
        self.running_loss = 0

    def train(self, data, target, h):
        # feed a few notes at a time
        h = tuple([e.data for e in h])
        losses = []
        batch_size = data.size(0)
        # feed in only a few notes at a time
        for i in range(0, len(data[0]), self.chunk_size):
            d_chunk = np.zeros((batch_size, self.chunk_size))
            t_chunk = np.zeros((batch_size, self.chunk_size))

            for j in range(batch_size):
                d_chunk[j] = data[j][i:i+self.chunk_size]
                t_chunk[j] = target[j][i:i+self.chunk_size]

            d_chunk = T.tensor(d_chunk).float().to(self.model.device)
            t_chunk = T.tensor(t_chunk).float().to(self.model.device)

            #print(f'd_chunk = {d_chunk.size()} h = {h[0].size()}, {h[1].size()}')
            out, h = self.model(d_chunk, h)
            #print(f't_chunk = {t_chunk.size()} out size = {out.size()}')
            loss = self.loss_function(out, t_chunk)
            losses.append(loss)

        self.optimizer.zero_grad()
        loss = T.mean(T.stack(losses))
        loss.backward()
        self.optimizer.step()
        self.running_loss += loss

    def test(self, data, hidden):
        hidden = tuple([e.data for e in hidden])
        data = T.tensor(data).float().to(self.model.device)
        batch_size = len(data)
        o = np.zeros((batch_size, len(data[0])))
        h = None
        #print("\nDEBUG-test")
        #print(f'data size: {data.size()}, batch={batch_size}, len={len(data[0])}')
        for i in range(0, len(data[0]), self.chunk_size):
            d_chunk = np.zeros((batch_size, self.chunk_size))
            for j in range(batch_size):
                d_chunk[j] = data[j][i:i+self.chunk_size]
            d_chunk = T.tensor(d_chunk).float().to(self.model.device)

            out, hid = self.model(d_chunk, hidden)
            #print(f'chunk out: {out.size()}')
            out = out.detach().numpy()
            #print(f'out: {out}')
            for j in range(batch_size):
                for k in range(self.chunk_size):
                    o[j][i+k] = out[j][k]
            h = hid
        #out, h = self.model(data, h)
        #print(f'test out:{o}')
        # o should be [batchsize, notes]
        return o, h

    def save(self, path):
        T.save(self.model.state_dict(), f'{path}/trained_lstm.pth')

    def load(self, path):
        print("TODO: Implement load")
