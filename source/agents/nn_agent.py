"""
nn_agent.py
Author: Michael Probst
Purpose: implements an agent which trains using a nn
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from agents.models import nn as NN

class NNAgent():
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.model = NN.NN(input_size, input_size)
        self.loss_function = nn.MSELoss()
        self.optimizer = T.optim.SGD(self.model.parameters(), lr=0.1)
        #print(self.model)
        self.running_loss = 0

    def train(self, data, target):
        data = T.tensor(data).float()
        target = T.tensor(target).float()
        #data.flatten()
        #target.flatten()
        #print(data)

        self.model.zero_grad()
        out = self.model(data)

        loss = self.loss_function(out, target)
        loss.backward()
        self.running_loss += loss
        #print(f'running loss = {self.running_loss}')
        self.optimizer.step()

    def test(self, data):
        data = T.tensor(data).float()
        return self.model(data)

    def save(self, path):
        T.save(self.model.state_dict(), f'{path}/trained_lstm.pth')

    def load(self, path):
        print("TODO: Implement load")
