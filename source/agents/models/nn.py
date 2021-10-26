"""
nn.py
Author: Michael Probst
Purpose: Implements a neural network for agents to use
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(NN, self).__init__()
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.fc1 = nn.Linear(self.dim_input, 64)    #first layer
        self.fc2 = nn.Linear(64, 32)                #second layer
        self.fc3 = nn.Linear(32, self.dim_output)   #output layer
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, data):
        data = T.tensor(data).float()
        x = F.relu(self.fc1(data))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
