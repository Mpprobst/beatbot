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
        self.fc1 = nn.Linear(self.dim_input, 512)       #first layer
        self.fc2 = nn.Linear(512, 1024)                  #second layer
        self.fc3 = nn.Linear(1024, 256)                  #third layer
        self.fc4 = nn.Linear(256, 128)                  #fourth layer

        self.fco = nn.Linear(128, self.dim_output)      #output layer
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, data):
        data = T.tensor(data).float()
        x = F.relu(self.fc1(data))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        out = self.fco(x)
        return out
