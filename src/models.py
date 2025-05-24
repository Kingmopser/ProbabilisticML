import numpy as np
import matplotlib as plt
import pandas as pd

from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim

# determining possiblities for acccelaration
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#Models applied on Simulation data will be Regressional task
#Models applied on real data will be classifications

#Standard NN, no need for extra Class since its a standard NN
def getNeuralNetwork():
    return nn.Sequential(
        nn.Linear(1,32),
        nn.ReLU(),
        nn.Linear(32,16),
        nn.ReLU(),
        nn.Linear(16,1)
    ).to(device)


# Partial stochastic NN: MAP estimated deterministic L-1 Layers and Bayesian Last Layer
class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(),
            nn.Linear(32, 16), # missing last layer
        )
    def forward(self,x):
        return self.head(x)

#Bayesian Last Layer definition using V.I "bayesian by backprop"
class BayesianLastLayer(nn.Module):
    def __init__(self,in_features,prior_sigma=1.0):
        super().__init__()
        self.w_mu = nn.Parameter(torch.zeros(in_features, 1))
        self.w_logvar = nn.Parameter(torch.full((in_features, 1), -5.0))
        self.b_mu = nn.Parameter(torch.zeros(1))
        self.b_logvar = nn.Parameter(torch.full((1,), -5.0))
        self.prior_sigma = prior_sigma

    def forward(self,x):




