import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets

from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#REGRESSION

# Partial stochastic NN: MAP estimated deterministic L-1 Layers and Bayesian Last Layer
class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
    def forward(self,x):
        return self.head(x)


#Bayesian Last Layer definition using V.I "bayesian by backprop"
class BayesianLastLayer(nn.Module):
    def __init__(self,in_features,prior_sigma=1.0):
        super().__init__()
        self.wMu = nn.Parameter(torch.zeros(in_features, 1))
        self.wLogVar = nn.Parameter(torch.full((in_features, 1), -5.0))
        self.bMu = nn.Parameter(torch.zeros(1))
        self.bLogVar = nn.Parameter(torch.full((1,), -5.0))
        self.priorSigma = prior_sigma

    def forward(self,x):
        #computing standard deviation
        wStd = torch.exp(0.5*self.wLogVar)
        bStd = torch.exp(0.5*self.bLogVar)
        #sampling weights and bias: reparametrization trick
        w = self.wMu + wStd * torch.randn_like(self.wMu)
        b = self.bMu + bStd * torch.randn_like(self.bMu)
        # linear layer (bayesian GLM with gaussian prior, L2 reg)
        return  x @ w + b

    def kl_term(self,mu,logvar):
        var = torch.exp(logvar)
        priorVar = self.priorSigma**2
        #formual for KL divergence of 2 gaussians
        return 0.5*((var+mu.pow(2)) / priorVar -1 +torch.log(priorVar)-logvar).sum()

    def kl_div(self):
        def kl_term(mu, logvar):
            var = torch.exp(logvar)
            priorVar = torch.tensor(self.priorSigma**2, device=mu.device, dtype=mu.dtype)
            # formual for KL divergence of 2 gaussians
            return 0.5 * ((var + mu.pow(2)) / priorVar - 1 + torch.log(priorVar) - logvar).sum()
        # KL weights + KL biases, since they're independent
        return  kl_term(self.wMu,self.wLogVar) + kl_term(self.bMu,self.bLogVar)

#JOINT TRAINING (Do bayesian Networks need to be fully stochastic)
#training last layer
def TrainLastLayer(base,lastLayer,loader,epochs=1000):
    optimizer= optim.Adam(lastLayer.parameters(),lr=5e-3)
    # autograd shouldn't touch the MAP trained weights, so set False to avoid optimizing
    for p in base.parameters():
        p.requires_grad = False
    N = len(loader.dataset)
    for epoch in range(epochs):
        for x,y in loader:
            optimizer.zero_grad()
            features = base(x)
            pred = lastLayer(features)
            #we are maximizing ELBO
            #neg log lik
            mse = F.mse_loss(pred,y)
            kl = lastLayer.kl_div() / N
            loss = mse + kl
            loss.backward()
            optimizer.step()
    return lastLayer

 # predicting Last Layer


def PredLastLayer(base, LastLayer, x, nSamples=100):
    base.eval()
    preds = []
    with torch.no_grad():
        feats = base(x)
        # predicitive sample-mean (Monte Carlo) and predicted sample-var
        for _ in range(nSamples):
            preds.append(LastLayer(feats).cpu().numpy())
    preds = np.stack(preds)
    mean = preds.mean(axis=0).squeeze()
    std = preds.std(axis=0).squeeze()
    return mean, std

#CLASSIFICATION










