import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets

from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
#REGRESSION
#Standard NN, no need for extra Class since its a standard NN
def GetNeuralNetwork():
    return nn.Sequential(
        nn.Linear(1,32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    ).to(device)

#training MAP NN:
def TrainNN(model,loader,epochs=1000):
    optimizer = optim.Adam(model.parameters(),lr=5e-3)
    for epoch in range(epochs):
        for x,y in loader:
            model.train()
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred,y)
            loss.backward()
            optimizer.step()

    return model

#predicting NN:
def PredNormal(model,x):
    model.eval()
    #forward pass
    with torch.no_grad():
        pred = model(x).cpu().numpy()
        # std = 0 because no epistemic uncertainty, model is confident over all, since determinstic MAP
    return pred.squeeze(),np.zeros_like(pred.squeeze())

#CLASSIFICATION



