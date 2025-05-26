import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets

from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# determining possiblities for acccelaration
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#Models applied on Simulation data will be Regressional task
#Models applied on real data will be classifications
np.random.seed(123)
#simulate data
X_train = np.random.uniform(-4,4,200).reshape(-1,1).astype(np.float32)
y_train = (X_train**3 + 0.5* np.random.randn(*X_train.shape)).astype(np.float32)

X_test = np.linspace(-8, 8, 500).reshape(-1, 1).astype(np.float32)
y_test = np.sin(X_test).astype(np.float32)

tenX = torch.from_numpy(X_train).to(device)
tenY = torch.from_numpy(y_train).to(device)
tenXTest = torch.from_numpy(X_test).to(device)
tenYTest = torch.from_numpy(y_test).to(device)
tenDF = TensorDataset(tenX,tenY)
trainLoader = DataLoader(tenDF,batch_size=32,shuffle=True)
InDist = (X_test >=-4) & (X_test <=4)

#Standard NN, no need for extra Class since its a standard NN
def GetNeuralNetwork():
    return nn.Sequential(
        nn.Linear(1,32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    ).to(device)


# Partial stochastic NN: MAP estimated deterministic L-1 Layers and Bayesian Last Layer
class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(),
            nn.Linear(32, 16), # missing last layer
            nn.ReLU(),
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

#predicting NN:
def PredNormal(model,x):
    model.eval()
    #forward pass
    with torch.no_grad():
        pred = model(x).cpu().numpy()
        # std = 0 because no epistemic uncertainty, model is confident over all, since determinstic MAP
    return pred.squeeze(),np.zeros_like(pred.squeeze())
#predicting Last Layer
def PredLastLayer(base,LastLayer,x, nSamples= 100):
    base.eval()
    preds = []
    with torch.no_grad():
        feats = base(x)
        #predicitive sample-mean (Monte Carlo) and predicted sample-var
        for _ in range(nSamples):
            preds.append(LastLayer(feats).cpu().numpy())
    preds = np.stack(preds)
    mean = preds.mean(axis=0).squeeze()
    std = preds.std(axis=0).squeeze()
    return mean,std

#NO TUNING NOR ANY OPTIMIZATION DONE

#  test run
mapNN = GetNeuralNetwork()
TrainNN(mapNN, trainLoader, epochs=600)
mean_map, std_map = PredNormal(mapNN, tenXTest)
plt.figure(figsize=(10,5))
plt.plot(X_test, X_test**3, 'g-', label='True')
plt.show()
plt.plot(X_test, mean_map, 'b--', label='MAP NN Mean')
plt.show()

# Base + bayesian Last Layer model
base = BaseNetwork().to(device)
head = nn.Linear(16,1).to(device)
baseMod = nn.Sequential(base,head)
TrainNN(baseMod,trainLoader,100)
# freezing base weights
for p in baseMod.parameters():
    p.requires_grad = False
lastLayer = BayesianLastLayer(16).to(device)
TrainLastLayer(base,lastLayer,trainLoader,epochs=20)
meanB, stdB = PredLastLayer(base,lastLayer,tenXTest)

plt.figure(figsize=(10,5))
plt.plot(X_test, X_test**3, 'g-', label='True')
plt.show()
plt.plot(X_test, meanB, 'b--', label='MAP NN Mean')
plt.show()


