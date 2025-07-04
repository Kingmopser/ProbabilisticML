import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#REGRESSION

# Partial stochastic NN: MAP estimated deterministic L-1 Layers and Bayesian Last Layer
class BaseNetwork(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(n_in,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
    def forward(self,x):
        return self.head(x)


#Bayesian Last Layer definition using V.I "bayesian by backprop"
class BayesianLastLayer(nn.Module):
    def __init__(self,in_features,out_features,logvals,prior_sigma=1.0):
        super().__init__()
        self.wMu = nn.Parameter(torch.zeros(in_features, out_features))
        self.wLogVar = nn.Parameter(torch.full((in_features, out_features), logvals))
        self.bMu = nn.Parameter(torch.zeros(out_features))
        self.bLogVar = nn.Parameter(torch.full((out_features,), logvals))
        self.priorSigma = prior_sigma

    def forward(self,x):
        #computing standard deviation
        wStd = torch.exp(0.5*self.wLogVar)
        bStd = torch.exp(0.5*self.bLogVar)
        #sampling weights and bias: reparametrization trick
        w = self.wMu + wStd * torch.randn_like(self.wMu)
        b = self.bMu + bStd * torch.randn_like(self.bMu)
        # linear layer (bayesian GLM with gaussian prior)
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
    optimizer= optim.Adam(lastLayer.parameters(),lr=1e-3)
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
            kl = (lastLayer.kl_div() / N)
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

# Partial stochastic NN: MAP estimated deterministic L-1 Layers and Bayesian Last Layer
class BaseNetworkCL(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(n_in,8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
    def forward(self,x):
        return self.head(x)


def TrainLastLayerCL(base,lastLayer,loader,beta=1.0,class_weights = None,learningrate= 1e-3,epochs=1000):
    optimizer= optim.Adam(lastLayer.parameters(),lr=learningrate)
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
            #neg log
            ce = F.cross_entropy(pred,y)
            kl = beta*( lastLayer.kl_div() /N)
            loss = ce + kl
            loss.backward()
            optimizer.step()
    return lastLayer


def PredLastLayerCl(base,lastLayer, x, nSamples=100, t=1 ):
    base.eval()
    preds = []
    with torch.no_grad():
        feats = base(x)
        # predicitive sample-mean (Monte Carlo) and predicted sample-var
        # Temperature Scaling because Logits saturate very quickly
        for _ in range(nSamples):
            logits = lastLayer(feats)
            preds.append(F.softmax(logits/t,dim=1).cpu().numpy())
    preds = np.stack(preds)
    mean_prob = preds.mean(axis=0)
    #Bayesian Agents: Attempt to UQ, Lisa Wimmers Slides
    #predictive entropy(Total Uncertainty)
    entropy = (-1)*np.sum(mean_prob*np.log(mean_prob + 1e-9),axis=1)
    #Expected_entropy(aleatoric)
    exp_entropy= -np.mean(np.sum(preds * np.log(preds + 1e-9), axis=2), axis=0)
    # mutual info(epistemic uncertainty)
    mutual_info = entropy - exp_entropy
    energy = -torch.logsumexp(logits, dim=1).cpu().numpy()
    return mean_prob, entropy, exp_entropy, mutual_info,energy









