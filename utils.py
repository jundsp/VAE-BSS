import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def mix_data(data):
    n = data.size(0)//2
    sources = torch.cat([data[:n],data[n:2*n]],1) / 2.0
    data = sources.sum(1).unsqueeze(1)
    return data, sources
    
# KL(q(x)||p(x)) where p(x) is Gaussian, q(x) is Gaussian
def KLD_gauss(mu,logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

# KL(q(x)||p(x)) where p(x) is Laplace, q(x) is Gaussian
def KLD_laplace(mu,logvar,scale=1.0):
    v = logvar.exp()
    y = mu/torch.sqrt(2*v)
    y2 = y.pow(2)
    t1 = -2*torch.exp(-y2)*torch.sqrt(2*v/np.pi)
    t2 = -2*mu*torch.erf(y)
    t3 = scale*torch.log(np.pi*v/(2.0*scale*scale))
    temp = scale+t1+t2+t3
    KLD = -1.0/(2*scale)*torch.sum(1+t1+t2+t3)
    return KLD

def vae_masks(mu_s, x):
    mu_sm = mu_s * (x / mu_s.sum(1).unsqueeze(1))
    mu_sm[torch.isnan(mu_sm)] = 0
    return mu_sm

# Optimal permutation based on MSE, with the Hungarian Algorithm (Assignment Problem)
def optimal_permute(y,x):
    n = x.size(0)
    nx = x.size(1)
    ny = y.size(1)
    z = torch.zeros_like(x)
    for i in range(n):
        cost = torch.zeros(ny,nx)
        for j in range(ny):
            cost[j] = (y[i,j].unsqueeze(0) - x[i,:]).pow(2).sum(-1).sum(-1)

        row_ind, col_ind = linear_sum_assignment(cost.detach().numpy().T)
        z[i] = y[i,col_ind]
    return z
