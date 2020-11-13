from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import os
import shutil
from model import *
import argparse
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser(description='VAE-BSS')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data-directory', type=str, default='data', metavar='fname',
                    help='Folder containing the data')
parser.add_argument('--sources', type=int, default=2,
                    help='Number of sources to infer')
parser.add_argument('--dimz', type=int, default=20,
                    help='Dimension of latent space.')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of data loading workers (parallel).')
parser.add_argument('--prior', type=str, default='laplace',
                    help='Prior over latent variables')
parser.add_argument('--samples', type=int, default=1,
                    help='Latent samples.')

def mix_data(data):
    n = data.size(0)//2
    sources = torch.cat([data[:n],data[n:2*n]],1) / 2.0
    data = sources.sum(1).unsqueeze(1)
    return data, sources

def colorize(input):
    temp = torch.from_numpy(cm_jet(input.detach())).permute(-1,0,1)[:3]
    return temp

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

args = parser.parse_args()
torch.manual_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_directory, train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_directory, train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)

dimx = int(28*28)

print('Loading MNIST Test Data Batch')
# Load MNIST Test Data
data, numbers = next(iter(test_loader))
# Randomly mix
x, s_tru = mix_data(data.to(device))


print('Evaluating mixture with K = 2, 3, and 4 assumed sources.')
# Evaluate for K = 2, 3, 4 model sources
for num_sources in range(2,5):
    print('K = ' + str(num_sources))

    # Load the Trained VAE Model
    model_vae = VAE(dimx=dimx,dimz=args.dimz,n_sources=num_sources,device=device).to(device)
    model_vae.load_state_dict(torch.load('saves/model_vae_K' + str(num_sources) + '.pt',map_location=torch.device('cpu')))
    model_vae.eval()

    # Load the Trained AE Model (latent sampling is deterministic, trained without KLD)
    model_ae = VAE(dimx=dimx,dimz=args.dimz,n_sources=num_sources,device=device,samples=0).to(device)
    model_ae.load_state_dict(torch.load('saves/model_ae_K' + str(num_sources) + '.pt',map_location=torch.device('cpu')))
    model_ae.eval()

    # VAE Evaluation
    x_vae, mu_z, logvar_z, s_vae = model_vae(x)
    x_vae = x_vae.view(-1,1,28,28)
    s_vae = s_vae.view(-1,num_sources,28,28)
    s_vae = optimal_permute(s_vae,s_tru)

    # Create masks
    s_vaem = vae_masks(s_vae,x)
    x_vaem = s_vaem.sum(1).unsqueeze(1)

    # AE Evaluation
    x_ae, mu_z, logvar_z, s_ae = model_ae(x)
    x_ae = x_ae.view(-1,1,28,28)
    s_ae = s_ae.view(-1,num_sources,28,28)
    s_ae = optimal_permute(s_ae,s_tru)

    # Save Results
    cm_jet = plt.get_cmap('jet')
    cmaplist = [cm_jet(i) for i in range(cm_jet.N)]
    cmaplist[0] = (0.0,0.0,0.0,1.0)
    cm_jet = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, cm_jet.N)

    root = 'results/K'+str(num_sources)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.mkdir(root)

    n = 10
    for i in range(n):
        dir = os.path.join(root,'Example_' + str(i+1))
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

        save_image(colorize(x[i].squeeze()), os.path.join(dir,'mix' +'_gt.png'), normalize=False)
        save_image(colorize(x_vae[i].squeeze()), os.path.join(dir,'mix' +'_vae.png'), normalize=False)
        save_image(colorize(x_vaem[i].squeeze()), os.path.join(dir,'mix' +'_vaem.png'), normalize=False)
        save_image(colorize(x_ae[i].squeeze()), os.path.join(dir,'mix' +'_ae.png'), normalize=False)
        for j in range(2):
        	save_image(colorize(s_tru[i,j].squeeze()), os.path.join(dir, 's' + str(j+1) +'_gt.png'), normalize=False)
        for j in range(2):
            save_image(colorize(s_vae[i,j].squeeze()), os.path.join(dir, 's' + str(j+1) +'_vae.png'), normalize=False)
            save_image(colorize(s_vaem[i,j].squeeze()), os.path.join(dir, 's' + str(j+1) +'_vaem.png'), normalize=False)
            save_image(colorize(s_ae[i,j].squeeze()), os.path.join(dir, 's' + str(j+1) +'_ae.png'), normalize=False)

print('Separated images saved in "results" directory.')
print('Evaluation complete.')