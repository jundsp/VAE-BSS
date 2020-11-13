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

parser = argparse.ArgumentParser(description='VAE-BSS')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data-directory', type=str, default='/Users/julian/Documents/Code/Compute_Canada/data', metavar='fname',
                    help='Folder containing the spectrogram data')
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

# Transition between latent spaces
data, numbers = next(iter(test_loader))
x,s_tru = mix_data(data.to(device))

dimx = int(28*28)
for num_sources in range(2,4):
    model = VAE(dimx=dimx,dimz=args.dimz,n_sources=num_sources,device=device)
    loss_function = Loss(sources=num_sources,likelihood='laplace')
    model.to(device)

    model.load_state_dict(torch.load('saves/model_K' + str(num_sources) + '.pt',map_location=torch.device('cpu')))
    model.eval()

    mu_x, mu_z, logvar_z, mu_s = model(x)

    mu_x = mu_x.view(-1,1,28,28)
    mu_s = mu_s.view(-1,num_sources,28,28)

    # Create masks
    mu_sm = vae_masks(mu_s,x)
    mu_xm = mu_sm.sum(1).unsqueeze(1)

    # Plots
    cm_jet = plt.get_cmap('jet')
    cmaplist = [cm_jet(i) for i in range(cm_jet.N)]
    cmaplist[0] = (0.0,0.0,0.0,1.0)
    cm_jet = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, cm_jet.N)

    root = 'results/K'+str(num_sources)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.mkdir(root)

    examples = 10
    jv = torch.randperm(x.size(0))[:examples]
    for v, i in enumerate(jv):
        dir = os.path.join(root,'Example_' + str(v+1))
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)
        save_image(colorize(mu_x[i].squeeze()), os.path.join(dir,'mix' +'_vae.png'), normalize=False)
        save_image(colorize(mu_xm[i].squeeze()), os.path.join(dir,'mix' +'_vaem.png'), normalize=False)
        save_image(colorize(x[i].squeeze()), os.path.join(dir,'mix' +'_gt.png'), normalize=False)
        for j in range(2):
        	save_image(colorize(s_tru[i,j].squeeze()), os.path.join(dir, 's' + str(j+1) +'_gt.png'), normalize=False)
        for j in range(num_sources):
            save_image(colorize(mu_s[i,j].squeeze()), os.path.join(dir, 's' + str(j+1) +'_vae.png'), normalize=False)
            save_image(colorize(mu_sm[i,j].squeeze()), os.path.join(dir, 's' + str(j+1) +'_vaem.png'), normalize=False)