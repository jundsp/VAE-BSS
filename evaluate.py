import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import os
import shutil
from src.model import *
from src.argparser import *
from src.utils import *
    
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

print('\nLoading and randomly mixing pairs of MNIST test data images.')
# Load MNIST Test Data
data, numbers = next(iter(test_loader))
# Randomly mix
x, s_tru = mix_data(data.to(device))

print('Evaluating mixtures with K = 2, 3, and 4 assumed sources:')
if os.path.exists('results'):
    shutil.rmtree('results')
os.mkdir('results')
# Evaluate for K = 2, 3, 4 model sources
for num_sources in range(2,5):
    print('\tK = ' + str(num_sources))

    # Load the Trained VAE Model
    model_vae = VAE(dimx=dimx,dimz=args.dimz,n_sources=num_sources,device=device,variational=True).to(device)
    model_vae.load_state_dict(torch.load('saves/pretrained/model_vae_K' + str(num_sources) + '.pt',map_location=torch.device('cpu')))
    model_vae.eval()

    # Load the Trained AE Model (latent sampling is deterministic, trained without KLD)
    model_ae = VAE(dimx=dimx,dimz=args.dimz,n_sources=num_sources,device=device,variational=False).to(device)
    model_ae.load_state_dict(torch.load('saves/pretrained/model_ae_K' + str(num_sources) + '.pt',map_location=torch.device('cpu')))
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

        nrows = 5
        n = 2*nrows
        togrid = torch.zeros(n,x.size(-2),x.size(-1))
        togrid[0::nrows] = x[i]
        togrid[1::nrows] = s_ae[i]
        togrid[2::nrows] = s_vae[i]
        togrid[3::nrows] = s_vaem[i]
        togrid[4::nrows] = s_tru[i]
        recon_grid = make_grid(togrid.detach().unsqueeze(1),nrow=nrows).cpu()
        # Todo: Save after plotting matplotlib, title the columns
        save_image(recon_grid,os.path.join(dir,'result.png'),padding=0)

print('\nMixed and Separated images saved in "results" directory....')
print('Evaluation complete.')