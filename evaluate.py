"""Evaluate VAE-BSS.

    python evaluate.py                                             # pretrained MNIST models
    python evaluate.py --checkpoint saves/synthetic_vae_K2.pt      # a model from train.py

Without a checkpoint, separates mixed MNIST test digits with the pretrained models in
saves/pretrained (AE and VAE, K = 2, 3, 4) and saves image grids in results/.

With a checkpoint from train.py, scores it on its dataset's test set (the dataset and model
size are read from the weights) and saves examples in results/eval_<checkpoint name>/: for the
synthetic notes, median SI-SDR, SIR and SAR of the VAE, the VAE with masking (VAEM) and the
ideal masks (Table III of the paper), with .wav examples; for MNIST, median PSNR of the VAE and
VAEM with image grids. Add ``--variational False`` for a model trained as an AE.
"""

import argparse
import os
import shutil
from types import SimpleNamespace

import torch

from src.datasets import dataset_for_dimx
from src.model import VAE
from src.utils import mix_data, optimal_permute, vae_masks

parser = argparse.ArgumentParser(description='Evaluate VAE-BSS')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Model saved by train.py (default: the pretrained MNIST models)')
parser.add_argument('--variational', type=str, default=True,
                    help='VAE if True, AE if False (for --checkpoint)')
parser.add_argument('--data-directory', type=str, default=None,
                    help='Data folder (default: the dataset\'s, data or data/synthetic)')
parser.add_argument('--out', type=str, default=None,
                    help='Results folder (default: results, or results/eval_<checkpoint name>)')
parser.add_argument('--examples', type=int, default=10, help='Examples to save (--checkpoint)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='S')
args = parser.parse_args()
torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}


def evaluate_checkpoint():
    state = torch.load(args.checkpoint, map_location='cpu')
    # The model's sizes, read from its weights.
    hidden = [state[f'Encoder.{i}.block.0.weight'].shape[0] for i in range(5)]
    dimx = state['Encoder.0.block.0.weight'].shape[1]
    dimz = state['Decoder.0.block.0.weight'].shape[1]
    sources = state['out_w.weight'].shape[0]
    variational = str(args.variational).lower() not in ('false', '0', 'no')
    dataset = dataset_for_dimx(dimx)
    dev = device
    if dataset.use_mps and not args.no_cuda and not args.cuda and torch.backends.mps.is_available():
        dev = torch.device('mps')
    model = VAE(dimx=dimx, dimz=dimz, n_sources=sources, device=dev, variational=variational,
                hidden=hidden).to(dev)
    model.load_state_dict(state)
    model.eval()
    run = SimpleNamespace(sources=sources, data_directory=args.data_directory
                          or dataset.defaults['data_directory'])
    name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out = args.out or os.path.join('results', 'eval_' + name)
    os.makedirs(out, exist_ok=True)
    print(f'{name}: {dataset.name}, K = {sources}, {"VAE" if variational else "AE"}')
    dataset.evaluate(model, run, out, dev, examples=args.examples)


if args.checkpoint:
    evaluate_checkpoint()
    raise SystemExit

from torchvision.utils import save_image, make_grid
from src.datasets.mnist import get_data_loaders

train_loader, test_loader = get_data_loaders(args.data_directory or 'data',args.batch_size,kwargs)

dimx = int(28*28)
dimz = 20

print('\nLoading and randomly mixing pairs of MNIST test data images.')
# Load MNIST Test Data
data, numbers = next(iter(test_loader))
# Randomly mix
x, s_tru = mix_data(data.to(device))

print('Evaluating mixtures with K = 2, 3, and 4 assumed sources:')
results = args.out or 'results'
if os.path.exists(results):
    shutil.rmtree(results)
os.makedirs(results)
# Evaluate for K = 2, 3, 4 model sources
for num_sources in range(2,5):
    print('\tK = ' + str(num_sources))

    # Load the Trained VAE Model
    model_vae = VAE(dimx=dimx,dimz=dimz,n_sources=num_sources,device=device,variational=True).to(device)
    model_vae.load_state_dict(torch.load('saves/pretrained/model_vae_K' + str(num_sources) + '.pt',map_location=torch.device('cpu')))
    model_vae.eval()

    # Load the Trained AE Model (latent sampling is deterministic, trained without KLD)
    model_ae = VAE(dimx=dimx,dimz=dimz,n_sources=num_sources,device=device,variational=False).to(device)
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
    root = os.path.join(results, 'K'+str(num_sources))
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

print(f'\nMixed and Separated images saved in "{results}" directory....')
print('Evaluation complete.')
