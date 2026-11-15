"""Mixed pairs of MNIST digits (paper, Sec. V-A), for train.py and evaluate.py.

The interface shared by all datasets is described in src/datasets/__init__.py.
"""

import os

import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ..utils import mix_data, optimal_permute, vae_masks

name = 'mnist'
dimx = 28 * 28

defaults = dict(
    data_directory='data',
    dimz=20,
    hidden=[700, 600, 500, 400, 300],
    batch_size=64,       # images per batch, so 32 mixtures
    learning_rate=2e-4,
    decay=0.9998,
    beta_max=0.5,
    warm_up=50,
    epochs=10,           # separation is clear within 10 epochs; the paper trained ~5000
    save_interval=1,
    plot_interval=1,
    eval_interval=0,     # no test scores while training (use evaluate.py --checkpoint)
)
use_mps = False          # CUDA or CPU, as the original MNIST training


def paths(args):
    """Results folder and checkpoint of a training run (never in saves/pretrained)."""
    return 'results', f"saves/model_{'vae' if args.variational else 'ae'}_K{args.sources}.pt"


def get_data_loaders(directory, batch_size, kwargs):
    train_set = datasets.MNIST(directory, train=True, download=True,
                               transform=transforms.ToTensor())
    test_set = datasets.MNIST(directory, train=False, download=True,
                              transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True,
                                              **kwargs)
    return train_loader, test_loader


def loaders(args, device):
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device.type == 'cuda' else {}
    return get_data_loaders(args.data_directory, args.batch_size, kwargs)


def prepare(batch, device):
    """Mix the first half of a batch of digits with the second half."""
    x, s = mix_data(batch[0].to(device))
    return x.view(-1, dimx), dict(s=s)


def _psnr(est, ref):
    mse = (est - ref).pow(2).mean((-2, -1)).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def scores(x, recons, extra):
    """PSNR (dB) of every true digit, for the VAE and VAEM estimates."""
    s = extra['s']
    s_vae = recons.view(-1, recons.shape[1], 28, 28)
    s_vaem = vae_masks(s_vae, x.view(-1, 1, 28, 28))
    return {f'{label} PSNR': _psnr(optimal_permute(est, s), s).flatten()
            for label, est in (('VAE', s_vae), ('VAEM', s_vaem))}


def figure(path, x, recon_x, recons, extra, n=6):
    """Rows: test mixtures. Columns: mixture, reconstruction, K sources."""
    n = min(x.size(0), n)
    ncols = 2 + recons.shape[1]
    comparison = torch.zeros(n * ncols, 1, 28, 28)
    comparison[::ncols] = x.view(x.size(0), 1, 28, 28)[:n]
    comparison[1::ncols] = recon_x.view(x.size(0), 1, 28, 28)[:n]
    for i in range(recons.shape[1]):
        comparison[(i + 2)::ncols] = recons[:, i].view(x.size(0), 1, 28, 28)[:n]
    save_image(comparison.cpu(), path, nrow=ncols)


@torch.no_grad()
def evaluate(model, args, out, device, examples=10, batch_size=1000):
    """Median PSNR of VAE and VAEM over the test set, and image grids of a few examples
    (columns: mixture, VAE sources, VAEM sources, true sources)."""
    _, test_loader = get_data_loaders(args.data_directory, batch_size, {})
    torch.manual_seed(0)
    medians = {}
    totals = {}
    for i, batch in enumerate(test_loader):
        x, extra = prepare(batch, device)
        recons = model(x)[3]
        for key, value in scores(x, recons, extra).items():
            totals.setdefault(key, []).append(value.cpu())
        if i == 0:
            s = extra['s']
            s_vae = optimal_permute(recons.view(-1, args.sources, 28, 28), s)
            s_vaem = vae_masks(s_vae, x.view(-1, 1, 28, 28))
            for j in range(min(examples, len(x))):
                grid = torch.cat([x.view(-1, 1, 28, 28)[j:j + 1], s_vae[j].unsqueeze(1),
                                  s_vaem[j].unsqueeze(1), s[j].unsqueeze(1)]).cpu()
                save_image(grid, os.path.join(out, f'example_{j + 1:02d}.png'),
                           nrow=len(grid), padding=1)
    for key, values in totals.items():
        medians[key] = torch.cat(values).median().item()
        print(f'{key}: {medians[key]:.2f} dB (median)')
    print(f'Examples saved in {out}')
    return medians
