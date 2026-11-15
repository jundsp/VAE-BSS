"""Train VAE-BSS from scratch, solely on data mixtures, in an unsupervised way.

    python train.py                                   # MNIST digits, as before
    python -m src.datasets.synthetic_notes            # once: synthetic notes in data/synthetic
    python train.py --dataset synthetic               # spectrograms of mixed notes

Options that are not given take the dataset's settings from the paper (src/argparser.py).
The model is saved to saves/ (MNIST: saves/model_<vae|ae>_K<sources>.pt, synthetic:
saves/synthetic_<vae|ae>_K<sources>.pt) and figures to results/ (synthetic:
results/synthetic_<vae|ae>_K<sources>/). The pretrained models in saves/pretrained are never
written.
"""

import csv
import os

import numpy as np
import torch
import torch.utils.data
from torch import optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.argparser import parse_args
from src.datasets import get_dataset
from src.model import VAE, Loss


def get_device(no_cuda, use_mps):
    if not no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if not no_cuda and use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def value(v):
    """A loss term as a float (the KLD is 0.0, not a tensor, for the AE)."""
    return v.item() if torch.is_tensor(v) else float(v)


def train(epoch):

    model.train()
    train_losses = torch.zeros(3)

    for batch_idx, batch in enumerate(train_loader):
        data = dataset.prepare(batch, device)[0]

        optimizer.zero_grad()
        recon_y, mu_z, logvar_z, _ = model(data)
        loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)
        loss.backward()
        optimizer.step()

        train_losses[0] += loss.item()
        train_losses[1] += ELL.item()
        train_losses[2] += value(KLD)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t -ELL: {:5.6f} \t KLD: {:5.6f} \t Loss: {:5.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                ELL.item() / len(data),value(KLD) / len(data),loss.item() / len(data)))

    train_losses /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_losses[0]))

    return train_losses


def test(epoch):

    model.eval()
    test_losses = torch.zeros(3)
    scores = {}
    evaluate = args.eval_interval > 0 and (epoch % args.eval_interval == 0 or epoch == args.epochs)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            data, extra = dataset.prepare(batch, device)

            recon_y, mu_z, logvar_z, recons = model(data)
            loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)

            test_losses[0] += loss.item()
            test_losses[1] += ELL.item()
            test_losses[2] += value(KLD)

            if evaluate:
                for key, v in dataset.scores(data, recons, extra).items():
                    scores.setdefault(key, []).append(v.cpu())

        # Separations of the last test batch.
        if epoch % args.plot_interval == 0 or epoch == args.epochs:
            dataset.figure(os.path.join(results, 'reconstruction_' + str(epoch) + '.png'),
                           data, recon_y, recons, extra)

        test_losses /= len(test_loader.dataset)
        medians = {key: torch.cat(v).median().item() for key, v in scores.items()}
        print('====> Test set loss: {:.4f}'.format(test_losses[0])
              + ''.join('   {}: {:.2f} dB'.format(key, v) for key, v in medians.items()))

    return test_losses, medians

def plot_losses(losses, epochs):
	plt.figure()
	plt.plot(np.array(range(1,epochs+1)),losses["train"][:epochs,0].view(-1),label="Train")
	plt.plot(np.array(range(1,epochs+1)),losses["test"][:epochs,0].view(-1),label="Test")
	plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend(), plt.xlim(1,max(epochs,2))
	plt.savefig(os.path.join(results, 'losses.png'))
	plt.close()


def plot_progress(rows):
    """Losses (log scale) and median test scores per epoch, for datasets scored in training."""
    base = ('epoch', 'beta', 'lr', 'train_loss', 'test_loss')
    keys = list(dict.fromkeys(k for r in rows for k in r if k not in base))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r['epoch'] for r in rows]
    ax1.semilogy(epochs, [r['train_loss'] for r in rows], label='Train')
    ax1.semilogy(epochs, [r['test_loss'] for r in rows], label='Test')
    ax1.set_xlabel('Epoch'), ax1.set_ylabel('Loss per mixture'), ax1.legend()
    for key in keys:
        points = [(r['epoch'], r[key]) for r in rows if key in r]
        ax2.plot(*zip(*points), 'o-', label=key)
    ax2.set_xlabel('Epoch'), ax2.set_ylabel('Median test score (dB)'), ax2.grid(alpha=0.3)
    if keys:
        ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results, 'progress.png'))
    plt.close(fig)


def write_log(rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(results, 'log.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)



if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args.dataset)
    torch.manual_seed(args.seed)

    device = get_device(args.no_cuda, dataset.use_mps)
    results, checkpoint = dataset.paths(args)
    pretrained = os.path.abspath(os.path.join('saves', 'pretrained'))
    if os.path.abspath(checkpoint).startswith(pretrained + os.sep):
        raise SystemExit('refusing to overwrite the pretrained models in saves/pretrained')
    os.makedirs(results, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)

    train_loader, test_loader = dataset.loaders(args, device)

    model = VAE(dimx=dataset.dimx,dimz=args.dimz,n_sources=args.sources,device=device,variational=args.variational,hidden=args.hidden).to(device)
    loss_function = Loss(sources=args.sources,likelihood='laplace',variational=args.variational,prior=args.prior,scale=args.scale)

    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.decay, last_epoch=-1)

    losses = {"train": torch.zeros(args.epochs,3), "test": torch.zeros(args.epochs,3)}
    log = []

    for epoch in range(1, args.epochs+1):
        beta = min(1.0,(epoch)/min(args.epochs,args.warm_up)) * args.beta_max

        losses["train"][epoch-1] = train(epoch)
        losses["test"][epoch-1], medians = test(epoch)
        log.append({'epoch': epoch, 'beta': beta, 'lr': optimizer.param_groups[0]['lr'],
                    'train_loss': losses["train"][epoch-1, 0].item(),
                    'test_loss': losses["test"][epoch-1, 0].item(), **medians})

        if optimizer.param_groups[0]['lr'] >= 1.0e-5:
            scheduler.step()

        with torch.no_grad():
            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), checkpoint)
                write_log(log)
                plot_losses(losses, epoch)
                if args.eval_interval > 0:
                    plot_progress(log)


    plot_losses(losses, args.epochs)
