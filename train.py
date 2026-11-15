"""Train VAE-BSS from scratch, solely on data mixtures, in an unsupervised way.

    python train.py                                   # MNIST digits, as before
    python -m src.datasets.synthetic_notes            # once: synthetic notes in data/audio_synthetic
    python train.py --dataset audio_synthetic         # spectrograms of mixed notes

Options that are not given take the dataset's settings from the paper (src/argparser.py).
The model is saved to saves/ (MNIST: saves/model_<vae|ae>_K<sources>.pt, audio_synthetic:
saves/audio_synthetic_<vae|ae>_K<sources>.pt) and figures to results/ (audio_synthetic:
results/audio_synthetic_<vae|ae>_K<sources>/). The pretrained models in saves/pretrained are never
written.
"""

import csv
import math
import json
import os
import sys
import time

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


class CosineLR(optim.lr_scheduler.LRScheduler):
    """Half a cosine from the base rate at epoch ``start`` to ``min_lr`` at epoch ``total``.

    ``restart`` begins a new cosine from the current rate, e.g. when a resumed run trains to
    a new number of epochs, so the rate never jumps."""

    def __init__(self, optimizer, total, min_lr, last_epoch=-1):
        self.total, self.min_lr, self.start = total, min_lr, 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # last_epoch counts finished epochs: 0 while training the first one, total - 1 the last.
        t = (self.last_epoch - self.start) / max(1, self.total - 1 - self.start)
        t = min(max(t, 0.0), 1.0)
        return [self.min_lr + (base - self.min_lr) * 0.5 * (1 + math.cos(math.pi * t))
                for base in self.base_lrs]

    def restart(self, lr, total, min_lr):
        self.total, self.min_lr, self.start = total, min_lr, self.last_epoch
        self.base_lrs = [lr] * len(self.optimizer.param_groups)
        for group, rate in zip(self.optimizer.param_groups, self.get_lr()):
            group['lr'] = rate
        self._last_lr = [g['lr'] for g in self.optimizer.param_groups]


def make_scheduler(optimizer, args):
    if args.scheduler == 'cosine':
        return CosineLR(optimizer, args.epochs, args.min_lr)
    return optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.decay, last_epoch=-1)


def log_line(message=''):
    """A timestamped line of the training log (standard output)."""
    print(time.strftime('%Y-%m-%d %H:%M:%S'), ' ', message, sep='')


def rule(title='', char='=', width=79):
    """A separator line with an optional title: ``==== title ====``."""
    if not title:
        return char * width
    side = max(width - len(title) - 2, 0)
    return '{} {} {}'.format(char * (side // 2), title, char * (side - side // 2))


def duration(seconds):
    """``3725`` -> ``1:02:05``; ``65`` -> ``1:05``."""
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return '{}:{:02d}:{:02d}'.format(hours, minutes, secs) if hours else '{}:{:02d}'.format(minutes, secs)


def losses_text(values):
    """``loss  -ell  kld`` of an epoch (per example)."""
    return 'loss {:.4f}  -ell {:.4f}  kld {:.4f}'.format(*values.tolist())


def log_header():
    """The settings of the run, at the start of the log."""
    log_line(rule('train.py'))
    log_line('dataset: {}, {} training / {} {} examples, {} batches of {} per epoch'.format(
        args.dataset, len(train_loader.dataset), len(test_loader.dataset), monitor,
        len(train_loader), args.batch_size))
    for key in getattr(dataset, 'config_keys', ()):
        log_line('processing: {} {:g}'.format(key, getattr(args, key)))
    log_line('device: {}'.format(device))
    log_line('model: {}, {} sources, latent {} per source, hidden {}, {:.1f} M parameters'.format(
        'VAE' if args.variational else 'AE', args.sources, args.dimz,
        ' '.join(map(str, args.hidden)), sum(p.numel() for p in model.parameters()) / 1e6))
    log_line('objective: Laplace likelihood (scale {:g}) + beta * KL to the {} prior, beta 0 to '
             '{:g} over {} epochs'.format(args.scale, args.prior, args.beta_max, args.warm_up))
    if args.scheduler == 'cosine':
        schedule = 'cosine to {:g} at epoch {}'.format(args.min_lr, args.epochs)
    else:
        schedule = 'x{:g} per epoch'.format(args.decay)
    log_line('optimiser: Adam, lr {:g} ({}), {} epochs'.format(args.learning_rate, schedule,
                                                                args.epochs))
    log_line('saving: {} and {}/ every {} epochs'.format(checkpoint, results, args.save_interval))
    log_line(rule())


def value(v):
    """A loss term as a float (the KLD is 0.0, not a tensor, for the AE)."""
    return v.item() if torch.is_tensor(v) else float(v)


def train(epoch):

    model.train()
    train_losses = torch.zeros(3)
    started = time.time()
    width = len(str(len(train_loader)))

    for batch_idx, batch in enumerate(train_loader):
        data = dataset.prepare(batch, device, args)[0]

        optimizer.zero_grad()
        recon_y, mu_z, logvar_z, _ = model(data)
        loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)
        loss.backward()
        optimizer.step()

        train_losses[0] += loss.item()
        train_losses[1] += ELL.item()
        train_losses[2] += value(KLD)

        if batch_idx % args.log_interval == 0:
            n = len(data)
            left = (time.time() - started) / (batch_idx + 1) * (len(train_loader) - batch_idx - 1)
            log_line('epoch {} [{:{}d}/{}] loss {:.4f} (-ell {:.4f}, kld {:.4f}) | lr {:.2e} | '
                     'ETA {}'.format(epoch, batch_idx, width, len(train_loader), loss.item() / n,
                                     ELL.item() / n, value(KLD) / n,
                                     optimizer.param_groups[0]['lr'], duration(left)))

    train_losses /= len(train_loader.dataset)

    return train_losses


def test(epoch):

    model.eval()
    test_losses = torch.zeros(3)
    scores = {}
    evaluate = args.eval_interval > 0 and (epoch % args.eval_interval == 0 or epoch == args.epochs)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            data, extra = dataset.prepare(batch, device, args)

            recon_y, mu_z, logvar_z, recons = model(data)
            loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)

            test_losses[0] += loss.item()
            test_losses[1] += ELL.item()
            test_losses[2] += value(KLD)

            if evaluate:
                for key, v in dataset.scores(data, recons, extra).items():
                    scores.setdefault(key, []).append(v.cpu())

        # Separations of the last batch of the monitored set (audio: validation; MNIST: test).
        if epoch % args.plot_interval == 0 or epoch == args.epochs:
            dataset.figure(os.path.join(results, 'reconstruction_' + str(epoch) + '.png'),
                           data, recon_y, recons, extra)

        test_losses /= len(test_loader.dataset)
        medians = {key: torch.cat(v).median().item() for key, v in scores.items()}

    return test_losses, medians

def plot_losses(losses, epochs):
	plt.figure()
	plt.plot(np.array(range(1,epochs+1)),losses["train"][:epochs,0].view(-1),label="Train")
	plt.plot(np.array(range(1,epochs+1)),losses["test"][:epochs,0].view(-1),label=monitor.capitalize())
	plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend(), plt.xlim(1,max(epochs,2))
	plt.savefig(os.path.join(results, 'losses.png'))
	plt.close()


def plot_progress(rows):
    """Losses (log scale) and median scores per epoch, for datasets scored in training.
    The log's test_loss column is the loss on the monitored set (audio: validation)."""
    base = ('epoch', 'beta', 'lr', 'seconds', 'train_loss', 'train_ell', 'train_kld',
            'test_loss', 'test_ell', 'test_kld')
    keys = list(dict.fromkeys(k for r in rows for k in r if k not in base))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r['epoch'] for r in rows]
    ax1.semilogy(epochs, [r['train_loss'] for r in rows], label='Train')
    ax1.semilogy(epochs, [r['test_loss'] for r in rows], label=monitor.capitalize())
    ax1.set_xlabel('Epoch'), ax1.set_ylabel('Loss per mixture'), ax1.legend()
    for key in keys:
        points = [(r['epoch'], r[key]) for r in rows if key in r]
        ax2.plot(*zip(*points), 'o-', label=key)
    ax2.set_xlabel('Epoch'), ax2.set_ylabel('Median {} score (dB)'.format(monitor)), ax2.grid(alpha=0.3)
    if keys:
        ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results, 'progress.png'))
    plt.close(fig)


def rng_states(device):
    states = {'cpu': torch.get_rng_state()}
    if device.type == 'cuda':
        states['cuda'] = torch.cuda.get_rng_state_all()
    elif device.type == 'mps':
        states['mps'] = torch.mps.get_rng_state()
    return states


def set_rng_states(states, device):
    torch.set_rng_state(states['cpu'])
    if device.type == 'cuda' and 'cuda' in states:
        torch.cuda.set_rng_state_all(states['cuda'])
    elif device.type == 'mps' and 'mps' in states:
        torch.mps.set_rng_state(states['mps'])


def save_training_state(path, epoch, losses, log, device):
    """Everything --resume needs to continue exactly where training stopped."""
    tmp = path + '.tmp'
    rng = rng_states(device)
    sampler = getattr(train_loader, 'sampler', None)
    if getattr(sampler, 'generator', None) is not None:  # a sampler with its own generator
        rng['sampler'] = sampler.generator.get_state()
    torch.save({'epoch': epoch, 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scheduler_kind': args.scheduler, 'losses': losses, 'log': log,
                'rng': rng}, tmp)
    os.replace(tmp, path)


def resume(checkpoint, state_path, losses, device):
    """Load the model and training state; returns the last finished epoch, the log and
    whether the saved schedule must be replaced (another kind of scheduler, or none saved)."""
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    if os.path.exists(state_path):
        # On the CPU: random states must be CPU tensors; the optimizer moves its own state.
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        optimizer.load_state_dict(state['optimizer'])
        done = state['epoch']
        same = state.get('scheduler_kind', 'step') == args.scheduler
        if same:
            scheduler.load_state_dict(state['scheduler'])
        else:  # e.g. a step-scheduled run continued with cosine: from the saved rate
            scheduler.last_epoch = done
        for key in losses:
            n = min(done, len(losses[key]))
            losses[key][:n] = state['losses'][key][:n]
        set_rng_states(state['rng'], device)
        if 'sampler' in state['rng']:
            train_loader.sampler.generator.set_state(state['rng']['sampler'])
        log_line('Resuming from epoch {} of {}'.format(done, checkpoint))
        return done, state['log'], not same
    # Older runs saved only the weights: continue from the log, with a fresh optimizer.
    with open(os.path.join(results, 'log.csv')) as f:
        log = [{k: float(v) for k, v in r.items() if v != ''} for r in csv.DictReader(f)]
    done = int(log[-1]['epoch'])
    for _ in range(done):
        if optimizer.param_groups[0]['lr'] >= 1.0e-5:
            scheduler.step()  # the schedule of the current options, as if run from the start
    for r in log[:len(losses['train'])]:
        losses['train'][int(r['epoch']) - 1, 0] = r['train_loss']
        losses['test'][int(r['epoch']) - 1, 0] = r['test_loss']
    log_line('Resuming from epoch {} of {} (weights only: no training state was saved, so the '
          'optimizer starts afresh and results differ from an uninterrupted run)'.format(
              done, checkpoint))
    return done, log, False


def write_log(rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(results, 'log.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)



if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)  # log lines appear at once, even in a file
    args = parse_args()
    dataset = get_dataset(args.dataset)
    monitor = getattr(dataset, 'monitor', 'test')  # the set scored while training
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
    scheduler = make_scheduler(optimizer, args)

    log_header()
    losses = {"train": torch.zeros(args.epochs,3), "test": torch.zeros(args.epochs,3)}
    log = []
    state_path = os.path.splitext(checkpoint)[0] + '.train.pt'
    start = 1
    if args.resume:
        if not os.path.exists(checkpoint):
            raise SystemExit('nothing to resume: {} does not exist'.format(checkpoint))
        done, log, replace = resume(checkpoint, state_path, losses, device)
        start = done + 1
        lr = optimizer.param_groups[0]['lr']
        if 'learning_rate' in args.given:  # continue at a new rate
            lr = args.learning_rate
            for group in optimizer.param_groups:
                group['lr'] = lr
        if args.scheduler == 'cosine' and (
                replace or 'learning_rate' in args.given or args.epochs != scheduler.total
                or args.min_lr != scheduler.min_lr):
            # A new cosine from the current rate down to --min-lr at the new last epoch.
            scheduler.restart(lr, args.epochs, args.min_lr)
            log_line('Cosine schedule from {:g} at epoch {} to {:g} at epoch {}'.format(
                lr, start, args.min_lr, args.epochs))
        elif 'learning_rate' in args.given:
            log_line('Learning rate set to {:g}'.format(lr))

    for epoch in range(start, args.epochs+1):
        beta = min(1.0,(epoch)/min(args.epochs,args.warm_up)) * args.beta_max
        started = time.time()

        losses["train"][epoch-1] = train(epoch)
        losses["test"][epoch-1], medians = test(epoch)
        seconds = time.time() - started
        lr = optimizer.param_groups[0]['lr']
        row = {'epoch': epoch, 'beta': beta, 'lr': lr, 'seconds': seconds}
        for split in ('train', 'test'):
            for i, term in enumerate(('loss', 'ell', 'kld')):
                row[split + '_' + term] = losses[split][epoch-1, i].item()
        log.append({**row, **medians})

        saving = epoch % args.save_interval == 0
        log_line(rule('epoch {}/{} done in {}'.format(epoch, args.epochs, duration(seconds)), '-'))
        log_line('train  ' + losses_text(losses["train"][epoch-1]))
        log_line('{:<6} '.format(monitor[:5]) + losses_text(losses["test"][epoch-1])
                 + ''.join('  {} {:.2f} dB'.format(key, v) for key, v in medians.items()))
        log_line('lr {:.2e} | beta {:.3f}{}'.format(
            lr, beta, ' | saved ' + checkpoint if saving else ''))

        if optimizer.param_groups[0]['lr'] >= 1.0e-5:
            scheduler.step()

        with torch.no_grad():
            if saving:
                torch.save(model.state_dict(), checkpoint)
                save_training_state(state_path, epoch, losses, log, device)
                if getattr(dataset, 'config_keys', None):
                    with open(os.path.splitext(checkpoint)[0] + '.json', 'w') as f:
                        json.dump({k: getattr(args, k) for k in dataset.config_keys}, f)
                write_log(log)
                plot_losses(losses, epoch)
                if args.eval_interval > 0:
                    plot_progress(log)


    plot_losses(losses, args.epochs)
