import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from model import *
from argparser import *
from utils import *

def train(epoch):

    model.train()
    train_losses = torch.zeros(3)

    for batch_idx, (data,_) in enumerate(train_loader):
        data = mix_data(data.to(device))[0].view(-1,dimx)

        optimizer.zero_grad()
        recon_y, mu_z, logvar_z, _ = model(data)
        loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)
        loss.backward()
        optimizer.step()

        train_losses[0] += loss.item()
        train_losses[1] += ELL.item()
        train_losses[2] += KLD.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t -ELL: {:5.6f} \t KLD: {:5.6f} \t Loss: {:5.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                ELL.item() / len(data),KLD.item() / len(data),loss.item() / len(data)))

    train_losses /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_losses[0]))

    return train_losses


def test(epoch):

    model.eval()
    test_losses = torch.zeros(3)

    with torch.no_grad():
        for i, (data,_) in enumerate(test_loader):
            data = mix_data(data.to(device))[0].view(-1,dimx)

            recon_y, mu_z, logvar_z, recons = model(data)
            loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)

            test_losses[0] += loss.item()
            test_losses[1] += ELL.item()
            test_losses[2] += KLD.item()

        n = min(data.size(0), 6)
        ncols = (2+args.sources)
        comparison = torch.zeros(n*ncols,1,28,28)
        comparison[::ncols] = data.view(data.size(0), 1, 28, 28)[:n]
        comparison[1::ncols] = recon_y.view(data.size(0), 1, 28, 28)[:n]
        for i in range(args.sources):
            comparison[(i+2)::ncols] = recons[:,i].view(data.size(0), 1, 28, 28)[:n]

        grid = make_grid(comparison,nrow=ncols)

        save_image(comparison.cpu(),'results/reconstruction_' + str(epoch) + '.png', nrow=ncols)

        test_losses /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_losses[0]))

    return test_losses

def plot_losses(losses):
	plt.figure()
	plt.plot(np.array(range(1,args.epochs+1)),losses["train"][:,0].view(-1),label="Train")
	plt.plot(np.array(range(1,args.epochs+1)),losses["test"][:,0].view(-1),label="Test")
	plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend(), plt.xlim(1,args.epochs)
	plt.savefig('results/losses.png')
	plt.close()



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
    batch_size=args.batch_size, shuffle=True, **kwargs)

# MNIST is 28 X 28
dimx = int(28*28)

model = VAE(dimx=dimx,dimz=args.dimz,n_sources=args.sources,device=device,variational=args.variational).to(device)
loss_function = Loss(sources=args.sources,likelihood='laplace',variational=args.variational,prior=args.prior,scale=args.scale)

optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.decay, last_epoch=-1)

losses = {"train": torch.zeros(args.epochs,3), "test": torch.zeros(args.epochs,3)}

for epoch in range(1, args.epochs+1):
    beta = min(1.0,(epoch)/min(args.epochs,args.warm_up)) * args.beta_max

    losses["train"][epoch-1] = train(epoch)
    losses["test"][epoch-1] = test(epoch)

    if optimizer.param_groups[0]['lr'] >= 1.0e-5:
        scheduler.step()

    with torch.no_grad():
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(),'saves/model_'+('vae' if args.variational else 'ae')+'_K' + str(args.sources) +  '.pt')


plot_losses(losses)
