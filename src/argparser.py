import argparse

''' 
Argument parser for model training and evaluation 
'''

parser = argparse.ArgumentParser(description='VAE-BSS')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data-directory', type=str, default='data', metavar='fname',
                    help='Folder containing the data')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--sources', type=int, default=2,
                    help='Number of sources to infer')
parser.add_argument('--dimz', type=int, default=20,
                    help='Dimension of latent space.')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of data loading workers (parallel).')
parser.add_argument('--prior', type=str, default='gauss',
                    help='Prior over latent variables')
parser.add_argument('--scale', type=float, default=0.5,
                    help='Scale of Laplace likelihood')
parser.add_argument('--learning-rate', type=float, default=.0002,
                    help='Learning rate')
parser.add_argument('--variational', type=str, default=True,
                    help='VAE if True, AE if False.')
parser.add_argument('--decay', type=float, default=.9998,
                    help='Decay rate of scheduler for optimizer.')
parser.add_argument('--beta-max', type=float, default=0.5,
                    help='beta-VAE max value of weight')
parser.add_argument('--epochs', type=int, default=10,
                    help='Training epochs')
parser.add_argument('--warm-up', type=int, default=50,
                    help='Number of iterations to warm up beta.')
parser.add_argument('--save-interval', type=int, default=1,
                    help='How many epochs to wait before saving model.')