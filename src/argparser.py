import argparse

from .datasets import DATASETS, get_dataset

'''
Argument parser for model training and evaluation.
Options marked "dataset default" take the settings of the chosen --dataset (the paper's
settings for that experiment, see `defaults` in src/datasets/mnist.py and audio_synthetic.py).
'''

parser = argparse.ArgumentParser(description='VAE-BSS')
parser.add_argument('--dataset', type=str, default='mnist', choices=sorted(DATASETS),
                    help='mnist: mixed pairs of digits (default). audio_synthetic: spectrograms of '
                         'mixed pairs of synthetic instrument notes')
parser.add_argument('--batch-size', type=int, default=None, metavar='N',
                    help='input batch size for training (dataset default; mnist: 64 images, '
                         'i.e. 32 mixtures; audio_synthetic: 128 mixtures)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data-directory', type=str, default=None, metavar='fname',
                    help='Folder containing the data (dataset default; mnist: data, audio_synthetic: '
                         'data/audio_synthetic)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--sources', type=int, default=2,
                    help='Number of sources to infer')
parser.add_argument('--dimz', type=int, default=None,
                    help='Dimension of latent space (dataset default; mnist: 20, audio_synthetic: 64)')
parser.add_argument('--hidden', type=int, nargs=5, default=None, metavar='N',
                    help='Hidden units of the five encoder layers, reversed in the decoder '
                         '(dataset default; mnist: 700 600 500 400 300, audio_synthetic: '
                         '1280 1024 768 512 256, half of the paper\'s 2560 2048 1536 1024 512)')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of data loading workers (parallel).')
parser.add_argument('--prior', type=str, default='gauss',
                    help='Prior over latent variables')
parser.add_argument('--scale', type=float, default=0.5,
                    help='Scale of Laplace likelihood')
parser.add_argument('--learning-rate', type=float, default=None,
                    help='Learning rate (dataset default; mnist: 2e-4, audio_synthetic: 1e-4)')
parser.add_argument('--variational', type=str, default=True,
                    help='VAE if True, AE if False.')
parser.add_argument('--decay', type=float, default=None,
                    help='Decay rate of scheduler for optimizer (dataset default; mnist: 0.9998, '
                         'audio_synthetic: 0.9999)')
parser.add_argument('--beta-max', type=float, default=None,
                    help='beta-VAE max value of weight (dataset default: 0.5)')
parser.add_argument('--epochs', type=int, default=None,
                    help='Training epochs (dataset default; mnist: 10, audio_synthetic: 5000)')
parser.add_argument('--warm-up', type=int, default=None,
                    help='Number of epochs to warm up beta (dataset default; mnist: 50, '
                         'audio_synthetic: 100)')
parser.add_argument('--save-interval', type=int, default=None,
                    help='How many epochs to wait before saving model (dataset default; '
                         'mnist: 1, audio_synthetic: 10)')
parser.add_argument('--resume', action='store_true', default=False,
                    help='Continue training from the saved model and training state (saved '
                         'next to it as <model>.train.pt); raise --epochs to train longer')
parser.add_argument('--compression', type=float, default=None, metavar='C',
                    help='Spectrogram magnitude compression |STFT|**C, as in the dissertation\'s '
                         'DVAE (audio only; dataset default: 1, no compression, as in the paper)')
parser.add_argument('--plot-interval', type=int, default=None,
                    help='How many epochs to wait before saving a figure of test separations '
                         '(dataset default; mnist: 1, audio_synthetic: 10)')
parser.add_argument('--eval-interval', type=int, default=None,
                    help='How many epochs to wait before scoring the test set, 0 for never '
                         '(dataset default; mnist: 0, audio_synthetic: 10)')


def parse_args(argv=None):
    """Parse the arguments and fill the unset ones with the dataset's defaults."""
    args = parser.parse_args(argv)
    for key, value in get_dataset(args.dataset).defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    # "--variational False" trains the AE; anything else (or nothing) the VAE.
    args.variational = str(args.variational).lower() not in ('false', '0', 'no')
    return args
