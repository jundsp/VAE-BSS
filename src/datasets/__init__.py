"""Datasets of mixtures for train.py and evaluate.py.

Each dataset module tells the shared scripts how to load and mix its data (``loaders``,
``prepare``), which settings to use by default (``defaults``, the paper's settings for that
experiment), how to score separated sources (``scores``) and how to draw and save examples
(``figure``, ``evaluate``). To add a dataset, write such a module and register it below.
"""

import importlib

DATASETS = {
    'mnist': 'src.datasets.mnist',          # mixed pairs of handwritten digits (images)
    'audio_synthetic': 'src.datasets.audio_synthetic',  # mixed pairs of synthetic notes (spectrograms)
}


def get_dataset(name):
    """The module of dataset ``name``, imported on demand (MNIST needs torchvision)."""
    return importlib.import_module(DATASETS[name])


def dataset_for_dimx(dimx):
    """The dataset whose inputs have ``dimx`` elements (784: mnist, 32768: audio_synthetic)."""
    for name in DATASETS:
        try:
            dataset = get_dataset(name)
        except ImportError:  # e.g. no torchvision, so no MNIST
            continue
        if dataset.dimx == dimx:
            return dataset
    raise ValueError(f'no dataset has inputs of {dimx} elements')
