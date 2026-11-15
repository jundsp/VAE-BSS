"""Datasets of random note pairs for the audio version (the synthetic notes or any folder of
mono .wav notes, such as MUMS).

As in the paper, sources are mixed at random: every epoch pairs each training note with a new
random partner, so the number of distinct mixtures grows with the square of the number of
notes. Validation and test pairs are fixed by a seed, so their scores are comparable between
runs. Notes of different splits are never mixed with one another.
"""

from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from scipy.io import wavfile

from ..processing import N_SAMPLES, SAMPLE_RATE


def load_note(path):
    """A note as float32 in [-1, 1], cut or zero-padded to the first ``N_SAMPLES`` samples."""
    rate, wave = wavfile.read(path)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {rate} Hz; the paper's STFT settings need {SAMPLE_RATE} Hz")
    if wave.dtype.kind == "i":
        wave = wave / float(np.iinfo(wave.dtype).max)
    if wave.ndim > 1:
        wave = wave.mean(1)
    out = np.zeros(N_SAMPLES, dtype=np.float32)
    n = min(len(wave), N_SAMPLES)
    out[:n] = wave[:n]
    return out


class NotePairs(torch.utils.data.Dataset):
    """Pairs of notes from one folder (searched recursively for .wav files), each item a
    ``(2, N_SAMPLES)`` tensor. With ``fixed=False`` (training), items are indexed by a pair
    of note indices, drawn by :class:`PairSampler`; with ``fixed=True`` (test), the notes are
    paired once from ``seed`` and item ``i`` is the ``i``-th pair (``len // 2`` items).
    """

    def __init__(self, folder, fixed=False, seed=0):
        self.files = sorted(Path(folder).rglob("*.wav"))
        if len(self.files) < 2:
            raise FileNotFoundError(
                f"no notes in {folder}; run `python -m src.datasets.synthetic_notes` first")
        self.fixed = fixed
        if fixed:
            order = np.random.default_rng(seed).permutation(len(self.files))
            self.pairs = order[: len(order) // 2 * 2].reshape(-1, 2)

    def __len__(self):
        return len(self.pairs) if self.fixed else len(self.files)

    def __getitem__(self, i):
        if self.fixed:
            a, b = self.pairs[i]
        else:
            a, b = i  # a pair of note indices from PairSampler
        return torch.from_numpy(np.stack([load_note(self.files[a]), load_note(self.files[b])]))

    def name(self, i):
        """File names of the notes of fixed pair ``i``."""
        return [self.files[j].stem for j in self.pairs[i]]


class PairSampler(torch.utils.data.Sampler):
    """Every note once per epoch, in random order, each with a random partner (never itself).

    The pairs are drawn in the main process from the sampler's own ``generator``, so they do
    not depend on the data-loading workers, and saving and restoring the generator's state
    (train.py --resume) continues the same sequence of training mixtures.
    """

    def __init__(self, num_notes, seed=0):
        self.num_notes = num_notes
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.num_notes

    def __iter__(self):
        order = torch.randperm(self.num_notes, generator=self.generator)
        partner = torch.randint(self.num_notes - 1, (self.num_notes,), generator=self.generator)
        partner += partner >= order  # skip the note itself
        return iter(zip(order.tolist(), partner.tolist()))


def get_loaders(directory, batch_size, kwargs, seed=0, train_seed=1):
    """Training loader (random pairs, ``train_seed``) on ``directory/train`` and validation
    loader (pairs fixed by ``seed``) on ``directory/valid``, to follow training. The test notes
    (``directory/test``) are left for evaluate.py."""
    train_set = NotePairs(Path(directory) / "train")
    if not (Path(directory) / "valid").is_dir():
        raise FileNotFoundError(
            f"no validation notes in {Path(directory) / 'valid'}; run "
            "`python -m src.datasets.synthetic_notes` (or, for your own notes, move some "
            "training notes there)")
    test_set = NotePairs(Path(directory) / "valid", fixed=True, seed=seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=PairSampler(len(train_set), train_seed),
        drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              **kwargs)
    return train_loader, test_loader
