"""Datasets of random note pairs for the audio version (the synthetic notes or any folder of
mono .wav notes, such as MUMS).

As in the paper, sources are mixed at random: every epoch pairs each training note with a new
random partner, so the number of distinct mixtures grows with the square of the number of
notes. The test pairs are fixed by a seed, so test scores are comparable between runs.
Training and test notes are never mixed with one another.
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
    """Pairs of notes from one folder (searched recursively for .wav files).

    Item ``i`` is note ``i`` and a partner note, stacked as a ``(2, N_SAMPLES)`` tensor.
    With ``fixed=False`` the partner is drawn at random on every access (training); with
    ``fixed=True`` the notes are paired once from ``seed`` and there are ``len // 2`` items.
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
            # Torch's generator is seeded differently in every worker and epoch.
            a = i
            b = int(torch.randint(len(self.files) - 1, ()))
            b += b >= a  # never the same note twice
        return torch.from_numpy(np.stack([load_note(self.files[a]), load_note(self.files[b])]))

    def name(self, i):
        """File names of the notes of fixed pair ``i``."""
        return [self.files[j].stem for j in self.pairs[i]]


def get_loaders(directory, batch_size, kwargs, seed=0):
    """Training loader (random pairs) on ``directory/train`` and test loader (fixed pairs)."""
    train_set = NotePairs(Path(directory) / "train")
    test_set = NotePairs(Path(directory) / "test", fixed=True, seed=seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              **kwargs)
    return train_loader, test_loader
