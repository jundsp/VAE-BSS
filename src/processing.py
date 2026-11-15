"""Audio processing of the VAE paper: fixed-size magnitude spectrograms of note mixtures.

Settings follow the EUSIPCO 2021 paper (Sec. V-A) and the dissertation (Sec. 6.2.6):

* 44.1 kHz audio, 2048-point STFT with a Hann window and a hop of 512 samples;
* only the first 256 frequency bins (0 to 5.5 kHz) and the first 128 frames (about 1.5 s);
* two sources are mixed as complex STFTs, the mixture's magnitude spectrogram is normalized
  to the range 0-1 and flattened into a 256 x 128 = 32768-dimensional input;
* separated audio is synthesized from an estimated magnitude with the mixture's phase and the
  inverse STFT, then scored with SI-SDR, SIR and SAR (see metrics.py).

Because the model sees magnitudes, the mixture is not exactly the sum of its sources: phase
differences between the sources change the mixture's magnitude (paper, Sec. V-A).
"""

import numpy as np
import torch

SAMPLE_RATE = 44100
N_FFT = 2048
HOP = 512
N_BINS = 256      # first 256 of the 1025 bins: up to 255 * 44100 / 2048 = 5.49 kHz
N_FRAMES = 128    # first 128 frames
# With a centred STFT, this many samples give exactly N_FRAMES frames (about 1.48 s).
N_SAMPLES = (N_FRAMES - 1) * HOP
DIMX = N_BINS * N_FRAMES

EPS = 1e-8


def stft(waves):
    """Cropped complex STFT, ``(..., N_SAMPLES)`` -> ``(..., N_BINS, N_FRAMES)``."""
    shape = waves.shape[:-1]
    window = torch.hann_window(N_FFT, device=waves.device)
    spec = torch.stft(waves.reshape(-1, waves.shape[-1]), N_FFT, HOP, window=window,
                      center=True, return_complex=True)
    return spec[:, :N_BINS, :N_FRAMES].reshape(*shape, N_BINS, N_FRAMES)


def istft(spec):
    """Waveform of a cropped STFT, with the discarded bins above 5.5 kHz set to zero."""
    shape = spec.shape[:-2]
    full = torch.zeros(int(np.prod(shape)), N_FFT // 2 + 1, N_FRAMES, dtype=spec.dtype,
                       device=spec.device)
    full[:, :N_BINS] = spec.reshape(-1, N_BINS, N_FRAMES)
    window = torch.hann_window(N_FFT, device=spec.device)
    waves = torch.istft(full, N_FFT, HOP, window=window, center=True, length=N_SAMPLES)
    return waves.reshape(*shape, N_SAMPLES)


def mix_audio(sources):
    """Mix source waveforms and build the model's input, like ``mix_data`` does for MNIST.

    Args:
        sources: ``(B, M, N_SAMPLES)`` waveforms of the M sources of each mixture.

    Returns:
        x: ``(B, DIMX)`` mixture magnitude spectrograms, each normalized to a maximum of 1.
        s: ``(B, M, N_BINS, N_FRAMES)`` source magnitudes on the same scale as ``x``
           (ground truth, used only for evaluation and plots, never for training).
        mix_spec: ``(B, N_BINS, N_FRAMES)`` complex STFT of the mixture, unnormalized; its
           phase and ``norm`` turn estimated magnitudes back into audio.
        src_spec: ``(B, M, N_BINS, N_FRAMES)`` complex STFTs of the sources.
        norm: ``(B, 1, 1)`` normalization constant of each mixture.
    """
    src_spec = stft(sources)
    mix_spec = src_spec.sum(1)
    mag = mix_spec.abs()
    norm = mag.amax(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    x = (mag / norm).reshape(-1, DIMX)
    s = src_spec.abs() / norm.unsqueeze(1)
    return x, s, mix_spec, src_spec, norm


def to_audio(mags, mix_spec, norm):
    """Waveforms from estimated magnitudes ``(B, K, N_BINS, N_FRAMES)`` with the mixture phase."""
    phase = torch.exp(1j * torch.angle(mix_spec)).unsqueeze(1)
    return istft(mags * norm.unsqueeze(1) * phase)


def ideal_masks(src_spec, mix_spec):
    """Ideal binary and ratio masks (upper bounds), as magnitudes on the mixture's scale."""
    mags = src_spec.abs()
    irm = mags / mags.sum(1, keepdim=True).clamp_min(EPS)
    ibm = (mags == mags.amax(1, keepdim=True)).to(mags.dtype)
    mix = mix_spec.abs().unsqueeze(1)
    return ibm * mix, irm * mix
