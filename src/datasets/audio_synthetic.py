"""Mixed pairs of synthetic notes as fixed-size magnitude spectrograms (paper, Sec. V-A;
dissertation, Sec. 6.2.6), for train.py and evaluate.py.

The notes come from synthetic_notes.py. Any folder of 44.1 kHz notes with train/ and test/
subfolders (e.g. MUMS, as in the paper) works with ``--data-directory``. The interface shared
by all datasets is described in src/datasets/__init__.py.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile

from ..utils import vae_masks
from ..metrics import bss_measures, match_sources
from ..processing import (DIMX, N_BINS, N_FRAMES, SAMPLE_RATE, ideal_masks, istft, mix_audio,
                         to_audio, wiener_audio)
from .notes import NotePairs, get_loaders

name = 'audio_synthetic'
dimx = DIMX  # 256 bins x 128 frames

# Settings of the paper's spectrogram experiment.
defaults = dict(
    data_directory='data/audio_synthetic',
    dimz=64,
    # Half the paper's hidden units (2560, 2048, 1536, 1024, 512), which gives 85 M instead
    # of 190 M parameters: most are in the first encoder and last decoder layers. Use
    # --hidden 2560 2048 1536 1024 512 for the paper's size.
    hidden=[1280, 1024, 768, 512, 256],
    batch_size=128,
    learning_rate=1e-4,
    decay=0.9999,      # 0.01 % per epoch
    beta_max=0.5,
    warm_up=100,       # epochs for beta to grow from 0 to 0.5
    epochs=5000,       # the paper trained until the loss converged, about 5000 epochs
    save_interval=10,
    plot_interval=10,
    eval_interval=10,  # median test SI-SDR of VAE and VAEM
    compression=1.0,   # |STFT|**c; the paper's VAE uses none, the dissertation's DVAE c = 0.5
)
use_mps = True         # Apple-silicon GPUs help with this large model
# Settings that change the processing, so evaluate.py needs them: saved next to the model.
config_keys = ('compression',)


def paths(args):
    """Results folder and checkpoint of a training run (never in saves/pretrained)."""
    c = '' if args.compression == 1 else f'c{args.compression:g}_'
    name = f"audio_synthetic_{c}{'vae' if args.variational else 'ae'}_K{args.sources}"
    return f'results/{name}', f'saves/{name}.pt'


def loaders(args, device):
    kwargs = {'num_workers': args.num_workers, 'persistent_workers': args.num_workers > 0}
    if device.type == 'cuda':
        kwargs['pin_memory'] = True
    return get_loaders(args.data_directory, args.batch_size, kwargs)


def prepare(batch, device, args):
    """Mix a batch of note pairs: the model's input ``x`` ``(B, DIMX)`` and the extras needed
    to score and plot it (true sources, mixture STFT, normalization and compression)."""
    x, s, mix_spec, src_spec, norm = mix_audio(batch.to(device), args.compression)
    return x, dict(s=s, mix_spec=mix_spec, src_spec=src_spec, norm=norm,
                   compression=args.compression)


def _sources(x, recons, n_sources):
    s_vae = recons.view(-1, n_sources, N_BINS, N_FRAMES)
    s_vaem = vae_masks(s_vae, x.view(-1, 1, N_BINS, N_FRAMES))
    return s_vae, s_vaem


def scores(x, recons, extra):
    """Median-able SI-SDR (dB) of every true source, for the VAE and VAEM estimates, and
    with compression also the Wiener-filtered estimates (VAEW, see processing.wiener_audio)."""
    s_vae, s_vaem = _sources(x, recons, recons.shape[1])
    refs = istft(extra['src_spec'])  # true sources, band-limited to 5.5 kHz like the model
    c = extra['compression']
    estimates = {'VAE': to_audio(s_vae, extra['mix_spec'], extra['norm'], c),
                 'VAEM': to_audio(s_vaem, extra['mix_spec'], extra['norm'], c)}
    if c != 1:  # without compression, VAEW is the same as VAEM
        estimates['VAEW'] = wiener_audio(s_vae, extra['mix_spec'])
    out = {}
    for label, est in estimates.items():
        est, _ = match_sources(est, refs)
        out[f'{label} SI-SDR'] = bss_measures(est, refs)[0].flatten()
    return out


def _show(ax, mag, title=None, compression=1.0):
    """Spectrogram in dB of the linear magnitude: compressed magnitudes (``|X|**c``, the
    model's domain) are decompressed first, so plots compare across compression settings."""
    mag = np.maximum(np.asarray(mag).reshape(N_BINS, N_FRAMES), 0) ** (1.0 / compression)
    db = 20 * np.log10(np.maximum(mag, 1e-4))
    ax.imshow(db, origin='lower', aspect='auto', vmin=-60, vmax=0, cmap='magma')
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def figure(path, x, recon_x, recons, extra, n=4):
    """Spectrograms in dB of linear magnitude (decompressed if the model is compressed).
    Rows: test mixtures. Columns: mixture, reconstruction (sum of the K decoded sources),
    the K decoded sources (VAE), the K mask-based sources (VAEM: each decoded source over
    their sum, a mask that sums to 1 over the sources, times the mixture) and the 2 true
    sources. Sources are ordered to match the true ones; extra model sources come last."""
    n = min(n, len(x))
    x, recon_x, recons = x[:n], recon_x[:n], recons[:n]
    k = recons.shape[1]
    s = extra['s'][:n]
    s_vae, s_vaem = _sources(x, recons, k)
    # Match on the VAE sources, so both rows share one order.
    _, index = match_sources(s_vae, s)
    order = torch.stack([torch.cat([i, torch.tensor([j for j in range(k) if j not in i],
                                                    dtype=i.dtype, device=i.device)])
                         for i in index])
    pick = order.view(n, k, 1, 1).expand(-1, -1, N_BINS, N_FRAMES)
    s_vae, s_vaem = s_vae.gather(1, pick), s_vaem.gather(1, pick)
    cols = [('mixture', x), ('reconstruction', recon_x)]
    cols += [(f'VAE {j + 1}', s_vae[:, j]) for j in range(k)]
    cols += [(f'VAEM {j + 1}', s_vaem[:, j]) for j in range(k)]
    cols += [(f'true {j + 1}', s[:, j]) for j in range(s.shape[1])]
    fig, axes = plt.subplots(n, len(cols), figsize=(2 * len(cols), 1.8 * n), squeeze=False)
    for r in range(n):
        for c, (title, spec) in enumerate(cols):
            _show(axes[r, c], spec[r].cpu().numpy(), title if r == 0 else None,
                  extra['compression'])
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _write_wav(path, wave, peak):
    wavfile.write(path, SAMPLE_RATE, np.round(32767 * wave / peak).astype(np.int16))


def _save_example(folder, names, mixture, rows, audio, compression=1.0):
    """Spectrograms and .wav files of one test mixture."""
    os.makedirs(folder, exist_ok=True)
    peak = max(1e-9, max(float(np.abs(w).max()) for w in audio.values()))
    for label, wave in audio.items():
        _write_wav(os.path.join(folder, f'{label}.wav'), wave, peak)
    with open(os.path.join(folder, 'notes.txt'), 'w') as f:
        f.write('\n'.join(names) + '\n')
    ncols = 1 + max(len(specs) for _, specs in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(2.2 * ncols, 2 * len(rows)),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    _show(axes[0, 0], mixture, 'mixture', compression)
    for r, (label, specs) in enumerate(rows):
        for j, spec in enumerate(specs):
            _show(axes[r, j + 1], spec, f'{label} {j + 1}', compression)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'spectrograms.png'), dpi=100)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, args, out, device, examples=10, batch_size=64):
    """Table III of the paper: median SI-SDR, SIR and SAR of VAE, VAEM, IBM and IRM over the
    test mixtures. Writes ``scores.csv`` and audio examples to ``out``; returns the medians."""
    test_set = NotePairs(os.path.join(args.data_directory, 'test'), fixed=True, seed=0)
    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    k = args.sources
    methods = ('VAE', 'VAEM', 'IBM', 'IRM')
    if args.compression != 1:  # Wiener masks differ from VAEM only with compression
        methods = ('VAE', 'VAEM', 'VAEW', 'IBM', 'IRM')
    measures = {m: [] for m in methods}
    records = []
    index = 0
    for batch in loader:
        x, extra = prepare(batch, device, args)
        mix_spec, src_spec, norm = extra['mix_spec'], extra['src_spec'], extra['norm']
        recons = model(x)[3]
        s_vae, s_vaem = _sources(x, recons, k)
        ibm, irm = ideal_masks(src_spec, mix_spec)
        phase = torch.exp(1j * torch.angle(mix_spec)).unsqueeze(1)

        refs = istft(src_spec)
        c = args.compression  # the masks (VAEM) are applied in the compressed domain
        estimates = {'VAE': to_audio(s_vae, mix_spec, norm, c),
                     'VAEM': to_audio(s_vaem, mix_spec, norm, c),
                     'IBM': istft(ibm * phase), 'IRM': istft(irm * phase)}
        if 'VAEW' in methods:
            estimates['VAEW'] = wiener_audio(s_vae, mix_spec)
        for m, est in estimates.items():
            matched, order = match_sources(est, refs)
            measures[m].append(torch.stack(bss_measures(matched, refs), -1).cpu())
            if m == 'VAEM':
                vaem_order = order.cpu()

        for b in range(len(x)):
            names = test_set.name(index)
            for j in range(2):
                rec = {'mixture': index + 1, 'note': names[j]}
                for m in methods:
                    values = measures[m][-1][b, j]
                    rec.update({f'{m}_{q}': round(float(v), 3)
                                for q, v in zip(('si_sdr', 'sir', 'sar'), values)})
                records.append(rec)
            if index < examples:
                # The sources matched to the true ones first, then any extra model sources.
                order = vaem_order[b].tolist()
                order += [j for j in range(k) if j not in order]
                audio = {'mixture': istft(mix_spec[b]).cpu().numpy()}
                audio.update({f'true_{j + 1}': refs[b, j].cpu().numpy() for j in range(2)})
                for j, src in enumerate(order):
                    audio[f'vae_{j + 1}'] = estimates['VAE'][b, src].cpu().numpy()
                    audio[f'vaem_{j + 1}'] = estimates['VAEM'][b, src].cpu().numpy()
                    if 'VAEW' in estimates:
                        audio[f'vaew_{j + 1}'] = estimates['VAEW'][b, src].cpu().numpy()
                rows = [('VAE', s_vae[b, order].cpu().numpy()),
                        ('VAEM', s_vaem[b, order].cpu().numpy()),
                        ('true', extra['s'][b].cpu().numpy())]
                _save_example(os.path.join(out, f'example_{index + 1:02d}'), names,
                              x[b].cpu().numpy(), rows, audio, args.compression)
            index += 1

    with open(os.path.join(out, 'scores.csv'), 'w') as f:
        f.write(','.join(records[0]) + '\n')
        f.writelines(','.join(str(v) for v in r.values()) + '\n' for r in records)

    table = {m: torch.cat(measures[m]).flatten(0, 1).median(0).values.tolist() for m in methods}
    print(f'\nMedian over {len(records)} test sources (dB):')
    print(f'{"":8s}' + ''.join(f'{m:>8s}' for m in methods))
    for i, q in enumerate(('SI-SDR', 'SIR', 'SAR')):
        print(f'{q:8s}' + ''.join(f'{table[m][i]:8.2f}' for m in methods))
    print(f'\nScores and {min(examples, len(test_set))} audio examples saved in {out}')
    return table
