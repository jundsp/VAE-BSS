# VAE-BSS

> **New: [DVAE-BSS](https://github.com/jundsp/DVAE-BSS)** adds a dynamical prior, so it separates audio of any length, with a pretrained model and a one-command `dvae-separate`. [Listen to examples](https://jundsp.github.io/DVAE-BSS/).
  
A PyTorch model of the variational auto-encoder for unsupervised blind source separation, with training and evaluation programs, as described in

* J. Neri, R. Badeau, P. Depalle, “[**Unsupervised Blind Source Separation with Variational Auto-Encoders**](https://hal.science/hal-03255341)”, 29th European Signal Processing Conference (EUSIPCO), Dublin, Ireland, August 2021. [doi:10.23919/EUSIPCO54536.2021.9616154](https://doi.org/10.23919/EUSIPCO54536.2021.9616154)

Includes pre-trained models located in `saves/pretrained` (~75 MB total).

Author: Julian Neri

<img src="docs/imgs/vae-bss-diagram.png" alt="Diagram" title="Diagram" width="700" />


## Instructions

### Installation

```bash
git clone https://github.com/jundsp/VAE-BSS.git
cd VAE-BSS
pip install -r requirements.txt
```

### Usage

Run either the `evaluate.py` or `train.py` files.
* For example, enter this command in terminal: `python3 evaluate.py`

**`evaluate.py`** uses models that we pre-trained in an unsupervised way to separate mixed MNIST handwritten digit images.
Results for K = 2, 3, 4 assumed model sources are saved in the results directory. The models include normal auto-encoders, variational auto-encoders, and the VAE with masking, to compare their qualities. The VAEs, with or without masking, perform high quality separation without knowing the true number of source images in the mixture and without ever having access to the true ground truth source images. With `--checkpoint saves/<model>.pt`, it instead scores a model trained by `train.py` on its dataset's test set.

**`train.py`** starts from scratch and trains a VAE model solely on data mixtures in an unsupervised way. It demonstrates the training procedure described in our paper. `--dataset` picks the data (`mnist`, the default, or `audio_synthetic`, see [Audio](#audio)), and options that are not given take that dataset's settings from the paper (see `python train.py --help` and `defaults` in `src/datasets/`). For MNIST, separation is clear within the first 10 epochs (the default). You should get about the same results as the pre-trained models if you train over a few thousand epochs, following the training settings defined in our paper. The other arguments allow you to try different hyperparameters and settings for model training / specification, such as the number of model sources (`--sources`), the dimension of the latent space, the prior probability, the batch size, or a deterministic auto-encoder (`--variational False`). MNIST models are saved to `saves/model_<vae|ae>_K<sources>.pt` and figures to `results/`; the pretrained models in `saves/pretrained` are never overwritten. `--resume` continues a run from its last save, with the optimizer, learning-rate schedule and random state saved next to the model (`<model>.train.pt`), so a resumed run gives the same results as an uninterrupted one; raise `--epochs` to train longer. With the cosine schedule (the audio default), a resumed run with a new `--epochs` starts a new cosine from the current learning rate down to `--min-lr` at the new last epoch; a run that already reached `--min-lr` stays there unless `--learning-rate` sets a new starting rate.

Either program will first download the MNIST data and prepare it as training and testing datasets. The datasets are saved in a directory that will be recalled for subsequent use.

### Audio

The audio version separates mixtures of two musical notes from their spectrograms, as in the paper's MUMS experiment (Section V and Table III; Section 6.2.6 of the [dissertation](https://hdl.handle.net/20.500.14905/94099)). MUMS is licensed, so this repository trains on the synthetic notes of [DVAE-BSS](https://github.com/jundsp/DVAE-BSS) instead: plucked and bowed strings, reeds, brass, bells and organs with random pitch, timbre, vibrato and decay. There is no pretrained audio model.

```bash
python -m src.datasets.synthetic_notes      # 3000 training, 750 validation and 750 test notes in data/audio_synthetic (500 MB, ~1 min)
python train.py --dataset audio_synthetic    # 1000 epochs, about an hour on an Apple-silicon GPU
python evaluate.py --checkpoint saves/audio_synthetic_c0.5_vae_K2.pt
```

<img src="docs/imgs/audio_training.gif" alt="Separation of synthetic note mixtures over training" title="Separation of synthetic note mixtures over training" width="800" />

Training, epochs 10 to 1000: four validation mixtures (rows), each with the mixture, the model's reconstruction, its two decoded sources (VAE), the mask-based sources (VAEM) and the true notes. Made with `python -m src.animate results/audio_synthetic_c0.5_vae_K2 -o docs/imgs/audio_training.gif`.

| 1000 epochs, test set (dB) | VAE | VAEM | VAEW | IBM | IRM |
|---|---|---|---|---|---|
| SI-SDR | 8.06 | 14.01 | 13.92 | 23.72 | 21.80 |
| SIR | 31.84 | 32.88 | 21.63 | 46.34 | 33.58 |
| SAR | 8.48 | 15.26 | 16.86 | 23.80 | 22.14 |

Medians over the 750 held-out test notes (375 fixed mixtures), which played no part in training or in choosing the settings; IBM and IRM are the ideal binary and ratio masks (upper bounds). The model is exactly what the commands above give, with the default settings below and seed 1; on the validation notes it scores 13.73 dB VAEM.

Listen to three test mixtures (the model's mask-based separations, VAEM, next to the true notes; each folder also has `spectrograms.png`):

| Test mixture | Mixture | Separated | True notes | VAEM SI-SDR (dB) |
|---|---|---|---|---|
| brass + plucked string | [mixture](docs/audio/example_1/mixture.wav) | [1](docs/audio/example_1/vaem_1.wav), [2](docs/audio/example_1/vaem_2.wav) | [1](docs/audio/example_1/true_1.wav), [2](docs/audio/example_1/true_2.wav) | 26.8, 16.4 |
| two bells | [mixture](docs/audio/example_2/mixture.wav) | [1](docs/audio/example_2/vaem_1.wav), [2](docs/audio/example_2/vaem_2.wav) | [1](docs/audio/example_2/true_1.wav), [2](docs/audio/example_2/true_2.wav) | 8.2, 4.9 |
| two reeds | [mixture](docs/audio/example_3/mixture.wav) | [1](docs/audio/example_3/vaem_1.wav), [2](docs/audio/example_3/vaem_2.wav) | [1](docs/audio/example_3/true_1.wav), [2](docs/audio/example_3/true_2.wav) | 29.6, 32.8 |

The two bells are a hard case. Over the whole test set, the median VAEM SI-SDR per family ranges from 10.0 dB (organs) and 12.2 dB (plucked strings) to 19.6 dB (reeds).

The notes are in three splits: the model trains on `train/`, is scored on `valid/` while training (the figures, `progress.png` and the settings below were chosen from these scores), and `evaluate.py` scores `test/`.

Processing follows the paper, with the two exceptions listed after:

* notes at 44.1 kHz; 2048-point STFT, Hann window, hop of 512 samples;
* the first 256 frequency bins (up to 5.5 kHz) and the first 128 frames (about 1.5 s) of each note;
* two notes picked at random are mixed (complex STFTs added), anew every epoch; the mixture's magnitude spectrogram is normalized to 0-1 and flattened into a 32768-dimensional input;
* a 64-dimensional latent variable per source, sigmoid outputs; Laplace likelihood with scale √0.5 and Gaussian prior; the paper's encoder layers have 2560, 2048, 1536, 1024 and 512 units, but the default here is half of each (89 M instead of 189 M parameters, almost all in the first encoder and last decoder layers); use `--hidden 2560 2048 1536 1024 512` for the paper's size;
* Adam, batch of 128 mixtures, β rising from 0 to 0.5 over the first 100 epochs;
* separated audio is synthesized from each estimated magnitude with the mixture's phase and the inverse STFT.

Three defaults differ from the paper, because the paper's settings did not separate these notes within 300 epochs:

* learning rate 1e-3 (the paper: 1e-4, `--learning-rate 1e-4`);
* a cosine schedule from that rate down to 1e-4 at the last epoch, 1000 by default (`--scheduler cosine --min-lr 1e-4`); the paper decayed the rate by 0.01 % per epoch (`--scheduler step --decay 0.9999`), which barely lowers it within a few thousand epochs;
* magnitudes compressed to |STFT|^0.5 before the 0-1 normalization, as the dissertation does for the DVAE (Section 6.3.7); the model, its masks (VAEM) and its loss then work in the compressed domain, and synthesis undoes the compression before the inverse STFT. The paper's VAE uses no compression (`--compression 1`).

Median validation SI-SDR of VAEM after 300 epochs: 0.9 dB with the paper's settings (stopped at epoch 120, no separation), 5.3 dB with learning rate 1e-3, and 11.3 dB with learning rate 1e-3 and compression 0.5 (14.0 dB on the test notes after 1000 epochs with all three, the table above).

The compression is saved next to the model (`saves/<name>.json`) for `evaluate.py`, and runs are named after it, e.g. `audio_synthetic_c0.5_vae_K2` (no `c` part without compression). With compression, the masked sources (VAEM) no longer add up to the mixture after decompression (the masks become m^(1/c)), so `evaluate.py` also reports VAEW: the ratio masks from the compressed estimates applied to the uncompressed mixture STFT, a generalized Wiener filter (Liutkus and Badeau, 2015) whose sources add up exactly to the mixture.

Training saves the model to `saves/audio_synthetic_c0.5_vae_K2.pt` and, every 10 epochs, to `results/audio_synthetic_c0.5_vae_K2/`: `reconstruction_<epoch>.png` (validation mixtures, reconstruction, separated and true spectrograms), `progress.png` (losses and median validation SI-SDR of VAE, VAEM and VAEW), `losses.png` and `log.csv`. `evaluate.py` reports median SI-SDR, SIR and SAR over the test sources for the VAE, the VAE with masking (VAEM), the Wiener-filtered VAE (VAEW) and the ideal binary and ratio masks (upper bounds), like Table III, and saves .wav examples. SIR and SAR are the scale-invariant versions (Le Roux et al., 2019) rather than bss_eval's, so only SI-SDR is directly comparable to the paper. The paper's MUMS results are not reproduced here: the data differ.

The default model has about 89 M parameters (a 0.36 GB checkpoint), so a GPU is recommended; Apple-silicon GPUs (MPS) are used when available. To train on your own notes (e.g. MUMS), put mono 44.1 kHz .wav files under `train/`, `valid/` and `test/` in one folder and pass it with `--data-directory`.

### Code layout

* `train.py`, `evaluate.py`: shared by all datasets.
* `src/model.py`: the VAE and its loss; `src/utils.py`: MNIST mixing, KL divergences, masks, permutation.
* `src/processing.py`: STFT processing of the audio version; `src/metrics.py`: SI-SDR, SIR, SAR.
* `src/datasets/`: one module per dataset (`mnist.py`, `audio_synthetic.py`) with its loaders, mixing, default settings, scores and figures; `notes.py` loads pairs of notes; `synthetic_notes.py` generates the notes.

<img src="docs/imgs/MNIST_K3_25.png" alt="MNIST K=3" title="MNIST K=3" width="600" />
