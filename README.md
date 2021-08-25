# VAE-BSS
  
A PyTorch model of the variational auto-encoder for unsupervised blind source separation, with training and evaluation programs, as described in

* J. Neri, R. Badeau, P. Depalle, “[**Unsupervised Blind Source Separation with Variational Auto-Encoders**](https://www.music.mcgill.ca/~julian/wp-content/uploads/2021/06/2021_eusipco_vae_bss_neri.pdf)”, 29th European Signal Processing Conference (EUSIPCO), Dublin, Ireland, August 2021.

Includes pre-trained models located in `saves/pretrained` (~80mb total).

Author: Julian Neri  
Webpage: [music.mcgill.ca/~julian/vae-bss](https://www.music.mcgill.ca/~julian/vae-bss)

## Instructions

### Installation

Install the python packages listed in the `requirements.txt` file in the main directory of this repository.
This can be done manually or with the following steps:

1. Open the unix shell (terminal for mac users)
2. Change working directory to "vae-bss-master"
3. Execute this command: `pip install -r requirements.txt`

### Usage

Run either the `evaluate.py` or `train.py` files.
* For example, enter this command in terminal: `python3 evaluate.py`

**`evaluate.py`** uses models that we pre-trained in an unsupervised way to separate mixed MNIST handwritten digit images.
Results for K = 2, 3, 4 assumed model sources are saved in the results directory. The models include normal auto-encoders, variational auto-encoders, and the VAE with masking, to compare their qualities. The VAEs, with or without masking, perform high quality separation without knowing the true number of source images in the mixture and without ever having access to the true ground truth source images. 

**`train.py`** starts from scratch and trains a VAE model solely on data mixtures in an unsupervised way. It demonstrates the training procedure described in our paper. The number of epochs for training / testing is defined in `argparser.py`. Separation is clear within the first 10 epochs. You should get about the same results as the pre-trained models if you train over a few thousand epochs, following the training settings defined in our paper. The other arguments allow you to try different hyperparameters and settings for model training / specification, such as the dimension of the latent space, the prior probability, and the batch size.

Either program will first download the MNIST data and prepare it as training and testing datasets. The datasets are saved in a directory that will be recalled for subsequent use.
