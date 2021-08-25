# vae-bss
  
Evaluates our pre-trained variational auto-encoder for unsupervised blind source separation, as described in

J. Neri, R. Badeau, P. Depalle, “**Unsupervised Blind Source Separation with Variational Auto-Encoders**”, 29th European Signal Processing Conference (EUSIPCO), Dublin, Ireland, August 2021.
  
 Authors: Julian Neri, Roland Badeau, Philippe Depalle

## Instructions

### Quick Start

To evaluate the VAE and AE models, perform the following steps.

1. Open the unix shell (terminal for mac users)
2. Change working directory to 'vae-bss-master'
3. Execute the shell script ./run.sh

A virtual environment will be created in the directory 'env'. Required packages are then installed. Then, the script 'evaluate.py' is run, which downloads the MNIST data and prepares it into training and validation sets.
A pre-trained (unsupervised) variational auto-encoder (see model.py)  is used to separate the mixed data.
Results for K = 2, 3, 4 assumed sources are saved in the results directory.

If you instead enter "./run.sh train" it will train a new model using the MNIST data and save the results and a trained model every epoch.

### Usage

If you already meet the package requirements listed in "requirements.txt", then you can simply run "evaluate.py" or "train.py".
The "train.py" file trains the VAE model on the MNIST data, demonstrating the training procedure described in our paper.
The number of epochs for training / testing is defined in "argparser.py". The other arguments defined in that file allow you to try different hyperparameters and settings for model training / specification, such as the dimension of the latent space, the prior probability, and the batch size.

