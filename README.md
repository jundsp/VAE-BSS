# vae-bss
  
Evaluates our pre-trained variational auto-encoder for unsupervised blind source separation, as described in

J. Neri, R. Badeau, P. Depalle, “**Unsupervised Blind Source Separation with Variational Auto-Encoders**”, 29th European Signal Processing Conference (EUSIPCO), Dublin, Ireland, August 2021.
  
 Authors: Julian Neri, Roland Badeau, Philippe Depalle

### Instructions

1. Open the unix shell (terminal for mac users)
2. Change working directory to 'vae-bss-master'
3. Execute the command: ./run.sh

### Summary
A virtual environment will be created in the directory 'env'. Required packages are then installed. Then, the script 'evaluate.py' is run, which downloads the MNIST data and prepares it into training and validation sets.
A pre-trained (unsupervised) variational auto-encoder (see model.py)  is used to separate the mixed data.
Results for K = 2, 3, 4 assumed sources are saved in the results directory.
