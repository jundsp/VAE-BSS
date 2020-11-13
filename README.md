# vae-bss
 
 Authors: Julian Neri, Roland Badeau, Philippe Depalle

### Instructions

1. Open the unix shell (terminal for mac users)
2. Change working directory to 'vae-bss-master'
3. Execute the command: ./run.sh

### Summary
A virtual environment will be created in the directory 'env'. Required packages are then installed and 'evaluate.py' is run.
MNIST data will be downloaded and mixed.
Pre-trained (unsupervised) VAE and AE models then separate the mixed data.
Results for K = 2, 3, 4 assumed sources are saved in the results directory.
