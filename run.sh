#!/bin/bash

printf "
Unsupervised Blind Source Separation with Variational Auto-Encoders
Author: Julian Neri
email: julian.neri@mcgill.ca \n
"

printf "Creating virtual environment and installing packages ... \n "



python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip

pip3 install -r requirements.txt

echo "Running Python script"

if [ "$1" == 'train' ]
then
    echo "Training the VAE model on MNIST data"
    python train.py
else
    echo "Evaluating the pre-trained model"
    python evaluate.py
fi

echo "Done!"
