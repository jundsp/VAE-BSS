#!/bin/bash

python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip

pip3 install -r requirements.txt

echo "Running Python script"

echo "Training the VAE model on MNIST data"
python train.py

echo "Evaluating the pre-trained model"
python evaluate.py

echo "Done!"