#!/bin/bash

python3 -m venv env
source env/bin/activate
pip install --upgrade pip

pip3 install torch
pip3 install numpy
pip3 install matplotlib

echo "Running Python script"
python train.py