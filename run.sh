#!/bin/bash

python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip

pip3 install torch
pip3 install torchvision
pip3 install numpy
pip3 install matplotlib
pip3 install scipy

echo "Running Python script"
python evaluate.py

echo "Done!"