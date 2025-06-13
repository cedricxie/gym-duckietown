#!/bin/bash
set -e

# Env Setup.
apt install python3.8-venv
python3.8 -m venv ".venv"
source .venv/bin/activate

echo "##############################"
python --version
echo "##############################"

sudo apt-get install freeglut3-dev

pip3 install -e .

pip install torch==1.7.0 matplotlib==3.2.0 numpy==1.21.4 pyglet==1.5.15 tqdm
