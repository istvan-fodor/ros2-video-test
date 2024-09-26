#!/bin/bash

deactivate
python -m venv venv
source ./venv/bin/activate

pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1