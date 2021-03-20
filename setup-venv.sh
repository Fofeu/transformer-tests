#!/bin/bash
git submodule update --init --recursive

virtualenv3 venv

source venv/bin/activate

echo "Assuming cuda 11.0, see pytorch website to change accordingly"

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html



pip install transformers datasets

cd apex

pip instal --requirements=./requirements.txt

echo "Installing apex without C++"

pip install -v --disable-pip-version-check --no-cache-dir ./

cd ..
