# Introduction

This repository contains my tests to use the huggingface transformers library.

# How to setup this repository

This setups a working venv (supposing you have CUDA 11.0 installed)

```python
./setup-venv.sh
source venv/bin/activate
```

Once the venv is setup, you just need

```python
source venv/bin/activate
```

# How to use this repository



```bash
cd datasets
# Preprocess the dataset
python lambada-gpt2.py
cd ..
cd gpt2
# Finetune GPT-2 with the dataset
python finetune.py
# Use the different sampling methods to produce text with the fine-tune model
python use-finetuned.py
```
