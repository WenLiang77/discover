#!/bin/bash

set -e

echo "========================================"
echo "Setting up university GPU environment"
echo "========================================"

echo "Current directory: $(pwd)"
echo "Python before setup:"
python --version || true

echo ""
echo "========================================"
echo "Creating virtual environment"
echo "========================================"

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate

echo "Python after activating venv:"
python --version
which python

echo ""
echo "========================================"
echo "Upgrading pip"
echo "========================================"

python -m pip install --upgrade pip setuptools wheel

echo ""
echo "========================================"
echo "Installing base packages"
echo "========================================"

python -m pip install \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    tqdm \
    requests \
    packaging

echo ""
echo "========================================"
echo "Installing LLM / LoRA packages"
echo "========================================"

python -m pip install \
    torch \
    transformers \
    accelerate \
    peft \
    datasets \
    safetensors \
    sentencepiece

echo ""
echo "========================================"
echo "Installing current repo in editable mode"
echo "========================================"

python -m export PYTHONPATH="$(pwd):$PYTHONPATH"

echo ""
echo "========================================"
echo "Setup finished"
echo "========================================"

python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import peft; print('peft:', peft.__version__)"
