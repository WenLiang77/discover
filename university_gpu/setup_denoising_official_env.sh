#!/bin/bash

set -e

echo "========================================"
echo "Setting up official denoising environment"
echo "========================================"

echo "Current directory: $(pwd)"
echo "Python:"
python3 --version || true

echo ""
echo "========================================"
echo "Create / activate virtual environment"
echo "========================================"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

python3 --version
which python3

echo ""
echo "========================================"
echo "Upgrade pip"
echo "========================================"

python3 -m pip install --upgrade pip setuptools wheel

echo ""
echo "========================================"
echo "Install current repo"
echo "========================================"

# python3 -m pip install -e .

echo ""
echo "========================================"
echo "Install official denoising requirements"
echo "========================================"

python3 -m pip install -r requirements/denoising/requirements-denoising.txt
python3 -m pip install accelerate peft sentencepiece

echo ""
echo "========================================"
echo "Install git dependencies"
echo "========================================"

python3 -m pip install git+https://github.com/czbiohub/simscity.git
python3 -m pip install --no-deps git+https://github.com/czbiohub/molecular-cross-validation.git

echo ""
echo "========================================"
echo "Clone and patch openproblems"
echo "========================================"

if [ ! -d "openproblems" ]; then
    git clone https://github.com/openproblems-bio/openproblems.git
fi

cd openproblems
git checkout v1.0.0
git apply ../requirements/denoising/openproblems_api_fix.patch || echo "Patch may already be applied."
cd ..

echo ""
echo "========================================"
echo "Install openproblems without dependencies"
echo "========================================"

python3 -m pip install --no-deps -e ./openproblems

echo ""
echo "========================================"
echo "Set OpenProblems cache directory"
echo "========================================"

mkdir -p .openproblems_cache
echo "export OPENPROBLEMS_CACHE_DIR=$(pwd)/.openproblems_cache" > university_gpu/activate_denoising_cache.sh

echo ""
echo "========================================"
echo "Check imports"
echo "========================================"

export OPENPROBLEMS_CACHE_DIR="$(pwd)/.openproblems_cache"

python3 -c "import torch; print('torch:', torch.__version__)"
python3 -c "import transformers; print('transformers:', transformers.__version__)"
python3 -c "import peft; print('peft:', peft.__version__)"
python3 -c "import scanpy; print('scanpy:', scanpy.__version__)"
python3 -c "import anndata; print('anndata:', anndata.__version__)"
python3 -c "import scprep; print('scprep import ok')"
python3 -c "import graphtools; print('graphtools import ok')"
python3 -c "import openproblems; print('openproblems import ok')"

echo ""
echo "========================================"
echo "Official denoising environment setup finished"
echo "========================================"