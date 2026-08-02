#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv_epidemic"

echo "============================================================"
echo "Setting up epidemic forecasting environment"
echo "============================================================"

cd "${REPO_ROOT}"

echo "Repository root: ${REPO_ROOT}"
echo "Python:"
python3 --version

echo
echo "============================================================"
echo "Create or activate virtual environment"
echo "============================================================"

if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
    echo "Created virtual environment: ${VENV_DIR}"
else
    echo "Reusing virtual environment: ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo "Active Python:"
which python
python --version

echo
echo "============================================================"
echo "Upgrade packaging tools"
echo "============================================================"

python -m pip install --upgrade pip setuptools wheel

echo
echo "============================================================"
echo "Install epidemic forecasting requirements"
echo "============================================================"

python -m pip install \
    -r "${SCRIPT_DIR}/requirements.txt"

echo
echo "============================================================"
echo "Check required imports"
echo "============================================================"

python - <<'PY'
import accelerate
import numpy
import pandas
import peft
import scipy
import sklearn
import torch
import transformers

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scipy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("peft:", peft.__version__)
print("CUDA available in current session:", torch.cuda.is_available())
PY

echo
echo "============================================================"
echo "Check epidemic forecasting package"
echo "============================================================"

python - <<'PY'
from epidemic_forecasting.tasks.covid19.task import create_covid19_task

task = create_covid19_task(
    dataset="uk",
    forecast_horizon=14,
    runtime_budget_seconds=20,
)

print("Task loaded successfully:")
print(task.describe())
PY

echo
echo "============================================================"
echo "Environment setup completed"
echo "============================================================"
echo
echo "Activate it later with:"
echo "source ${VENV_DIR}/bin/activate"