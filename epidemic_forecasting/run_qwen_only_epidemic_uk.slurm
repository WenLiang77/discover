#!/bin/bash

#SBATCH --job-name=qwen_epi_uk
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=02:00:00
#SBATCH --output=epidemic_forecasting/results/qwen_epi_uk_%j.out
#SBATCH --error=epidemic_forecasting/results/qwen_epi_uk_%j.err

set -euo pipefail

echo "============================================================"
echo "Qwen-only COVID-19 UK forecasting smoke test"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Node: $(hostname)"
echo "Start time: $(date --iso-8601=seconds)"
echo "Submission directory: ${SLURM_SUBMIT_DIR}"

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

if [[ ! -f "epidemic_forecasting/run_baseline.py" ]]; then
    echo "ERROR: epidemic_forecasting/run_baseline.py was not found."
    echo "Submit this job from the discover repository root."
    exit 1
fi

module load cray-python/3.11.7

VENV_DIR="${REPO_ROOT}/.venv_epidemic"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: ${VENV_DIR} was not found."
    echo "Run this first from the repository root:"
    echo "bash epidemic_forecasting/setup_epidemic_env.sh"
    exit 1
fi

source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-8}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "${HF_HOME}"

MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET="uk"
FORECAST_HORIZON="14"
ATTEMPTS="5"
SEED="42"
TEMPERATURE="0.7"
TOP_P="0.95"
MAX_NEW_TOKENS="1536"
MAX_INPUT_TOKENS="8192"
RUNTIME_BUDGET_SECONDS="400"
OUTPUT_DIR="${REPO_ROOT}/epidemic_forecasting/results/baseline/covid19_uk_14step/job_${SLURM_JOB_ID}"

echo
echo "============================================================"
echo "Environment"
echo "============================================================"
echo "Python executable: $(which python)"
python --version
echo "PyTorch and CUDA:"
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY

echo
echo "nvidia-smi:"
nvidia-smi --list-gpus
nvidia-smi

echo
echo "============================================================"
echo "Experiment configuration"
echo "============================================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET}"
echo "Forecast horizon: ${FORECAST_HORIZON}"
echo "Attempts: ${ATTEMPTS}"
echo "Seed: ${SEED}"
echo "Temperature: ${TEMPERATURE}"
echo "Top-p: ${TOP_P}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Max input tokens: ${MAX_INPUT_TOKENS}"
echo "Candidate runtime budget: ${RUNTIME_BUDGET_SECONDS} seconds"
echo "Output directory: ${OUTPUT_DIR}"

echo
echo "============================================================"
echo "Starting experiment"
echo "============================================================"

srun python -m epidemic_forecasting.run_baseline \
    --dataset "${DATASET}" \
    --forecast-horizon "${FORECAST_HORIZON}" \
    --model-name "${MODEL_NAME}" \
    --attempts "${ATTEMPTS}" \
    --seed "${SEED}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-input-tokens "${MAX_INPUT_TOKENS}" \
    --runtime-budget-seconds "${RUNTIME_BUDGET_SECONDS}" \
    --mase-seasonality 1 \
    --output-dir "${OUTPUT_DIR}"

echo
echo "============================================================"
echo "Experiment completed"
echo "============================================================"
echo "End time: $(date --iso-8601=seconds)"
echo "Results: ${OUTPUT_DIR}"
