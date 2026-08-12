#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=us_ttt_7b_100
#SBATCH --time=05:00:00
#SBATCH --output=epidemic_forecasting/results/logs/covid19/us_ttt_7b_100_%j.out
#SBATCH --error=epidemic_forecasting/results/logs/covid19/us_ttt_7b_100_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

if [[ ! -d ".venv_epidemic" ]]; then
    echo "ERROR: .venv_epidemic was not found."
    echo "Run: bash epidemic_forecasting/setup_epidemic_env.sh"
    exit 1
fi

source ".venv_epidemic/bin/activate"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "${HF_HOME}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Python: $(which python)"
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY
nvidia-smi --list-gpus

OUTPUT_DIR="${REPO_ROOT}/epidemic_forecasting/results/covid19/us_14days/ttt/job_${SLURM_JOB_ID}"

srun python -m epidemic_forecasting.run_ttt \
    --dataset us \
    --forecast-horizon 14 \
    --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
    --steps 10 \
    --parents-per-step 1 \
    --rollouts-per-parent 10 \
    --seed 42 \
    --temperature 0.7 \
    --top-p 0.95 \
    --max-new-tokens 2048 \
    --max-input-tokens 8192 \
    --training-max-length 2048 \
    --runtime-budget-seconds 400 \
    --learning-rate 1e-4 \
    --lora-rank 8 \
    --puct-c 1.0 \
    --topk-children 2 \
    --duplicate-penalty 0 \
    --output-dir "${OUTPUT_DIR}"

echo "End: $(date --iso-8601=seconds)"
