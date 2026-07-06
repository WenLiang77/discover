#!/bin/bash
#SBATCH --job-name=qwen_only_denoise
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=university_gpu/logs/qwen_only_denoise_%j.out
#SBATCH --error=university_gpu/logs/qwen_only_denoise_%j.err

set -e

module load cray-python/3.11.7

echo "=== Qwen-only official denoising baseline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Current directory: $(pwd)"

mkdir -p university_gpu/logs
mkdir -p university_gpu/outputs
mkdir -p .openproblems_cache

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found."
    echo "Please run first:"
    echo "bash university_gpu/setup_denoising_official_env.sh"
    exit 1
fi

source .venv/bin/activate

echo "Python:"
which python3
python3 --version

echo "GPU:"
nvidia-smi || true

export OPENPROBLEMS_CACHE_DIR="$(pwd)/.openproblems_cache"

export LOCAL_TTT_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
export LOCAL_QWEN_ONLY_ATTEMPTS="50"
export LOCAL_TTT_MAX_NEW_TOKENS="1536"
export LOCAL_QWEN_ONLY_TEMPERATURE="0.7"
export LOCAL_QWEN_ONLY_TOP_P="0.95"
export LOCAL_QWEN_ONLY_SEED="42"
export LOCAL_QWEN_ONLY_OUTPUT_DIR="university_gpu/outputs/qwen_only_denoising_official_${SLURM_JOB_ID}"

echo "Experiment config:"
echo "LOCAL_TTT_MODEL=$LOCAL_TTT_MODEL"
echo "LOCAL_QWEN_ONLY_ATTEMPTS=$LOCAL_QWEN_ONLY_ATTEMPTS"
echo "LOCAL_TTT_MAX_NEW_TOKENS=$LOCAL_TTT_MAX_NEW_TOKENS"
echo "LOCAL_QWEN_ONLY_TEMPERATURE=$LOCAL_QWEN_ONLY_TEMPERATURE"
echo "LOCAL_QWEN_ONLY_TOP_P=$LOCAL_QWEN_ONLY_TOP_P"
echo "LOCAL_QWEN_ONLY_SEED=$LOCAL_QWEN_ONLY_SEED"
echo "LOCAL_QWEN_ONLY_OUTPUT_DIR=$LOCAL_QWEN_ONLY_OUTPUT_DIR"

export PYTHONPATH="$(pwd):$PYTHONPATH"

python3 -m university_gpu.local_qwen_only_denoising \
  --model-name "$LOCAL_TTT_MODEL" \
  --eval-mode official \
  --attempts "$LOCAL_QWEN_ONLY_ATTEMPTS" \
  --max-new-tokens "$LOCAL_TTT_MAX_NEW_TOKENS" \
  --temperature "$LOCAL_QWEN_ONLY_TEMPERATURE" \
  --top-p "$LOCAL_QWEN_ONLY_TOP_P" \
  --seed "$LOCAL_QWEN_ONLY_SEED" \
  --output-dir "$LOCAL_QWEN_ONLY_OUTPUT_DIR"

echo "End time: $(date)"
echo "Done."