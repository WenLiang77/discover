#!/bin/bash
#SBATCH --job-name=official_ttt_denoise
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=university_gpu/logs/official_ttt_denoise_%j.out
#SBATCH --error=university_gpu/logs/official_ttt_denoise_%j.err

set -e

echo "========================================"
echo "Official Tiny Local TTT-Denoising"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Current directory: $(pwd)"

mkdir -p university_gpu/logs
mkdir -p university_gpu/outputs
mkdir -p .openproblems_cache

echo ""
echo "========================================"
echo "Activate Python environment"
echo "========================================"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found."
    echo "Please run first:"
    echo "bash university_gpu/setup_denoising_official_env.sh"
    exit 1
fi

source .venv/bin/activate

echo "Python executable:"
which python

echo "Python version:"
python --version

echo ""
echo "========================================"
echo "GPU information"
echo "========================================"
nvidia-smi || true

echo ""
echo "========================================"
echo "Experiment configuration"
echo "========================================"

export OPENPROBLEMS_CACHE_DIR="$(pwd)/.openproblems_cache"

export LOCAL_TTT_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
export LOCAL_TTT_EVAL_MODE="official"
export LOCAL_TTT_STEPS="3"
export LOCAL_TTT_ROLLOUTS="3"
export LOCAL_TTT_MAX_NEW_TOKENS="1024"
export LOCAL_TTT_LORA_RANK="8"
export LOCAL_TTT_LR="1e-4"
export LOCAL_TTT_OUTPUT_DIR="university_gpu/outputs/local_ttt_denoising_official"

echo "OPENPROBLEMS_CACHE_DIR=$OPENPROBLEMS_CACHE_DIR"
echo "LOCAL_TTT_MODEL=$LOCAL_TTT_MODEL"
echo "LOCAL_TTT_EVAL_MODE=$LOCAL_TTT_EVAL_MODE"
echo "LOCAL_TTT_STEPS=$LOCAL_TTT_STEPS"
echo "LOCAL_TTT_ROLLOUTS=$LOCAL_TTT_ROLLOUTS"
echo "LOCAL_TTT_MAX_NEW_TOKENS=$LOCAL_TTT_MAX_NEW_TOKENS"
echo "LOCAL_TTT_LORA_RANK=$LOCAL_TTT_LORA_RANK"
echo "LOCAL_TTT_LR=$LOCAL_TTT_LR"
echo "LOCAL_TTT_OUTPUT_DIR=$LOCAL_TTT_OUTPUT_DIR"

echo ""
echo "========================================"
echo "Run official tiny local TTT denoising"
echo "========================================"

python university_gpu/local_ttt_denoising_tiny.py \
    --model-name "$LOCAL_TTT_MODEL" \
    --eval-mode "$LOCAL_TTT_EVAL_MODE" \
    --steps "$LOCAL_TTT_STEPS" \
    --rollouts-per-step "$LOCAL_TTT_ROLLOUTS" \
    --max-new-tokens "$LOCAL_TTT_MAX_NEW_TOKENS" \
    --lora-rank "$LOCAL_TTT_LORA_RANK" \
    --learning-rate "$LOCAL_TTT_LR" \
    --output-dir "$LOCAL_TTT_OUTPUT_DIR"

echo ""
echo "========================================"
echo "Finished"
echo "========================================"
echo "End time: $(date)"