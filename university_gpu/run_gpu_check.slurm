#!/bin/bash
#SBATCH --job-name=ttt_gpu_check
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=university_gpu/logs/gpu_check_%j.out
#SBATCH --error=university_gpu/logs/gpu_check_%j.err

set -e

echo "========================================"
echo "TTT-Discover University GPU Check"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "Start time: $(date)"

mkdir -p university_gpu/logs

echo ""
echo "========================================"
echo "nvidia-smi"
echo "========================================"
nvidia-smi || echo "nvidia-smi failed or not available"

echo ""
echo "========================================"
echo "Python version"
echo "========================================"
python --version

echo ""
echo "========================================"
echo "Running PyTorch GPU check"
echo "========================================"
python university_gpu/check_torch_gpu.py

echo ""
echo "========================================"
echo "Finished"
echo "========================================"
echo "End time: $(date)"