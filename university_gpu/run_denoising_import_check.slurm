#!/bin/bash
#SBATCH --job-name=denoise_import_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=university_gpu/logs/denoise_import_%j.out
#SBATCH --error=university_gpu/logs/denoise_import_%j.err

set -e

echo "========================================"
echo "TTT-Discover Denoising Import Check"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "Start time: $(date)"

mkdir -p university_gpu/logs

echo ""
echo "========================================"
echo "Python version"
echo "========================================"
python --version

echo ""
echo "========================================"
echo "Running denoising import check"
echo "========================================"
python university_gpu/test_import_denoising.py

echo ""
echo "========================================"
echo "Finished"
echo "========================================"
echo "End time: $(date)"