#!/bin/bash

#SBATCH --job-name=tfidf_base_v3
#SBATCH --partition=cpu-dev
#SBATCH --output=01_event_aware_v3/logs/tfidf_base_v3_%j.out
#SBATCH --error=01_event_aware_v3/logs/tfidf_base_v3_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42

echo "=============================================="
echo "CLASSICAL TF-IDF BASELINES v3"
echo "=============================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-NA}"

python -u \
01_event_aware_v3/04_classical_baselines/06_classical_tfidf_baselines_v3.py

echo ""
echo "Finished: $(date)"
