#!/bin/bash
#SBATCH -J q1_gen_gap
#SBATCH -p cpu-dev
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH -o generalization_gap_%j.out
#SBATCH -e generalization_gap_%j.err

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42

python \
01_event_aware_v3/12_posthoc_generalization_gap_v1/01_posthoc_generalization_gap_v1_2.py
