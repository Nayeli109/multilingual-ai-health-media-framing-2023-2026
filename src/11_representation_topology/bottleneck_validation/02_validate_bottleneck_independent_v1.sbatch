#!/bin/bash
#SBATCH -J q1_bn_valid
#SBATCH -p cpu-dev
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH -o bottleneck_validation_%j.out
#SBATCH -e bottleneck_validation_%j.err

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42

python \
01_event_aware_v3/11_posthoc_representation_analysis_v1/bottleneck_validation_v1/01_validate_bottleneck_independent_v1.py
