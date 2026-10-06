#!/bin/bash

#SBATCH --job-name=eveB_cov
#SBATCH --partition=cpu-dev
#SBATCH --output=01_event_aware_v3/logs/eveB_cov_%j.out
#SBATCH --error=01_event_aware_v3/logs/eveB_cov_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

python -u \
01_event_aware_v3/08_eve_frame_stage_b_event_context/44_event_context_coverage_v3.py
