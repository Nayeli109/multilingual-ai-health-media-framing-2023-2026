#!/bin/bash

#SBATCH --job-name=eveA_stats
#SBATCH --partition=cpu-dev
#SBATCH --output=01_event_aware_v3/logs/eveA_stats_%j.out
#SBATCH --error=01_event_aware_v3/logs/eveA_stats_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u \
01_event_aware_v3/07_eve_frame_stage_a_no_event/42_analyze_stage_a_vs_base_v3.py
