#!/bin/bash

#SBATCH --job-name=eve_verifyctx
#SBATCH --partition=cpu-dev
#SBATCH --output=01_event_aware_v3/logs/eve_verifyctx_%j.out
#SBATCH --error=01_event_aware_v3/logs/eve_verifyctx_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

python -u \
01_event_aware_v3/08_eve_frame_stage_b_event_context/55_apply_verified_event_context_v3.py
