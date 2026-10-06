#!/bin/bash

#SBATCH --job-name=ft_stats_v3
#SBATCH --partition=cpu-dev
#SBATCH --output=01_event_aware_v3/logs/ft_stats_v3_%j.out
#SBATCH --error=01_event_aware_v3/logs/ft_stats_v3_%j.err
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
01_event_aware_v3/06_supervised_analysis_v3/35_supervised_oof_statistics_v3.py
