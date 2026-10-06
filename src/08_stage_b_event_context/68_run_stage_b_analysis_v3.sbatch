#!/bin/bash

#SBATCH --job-name=eveB_stats_v3
#SBATCH --output=01_event_aware_v3/logs/eveB_stats_v3_%j.out
#SBATCH --error=01_event_aware_v3/logs/eveB_stats_v3_%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42

DEST="01_event_aware_v3/08_eve_frame_stage_b_event_context"
FREEZE="$DEST/FROZEN_STAGE_B_STATISTICAL_ANALYSIS_SHA256SUMS_v3.txt"
SCRIPT="$DEST/67_analyze_stage_b_v3.py"

echo "============================================================"
echo "STAGE-B FROZEN STATISTICAL ANALYSIS"
echo "Node    : $(hostname)"
echo "Started : $(date)"
echo "============================================================"

echo ""
echo "========== FREEZE CHECK =========="

sha256sum -c "$FREEZE"

echo ""
echo "========== FULL PREDECLARED ANALYSIS =========="

python -u "$SCRIPT"

echo ""
echo "============================================================"
echo "STAGE-B ANALYSIS FINISHED"
echo "Finished : $(date)"
echo "============================================================"
