#!/bin/bash
#SBATCH --job-name=eve_v4_confirm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=01_event_aware_v3/09_eve_frame_v4_architecture_sensitivity/logs_v4/v4_confirmatory_%j.out
#SBATCH --error=01_event_aware_v3/09_eve_frame_v4_architecture_sensitivity/logs_v4/v4_confirmatory_%j.err

set -euo pipefail

cd "$HOME/Q1_MediaDiscourse_2026"

V4="01_event_aware_v3/09_eve_frame_v4_architecture_sensitivity"

SCRIPT="$V4/73_analyze_eve_frame_v4.py"
FREEZE="$V4/FROZEN_V4_ANALYSIS_SCRIPT_SHA256SUMS_v4.txt"

OUTDIR="$V4/v4_confirmatory_analysis_v4"
TMPDIR="$V4/v4_confirmatory_analysis_v4_tmp"

PY="$HOME/media_q1_env/bin/python"

export PYTHONHASHSEED=42

# The inferential implementation itself is serial.
# Prevent implicit BLAS/OpenMP oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "V4 CONFIRMATORY ANALYSIS"
echo "============================================================"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"
echo "PYTHON=$PY"
echo ""

test -x "$PY"

sha256sum -c "$FREEZE"

if [ -e "$OUTDIR" ] || [ -e "$TMPDIR" ]; then
    echo "STOP: confirmatory output already exists."
    exit 1
fi

echo ""
echo "FROZEN_ANALYSIS_PACKAGE=OK"
echo "EVENTGOLD_STATUS=SEALED_NOT_ACCESSED"
echo ""
echo "BEGIN_CONFIRMATORY_INFERENCE"

"$PY" "$SCRIPT"

echo ""
echo "END_CONFIRMATORY_INFERENCE"
echo ""

sha256sum -c "$FREEZE"

test -d "$OUTDIR"
test ! -e "$TMPDIR"

echo ""
echo "FINAL_ANALYSIS_DIRECTORY_PRESENT=YES"
echo "TEMP_DIRECTORY_ABSENT=YES"
echo "EVENTGOLD_STATUS=SEALED_NOT_ACCESSED"
