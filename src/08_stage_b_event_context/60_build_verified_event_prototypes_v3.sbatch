#!/bin/bash

#SBATCH --job-name=eve_proto
#SBATCH --partition=cpu-dev
#SBATCH --output=01_event_aware_v3/logs/eve_proto_%j.out
#SBATCH --error=01_event_aware_v3/logs/eve_proto_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42

DEST="01_event_aware_v3/08_eve_frame_stage_b_event_context"

SCRIPT="$DEST/59_build_verified_event_prototypes_v3.py"

EXPECTED_SCRIPT_SHA="0abbb18610660075b7995f7b0ea40d90ddecbec411a2a621aeb7ffe0832137ce"

OBSERVED_SCRIPT_SHA=$(sha256sum "$SCRIPT" | awk '{print $1}')

if [ "$OBSERVED_SCRIPT_SHA" != "$EXPECTED_SCRIPT_SHA" ]; then
    echo "ERROR: prototype-builder SHA256 mismatch."
    echo "Expected: $EXPECTED_SCRIPT_SHA"
    echo "Observed: $OBSERVED_SCRIPT_SHA"
    exit 1
fi

echo "============================================================"
echo "EVE-FRAME STAGE B — VERIFIED EVENT PROTOTYPE CONSTRUCTION"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Started     : $(date)"
echo "Script SHA  : $OBSERVED_SCRIPT_SHA"
echo "============================================================"

python -u "$SCRIPT"

echo "============================================================"
echo "Finished    : $(date)"
echo "============================================================"
