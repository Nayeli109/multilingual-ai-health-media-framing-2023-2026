#!/bin/bash

#SBATCH --job-name=eve_ctxemb
#SBATCH --partition=gpu
#SBATCH --output=01_event_aware_v3/logs/eve_ctxemb_%j.out
#SBATCH --error=01_event_aware_v3/logs/eve_ctxemb_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100-sxm4-40gb:1

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026

source ~/media_q1_env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONHASHSEED=42

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export CUBLAS_WORKSPACE_CONFIG=:4096:8

DEST="01_event_aware_v3/08_eve_frame_stage_b_event_context"

SCRIPT="$DEST/57_extract_context_embeddings_xlmr_v3.py"

EXPECTED_SCRIPT_SHA="28fa7640b22dd8162db4a086bf19a6b56b8c1caa0df0ef588297cdc1767a9ffb"

OBSERVED_SCRIPT_SHA=$(sha256sum "$SCRIPT" | awk '{print $1}')

if [ "$OBSERVED_SCRIPT_SHA" != "$EXPECTED_SCRIPT_SHA" ]; then
    echo "ERROR: extractor SHA256 mismatch."
    echo "Expected: $EXPECTED_SCRIPT_SHA"
    echo "Observed: $OBSERVED_SCRIPT_SHA"
    exit 1
fi

echo "============================================================"
echo "EVE-FRAME STAGE B — FROZEN CONTEXT EMBEDDING EXTRACTION"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "GPU         : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started     : $(date)"
echo "Script SHA  : $OBSERVED_SCRIPT_SHA"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total,driver_version \
--format=csv,noheader

python -u "$SCRIPT"

echo "============================================================"
echo "Finished    : $(date)"
echo "============================================================"
