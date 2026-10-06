#!/bin/bash

#SBATCH --job-name=eveA_v3
#SBATCH --partition=gpu
#SBATCH --output=01_event_aware_v3/logs/eveA_v3_%A_%a.out
#SBATCH --error=01_event_aware_v3/logs/eveA_v3_%A_%a.err
#SBATCH --array=0-59%1
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

FREEZE="01_event_aware_v3/07_eve_frame_stage_a_no_event/FROZEN_STAGE_A_SHA256SUMS_v3.txt"

sha256sum -c "$FREEZE"

MANIFEST="01_event_aware_v3/07_eve_frame_stage_a_no_event/stage_a_manifest_v3.csv"

CONFIG=$(python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
import pandas as pd

idx = int(sys.argv[1])

p = (
    "01_event_aware_v3/07_eve_frame_stage_a_no_event/"
    "stage_a_manifest_v3.csv"
)

df = pd.read_csv(p)

if idx < 0 or idx >= len(df):
    raise SystemExit(
        f"Invalid array index {idx}"
    )

r = df.iloc[idx]

if int(r["run_id"]) != idx:
    raise SystemExit(
        "Manifest/index mismatch."
    )

print(
    f'{r["target_task"]}|'
    f'{int(r["outer_fold"])}|'
    f'{int(r["model_seed"])}'
)
PY
)

IFS='|' read -r TASK FOLD SEED <<< "$CONFIG"

echo "=============================================="
echo "EVE-FRAME STAGE A"
echo "Array index : $SLURM_ARRAY_TASK_ID"
echo "Target task : $TASK"
echo "Outer fold  : $FOLD"
echo "Seed        : $SEED"
echo "Node        : $(hostname)"
echo "Started     : $(date)"
echo "=============================================="

python -u \
01_event_aware_v3/07_eve_frame_stage_a_no_event/37_train_eve_frame_stage_a_v3.py \
--target-task "$TASK" \
--outer-fold "$FOLD" \
--seed "$SEED" \
--skip-if-valid

echo "Finished: $(date)"
