#!/bin/bash

#SBATCH --job-name=eveB_v3
#SBATCH --partition=gpu
#SBATCH --output=01_event_aware_v3/logs/eveB_v3_%A_%a.out
#SBATCH --error=01_event_aware_v3/logs/eveB_v3_%A_%a.err
#SBATCH --array=0-119%1
#SBATCH --time=01:30:00
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

FREEZE="$DEST/FROZEN_STAGE_B_IMPLEMENTATION_SHA256SUMS_v3.txt"

sha256sum -c "$FREEZE"

MANIFEST="$DEST/stage_b_manifest_v3.csv"

CONFIG=$(python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
import pandas as pd

idx = int(sys.argv[1])

p = (
    "01_event_aware_v3/08_eve_frame_stage_b_event_context/"
    "stage_b_manifest_v3.csv"
)

df = pd.read_csv(p)

if idx < 0 or idx >= len(df):
    raise SystemExit(
        f"Invalid Stage-B array index: {idx}"
    )

r = df.iloc[idx]

if int(r["run_id"]) != idx:
    raise SystemExit(
        "Stage-B manifest/index mismatch."
    )

print(
    f'{r["arm"]}|'
    f'{r["target_task"]}|'
    f'{int(r["outer_fold"])}|'
    f'{int(r["model_seed"])}'
)
PY
)

IFS='|' read -r ARM TASK FOLD SEED <<< "$CONFIG"

echo "============================================================"
echo "EVE-FRAME STAGE B"
echo "Array index : $SLURM_ARRAY_TASK_ID"
echo "Arm         : $ARM"
echo "Target task : $TASK"
echo "Outer fold  : $FOLD"
echo "Seed        : $SEED"
echo "Node        : $(hostname)"
echo "Started     : $(date)"
echo "============================================================"

python -u "$DEST/63_train_eve_frame_stage_b_v3.py" --arm "$ARM" --target-task "$TASK" --outer-fold "$FOLD" --seed "$SEED" --skip-if-valid

echo "Finished: $(date)"
