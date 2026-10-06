#!/bin/bash

#SBATCH --job-name=eve_v4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00

#SBATCH --output=01_event_aware_v3/09_eve_frame_v4_architecture_sensitivity/logs_v4/v4_%A_%a.out
#SBATCH --error=01_event_aware_v3/09_eve_frame_v4_architecture_sensitivity/logs_v4/v4_%A_%a.err

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

V4="01_event_aware_v3/09_eve_frame_v4_architecture_sensitivity"

TRAINER="$V4/70_train_eve_frame_v4.py"
TRAINER_FREEZE="$V4/FROZEN_V4_TRAINER_SHA256SUMS_v4.txt"

MANIFEST="$V4/V4_RUN_MANIFEST_v4.csv"
MANIFEST_SHA="$V4/V4_RUN_MANIFEST_v4.sha256"

echo "============================================================"
echo "EVE-FRAME V4 TRAINING"
echo "============================================================"
echo "Job     : $SLURM_JOB_ID"
echo "Array   : $SLURM_ARRAY_TASK_ID"
echo "Node    : $(hostname)"
echo "Started : $(date)"
echo ""

echo "========== FROZEN CODE CHECK =========="

sha256sum -c "$TRAINER_FREEZE"
sha256sum -c "$MANIFEST_SHA"

echo ""
echo "========== LOAD MANIFEST ROW =========="

eval "$(
python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv
import shlex
import sys

path = sys.argv[1]
idx = int(sys.argv[2])

with open(
    path,
    newline="",
    encoding="utf-8",
) as f:
    rows = list(
        csv.DictReader(f)
    )

if idx < 0 or idx >= len(rows):
    raise RuntimeError(
        f"Invalid array index: {idx}"
    )

row = rows[idx]

if int(row["array_index"]) != idx:
    raise RuntimeError(
        "Manifest index mismatch."
    )

mapping = {
    "ARCH":
        row["architecture"],

    "ARM":
        row["arm"],

    "TASK":
        row["target_task"],

    "FOLD":
        row["outer_fold"],

    "SEED":
        row["seed"],

    "ARCH_LABEL":
        row["architecture_label"],
}

for key, value in mapping.items():
    print(
        f"{key}={shlex.quote(str(value))}"
    )
PY
)"

echo "Architecture : $ARCH"
echo "Arm          : $ARM"
echo "Task         : $TASK"
echo "Fold         : $FOLD"
echo "Seed         : $SEED"
echo "Arch label   : $ARCH_LABEL"

echo ""
echo "========== TRAIN =========="

python -u "$TRAINER" \
    --architecture "$ARCH" \
    --arm "$ARM" \
    --target-task "$TASK" \
    --outer-fold "$FOLD" \
    --seed "$SEED" \
    --skip-if-valid

echo ""
echo "============================================================"
echo "V4 CONFIGURATION COMPLETE"
echo "Finished : $(date)"
echo "============================================================"
