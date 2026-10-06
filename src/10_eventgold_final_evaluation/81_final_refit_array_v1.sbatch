#!/bin/bash
#SBATCH --job-name=finalA_v1
#SBATCH --partition=gpu
#SBATCH --output=01_event_aware_v3/10_eventgold_final_evaluation/logs_final_refit_v1/finalA_v1_%A_%a.out
#SBATCH --error=01_event_aware_v3/10_eventgold_final_evaluation/logs_final_refit_v1/finalA_v1_%A_%a.err
#SBATCH --array=0-14%1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100-sxm4-40gb:1

set -euo pipefail

cd "$HOME/Q1_MediaDiscourse_2026"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONHASHSEED=42
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

FINAL="01_event_aware_v3/10_eventgold_final_evaluation"

TRAINER="$FINAL/80_train_final_stage_a_refit_v1.py"
MANIFEST="$FINAL/FINAL_REFIT_RUN_MANIFEST_v1.csv"
FREEZE="$FINAL/FROZEN_FINAL_REFIT_IMPLEMENTATION_SHA256SUMS_v1.txt"

PY="$HOME/media_q1_env/bin/python"

echo "============================================================"
echo "FINAL STAGE-A REFIT"
echo "============================================================"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-NA}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"

echo ""
echo "VERIFY FROZEN IMPLEMENTATION"
sha256sum -c "$FREEZE"

CONFIG="$(
"$PY" - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv
import sys

manifest = sys.argv[1]
idx = int(sys.argv[2])

with open(manifest, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 15:
    raise SystemExit(
        f"Expected 15 manifest rows; got {len(rows)}"
    )

if not (0 <= idx < len(rows)):
    raise SystemExit(
        f"Invalid array index: {idx}"
    )

r = rows[idx]

print(
    "\t".join([
        r["task"],
        r["seed"],
        r["final_refit_epoch"],
        r["scheduler_horizon_epochs"],
        r["run_name"],
    ])
)
PY
)"

IFS=$'\t' read -r TASK SEED EPOCH SCHED RUN_NAME <<< "$CONFIG"

echo ""
echo "TASK=$TASK"
echo "SEED=$SEED"
echo "FINAL_REFIT_EPOCH=$EPOCH"
echo "SCHEDULER_HORIZON=$SCHED"
echo "RUN_NAME=$RUN_NAME"
echo "EVENTGOLD_STATUS=SEALED_NOT_ACCESSED"

test "$SCHED" = "8"

echo ""
echo "BEGIN_FINAL_REFIT"

"$PY" -u "$TRAINER" \
    --target-task "$TASK" \
    --seed "$SEED" \
    --skip-if-valid

echo ""
echo "END_FINAL_REFIT"

echo ""
echo "VERIFY IMPLEMENTATION REMAINS FROZEN"
sha256sum -c "$FREEZE"

echo ""
echo "FINAL_REFIT_ARRAY_TASK_COMPLETE=YES"
echo "EVENTGOLD_STATUS=SEALED_NOT_ACCESSED"
