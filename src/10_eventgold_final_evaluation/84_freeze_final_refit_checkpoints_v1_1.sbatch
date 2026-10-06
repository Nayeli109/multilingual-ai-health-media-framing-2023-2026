#!/bin/bash
#SBATCH --job-name=freeze_finalA
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=01_event_aware_v3/10_eventgold_final_evaluation/freeze_final_refit_%j.out
#SBATCH --error=01_event_aware_v3/10_eventgold_final_evaluation/freeze_final_refit_%j.err

set -euo pipefail

cd "$HOME/Q1_MediaDiscourse_2026"

PY="$HOME/media_q1_env/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

FINAL="01_event_aware_v3/10_eventgold_final_evaluation"
RUNROOT="$FINAL/final_refit_runs_v1"

IMPL_FREEZE="$FINAL/FROZEN_FINAL_REFIT_IMPLEMENTATION_SHA256SUMS_v1.txt"
EXEC_FREEZE="$FINAL/FROZEN_FINAL_REFIT_EXECUTION_SHA256SUMS_v1.txt"

MANIFEST_OUT="$FINAL/FINAL_REFIT_CHECKPOINT_MANIFEST_v1.csv"
CLOSURE="$FINAL/FINAL_REFIT_CHECKPOINT_CLOSURE_v1.txt"
FREEZE="$FINAL/FROZEN_FINAL_REFIT_CHECKPOINTS_SHA256SUMS_v1.txt"

AMEND="$FINAL/FINAL_REFIT_FREEZE_AMENDMENT_01.txt"

echo "============================================================"
echo "FINAL REFIT CHECKPOINT FREEZE"
echo "============================================================"

echo "JOB_ID=${SLURM_JOB_ID:-NA}"
echo "HOST=$(hostname)"

echo ""
echo "VERIFY PRE-EXISTING FREEZES"

sha256sum -c "$IMPL_FREEZE"
sha256sum -c "$EXEC_FREEZE"

for f in \
    "$MANIFEST_OUT" \
    "$CLOSURE" \
    "$FREEZE"
do
    if [ -e "$f" ]; then
        echo "STOP: output already exists: $f"
        exit 1
    fi
done


echo ""
echo "CREATE CHECKPOINT MANIFEST"

"$PY" - <<'PYTHON_EOF'
from pathlib import Path
import pandas as pd
import json
import hashlib
import csv

FINAL = Path(
    "01_event_aware_v3/"
    "10_eventgold_final_evaluation"
)

RUNROOT = (
    FINAL
    / "final_refit_runs_v1"
)

SRC = pd.read_csv(
    FINAL
    / "FINAL_REFIT_RUN_MANIFEST_v1.csv"
)

OUT = (
    FINAL
    / "FINAL_REFIT_CHECKPOINT_MANIFEST_v1.csv"
)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


assert len(SRC) == 15

rows = []

for r in SRC.itertuples(index=False):

    run = (
        RUNROOT
        / str(r.run_name)
    )

    checkpoint = (
        run
        / "model_state_dict_v1.pt"
    )

    summary = (
        run
        / "run_summary_final_refit_v1.json"
    )

    labels = (
        run
        / "label_maps_v1.json"
    )

    chunking = (
        run
        / "document_chunking_v1.csv"
    )

    for p in [
        checkpoint,
        summary,
        labels,
        chunking,
    ]:
        assert p.is_file(), p
        assert p.stat().st_size > 0, p

    x = json.loads(
        summary.read_text(
            encoding="utf-8"
        )
    )

    checkpoint_sha = sha256(
        checkpoint
    )

    summary_sha = sha256(
        summary
    )

    labels_sha = sha256(
        labels
    )

    chunking_sha = sha256(
        chunking
    )

    assert x["status"] == "COMPLETE"

    assert (
        x["architecture"]
        == "EVE_FRAME_STAGE_A_NO_EVENT"
    )

    assert (
        x["stage"]
        == "FINAL_REFIT_PRE_EVENTGOLD"
    )

    assert (
        x["eventgold_status"]
        == "SEALED_NOT_ACCESSED"
    )

    assert (
        x["target_task"]
        == str(r.task)
    )

    assert (
        int(x["seed"])
        == int(r.seed)
    )

    assert (
        int(x["final_refit_epoch"])
        == int(r.final_refit_epoch)
    )

    assert (
        x["checkpoint_sha256"]
        == checkpoint_sha
    )

    assert (
        x["label_maps_sha256"]
        == labels_sha
    )

    assert (
        x["document_chunking_sha256"]
        == chunking_sha
    )

    rows.append({
        "run_name":
            str(r.run_name),
        "task":
            str(r.task),
        "seed":
            int(r.seed),
        "final_refit_epoch":
            int(r.final_refit_epoch),
        "checkpoint_path":
            str(checkpoint),
        "checkpoint_sha256":
            checkpoint_sha,
        "checkpoint_bytes":
            int(
                checkpoint.stat().st_size
            ),
        "summary_sha256":
            summary_sha,
        "label_maps_sha256":
            labels_sha,
        "document_chunking_sha256":
            chunking_sha,
    })


with OUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(rows)


print(
    "CHECKPOINT_MANIFEST_ROWS=15"
)

print(
    "CHECKPOINT_HASHES_VERIFIED=15"
)

print(
    "EVENTGOLD_STATUS="
    "SEALED_NOT_ACCESSED"
)
PYTHON_EOF


echo ""
echo "WRITE CHECKPOINT CLOSURE"

cat > "$CLOSURE" <<'CLOSURE_EOF'
EVE-FRAME — FINAL REFIT CHECKPOINT CLOSURE v1
==============================================

STATUS
------
FINAL PRE-EVENTGOLD MODEL PACKAGE CLOSED.

FINAL ARCHITECTURE
------------------
STAGE_A_NO_EVENT

TRAINING DATA
-------------
LegacyAux-199 only.

FINAL LOGICAL MODELS
--------------------
15

Tasks:
primary_frame
stance
misinformation_relation

Seeds:
11
29
47
83
131

FINAL REFIT EXECUTION
---------------------
SLURM array:
27782

All 15 array tasks:
COMPLETED

All 15 exit codes:
0:0

Final run directories:
15

Temporary directories:
0

Run summaries:
15

Checkpoints:
15

CHECKPOINT AUDIT
----------------
Audit SLURM job:
27808

State:
COMPLETED

Exit code:
0:0

Audited checkpoints:
15

Checkpoint bundle metadata audit:
PASS

Same-seed plus same-epoch state_dict equivalence:
PASS

Audit stderr:
EMPTY

DETERMINISM
-----------
Different logical checkpoint bundle files may have different
byte-level hashes because the bundles include task-specific
metadata.

Where seed and final epoch were identical, trained multitask
state_dict tensors were independently verified to be identical.

MODEL POLICY
------------
No checkpoint is selected, discarded, reweighted, modified, or
retrained after final refitting.

All five frozen logical seed-specific models per task proceed to
the one-shot EventGold evaluation ensemble.

No training-set performance metric is used for post-refit model
selection.

DEVELOPMENT STATUS
------------------
Architecture development is closed.

No V5 development, retraining, epoch modification, seed
reselection, threshold optimization, hyperparameter search, or
result-dependent model modification is permitted after observing
EventGold.

FREEZE AMENDMENT
----------------
A pre-submission shell-heredoc error in the first freeze-wrapper
construction was documented in:

FINAL_REFIT_FREEZE_AMENDMENT_01.txt

The failed attempt did not submit a SLURM job, modify any
checkpoint, or access EventGold.

EVENTGOLD
---------
At checkpoint closure:

SEALED_NOT_ACCESSED

EventGold-35 has contributed to none of:
architecture selection,
training,
epoch selection,
class weighting,
checkpoint selection,
or model modification.
CLOSURE_EOF


echo ""
echo "CREATE COMPLETE SHA256 FREEZE"

{
    sha256sum \
        "$FINAL/FROZEN_FINAL_REFIT_IMPLEMENTATION_SHA256SUMS_v1.txt" \
        "$FINAL/FROZEN_FINAL_REFIT_EXECUTION_SHA256SUMS_v1.txt" \
        "$FINAL/FINAL_STAGE_A_REFIT_PROTOCOL_v1.txt" \
        "$FINAL/FINAL_REFIT_EPOCHS_v1.csv" \
        "$FINAL/FINAL_REFIT_EPOCH_RULE_v1.txt" \
        "$FINAL/FINAL_REFIT_RUN_MANIFEST_v1.csv" \
        "$FINAL/80_train_final_stage_a_refit_v1.py" \
        "$FINAL/81_final_refit_array_v1.sbatch" \
        "$FINAL/FINAL_REFIT_EXECUTION_LOCK_v1.txt" \
        "$AMEND" \
        "$MANIFEST_OUT" \
        "$CLOSURE"

    while IFS= read -r f
    do
        sha256sum "$f"
    done < <(
        find "$RUNROOT" \
            -mindepth 2 \
            -maxdepth 2 \
            -type f \
            \( \
                -name 'model_state_dict_v1.pt' \
                -o -name 'run_summary_final_refit_v1.json' \
                -o -name 'label_maps_v1.json' \
                -o -name 'document_chunking_v1.csv' \
            \) \
            | sort
    )
} > "$FREEZE"


echo ""
echo "VERIFY COMPLETE FREEZE"

sha256sum -c "$FREEZE"


echo ""
echo "============================================================"
echo "FINAL CHECKPOINT FREEZE COMPLETE"
echo "============================================================"

echo \
"FINAL_REFIT_CHECKPOINT_MANIFEST_SHA=$(sha256sum "$MANIFEST_OUT" | awk '{print $1}')"

echo \
"FINAL_REFIT_CHECKPOINT_CLOSURE_SHA=$(sha256sum "$CLOSURE" | awk '{print $1}')"

echo \
"FINAL_REFIT_CHECKPOINT_FREEZE_SHA=$(sha256sum "$FREEZE" | awk '{print $1}')"

echo "FINAL_CHECKPOINT_PACKAGE_FROZEN=YES"
echo "FINAL_LOGICAL_MODELS=15"
echo "EVENTGOLD_STATUS=SEALED_NOT_ACCESSED"
