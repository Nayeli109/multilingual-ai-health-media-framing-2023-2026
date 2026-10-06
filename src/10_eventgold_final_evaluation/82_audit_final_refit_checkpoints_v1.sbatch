#!/bin/bash
#SBATCH --job-name=audit_finalA
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=01_event_aware_v3/10_eventgold_final_evaluation/audit_final_refit_checkpoints_%j.out
#SBATCH --error=01_event_aware_v3/10_eventgold_final_evaluation/audit_final_refit_checkpoints_%j.err

set -euo pipefail

cd "$HOME/Q1_MediaDiscourse_2026"

PY="$HOME/media_q1_env/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PY" - <<'PY'
from pathlib import Path
import pandas as pd
import torch
import hashlib
import gc

FINAL = Path(
    "01_event_aware_v3/"
    "10_eventgold_final_evaluation"
)

RUNROOT = FINAL / "final_refit_runs_v1"

manifest = pd.read_csv(
    FINAL / "FINAL_REFIT_RUN_MANIFEST_v1.csv"
)

assert len(manifest) == 15


def state_hash(sd):
    h = hashlib.sha256()

    for key in sorted(sd):
        t = sd[key].detach().cpu().contiguous()

        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(
            t.view(torch.uint8)
             .numpy()
             .tobytes()
        )

    return h.hexdigest()


records = []

for i, r in enumerate(
    manifest.itertuples(index=False),
    start=1,
):
    path = (
        RUNROOT
        / str(r.run_name)
        / "model_state_dict_v1.pt"
    )

    bundle = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    assert bundle["architecture"] == "EVE_FRAME_STAGE_A_NO_EVENT"
    assert bundle["stage"] == "FINAL_REFIT_PRE_EVENTGOLD"
    assert bundle["target_task"] == str(r.task)
    assert int(bundle["seed"]) == int(r.seed)
    assert int(bundle["final_refit_epoch"]) == int(r.final_refit_epoch)
    assert int(bundle["scheduler_horizon_epochs"]) == 8

    sh = state_hash(
        bundle["state_dict"]
    )

    records.append({
        "run_name": str(r.run_name),
        "seed": int(r.seed),
        "epoch": int(r.final_refit_epoch),
        "state_hash": sh,
    })

    print(
        f"PASS {i:02d}/15 "
        f"{r.run_name} "
        f"state={sh[:16]}...",
        flush=True,
    )

    del bundle
    gc.collect()


df = pd.DataFrame(records)

print()
print("SAME_SEED_EPOCH_EQUIVALENCE")

for (seed, epoch), g in df.groupby(
    ["seed", "epoch"],
    sort=True,
):
    ok = g["state_hash"].nunique() == 1

    print(
        f"seed={seed} "
        f"epoch={epoch} "
        f"n={len(g)} "
        f"identical={ok}"
    )

    assert ok


print()
print("CHECKPOINT_BUNDLE_METADATA_AUDIT=PASS")
print("SAME_SEED_EPOCH_STATE_DICT_EQUIVALENCE=PASS")
print("AUDITED_CHECKPOINTS=15")
print("EVENTGOLD_STATUS=SEALED_NOT_ACCESSED")
PY
