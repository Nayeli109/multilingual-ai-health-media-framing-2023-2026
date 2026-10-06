from pathlib import Path
import hashlib
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path.cwd()

DATA = (
    ROOT
    / "01_event_aware_v3/03_scalegold_active_learning/"
      "LegacyAux_leakage_free_v3.csv"
)

OUTDIR = (
    ROOT
    / "01_event_aware_v3/04_classical_baselines/"
      "frozen_folds_v3"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

TASKS = {
    "primary_frame":
        "final_primary_frame_validated",
    "stance":
        "final_stance_validated",
    "misinformation_relation":
        "final_misinformation_relation_validated",
}

SEED = 42
N_SPLITS = 4

df = pd.read_csv(DATA, low_memory=False)

if len(df) != 199:
    raise RuntimeError(f"Expected n=199, observed {len(df)}")

rows = []

for task, target in TASKS.items():

    y = df[target].astype(str).to_numpy()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    for fold, (_, test_idx) in enumerate(
        cv.split(df["doc_id"], y),
        start=1,
    ):
        for idx in test_idx:
            rows.append({
                "task": task,
                "doc_id": df.iloc[idx]["doc_id"],
                "fold": fold,
                "gold": y[idx],
            })

out = pd.DataFrame(rows)

for task in TASKS:
    part = out[out["task"] == task]

    if len(part) != 199:
        raise RuntimeError(
            f"{task}: expected 199 fold assignments"
        )

    if part["doc_id"].duplicated().any():
        raise RuntimeError(
            f"{task}: duplicate doc_id fold assignment"
        )

path = OUTDIR / "LegacyAux199_FROZEN_FOLDS_v3.csv"

out.to_csv(path, index=False)

digest = hashlib.sha256(
    path.read_bytes()
).hexdigest()

meta = {
    "dataset": "LegacyAux-199",
    "n": 199,
    "n_splits": 4,
    "seed": 42,
    "stratification": "task-specific target label",
    "sha256": digest,
}

(
    OUTDIR / "LegacyAux199_FROZEN_FOLDS_QC_v3.json"
).write_text(
    json.dumps(meta, indent=2),
    encoding="utf-8",
)

print("Frozen fold rows:", len(out))
print("Expected:", 199 * 3)
print("SHA256:", digest)

print(
    out.groupby(["task", "fold"])
       .size()
       .to_string()
)
