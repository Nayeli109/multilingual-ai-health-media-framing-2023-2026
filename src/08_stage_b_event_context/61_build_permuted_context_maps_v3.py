from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path.cwd()

DEST = (
    ROOT
    / "01_event_aware_v3/08_eve_frame_stage_b_event_context"
)

SPLITS = (
    ROOT
    / "01_event_aware_v3/05_supervised_transformers/"
      "frozen_finetune_design_v3/"
      "finetune_nested_splits_v3.csv"
)

AVAIL = (
    DEST
    / "event_prototypes_v3/"
      "context_availability_legacyaux199_v3.csv"
)

PROTOMAN = (
    DEST
    / "event_prototypes_v3/"
      "verified_event_prototype_manifest_v3.csv"
)

PROTOCOL = (
    DEST
    / "EVE_FRAME_STAGE_B_PERMUTED_CONTEXT_MAPPING_PROTOCOL_v3.txt"
)

PROTO_CLOSURE = (
    DEST
    / "EVENT_PROTOTYPE_FINAL_CLOSURE_v3.txt"
)

OUT = DEST / "control_maps_v3"
TMP = DEST / "control_maps_v3_tmp"

TASKS = [
    "primary_frame",
    "stance",
    "misinformation_relation",
]

EXPECTED = {
    "splits":
        "3aa88ac76bf63dd2e39f3a2943ba46bca6b528a427678b49ef1174f4f8cd0584",
    "availability":
        "b2c976c5d2ab5c578774ae5203c5006328618a8ed6d4f4a6aef6d1bb8833bf74",
    "prototype_manifest":
        "50e868bf6e007272a07e7c97ffe277a9f6406f9167e48950ca18b2eaf2f556e8",
    "protocol":
        "cc5bae3423077a99f61dcf74dcc4eb2740edd42f6c63e18ddee79fe840770691",
    "prototype_closure":
        "32afbc814bf2e7fc5b1e18df6725d623cbee2ab58452942105272b9fcc850fe3",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path, expected, label):
    got = sha256(path)
    if got != expected:
        raise RuntimeError(
            f"{label} SHA mismatch\n"
            f"Expected: {expected}\n"
            f"Observed: {got}"
        )


def stable_index(task, fold, phase, doc_id, n):
    key = (
        f"{task}|{fold}|{phase}|{doc_id}"
    ).encode("utf-8")
    value = int(
        hashlib.sha256(key).hexdigest()[:16],
        16,
    )
    return value % n


if OUT.exists() or TMP.exists():
    raise RuntimeError(
        "control_maps_v3 or temporary output already exists."
    )

check(SPLITS, EXPECTED["splits"], "Nested splits")
check(AVAIL, EXPECTED["availability"], "Availability")
check(PROTOMAN, EXPECTED["prototype_manifest"], "Prototype manifest")
check(PROTOCOL, EXPECTED["protocol"], "Mapping protocol")
check(
    PROTO_CLOSURE,
    EXPECTED["prototype_closure"],
    "Prototype closure",
)

splits = pd.read_csv(SPLITS, low_memory=False)
avail = pd.read_csv(AVAIL, dtype={"legacy_doc_id": str})
protoman = pd.read_csv(PROTOMAN, dtype={"legacy_doc_id": str})

if "doc_id" not in splits.columns:
    raise RuntimeError("Nested splits have no doc_id column.")

splits["doc_id"] = splits["doc_id"].astype(str)

# Detect task column.
if "target_task" in splits.columns:
    task_col = "target_task"
elif "task" in splits.columns:
    task_col = "task"
else:
    candidates = []
    for c in splits.columns:
        vals = set(
            splits[c]
            .dropna()
            .astype(str)
            .str.strip()
        )
        if set(TASKS).issubset(vals):
            candidates.append(c)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cannot uniquely identify task column: {candidates}"
        )
    task_col = candidates[0]

# Detect role column.
if "split_role" in splits.columns:
    role_col = "split_role"
elif "role" in splits.columns:
    role_col = "role"
else:
    required_roles = {
        "inner_train",
        "inner_dev",
        "outer_test",
    }
    candidates = []
    for c in splits.columns:
        vals = set(
            splits[c]
            .dropna()
            .astype(str)
            .str.strip()
        )
        if required_roles.issubset(vals):
            candidates.append(c)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cannot uniquely identify role column: {candidates}"
        )
    role_col = candidates[0]

# Detect fold column.
if "outer_fold" in splits.columns:
    fold_col = "outer_fold"
else:
    candidates = []
    for c in splits.columns:
        if c in {"doc_id", task_col, role_col}:
            continue
        numeric = pd.to_numeric(
            splits[c],
            errors="coerce",
        ).dropna()
        if len(numeric) == 0:
            continue
        vals = set(numeric.astype(int).unique())
        if vals.issubset({1, 2, 3, 4}) and len(vals) >= 2:
            candidates.append(c)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cannot uniquely identify fold column: {candidates}"
        )
    fold_col = candidates[0]

splits[task_col] = (
    splits[task_col].astype(str).str.strip()
)
splits[role_col] = (
    splits[role_col].astype(str).str.strip()
)
splits["_outer_fold_int"] = pd.to_numeric(
    splits[fold_col],
    errors="raise",
).astype(int)

avail["context_available"] = pd.to_numeric(
    avail["context_available"],
    errors="raise",
).astype(int)

avail["prototype_row_index"] = pd.to_numeric(
    avail["prototype_row_index"],
    errors="coerce",
).astype("Int64")

if len(avail) != 199:
    raise RuntimeError(f"Expected 199 availability rows, got {len(avail)}.")

if avail["legacy_doc_id"].nunique() != 199:
    raise RuntimeError("Availability doc IDs are not unique.")

if int(avail["context_available"].sum()) != 51:
    raise RuntimeError("Expected exactly 51 context-available documents.")

if len(protoman) != 51:
    raise RuntimeError("Expected exactly 51 prototype-manifest rows.")

availability = {
    str(r.legacy_doc_id): int(r.context_available)
    for r in avail.itertuples(index=False)
}

true_row = {}
for r in avail.itertuples(index=False):
    doc = str(r.legacy_doc_id)
    if int(r.context_available) == 1:
        if pd.isna(r.prototype_row_index):
            raise RuntimeError(
                f"Context document {doc} lacks prototype row."
            )
        true_row[doc] = int(r.prototype_row_index)
    else:
        true_row[doc] = None

all_ids = set(availability)
rows = []
summary = []


def add_phase(
    task,
    fold,
    phase,
    phase_ids,
    source_pool_ids,
    training_derangement,
):
    phase_ids = sorted(phase_ids)

    source_context = sorted(
        d for d in source_pool_ids
        if availability[d] == 1
    )

    if training_derangement:
        context_phase = sorted(
            d for d in phase_ids
            if availability[d] == 1
        )

        if context_phase and len(context_phase) < 2:
            raise RuntimeError(
                f"Cannot derange single context document: "
                f"{task} fold={fold} phase={phase}"
            )

        rotation = {}
        if context_phase:
            for i, doc in enumerate(context_phase):
                source = context_phase[
                    (i + 1) % len(context_phase)
                ]
                if source == doc:
                    raise RuntimeError("Fixed point detected.")
                rotation[doc] = source
    else:
        heldout_context = [
            d for d in phase_ids
            if availability[d] == 1
        ]
        if heldout_context and not source_context:
            raise RuntimeError(
                f"No training prototype pool for held-out context: "
                f"{task} fold={fold} phase={phase}"
            )

    for doc in phase_ids:
        available = availability[doc]

        if available == 0:
            source_doc = None
            assigned_row = None
            assignment_type = "NO_CONTEXT"

        elif training_derangement:
            source_doc = rotation[doc]
            assigned_row = true_row[source_doc]
            assignment_type = "CYCLIC_DERANGEMENT"

        else:
            idx = stable_index(
                task,
                fold,
                phase,
                doc,
                len(source_context),
            )
            source_doc = source_context[idx]
            assigned_row = true_row[source_doc]
            assignment_type = "TRAIN_POOL_HASH_ASSIGNMENT"

        rows.append({
            "target_task": task,
            "outer_fold": fold,
            "phase": phase,
            "legacy_doc_id": doc,
            "context_available": available,
            "true_prototype_row_index": true_row[doc],
            "assigned_source_legacy_doc_id": source_doc,
            "assigned_prototype_row_index": assigned_row,
            "assignment_type": assignment_type,
        })


for task in TASKS:
    for fold in [1, 2, 3, 4]:

        sub = splits[
            splits[task_col].eq(task)
            & splits["_outer_fold_int"].eq(fold)
        ].copy()

        if sub.empty:
            raise RuntimeError(
                f"No split rows for {task}, fold {fold}."
            )

        inner_train = set(
            sub.loc[
                sub[role_col].eq("inner_train"),
                "doc_id",
            ]
        )

        inner_dev = set(
            sub.loc[
                sub[role_col].eq("inner_dev"),
                "doc_id",
            ]
        )

        outer_test = set(
            sub.loc[
                sub[role_col].eq("outer_test"),
                "doc_id",
            ]
        )

        if not inner_train or not inner_dev or not outer_test:
            raise RuntimeError(
                f"Incomplete split for {task}, fold {fold}."
            )

        if (
            inner_train & inner_dev
            or inner_train & outer_test
            or inner_dev & outer_test
        ):
            raise RuntimeError(
                f"Split overlap for {task}, fold {fold}."
            )

        if (inner_train | inner_dev | outer_test) != all_ids:
            raise RuntimeError(
                f"Split does not partition all 199 docs: "
                f"{task}, fold {fold}."
            )

        outer_train = inner_train | inner_dev

        add_phase(
            task, fold,
            "inner_train",
            inner_train,
            inner_train,
            True,
        )

        add_phase(
            task, fold,
            "inner_dev",
            inner_dev,
            inner_train,
            False,
        )

        add_phase(
            task, fold,
            "outer_train",
            outer_train,
            outer_train,
            True,
        )

        add_phase(
            task, fold,
            "outer_test",
            outer_test,
            outer_train,
            False,
        )

        summary.append({
            "target_task": task,
            "outer_fold": fold,
            "n_inner_train": len(inner_train),
            "n_inner_dev": len(inner_dev),
            "n_outer_train": len(outer_train),
            "n_outer_test": len(outer_test),
            "context_inner_train": sum(
                availability[d] for d in inner_train
            ),
            "context_inner_dev": sum(
                availability[d] for d in inner_dev
            ),
            "context_outer_train": sum(
                availability[d] for d in outer_train
            ),
            "context_outer_test": sum(
                availability[d] for d in outer_test
            ),
        })

mapping = pd.DataFrame(rows)

if mapping.duplicated(
    ["target_task", "outer_fold", "phase", "legacy_doc_id"]
).any():
    raise RuntimeError("Duplicate mapping key.")

ctx = mapping["context_available"].eq(1)
noctx = ~ctx

if mapping.loc[ctx, "assigned_prototype_row_index"].isna().any():
    raise RuntimeError(
        "A context-available mapping lacks assigned prototype."
    )

if mapping.loc[noctx, "assigned_prototype_row_index"].notna().any():
    raise RuntimeError(
        "A NO_CONTEXT mapping received a prototype."
    )

training = mapping["phase"].isin(
    ["inner_train", "outer_train"]
) & ctx

if (
    mapping.loc[training, "legacy_doc_id"]
    == mapping.loc[training, "assigned_source_legacy_doc_id"]
).any():
    raise RuntimeError("Training fixed point detected.")

TMP.mkdir(parents=True, exist_ok=False)

map_path = TMP / "permuted_context_mapping_v3.csv"
summary_path = TMP / "permuted_context_mapping_summary_v3.csv"
results_path = (
    TMP
    / "EVE_FRAME_STAGE_B_PERMUTED_CONTEXT_MAPPING_RESULTS_v3.json"
)

mapping.to_csv(map_path, index=False)
pd.DataFrame(summary).to_csv(summary_path, index=False)

results = {
    "status": "PERMUTED_CONTEXT_MAPPING_COMPLETE",
    "task_column": task_col,
    "role_column": role_col,
    "fold_column": fold_col,
    "rows": int(len(mapping)),
    "tasks": TASKS,
    "outer_folds": [1, 2, 3, 4],
    "seed_dependent": False,
    "main_task_labels_used": False,
    "eventgold_status": "SEALED_NOT_ACCESSED",
    "protocol_sha256": sha256(PROTOCOL),
    "split_sha256": sha256(SPLITS),
    "availability_sha256": sha256(AVAIL),
    "prototype_manifest_sha256": sha256(PROTOMAN),
}

results_path.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

files = sorted(
    [p for p in TMP.iterdir() if p.is_file()],
    key=lambda x: x.name,
)

sums = (
    TMP
    / "EVE_FRAME_STAGE_B_PERMUTED_CONTEXT_MAPPING_SHA256SUMS_v3.txt"
)

sums.write_text(
    "".join(
        f"{sha256(p)}  {p.name}\n"
        for p in files
    ),
    encoding="utf-8",
)

TMP.rename(OUT)

print("=" * 80)
print("PERMUTED CONTEXT MAPS COMPLETE")
print("=" * 80)
print("Detected task column:", task_col)
print("Detected role column:", role_col)
print("Detected fold column:", fold_col)
print("Mapping rows:", len(mapping))
print("Mapping SHA256:", sha256(OUT / map_path.name))
print("Summary SHA256:", sha256(OUT / summary_path.name))
print("Results SHA256:", sha256(OUT / results_path.name))
print("✅ Same availability mask preserved.")
print("✅ No training fixed points.")
print("✅ Held-out prototypes sourced only from training pools.")
print("✅ Mapping independent of model seed.")
print("✅ No task labels used.")
print("✅ EventGold not accessed.")
