from pathlib import Path
import hashlib
import json
import math

import pandas as pd


ROOT = Path.cwd()

DEST = (
    ROOT
    / "01_event_aware_v3/08_eve_frame_stage_b_event_context"
)

MANIFEST = DEST / "stage_b_manifest_v3.csv"

SPLITS = (
    ROOT
    / "01_event_aware_v3/05_supervised_transformers/"
      "frozen_finetune_design_v3/"
      "finetune_nested_splits_v3.csv"
)

BASE = DEST / "62_stage_b_base_from_stage_a_v3.py"
WRAPPER = DEST / "63_train_eve_frame_stage_b_v3.py"

RUNROOT = DEST / "stage_b_runs_v3"

OUT = DEST / "stage_b_run_audit_v3"
TMP = DEST / "stage_b_run_audit_v3_tmp"


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


if OUT.exists() or TMP.exists():
    raise RuntimeError(
        "Audit output already exists. "
        "Do not overwrite without inspection."
    )


manifest = pd.read_csv(
    MANIFEST
)

splits = pd.read_csv(
    SPLITS,
    dtype={"doc_id": str},
)


# ------------------------------------------------------------
# Manifest integrity
# ------------------------------------------------------------

if len(manifest) != 120:
    raise RuntimeError(
        f"Expected 120 manifest rows, got {len(manifest)}."
    )

if manifest["run_id"].tolist() != list(range(120)):
    raise RuntimeError(
        "run_id is not exactly 0..119."
    )

if manifest.duplicated(
    [
        "arm",
        "target_task",
        "outer_fold",
        "model_seed",
    ]
).any():
    raise RuntimeError(
        "Duplicate Stage-B configuration."
    )


expected_arms = {
    "VERIFIED_CONTEXT",
    "PERMUTED_CONTEXT",
}

if set(manifest["arm"]) != expected_arms:
    raise RuntimeError(
        f"Unexpected arms: {set(manifest['arm'])}"
    )


# ------------------------------------------------------------
# Split integrity
# ------------------------------------------------------------

required_split_cols = {
    "task",
    "outer_fold",
    "doc_id",
    "role",
}

missing = required_split_cols - set(splits.columns)

if missing:
    raise RuntimeError(
        f"Missing split columns: {sorted(missing)}"
    )

splits["outer_fold"] = pd.to_numeric(
    splits["outer_fold"],
    errors="raise",
).astype(int)


# ------------------------------------------------------------
# Frozen code hashes
# ------------------------------------------------------------

base_sha = sha256(BASE)
wrapper_sha = sha256(WRAPPER)


# ------------------------------------------------------------
# Per-run artifact audit
# ------------------------------------------------------------

records = []
errors = []


for r in manifest.itertuples(index=False):

    run_id = int(r.run_id)
    arm = str(r.arm)
    task = str(r.target_task)
    fold = int(r.outer_fold)
    seed = int(r.model_seed)

    arm_dir = arm.lower()

    run_name = (
        f"{task}"
        f"__fold{fold}"
        f"__seed{seed}"
    )

    run_dir = (
        RUNROOT
        / arm_dir
        / run_name
    )

    summary_path = (
        run_dir
        / "run_summary_v3.json"
    )

    pred_path = (
        run_dir
        / "outer_test_predictions_v3.csv"
    )

    history_path = (
        run_dir
        / "inner_selection_history_v3.csv"
    )

    run_errors = []

    for path, label in [
        (summary_path, "summary"),
        (pred_path, "predictions"),
        (history_path, "selection_history"),
    ]:
        if not path.exists():
            run_errors.append(
                f"missing_{label}"
            )

    if run_errors:
        errors.append({
            "run_id": run_id,
            "arm": arm,
            "task": task,
            "fold": fold,
            "seed": seed,
            "errors": "|".join(run_errors),
        })
        continue


    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    expected_arch = (
        f"EVE_FRAME_STAGE_B_{arm}"
    )


    # --------------------------------------------------------
    # Summary metadata
    # --------------------------------------------------------

    checks = {
        "status":
            summary.get("status") == "COMPLETE",

        "architecture":
            summary.get("architecture") == expected_arch,

        "stage":
            summary.get("stage") == "B",

        "event_context":
            summary.get("event_context") == arm,

        "event_mapping_overlap":
            summary.get("event_mapping_overlap")
            == "51_of_199_verified_context_available",

        "target_task":
            summary.get("target_task") == task,

        "outer_fold":
            int(summary.get("outer_fold", -1)) == fold,

        "seed":
            int(summary.get("seed", -1)) == seed,

        "initialization":
            summary.get("initialization") == "XLMR_BASE",

        "n_documents":
            int(summary.get("n_documents", -1)) == 199,

        "eventgold":
            summary.get("eventgold_status")
            == "SEALED_NOT_ACCESSED",

        # This field comes from the inherited base module.
        "base_trainer_sha":
            summary.get("trainer_sha256") == base_sha,
    }

    for name, ok in checks.items():
        if not ok:
            run_errors.append(
                f"summary_{name}"
            )


    # --------------------------------------------------------
    # Expected held-out documents from frozen split
    # --------------------------------------------------------

    split = splits[
        splits["task"].astype(str).eq(task)
        & splits["outer_fold"].eq(fold)
    ].copy()

    if len(split) != 199:
        run_errors.append(
            f"split_rows_{len(split)}"
        )

    test_ids = set(
        split.loc[
            split["role"].eq("outer_test"),
            "doc_id",
        ].astype(str)
    )


    # --------------------------------------------------------
    # Prediction artifact
    # --------------------------------------------------------

    pred = pd.read_csv(
        pred_path,
        dtype={"doc_id": str},
    )

    if "doc_id" not in pred.columns:
        run_errors.append(
            "pred_missing_doc_id"
        )

    else:
        if pred["doc_id"].duplicated().any():
            run_errors.append(
                "pred_duplicate_doc_id"
            )

        pred_ids = set(
            pred["doc_id"].astype(str)
        )

        if pred_ids != test_ids:
            run_errors.append(
                "pred_outer_test_doc_set_mismatch"
            )

    if len(pred) != len(test_ids):
        run_errors.append(
            "pred_row_count_mismatch"
        )

    if int(
        summary.get(
            "n_outer_test",
            -1,
        )
    ) != len(test_ids):
        run_errors.append(
            "summary_n_outer_test_mismatch"
        )


    # --------------------------------------------------------
    # Prediction provenance metadata
    # --------------------------------------------------------

    expected_pred_values = {
        "task": task,
        "outer_fold": fold,
        "architecture": expected_arch,
        "initialization": "XLMR_BASE",
        "seed": seed,
    }

    for col, expected_value in expected_pred_values.items():

        if col not in pred.columns:
            run_errors.append(
                f"pred_missing_{col}"
            )
            continue

        values = set(
            pred[col]
            .dropna()
            .astype(str)
        )

        if col in {
            "outer_fold",
            "seed",
        }:
            expected_set = {
                str(int(expected_value))
            }
        else:
            expected_set = {
                str(expected_value)
            }

        if values != expected_set:
            run_errors.append(
                f"pred_{col}_mismatch"
            )


    # --------------------------------------------------------
    # Numeric sanity only — no performance interpretation
    # --------------------------------------------------------

    finite_fields = [
        "selected_dev_macro_f1",
        "outer_test_macro_f1",
        "outer_test_balanced_accuracy",
        "outer_test_accuracy",
        "outer_test_weighted_f1",
        "elapsed_seconds",
        "peak_gpu_memory_gb",
    ]

    for field in finite_fields:

        value = summary.get(field)

        try:
            finite = math.isfinite(
                float(value)
            )
        except Exception:
            finite = False

        if not finite:
            run_errors.append(
                f"nonfinite_{field}"
            )


    selected_epoch = int(
        summary.get(
            "selected_epoch",
            -1,
        )
    )

    if not 1 <= selected_epoch <= 8:
        run_errors.append(
            "selected_epoch_out_of_range"
        )


    n_inner_train = int(
        summary.get(
            "n_inner_train",
            -1,
        )
    )

    n_inner_dev = int(
        summary.get(
            "n_inner_dev",
            -1,
        )
    )

    n_outer_train = int(
        summary.get(
            "n_outer_train_refit",
            -1,
        )
    )

    n_outer_test = int(
        summary.get(
            "n_outer_test",
            -1,
        )
    )

    if (
        n_inner_train
        + n_inner_dev
        != n_outer_train
    ):
        run_errors.append(
            "train_partition_count_mismatch"
        )

    if (
        n_outer_train
        + n_outer_test
        != 199
    ):
        run_errors.append(
            "outer_partition_count_mismatch"
        )


    record = {
        "run_id": run_id,
        "arm": arm,
        "target_task": task,
        "outer_fold": fold,
        "seed": seed,
        "valid": len(run_errors) == 0,
        "n_outer_test": len(pred),
        "selected_epoch": selected_epoch,
        "summary_sha256":
            sha256(summary_path),
        "predictions_sha256":
            sha256(pred_path),
        "selection_history_sha256":
            sha256(history_path),
        "errors":
            "|".join(run_errors),
    }

    records.append(record)

    if run_errors:
        errors.append(record)


audit = pd.DataFrame(records)

if len(audit) != 120:
    raise RuntimeError(
        f"Only {len(audit)} complete run records audited."
    )


# ------------------------------------------------------------
# Global structural checks
# ------------------------------------------------------------

if errors:
    print(
        pd.DataFrame(errors).to_string(
            index=False
        )
    )

    raise RuntimeError(
        f"{len(errors)} Stage-B runs failed structural audit."
    )


if int(audit["valid"].sum()) != 120:
    raise RuntimeError(
        "Expected exactly 120 valid Stage-B runs."
    )


group_counts = (
    audit
    .groupby(
        [
            "arm",
            "target_task",
        ]
    )
    .size()
)

if not (
    group_counts == 20
).all():
    raise RuntimeError(
        "Expected 20 runs per arm/task."
    )


actual_summary_files = list(
    RUNROOT.rglob(
        "run_summary_v3.json"
    )
)

actual_prediction_files = list(
    RUNROOT.rglob(
        "outer_test_predictions_v3.csv"
    )
)

actual_history_files = list(
    RUNROOT.rglob(
        "inner_selection_history_v3.csv"
    )
)

if len(actual_summary_files) != 120:
    raise RuntimeError(
        "Unexpected total summary-file count."
    )

if len(actual_prediction_files) != 120:
    raise RuntimeError(
        "Unexpected total prediction-file count."
    )

if len(actual_history_files) != 120:
    raise RuntimeError(
        "Unexpected total selection-history count."
    )


# ------------------------------------------------------------
# Write audit only after all checks pass
# ------------------------------------------------------------

TMP.mkdir(
    parents=True,
    exist_ok=False,
)

audit_path = (
    TMP
    / "stage_b_run_audit_v3.csv"
)

results_path = (
    TMP
    / "STAGE_B_RUN_AUDIT_RESULTS_v3.json"
)

audit.sort_values(
    "run_id"
).to_csv(
    audit_path,
    index=False,
)

results = {
    "status":
        "STAGE_B_RUN_AUDIT_COMPLETE",

    "expected_runs":
        120,

    "audited_runs":
        120,

    "valid_runs":
        120,

    "invalid_runs":
        0,

    "arms": {
        arm: int(
            (
                audit["arm"]
                == arm
            ).sum()
        )
        for arm in sorted(
            audit["arm"].unique()
        )
    },

    "runs_per_arm_task": {
        f"{arm}|{task}":
            int(value)
        for (
            arm,
            task,
        ), value
        in group_counts.items()
    },

    "main_performance_aggregation_performed":
        False,

    "eventgold_status":
        "SEALED_NOT_ACCESSED",

    "base_module_sha256":
        base_sha,

    "execution_wrapper_sha256":
        wrapper_sha,

    "manifest_sha256":
        sha256(MANIFEST),

    "splits_sha256":
        sha256(SPLITS),
}

results_path.write_text(
    json.dumps(
        results,
        indent=2,
    ),
    encoding="utf-8",
)

files = sorted(
    [
        p
        for p in TMP.iterdir()
        if p.is_file()
    ],
    key=lambda p: p.name,
)

checksum_path = (
    TMP
    / "STAGE_B_RUN_AUDIT_SHA256SUMS_v3.txt"
)

checksum_path.write_text(
    "".join(
        f"{sha256(p)}  {p.name}\n"
        for p in files
    ),
    encoding="utf-8",
)

TMP.rename(OUT)

print("=" * 88)
print("STAGE-B STRUCTURAL RUN AUDIT COMPLETE")
print("=" * 88)
print("Expected runs : 120")
print("Audited runs  :", len(audit))
print("Valid runs    :", int(audit["valid"].sum()))
print("Invalid runs  :", int((~audit["valid"]).sum()))
print()
print(group_counts)
print()
print("Base SHA      :", base_sha)
print("Wrapper SHA   :", wrapper_sha)
print()
print("No performance aggregation performed.")
print("EventGold remains SEALED_NOT_ACCESSED.")
print()
print("✅ ALL 120 STAGE-B RUNS STRUCTURALLY VALID.")
