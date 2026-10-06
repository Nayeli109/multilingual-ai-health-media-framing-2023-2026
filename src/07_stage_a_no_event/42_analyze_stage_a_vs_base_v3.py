from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


ROOT = Path.cwd()

STAGE = (
    ROOT
    / "01_event_aware_v3/07_eve_frame_stage_a_no_event"
)

RUNROOT = STAGE / "runs_v3"

MANIFEST = STAGE / "stage_a_manifest_v3.csv"

AUDIT = (
    STAGE
    / "audit_v3/stage_a_audit_summary_v3.json"
)

TRAINER = (
    STAGE
    / "37_train_eve_frame_stage_a_v3.py"
)

PLAN = (
    STAGE
    / "STAGE_A_COMPARISON_ANALYSIS_PLAN_v3.txt"
)

LOCK = (
    STAGE
    / "STAGE_A_STATISTICAL_IMPLEMENTATION_LOCK_v3.txt"
)

DATA = (
    ROOT
    / "01_event_aware_v3/03_scalegold_active_learning/"
      "LegacyAux_leakage_free_v3.csv"
)

SPLITS = (
    ROOT
    / "01_event_aware_v3/05_supervised_transformers/"
      "frozen_finetune_design_v3/"
      "finetune_nested_splits_v3.csv"
)

BASE_ANALYSIS = (
    ROOT
    / "01_event_aware_v3/06_supervised_analysis_v3"
)

BASE_ENSEMBLE = (
    BASE_ANALYSIS
    / "results_v3/oof_ensemble_predictions_v3.csv"
)

BASE_SEED_STABILITY = (
    BASE_ANALYSIS
    / "results_v3/seed_stability_v3.csv"
)

BASE_CLOSURE = (
    BASE_ANALYSIS
    / "SUPERVISED_BASE_DAPT_CLOSURE_v3.txt"
)

OUTDIR = (
    STAGE
    / "comparison_results_v3"
)

TMPDIR = (
    STAGE
    / "comparison_results_v3_tmp"
)


EXPECTED_PLAN_SHA = (
    "423ba34ca3b44da3dd65c91dc5ea35ec41eec510008a581a5d4af1c9d2738bcf"
)

EXPECTED_LOCK_SHA = (
    "1eb61ad1c00f749d657f03497f26c9664c6d22bb700d0c111d48d7b618986084"
)

EXPECTED_TRAINER_SHA = (
    "62f5f55fcdeaf4e5c947313bcc326daa8ceb225e2ab954ffa7b24430e1e760b9"
)

EXPECTED_DATA_SHA = (
    "81523ab0ada98c4ef3f8b2adeed03f888dd94c8a746dc9c80661cf4ceb16f583"
)

EXPECTED_SPLITS_SHA = (
    "3aa88ac76bf63dd2e39f3a2943ba46bca6b528a427678b49ef1174f4f8cd0584"
)

EXPECTED_BASE_ENSEMBLE_SHA = (
    "b725357f84ba77cc47ba2484f34f5fea57a069e997d87a8d7fd5cf8ff2ff5cab"
)

EXPECTED_BASE_SEED_SHA = (
    "244f7d1b64f195feffd6c08ebb822e1f7b68e19899a28d049a26d1daf3c73fcb"
)

EXPECTED_BASE_CLOSURE_SHA = (
    "7ff55b374923c6a3650f399d5b9299ab5bcda6937709adf67356d98d438f0546"
)

TASKS = [
    "primary_frame",
    "stance",
    "misinformation_relation",
]

SEEDS = [
    11,
    29,
    47,
    83,
    131,
]

N_DOCS = 199
BOOTSTRAP_B = 10_000
RANDOMIZATION_B = 10_000
RNG_SEED = 20260921


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def macro_f1_fast(
    y_true,
    y_pred,
    n_classes,
):
    cm = np.bincount(
        y_true * n_classes + y_pred,
        minlength=n_classes * n_classes,
    ).reshape(
        n_classes,
        n_classes,
    )

    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    denom = (
        2.0 * tp
        + fp
        + fn
    )

    f1 = np.divide(
        2.0 * tp,
        denom,
        out=np.zeros_like(tp),
        where=denom != 0,
    )

    return float(
        f1.mean()
    )


def metric_dict(
    y_true,
    y_pred,
    labels,
):
    return {
        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                )
            ),
    }


def bh_fdr(p_values):
    p = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p)

    order = np.argsort(
        p
    )

    ranked = p[
        order
    ]

    q_ranked = (
        ranked
        * m
        / np.arange(
            1,
            m + 1,
        )
    )

    q_ranked = (
        np.minimum.accumulate(
            q_ranked[::-1]
        )[::-1]
    )

    q_ranked = np.clip(
        q_ranked,
        0.0,
        1.0,
    )

    q = np.empty(
        m,
        dtype=float,
    )

    q[
        order
    ] = q_ranked

    return q


def extract_gold_map(
    df,
    task,
):
    x = df[
        df["task"] == task
    ][
        [
            "gold_id",
            "gold_label",
        ]
    ].drop_duplicates()

    if (
        x["gold_id"].duplicated().any()
        or
        x["gold_label"].duplicated().any()
    ):
        raise RuntimeError(
            f"{task}: gold ID/label mapping "
            "is not one-to-one."
        )

    m = {
        int(r.gold_id):
            str(r.gold_label)
        for r
        in x.itertuples(
            index=False
        )
    }

    ids = sorted(
        m
    )

    if ids != list(
        range(
            len(ids)
        )
    ):
        raise RuntimeError(
            f"{task}: class IDs are not contiguous."
        )

    return m


# ============================================================
# 1. PRE-RESULT FROZEN INPUT VALIDATION
# ============================================================

checks = {
    "analysis_plan":
        (
            PLAN,
            EXPECTED_PLAN_SHA,
        ),

    "implementation_lock":
        (
            LOCK,
            EXPECTED_LOCK_SHA,
        ),

    "stage_a_trainer":
        (
            TRAINER,
            EXPECTED_TRAINER_SHA,
        ),

    "development_data":
        (
            DATA,
            EXPECTED_DATA_SHA,
        ),

    "frozen_splits":
        (
            SPLITS,
            EXPECTED_SPLITS_SHA,
        ),

    "frozen_base_ensemble":
        (
            BASE_ENSEMBLE,
            EXPECTED_BASE_ENSEMBLE_SHA,
        ),

    "frozen_base_seed_stability":
        (
            BASE_SEED_STABILITY,
            EXPECTED_BASE_SEED_SHA,
        ),

    "base_dapt_closure":
        (
            BASE_CLOSURE,
            EXPECTED_BASE_CLOSURE_SHA,
        ),
}

for name, (
    path,
    expected,
) in checks.items():

    observed = sha256(
        path
    )

    if observed != expected:
        raise RuntimeError(
            f"{name} SHA mismatch\n"
            f"Expected: {expected}\n"
            f"Observed: {observed}"
        )


if OUTDIR.exists():
    raise RuntimeError(
        f"Final output directory already exists: "
        f"{OUTDIR}\n"
        "Refusing to overwrite a prior analysis."
    )

if TMPDIR.exists():
    raise RuntimeError(
        f"Temporary output directory already exists: "
        f"{TMPDIR}\n"
        "Inspect/remove it manually before rerunning."
    )

TMPDIR.mkdir(
    parents=True,
    exist_ok=False,
)


audit = json.loads(
    AUDIT.read_text()
)

required_audit = {
    "expected_runs":
        60,

    "summaries_found":
        60,

    "valid_runs":
        60,

    "missing_or_invalid_runs":
        0,

    "all_60_valid":
        True,
}

for key, expected in (
    required_audit.items()
):
    if audit.get(key) != expected:
        raise RuntimeError(
            f"Stage-A audit mismatch: "
            f"{key}={audit.get(key)!r}; "
            f"expected {expected!r}"
        )


manifest = pd.read_csv(
    MANIFEST
)

if len(manifest) != 60:
    raise RuntimeError(
        f"Expected 60 Stage-A runs; "
        f"got {len(manifest)}"
    )

if manifest[
    "run_id"
].tolist() != list(
    range(60)
):
    raise RuntimeError(
        "Stage-A run_id must be exactly 0..59."
    )

if sorted(
    manifest[
        "target_task"
    ].unique().tolist()
) != sorted(
    TASKS
):
    raise RuntimeError(
        "Unexpected target-task set."
    )

if sorted(
    manifest[
        "model_seed"
    ].unique().tolist()
) != SEEDS:
    raise RuntimeError(
        "Unexpected Stage-A seed set."
    )

if sorted(
    manifest[
        "outer_fold"
    ].unique().tolist()
) != [
    1,
    2,
    3,
    4,
]:
    raise RuntimeError(
        "Unexpected Stage-A fold set."
    )


# ============================================================
# 2. LOAD + VALIDATE 60 STAGE-A PREDICTION FILES
# ============================================================

stage_frames = []
input_hash_rows = []
task_prob_counts = {
    task: set()
    for task in TASKS
}


for r in manifest.itertuples(
    index=False
):

    run_name = (
        f"{r.target_task}"
        f"__fold{int(r.outer_fold)}"
        f"__seed{int(r.model_seed)}"
    )

    run_dir = (
        RUNROOT
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

    if (
        not summary_path.exists()
        or
        not pred_path.exists()
    ):
        raise RuntimeError(
            f"Missing Stage-A output: {run_name}"
        )

    summary = json.loads(
        summary_path.read_text()
    )

    required_summary = [
        summary.get("status")
            == "COMPLETE",

        summary.get("architecture")
            == "EVE_FRAME_STAGE_A_NO_EVENT",

        summary.get("event_context")
            == "DISABLED",

        summary.get("eventgold_status")
            == "SEALED_NOT_ACCESSED",

        summary.get("target_task")
            == r.target_task,

        int(
            summary.get(
                "outer_fold",
                -1,
            )
        )
            == int(
                r.outer_fold
            ),

        int(
            summary.get(
                "seed",
                -1,
            )
        )
            == int(
                r.model_seed
            ),

        summary.get("trainer_sha256")
            == EXPECTED_TRAINER_SHA,

        summary.get("data_sha256")
            == EXPECTED_DATA_SHA,

        summary.get("splits_sha256")
            == EXPECTED_SPLITS_SHA,
    ]

    if not all(
        required_summary
    ):
        raise RuntimeError(
            f"Invalid Stage-A summary: {run_name}"
        )

    df = pd.read_csv(
        pred_path
    )

    prob_cols = sorted(
        [
            c
            for c
            in df.columns
            if c.startswith(
                "prob_class_"
            )
        ],
        key=lambda x:
            int(
                x.split("_")[-1]
            ),
    )

    if not prob_cols:
        raise RuntimeError(
            f"{run_name}: no probability columns."
        )

    prob_ids = [
        int(
            c.split("_")[-1]
        )
        for c in prob_cols
    ]

    if prob_ids != list(
        range(
            len(prob_ids)
        )
    ):
        raise RuntimeError(
            f"{run_name}: non-contiguous "
            "probability columns."
        )

    task_prob_counts[
        r.target_task
    ].add(
        len(prob_cols)
    )

    required_cols = {
        "doc_id",
        "task",
        "outer_fold",
        "architecture",
        "initialization",
        "seed",
        "gold_id",
        "gold_label",
        "prediction_id",
        "prediction_label",
        *prob_cols,
    }

    missing = (
        required_cols
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{run_name}: missing columns "
            f"{sorted(missing)}"
        )

    if set(
        df["task"].astype(str)
    ) != {
        str(
            r.target_task
        )
    }:
        raise RuntimeError(
            f"{run_name}: task mismatch."
        )

    if set(
        df[
            "architecture"
        ].astype(str)
    ) != {
        "EVE_FRAME_STAGE_A_NO_EVENT"
    }:
        raise RuntimeError(
            f"{run_name}: architecture mismatch."
        )

    if set(
        df[
            "initialization"
        ].astype(str)
    ) != {
        "XLMR_BASE"
    }:
        raise RuntimeError(
            f"{run_name}: initialization mismatch."
        )

    if set(
        df[
            "outer_fold"
        ].astype(int)
    ) != {
        int(
            r.outer_fold
        )
    }:
        raise RuntimeError(
            f"{run_name}: fold mismatch."
        )

    if set(
        df[
            "seed"
        ].astype(int)
    ) != {
        int(
            r.model_seed
        )
    }:
        raise RuntimeError(
            f"{run_name}: seed mismatch."
        )

    if df[
        "doc_id"
    ].astype(str).duplicated().any():
        raise RuntimeError(
            f"{run_name}: duplicate doc_id."
        )

    probs = df[
        prob_cols
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        probs
    ).all():
        raise RuntimeError(
            f"{run_name}: non-finite probabilities."
        )

    if (
        (probs < -1e-8).any()
        or
        (probs > 1.0 + 1e-8).any()
    ):
        raise RuntimeError(
            f"{run_name}: probability outside [0,1]."
        )

    if not np.allclose(
        probs.sum(
            axis=1
        ),
        1.0,
        atol=5e-5,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{run_name}: probabilities do not sum to 1."
        )

    if not np.array_equal(
        np.argmax(
            probs,
            axis=1,
        ),
        df[
            "prediction_id"
        ].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            f"{run_name}: prediction != argmax."
        )

    z = df.copy()

    z[
        "doc_id"
    ] = z[
        "doc_id"
    ].astype(str)

    z[
        "run_id"
    ] = int(
        r.run_id
    )

    stage_frames.append(
        z
    )

    input_hash_rows.append({
        "run_id":
            int(
                r.run_id
            ),

        "run_name":
            run_name,

        "summary_sha256":
            sha256(
                summary_path
            ),

        "prediction_sha256":
            sha256(
                pred_path
            ),
    })


for task in TASKS:
    if len(
        task_prob_counts[
            task
        ]
    ) != 1:
        raise RuntimeError(
            f"{task}: inconsistent number "
            "of probability columns."
        )


stage_raw = pd.concat(
    stage_frames,
    ignore_index=True,
    sort=False,
)

if len(
    stage_raw
) != 2985:
    raise RuntimeError(
        f"Expected 2985 Stage-A prediction rows; "
        f"got {len(stage_raw)}"
    )


pd.DataFrame(
    input_hash_rows
).sort_values(
    "run_id"
).to_csv(
    TMPDIR
    / "stage_a_input_prediction_hashes_v3.csv",
    index=False,
)


# ============================================================
# 3. LOAD FROZEN BASE ENSEMBLE + LABEL MAPS
# ============================================================

base_all = pd.read_csv(
    BASE_ENSEMBLE
)

base = base_all[
    base_all[
        "initialization"
    ]
    == "XLMR_BASE"
].copy()

base[
    "doc_id"
] = base[
    "doc_id"
].astype(str)

if len(
    base
) != (
    len(TASKS)
    * N_DOCS
):
    raise RuntimeError(
        f"Expected 597 BASE OOF rows; "
        f"got {len(base)}"
    )


base_maps = {}
stage_maps = {}


for task in TASKS:

    base_maps[
        task
    ] = extract_gold_map(
        base,
        task,
    )

    stage_maps[
        task
    ] = extract_gold_map(
        stage_raw,
        task,
    )

    if set(
        base_maps[
            task
        ].values()
    ) != set(
        stage_maps[
            task
        ].values()
    ):
        raise RuntimeError(
            f"{task}: Stage-A and BASE "
            "class-label sets differ."
        )


# ============================================================
# 4. STAGE-A SEED-SPECIFIC OOF STABILITY
# ============================================================

stage_seed_rows = []


for task in TASKS:

    stage_map = (
        stage_maps[
            task
        ]
    )

    labels = sorted(
        stage_map
    )

    for seed in SEEDS:

        g = stage_raw[
            (
                stage_raw["task"]
                == task
            )
            &
            (
                stage_raw["seed"]
                == seed
            )
        ].copy()

        if len(g) != N_DOCS:
            raise RuntimeError(
                f"{task}/seed{seed}: "
                f"expected 199 rows; got {len(g)}"
            )

        if g[
            "doc_id"
        ].duplicated().any():
            raise RuntimeError(
                f"{task}/seed{seed}: "
                "duplicate OOF document."
            )

        m = metric_dict(
            g[
                "gold_id"
            ].astype(int).to_numpy(),

            g[
                "prediction_id"
            ].astype(int).to_numpy(),

            labels,
        )

        stage_seed_rows.append({
            "task":
                task,

            "system":
                "EVE_FRAME_STAGE_A_NO_EVENT",

            "seed":
                seed,

            "n_docs":
                len(g),

            **m,
        })


stage_seed = pd.DataFrame(
    stage_seed_rows
)

stage_seed.to_csv(
    TMPDIR
    / "stage_a_seed_stability_v3.csv",
    index=False,
)


base_seed = pd.read_csv(
    BASE_SEED_STABILITY
)

base_seed = base_seed[
    base_seed[
        "initialization"
    ]
    == "XLMR_BASE"
].copy()

base_seed[
    "system"
] = "XLMR_BASE_SINGLE_TASK"

base_seed = base_seed[
    [
        "task",
        "system",
        "seed",
        "n_docs",
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "weighted_f1",
    ]
]


seed_comparison = pd.concat(
    [
        base_seed,
        stage_seed,
    ],
    ignore_index=True,
)

seed_comparison.to_csv(
    TMPDIR
    / "seed_stability_comparison_v3.csv",
    index=False,
)


# ============================================================
# 5. BUILD FIVE-SEED STAGE-A OOF ENSEMBLE
#
# Stage-A probabilities are remapped to BASE class-ID order
# by class label before averaging/comparison.
# ============================================================

stage_ensemble_rows = []


for task in TASKS:

    base_map = (
        base_maps[
            task
        ]
    )

    stage_map = (
        stage_maps[
            task
        ]
    )

    base_ids = sorted(
        base_map
    )

    stage_label_to_id = {
        label: class_id
        for class_id, label
        in stage_map.items()
    }

    stage_ids_in_base_order = [
        stage_label_to_id[
            base_map[
                base_id
            ]
        ]
        for base_id
        in base_ids
    ]

    native_prob_cols = [
        f"prob_class_{i}"
        for i in sorted(
            stage_map
        )
    ]

    all_doc_ids = []

    for fold in [
        1,
        2,
        3,
        4,
    ]:

        g = stage_raw[
            (
                stage_raw[
                    "task"
                ]
                == task
            )
            &
            (
                stage_raw[
                    "outer_fold"
                ]
                == fold
            )
        ].copy()

        for doc_id, d in g.groupby(
            "doc_id",
            sort=True,
        ):

            if len(d) != 5:
                raise RuntimeError(
                    f"{task}/fold{fold}/{doc_id}: "
                    f"expected 5 seeds, got {len(d)}"
                )

            if set(
                d[
                    "seed"
                ].astype(int)
            ) != set(
                SEEDS
            ):
                raise RuntimeError(
                    f"{task}/fold{fold}/{doc_id}: "
                    "seed set mismatch."
                )

            if d[
                "gold_id"
            ].nunique() != 1:
                raise RuntimeError(
                    f"{task}/fold{fold}/{doc_id}: "
                    "inconsistent gold ID."
                )

            if d[
                "gold_label"
            ].nunique() != 1:
                raise RuntimeError(
                    f"{task}/fold{fold}/{doc_id}: "
                    "inconsistent gold label."
                )

            native_mean = (
                d[
                    native_prob_cols
                ]
                .to_numpy(
                    dtype=float
                )
                .mean(
                    axis=0
                )
            )

            aligned_prob = np.asarray(
                [
                    native_mean[
                        stage_id
                    ]
                    for stage_id
                    in stage_ids_in_base_order
                ],
                dtype=float,
            )

            gold_label = str(
                d[
                    "gold_label"
                ].iloc[0]
            )

            base_label_to_id = {
                label: class_id
                for class_id, label
                in base_map.items()
            }

            gold_id = int(
                base_label_to_id[
                    gold_label
                ]
            )

            pred_id = int(
                np.argmax(
                    aligned_prob
                )
            )

            row = {
                "task":
                    task,

                "system":
                    "EVE_FRAME_STAGE_A_NO_EVENT",

                "doc_id":
                    str(
                        doc_id
                    ),

                "outer_fold":
                    int(
                        fold
                    ),

                "gold_id":
                    gold_id,

                "gold_label":
                    base_map[
                        gold_id
                    ],

                "prediction_id":
                    pred_id,

                "prediction_label":
                    base_map[
                        pred_id
                    ],

                "n_seeds":
                    5,
            }

            for class_id, value in zip(
                base_ids,
                aligned_prob,
            ):
                row[
                    f"prob_class_{class_id}"
                ] = float(
                    value
                )

            stage_ensemble_rows.append(
                row
            )

            all_doc_ids.append(
                str(
                    doc_id
                )
            )

    if len(
        all_doc_ids
    ) != N_DOCS:
        raise RuntimeError(
            f"{task}: Stage-A ensemble "
            f"expected 199 docs; got "
            f"{len(all_doc_ids)}"
        )

    if len(
        set(
            all_doc_ids
        )
    ) != N_DOCS:
        raise RuntimeError(
            f"{task}: document appears in "
            "multiple outer folds."
        )


stage_ensemble = pd.DataFrame(
    stage_ensemble_rows
)

if len(
    stage_ensemble
) != 597:
    raise RuntimeError(
        f"Expected 597 Stage-A ensemble rows; "
        f"got {len(stage_ensemble)}"
    )


stage_ensemble.to_csv(
    TMPDIR
    / "stage_a_oof_ensemble_predictions_v3.csv",
    index=False,
)


# ============================================================
# 6. TASK + CLASS METRICS
# ============================================================

task_metric_rows = []
class_metric_rows = []
cm_rows = []


systems = [
    (
        "XLMR_BASE_SINGLE_TASK",
        base,
    ),

    (
        "EVE_FRAME_STAGE_A_NO_EVENT",
        stage_ensemble,
    ),
]


for task in TASKS:

    label_map = (
        base_maps[
            task
        ]
    )

    labels = sorted(
        label_map
    )

    for system_name, table in systems:

        g = table[
            table[
                "task"
            ]
            == task
        ].copy()

        if len(g) != N_DOCS:
            raise RuntimeError(
                f"{task}/{system_name}: "
                "expected 199 rows."
            )

        y = g[
            "gold_id"
        ].astype(int).to_numpy()

        pred = g[
            "prediction_id"
        ].astype(int).to_numpy()

        metrics = metric_dict(
            y,
            pred,
            labels,
        )

        task_metric_rows.append({
            "task":
                task,

            "system":
                system_name,

            "n_docs":
                len(g),

            "n_classes":
                len(
                    labels
                ),

            **metrics,
        })

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y,
                pred,
                labels=labels,
                zero_division=0,
            )
        )

        for i, class_id in enumerate(
            labels
        ):

            class_metric_rows.append({
                "task":
                    task,

                "system":
                    system_name,

                "class_id":
                    int(
                        class_id
                    ),

                "class_label":
                    label_map[
                        class_id
                    ],

                "support":
                    int(
                        support[i]
                    ),

                "precision":
                    float(
                        precision[i]
                    ),

                "recall":
                    float(
                        recall[i]
                    ),

                "f1":
                    float(
                        f1[i]
                    ),
            })

        cm = confusion_matrix(
            y,
            pred,
            labels=labels,
        )

        for i, gold_id in enumerate(
            labels
        ):
            for j, pred_id in enumerate(
                labels
            ):

                cm_rows.append({
                    "task":
                        task,

                    "system":
                        system_name,

                    "gold_id":
                        int(
                            gold_id
                        ),

                    "gold_label":
                        label_map[
                            gold_id
                        ],

                    "prediction_id":
                        int(
                            pred_id
                        ),

                    "prediction_label":
                        label_map[
                            pred_id
                        ],

                    "count":
                        int(
                            cm[i, j]
                        ),
                })


task_metrics = pd.DataFrame(
    task_metric_rows
)

per_class = pd.DataFrame(
    class_metric_rows
)

confusions = pd.DataFrame(
    cm_rows
)


task_metrics.to_csv(
    TMPDIR
    / "task_metrics_comparison_v3.csv",
    index=False,
)

per_class.to_csv(
    TMPDIR
    / "per_class_metrics_comparison_v3.csv",
    index=False,
)

confusions.to_csv(
    TMPDIR
    / "confusion_matrices_comparison_v3.csv",
    index=False,
)


# ============================================================
# 7. PREDECLARED MINORITY-CLASS RECOVERY DIAGNOSTIC
# ============================================================

minority_rows = []


for task in TASKS:

    b = per_class[
        (
            per_class[
                "task"
            ]
            == task
        )
        &
        (
            per_class[
                "system"
            ]
            == "XLMR_BASE_SINGLE_TASK"
        )
    ].copy()

    s = per_class[
        (
            per_class[
                "task"
            ]
            == task
        )
        &
        (
            per_class[
                "system"
            ]
            == "EVE_FRAME_STAGE_A_NO_EVENT"
        )
    ].copy()

    merged = b.merge(
        s,
        on=[
            "task",
            "class_id",
            "class_label",
        ],
        suffixes=(
            "_base",
            "_stage_a",
        ),
    )

    zero = merged[
        np.isclose(
            merged[
                "recall_base"
            ],
            0.0,
            atol=1e-15,
        )
    ].copy()

    for r in zero.itertuples(
        index=False
    ):

        minority_rows.append({
            "task":
                task,

            "class_id":
                int(
                    r.class_id
                ),

            "class_label":
                r.class_label,

            "support":
                int(
                    r.support_base
                ),

            "base_recall":
                float(
                    r.recall_base
                ),

            "stage_a_recall":
                float(
                    r.recall_stage_a
                ),

            "delta_recall":
                float(
                    r.recall_stage_a
                    - r.recall_base
                ),

            "base_f1":
                float(
                    r.f1_base
                ),

            "stage_a_f1":
                float(
                    r.f1_stage_a
                ),

            "delta_f1":
                float(
                    r.f1_stage_a
                    - r.f1_base
                ),

            "recovered_nonzero_recall":
                bool(
                    r.recall_stage_a
                    > 0
                ),
        })


minority = pd.DataFrame(
    minority_rows
)

minority.to_csv(
    TMPDIR
    / "minority_class_recovery_v3.csv",
    index=False,
)


# ============================================================
# 8. PAIRED OOF COMPARISON + INFERENCE
# ============================================================

rng = np.random.default_rng(
    RNG_SEED
)

paired_rows = []
paired_article_rows = []
bootstrap_rows = []
randomization_rows = []


for task in TASKS:

    label_map = (
        base_maps[
            task
        ]
    )

    labels = sorted(
        label_map
    )

    n_classes = len(
        labels
    )

    prob_cols = [
        f"prob_class_{i}"
        for i in labels
    ]

    b = (
        base[
            base[
                "task"
            ]
            == task
        ]
        .copy()
        .set_index(
            "doc_id"
        )
        .sort_index()
    )

    s = (
        stage_ensemble[
            stage_ensemble[
                "task"
            ]
            == task
        ]
        .copy()
        .set_index(
            "doc_id"
        )
        .sort_index()
    )

    if list(
        b.index
    ) != list(
        s.index
    ):
        raise RuntimeError(
            f"{task}: BASE and Stage-A "
            "document sets differ."
        )

    if not np.array_equal(
        b[
            "gold_id"
        ].astype(int).to_numpy(),

        s[
            "gold_id"
        ].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            f"{task}: BASE and Stage-A "
            "gold labels differ."
        )

    if not np.array_equal(
        b[
            "outer_fold"
        ].astype(int).to_numpy(),

        s[
            "outer_fold"
        ].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            f"{task}: BASE and Stage-A "
            "outer-fold assignments differ."
        )

    y = b[
        "gold_id"
    ].astype(int).to_numpy()

    base_prob = b[
        prob_cols
    ].to_numpy(
        dtype=float
    )

    stage_prob = s[
        prob_cols
    ].to_numpy(
        dtype=float
    )

    base_pred = np.argmax(
        base_prob,
        axis=1,
    ).astype(int)

    stage_pred = np.argmax(
        stage_prob,
        axis=1,
    ).astype(int)

    base_macro = macro_f1_fast(
        y,
        base_pred,
        n_classes,
    )

    stage_macro = macro_f1_fast(
        y,
        stage_pred,
        n_classes,
    )

    observed_delta = (
        stage_macro
        - base_macro
    )

    # Independent consistency check against sklearn.
    if not math.isclose(
        base_macro,
        f1_score(
            y,
            base_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{task}: BASE Macro-F1 "
            "implementation mismatch."
        )

    if not math.isclose(
        stage_macro,
        f1_score(
            y,
            stage_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{task}: Stage-A Macro-F1 "
            "implementation mismatch."
        )

    # --------------------------------------------------------
    # Article-paired table
    # --------------------------------------------------------

    for i, doc_id in enumerate(
        b.index
    ):

        paired_article_rows.append({
            "task":
                task,

            "doc_id":
                str(
                    doc_id
                ),

            "outer_fold":
                int(
                    b[
                        "outer_fold"
                    ].iloc[i]
                ),

            "gold_id":
                int(
                    y[i]
                ),

            "gold_label":
                label_map[
                    int(
                        y[i]
                    )
                ],

            "base_prediction_id":
                int(
                    base_pred[i]
                ),

            "base_prediction_label":
                label_map[
                    int(
                        base_pred[i]
                    )
                ],

            "stage_a_prediction_id":
                int(
                    stage_pred[i]
                ),

            "stage_a_prediction_label":
                label_map[
                    int(
                        stage_pred[i]
                    )
                ],

            "base_correct":
                bool(
                    base_pred[i]
                    == y[i]
                ),

            "stage_a_correct":
                bool(
                    stage_pred[i]
                    == y[i]
                ),

            "predictions_differ":
                bool(
                    base_pred[i]
                    != stage_pred[i]
                ),

            "base_max_probability":
                float(
                    base_prob[
                        i
                    ].max()
                ),

            "stage_a_max_probability":
                float(
                    stage_prob[
                        i
                    ].max()
                ),
        })

    # --------------------------------------------------------
    # Paired class-stratified bootstrap
    # --------------------------------------------------------

    class_indices = [
        np.flatnonzero(
            y == class_id
        )
        for class_id
        in labels
    ]

    if any(
        len(x) == 0
        for x in class_indices
    ):
        raise RuntimeError(
            f"{task}: zero-support "
            "declared class."
        )

    boot_deltas = np.empty(
        BOOTSTRAP_B,
        dtype=float,
    )

    for replicate in range(
        BOOTSTRAP_B
    ):

        idx = np.concatenate(
            [
                rng.choice(
                    class_idx,
                    size=len(
                        class_idx
                    ),
                    replace=True,
                )
                for class_idx
                in class_indices
            ]
        )

        boot_deltas[
            replicate
        ] = (
            macro_f1_fast(
                y[
                    idx
                ],
                stage_pred[
                    idx
                ],
                n_classes,
            )
            -
            macro_f1_fast(
                y[
                    idx
                ],
                base_pred[
                    idx
                ],
                n_classes,
            )
        )

    ci_low, ci_high = np.percentile(
        boot_deltas,
        [
            2.5,
            97.5,
        ],
    )

    for replicate, value in enumerate(
        boot_deltas,
        start=1,
    ):

        bootstrap_rows.append({
            "task":
                task,

            "replicate":
                replicate,

            "delta_macro_f1_stage_a_minus_base":
                float(
                    value
                ),
        })

    # --------------------------------------------------------
    # Paired probability-vector randomization
    # --------------------------------------------------------

    null_deltas = np.empty(
        RANDOMIZATION_B,
        dtype=float,
    )

    for replicate in range(
        RANDOMIZATION_B
    ):

        swap = (
            rng.random(
                len(y)
            )
            < 0.5
        )

        randomized_base_prob = np.where(
            swap[
                :,
                None
            ],
            stage_prob,
            base_prob,
        )

        randomized_stage_prob = np.where(
            swap[
                :,
                None
            ],
            base_prob,
            stage_prob,
        )

        randomized_base_pred = np.argmax(
            randomized_base_prob,
            axis=1,
        ).astype(int)

        randomized_stage_pred = np.argmax(
            randomized_stage_prob,
            axis=1,
        ).astype(int)

        null_deltas[
            replicate
        ] = (
            macro_f1_fast(
                y,
                randomized_stage_pred,
                n_classes,
            )
            -
            macro_f1_fast(
                y,
                randomized_base_pred,
                n_classes,
            )
        )

    exceed = int(
        np.sum(
            np.abs(
                null_deltas
            )
            >= abs(
                observed_delta
            )
        )
    )

    randomization_p = (
        1
        + exceed
    ) / (
        RANDOMIZATION_B
        + 1
    )

    for replicate, value in enumerate(
        null_deltas,
        start=1,
    ):

        randomization_rows.append({
            "task":
                task,

            "replicate":
                replicate,

            "null_delta_macro_f1_stage_a_minus_base":
                float(
                    value
                ),
        })

    paired_rows.append({
        "task":
            task,

        "n_docs":
            len(y),

        "n_classes":
            n_classes,

        "base_macro_f1":
            float(
                base_macro
            ),

        "stage_a_macro_f1":
            float(
                stage_macro
            ),

        "observed_delta_stage_a_minus_base":
            float(
                observed_delta
            ),

        "bootstrap_mean_delta":
            float(
                boot_deltas.mean()
            ),

        "bootstrap_ci95_low":
            float(
                ci_low
            ),

        "bootstrap_ci95_high":
            float(
                ci_high
            ),

        "bootstrap_prob_delta_gt_0":
            float(
                np.mean(
                    boot_deltas > 0
                )
            ),

        "randomization_B":
            RANDOMIZATION_B,

        "randomization_exceed_two_sided":
            exceed,

        "randomization_p_two_sided":
            float(
                randomization_p
            ),
    })


paired = pd.DataFrame(
    paired_rows
)

paired[
    "bh_q"
] = bh_fdr(
    paired[
        "randomization_p_two_sided"
    ].to_numpy()
)


paired.to_csv(
    TMPDIR
    / "paired_inference_stage_a_vs_base_v3.csv",
    index=False,
)

pd.DataFrame(
    paired_article_rows
).to_csv(
    TMPDIR
    / "paired_oof_articles_v3.csv",
    index=False,
)

pd.DataFrame(
    bootstrap_rows
).to_csv(
    TMPDIR
    / "bootstrap_deltas_stage_a_vs_base_v3.csv",
    index=False,
)

pd.DataFrame(
    randomization_rows
).to_csv(
    TMPDIR
    / "randomization_null_deltas_stage_a_vs_base_v3.csv",
    index=False,
)


# ============================================================
# 9. MINORITY-CLASS SUMMARY
# ============================================================

minority_summary = (
    minority
    .groupby(
        "task",
        as_index=False,
    )
    .agg(
        previously_zero_recall_classes=(
            "class_id",
            "size",
        ),

        recovered_nonzero_recall_classes=(
            "recovered_nonzero_recall",
            "sum",
        ),
    )
)

minority_summary[
    "recovery_fraction"
] = (
    minority_summary[
        "recovered_nonzero_recall_classes"
    ]
    /
    minority_summary[
        "previously_zero_recall_classes"
    ]
)

minority_summary.to_csv(
    TMPDIR
    / "minority_class_recovery_summary_v3.csv",
    index=False,
)


# ============================================================
# 10. FINAL PROVENANCE + HASHES
# ============================================================

summary = {
    "analysis_status":
        "COMPLETE",

    "comparison":
        "EVE_FRAME_STAGE_A_NO_EVENT_vs_XLMR_BASE_SINGLE_TASK",

    "development_dataset":
        "LegacyAux-199",

    "n_docs_per_task":
        199,

    "n_stage_a_runs":
        60,

    "stage_a_prediction_rows":
        2985,

    "stage_a_oof_ensemble_rows":
        597,

    "bootstrap_replicates_per_task":
        BOOTSTRAP_B,

    "randomization_replicates_per_task":
        RANDOMIZATION_B,

    "rng":
        "numpy.default_rng",

    "rng_master_seed":
        RNG_SEED,

    "analysis_plan_sha256":
        sha256(
            PLAN
        ),

    "implementation_lock_sha256":
        sha256(
            LOCK
        ),

    "stage_a_protocol_sha256":
        sha256(
            STAGE
            / "EVE_FRAME_STAGE_A_PROTOCOL_v3.txt"
        ),

    "stage_a_manifest_sha256":
        sha256(
            MANIFEST
        ),

    "stage_a_audit_sha256":
        sha256(
            AUDIT
        ),

    "stage_a_trainer_sha256":
        sha256(
            TRAINER
        ),

    "development_data_sha256":
        sha256(
            DATA
        ),

    "frozen_splits_sha256":
        sha256(
            SPLITS
        ),

    "base_oof_ensemble_sha256":
        sha256(
            BASE_ENSEMBLE
        ),

    "base_seed_stability_sha256":
        sha256(
            BASE_SEED_STABILITY
        ),

    "base_dapt_closure_sha256":
        sha256(
            BASE_CLOSURE
        ),

    "analysis_script_sha256":
        sha256(
            Path(
                __file__
            )
        ),

    "eventgold_status":
        "SEALED_NOT_ACCESSED",

    "inferential_note":
        (
            "Development-set architectural comparison; "
            "not independent external validation."
        ),
}


(
    TMPDIR
    / "analysis_summary_v3.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)


result_files = sorted(
    p
    for p
    in TMPDIR.iterdir()
    if (
        p.is_file()
        and
        p.name
        != "STAGE_A_COMPARISON_RESULTS_SHA256SUMS_v3.txt"
    )
)

hash_path = (
    TMPDIR
    / "STAGE_A_COMPARISON_RESULTS_SHA256SUMS_v3.txt"
)

hash_path.write_text(
    "".join(
        f"{sha256(p)}  {p.name}\n"
        for p
        in result_files
    )
)


# Atomic promotion from temporary to final results directory.
TMPDIR.rename(
    OUTDIR
)


print("=" * 100)
print(
    "EVE-FRAME STAGE A vs FROZEN BASE — COMPLETE"
)
print("=" * 100)

print()
print(
    "TASK-LEVEL OOF METRICS"
)

print(
    task_metrics.to_string(
        index=False
    )
)

print()
print(
    "PAIRED PRIMARY INFERENCE"
)

print(
    paired.to_string(
        index=False
    )
)

print()
print(
    "PREVIOUSLY ZERO-RECALL CLASS RECOVERY"
)

print(
    minority.to_string(
        index=False
    )
)

print()
print(
    "MINORITY RECOVERY SUMMARY"
)

print(
    minority_summary.to_string(
        index=False
    )
)

print()
print(
    "✅ EventGold-35 was not accessed."
)

print(
    "✅ Development-set comparison only."
)
