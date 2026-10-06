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

MANIFEST = (
    ROOT
    / "01_event_aware_v3/05_supervised_transformers/"
      "frozen_finetune_design_v3/finetune_run_manifest_v3.csv"
)

RUNROOT = (
    ROOT
    / "01_event_aware_v3/05_supervised_transformers/"
      "finetune_runs_final_v3"
)

ANALYSIS_DIR = (
    ROOT
    / "01_event_aware_v3/06_supervised_analysis_v3"
)

OUTDIR = ANALYSIS_DIR / "results_v3"
OUTDIR.mkdir(parents=True, exist_ok=True)

PLAN = ANALYSIS_DIR / "STATISTICAL_ANALYSIS_PLAN_v3.txt"
LOCK = ANALYSIS_DIR / "STATISTICAL_IMPLEMENTATION_LOCK_v3.txt"

FULL_AUDIT = (
    ROOT
    / "01_event_aware_v3/05_supervised_transformers/"
      "full_finetune_audit_v3/"
      "full_finetune_audit_summary_v3.json"
)

EXPECTED_PLAN_SHA = (
    "cdfa108fb753af10e1ce6aad096f1f1570282f5a75d247a4f391bf09f150407c"
)

EXPECTED_LOCK_SHA = (
    "609dbfd0a7c558d6fc54e54c496563678d591e906c1e6c1bdc6491d7fe81ccbd"
)

EXPECTED_MANIFEST_SHA = (
    "149089f3237ce36c34e917ed93cf05870c5396743355aa81850469dea7efe85f"
)

EXPECTED_SEEDS = [11, 29, 47, 83, 131]

EXPECTED_INITS = [
    "XLMR_BASE",
    "XLMR_DAPT_INDUCTIVE",
]

EXPECTED_TASKS = [
    "primary_frame",
    "stance",
    "misinformation_relation",
]

EXPECTED_N_DOCS = 199

BOOTSTRAP_B = 10_000
RANDOMIZATION_B = 10_000
RNG_SEED = 20260920


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def macro_f1_fast(y_true, y_pred, n_classes):
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

    denom = 2.0 * tp + fp + fn

    f1 = np.divide(
        2.0 * tp,
        denom,
        out=np.zeros_like(tp),
        where=denom != 0,
    )

    return float(f1.mean())


def compute_metrics(y_true, y_pred, labels):
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
    p = np.asarray(p_values, dtype=float)

    m = len(p)
    order = np.argsort(p)
    ranked = p[order]

    q_ranked = (
        ranked
        * m
        / np.arange(1, m + 1)
    )

    q_ranked = np.minimum.accumulate(
        q_ranked[::-1]
    )[::-1]

    q_ranked = np.clip(
        q_ranked,
        0.0,
        1.0,
    )

    q = np.empty(m, dtype=float)
    q[order] = q_ranked

    return q


# ============================================================
# 1. Frozen-input integrity
# ============================================================

if sha256(PLAN) != EXPECTED_PLAN_SHA:
    raise RuntimeError(
        "Statistical analysis plan SHA256 mismatch."
    )

if sha256(LOCK) != EXPECTED_LOCK_SHA:
    raise RuntimeError(
        "Statistical implementation lock SHA256 mismatch."
    )

if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA:
    raise RuntimeError(
        "Run manifest SHA256 mismatch."
    )

if not FULL_AUDIT.exists():
    raise RuntimeError(
        "Full fine-tuning audit summary not found."
    )

full_audit = json.loads(
    FULL_AUDIT.read_text()
)

required_audit = {
    "expected_runs": 120,
    "summaries_found": 120,
    "valid_runs": 120,
    "missing_runs": 0,
    "invalid_runs": 0,
    "all_120_valid": True,
}

for key, expected in required_audit.items():
    if full_audit.get(key) != expected:
        raise RuntimeError(
            f"Full fine-tuning audit failed: "
            f"{key}={full_audit.get(key)!r}; "
            f"expected {expected!r}"
        )

manifest = pd.read_csv(MANIFEST)

if len(manifest) != 120:
    raise RuntimeError(
        f"Expected 120 manifest rows; got {len(manifest)}"
    )

if manifest["run_id"].tolist() != list(range(120)):
    raise RuntimeError(
        "Manifest run_id is not exactly 0..119."
    )

if sorted(
    manifest["model_seed"].unique().tolist()
) != EXPECTED_SEEDS:
    raise RuntimeError(
        "Unexpected seed set."
    )

if sorted(
    manifest["initialization"].unique().tolist()
) != sorted(EXPECTED_INITS):
    raise RuntimeError(
        "Unexpected initialization set."
    )

if sorted(
    manifest["task"].unique().tolist()
) != sorted(EXPECTED_TASKS):
    raise RuntimeError(
        "Unexpected task set."
    )


# ============================================================
# 2. Load + validate all 120 prediction files
# ============================================================

all_frames = []
run_metric_rows = []
label_maps = {}
input_hash_rows = []


for r in manifest.itertuples(index=False):

    run_name = (
        f"{r.task}"
        f"__fold{int(r.outer_fold)}"
        f"__{r.initialization}"
        f"__seed{int(r.model_seed)}"
    )

    run_dir = RUNROOT / run_name

    pred_path = (
        run_dir
        / "outer_test_predictions_v3.csv"
    )

    label_path = (
        run_dir
        / "label_map_v3.json"
    )

    if not pred_path.exists():
        raise RuntimeError(
            f"Missing predictions: {pred_path}"
        )

    if not label_path.exists():
        raise RuntimeError(
            f"Missing label map: {label_path}"
        )

    input_hash_rows.append(
        {
            "run_id":
                int(r.run_id),

            "run_name":
                run_name,

            "prediction_sha256":
                sha256(pred_path),

            "label_map_sha256":
                sha256(label_path),
        }
    )

    lm_raw = json.loads(
        label_path.read_text()
    )

    lm = {
        int(k): str(v)
        for k, v in lm_raw.items()
    }

    labels = sorted(lm)

    if labels != list(range(len(labels))):
        raise RuntimeError(
            f"{run_name}: class IDs are not contiguous."
        )

    if r.task not in label_maps:
        label_maps[r.task] = lm

    elif label_maps[r.task] != lm:
        raise RuntimeError(
            f"Inconsistent label map for {r.task}"
        )

    prob_cols = [
        f"prob_class_{i}"
        for i in labels
    ]

    df = pd.read_csv(pred_path)

    required = {
        "doc_id",
        "task",
        "outer_fold",
        "initialization",
        "seed",
        "gold_id",
        "gold_label",
        "prediction_id",
        "prediction_label",
        *prob_cols,
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{run_name}: missing columns "
            f"{sorted(missing)}"
        )

    if set(df["task"].astype(str)) != {
        str(r.task)
    }:
        raise RuntimeError(
            f"{run_name}: task mismatch."
        )

    if set(df["outer_fold"].astype(int)) != {
        int(r.outer_fold)
    }:
        raise RuntimeError(
            f"{run_name}: fold mismatch."
        )

    if set(df["initialization"].astype(str)) != {
        str(r.initialization)
    }:
        raise RuntimeError(
            f"{run_name}: initialization mismatch."
        )

    if set(df["seed"].astype(int)) != {
        int(r.model_seed)
    }:
        raise RuntimeError(
            f"{run_name}: seed mismatch."
        )

    if df["doc_id"].duplicated().any():
        raise RuntimeError(
            f"{run_name}: duplicate doc_id."
        )

    if df[
        [
            "gold_id",
            "prediction_id",
            *prob_cols,
        ]
    ].isna().any().any():
        raise RuntimeError(
            f"{run_name}: NaN detected."
        )

    probs = df[
        prob_cols
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(probs).all():
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
        probs.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=5e-5,
    ):
        raise RuntimeError(
            f"{run_name}: probabilities do not sum to 1."
        )

    pred_from_prob = np.argmax(
        probs,
        axis=1,
    )

    if not np.array_equal(
        pred_from_prob.astype(int),
        df["prediction_id"].to_numpy(
            dtype=int
        ),
    ):
        raise RuntimeError(
            f"{run_name}: prediction_id != argmax."
        )

    gold_ids = df[
        "gold_id"
    ].astype(int).to_numpy()

    if not set(gold_ids).issubset(
        set(labels)
    ):
        raise RuntimeError(
            f"{run_name}: gold class outside label map."
        )

    expected_gold_labels = np.array(
        [
            lm[int(x)]
            for x in gold_ids
        ]
    )

    if not np.array_equal(
        expected_gold_labels,
        df["gold_label"].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            f"{run_name}: gold label mismatch."
        )

    metrics = compute_metrics(
        gold_ids,
        df[
            "prediction_id"
        ].astype(int).to_numpy(),
        labels,
    )

    run_metric_rows.append(
        {
            "run_id":
                int(r.run_id),

            "task":
                str(r.task),

            "outer_fold":
                int(r.outer_fold),

            "initialization":
                str(r.initialization),

            "seed":
                int(r.model_seed),

            "n_test":
                len(df),

            **metrics,
        }
    )

    keep = [
        "doc_id",
        "task",
        "outer_fold",
        "initialization",
        "seed",
        "gold_id",
        "gold_label",
        "prediction_id",
        "prediction_label",
        *prob_cols,
    ]

    z = df[keep].copy()
    z["run_id"] = int(r.run_id)

    all_frames.append(z)


raw = pd.concat(
    all_frames,
    ignore_index=True,
    sort=False,
)

expected_raw_rows = (
    len(EXPECTED_TASKS)
    * len(EXPECTED_INITS)
    * len(EXPECTED_SEEDS)
    * EXPECTED_N_DOCS
)

if len(raw) != expected_raw_rows:
    raise RuntimeError(
        f"Expected {expected_raw_rows} total "
        f"seed-level OOF rows; got {len(raw)}"
    )


run_metrics = pd.DataFrame(
    run_metric_rows
).sort_values("run_id")

run_metrics.to_csv(
    OUTDIR / "run_level_metrics_v3.csv",
    index=False,
)


pd.DataFrame(
    input_hash_rows
).sort_values(
    "run_id"
).to_csv(
    OUTDIR / "input_prediction_hashes_v3.csv",
    index=False,
)


(
    OUTDIR
    / "label_maps_v3.json"
).write_text(
    json.dumps(
        label_maps,
        indent=2,
        ensure_ascii=False,
    )
)


# ============================================================
# 3. Seed-level full OOF stability
# ============================================================

seed_rows = []


for task in EXPECTED_TASKS:

    labels = sorted(
        label_maps[task]
    )

    for init in EXPECTED_INITS:

        for seed in EXPECTED_SEEDS:

            g = raw[
                (
                    raw["task"]
                    == task
                )
                &
                (
                    raw["initialization"]
                    == init
                )
                &
                (
                    raw["seed"]
                    == seed
                )
            ].copy()

            if len(g) != EXPECTED_N_DOCS:
                raise RuntimeError(
                    f"{task}/{init}/seed{seed}: "
                    f"expected 199 OOF rows; "
                    f"got {len(g)}"
                )

            if g["doc_id"].duplicated().any():
                raise RuntimeError(
                    f"{task}/{init}/seed{seed}: "
                    "duplicate OOF document."
                )

            m = compute_metrics(
                g["gold_id"].astype(int).to_numpy(),
                g["prediction_id"].astype(int).to_numpy(),
                labels,
            )

            seed_rows.append(
                {
                    "task":
                        task,

                    "initialization":
                        init,

                    "seed":
                        seed,

                    "n_docs":
                        len(g),

                    **m,
                }
            )


seed_stability = pd.DataFrame(seed_rows)

seed_stability.to_csv(
    OUTDIR / "seed_stability_v3.csv",
    index=False,
)


# ============================================================
# 4. Five-seed probability ensembles
# ============================================================

ensemble_rows = []


for task in EXPECTED_TASKS:

    labels = sorted(
        label_maps[task]
    )

    prob_cols = [
        f"prob_class_{i}"
        for i in labels
    ]

    for init in EXPECTED_INITS:

        all_doc_ids = []

        for fold in [1, 2, 3, 4]:

            g = raw[
                (
                    raw["task"]
                    == task
                )
                &
                (
                    raw["initialization"]
                    == init
                )
                &
                (
                    raw["outer_fold"]
                    == fold
                )
            ].copy()

            if set(
                g["seed"].astype(int).unique()
            ) != set(EXPECTED_SEEDS):
                raise RuntimeError(
                    f"{task}/{init}/fold{fold}: "
                    "incomplete seed set."
                )

            for doc_id, d in g.groupby(
                "doc_id",
                sort=True,
            ):

                if len(d) != 5:
                    raise RuntimeError(
                        f"{task}/{init}/fold{fold}/{doc_id}: "
                        f"expected 5 seeds; got {len(d)}"
                    )

                if set(
                    d["seed"].astype(int)
                ) != set(EXPECTED_SEEDS):
                    raise RuntimeError(
                        f"{task}/{init}/fold{fold}/{doc_id}: "
                        "seed mismatch."
                    )

                if d["gold_id"].nunique() != 1:
                    raise RuntimeError(
                        f"{task}/{init}/fold{fold}/{doc_id}: "
                        "inconsistent gold_id."
                    )

                if d["gold_label"].nunique() != 1:
                    raise RuntimeError(
                        f"{task}/{init}/fold{fold}/{doc_id}: "
                        "inconsistent gold_label."
                    )

                p = d[
                    prob_cols
                ].to_numpy(
                    dtype=float
                ).mean(axis=0)

                pred_id = int(
                    np.argmax(p)
                )

                gold_id = int(
                    d["gold_id"].iloc[0]
                )

                row = {
                    "task":
                        task,

                    "initialization":
                        init,

                    "doc_id":
                        str(doc_id),

                    "outer_fold":
                        int(fold),

                    "gold_id":
                        gold_id,

                    "gold_label":
                        label_maps[
                            task
                        ][gold_id],

                    "prediction_id":
                        pred_id,

                    "prediction_label":
                        label_maps[
                            task
                        ][pred_id],

                    "n_seeds":
                        5,
                }

                for col, val in zip(
                    prob_cols,
                    p,
                ):
                    row[col] = float(val)

                ensemble_rows.append(row)
                all_doc_ids.append(
                    str(doc_id)
                )

        if len(all_doc_ids) != EXPECTED_N_DOCS:
            raise RuntimeError(
                f"{task}/{init}: "
                f"expected 199 ensemble docs; "
                f"got {len(all_doc_ids)}"
            )

        if len(
            set(all_doc_ids)
        ) != EXPECTED_N_DOCS:
            raise RuntimeError(
                f"{task}/{init}: "
                "document overlap across folds."
            )


ensemble = pd.DataFrame(
    ensemble_rows
)

expected_ensemble_rows = (
    len(EXPECTED_TASKS)
    * len(EXPECTED_INITS)
    * EXPECTED_N_DOCS
)

if len(ensemble) != expected_ensemble_rows:
    raise RuntimeError(
        "Unexpected ensemble table size."
    )


ensemble.to_csv(
    OUTDIR
    / "oof_ensemble_predictions_v3.csv",
    index=False,
)


# ============================================================
# 5. Ensemble metrics + class-level errors
# ============================================================

task_metric_rows = []
class_metric_rows = []
cm_rows = []


for task in EXPECTED_TASKS:

    labels = sorted(
        label_maps[task]
    )

    for init in EXPECTED_INITS:

        g = ensemble[
            (
                ensemble["task"] == task
            )
            &
            (
                ensemble["initialization"]
                == init
            )
        ].copy()

        if len(g) != EXPECTED_N_DOCS:
            raise RuntimeError(
                f"{task}/{init}: "
                "ensemble does not contain 199 docs."
            )

        y = g[
            "gold_id"
        ].astype(int).to_numpy()

        pred = g[
            "prediction_id"
        ].astype(int).to_numpy()

        metrics = compute_metrics(
            y,
            pred,
            labels,
        )

        task_metric_rows.append(
            {
                "task":
                    task,

                "initialization":
                    init,

                "n_docs":
                    len(g),

                "n_classes":
                    len(labels),

                **metrics,
            }
        )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y,
                pred,
                labels=labels,
                zero_division=0,
            )
        )

        for i, cls in enumerate(labels):

            class_metric_rows.append(
                {
                    "task":
                        task,

                    "initialization":
                        init,

                    "class_id":
                        int(cls),

                    "class_label":
                        label_maps[
                            task
                        ][cls],

                    "precision":
                        float(precision[i]),

                    "recall":
                        float(recall[i]),

                    "f1":
                        float(f1[i]),

                    "support":
                        int(support[i]),
                }
            )

        cm = confusion_matrix(
            y,
            pred,
            labels=labels,
        )

        for i, gold_cls in enumerate(labels):
            for j, pred_cls in enumerate(labels):

                cm_rows.append(
                    {
                        "task":
                            task,

                        "initialization":
                            init,

                        "gold_id":
                            int(gold_cls),

                        "gold_label":
                            label_maps[
                                task
                            ][gold_cls],

                        "prediction_id":
                            int(pred_cls),

                        "prediction_label":
                            label_maps[
                                task
                            ][pred_cls],

                        "count":
                            int(cm[i, j]),
                    }
                )


task_metrics = pd.DataFrame(
    task_metric_rows
)

task_metrics.to_csv(
    OUTDIR / "task_metrics_v3.csv",
    index=False,
)


pd.DataFrame(
    class_metric_rows
).to_csv(
    OUTDIR / "per_class_metrics_v3.csv",
    index=False,
)


pd.DataFrame(
    cm_rows
).to_csv(
    OUTDIR / "confusion_matrices_v3.csv",
    index=False,
)


# ============================================================
# 6. Paired article-level inference
# ============================================================

paired_rows = []
bootstrap_rows = []
randomization_rows = []


for task_i, task in enumerate(
    EXPECTED_TASKS
):

    labels = sorted(
        label_maps[task]
    )

    n_classes = len(labels)

    prob_cols = [
        f"prob_class_{i}"
        for i in labels
    ]

    base = (
        ensemble[
            (
                ensemble["task"] == task
            )
            &
            (
                ensemble["initialization"]
                == "XLMR_BASE"
            )
        ]
        .copy()
        .set_index("doc_id")
        .sort_index()
    )

    dapt = (
        ensemble[
            (
                ensemble["task"] == task
            )
            &
            (
                ensemble["initialization"]
                == "XLMR_DAPT_INDUCTIVE"
            )
        ]
        .copy()
        .set_index("doc_id")
        .sort_index()
    )

    if list(base.index) != list(dapt.index):
        raise RuntimeError(
            f"{task}: BASE/DAPT document sets differ."
        )

    if not np.array_equal(
        base[
            "gold_id"
        ].astype(int).to_numpy(),

        dapt[
            "gold_id"
        ].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            f"{task}: BASE/DAPT gold labels differ."
        )

    if not np.array_equal(
        base[
            "outer_fold"
        ].astype(int).to_numpy(),

        dapt[
            "outer_fold"
        ].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            f"{task}: BASE/DAPT folds differ."
        )

    y = base[
        "gold_id"
    ].astype(int).to_numpy()

    base_prob = base[
        prob_cols
    ].to_numpy(dtype=float)

    dapt_prob = dapt[
        prob_cols
    ].to_numpy(dtype=float)

    base_pred = np.argmax(
        base_prob,
        axis=1,
    ).astype(int)

    dapt_pred = np.argmax(
        dapt_prob,
        axis=1,
    ).astype(int)

    base_macro = macro_f1_fast(
        y,
        base_pred,
        n_classes,
    )

    dapt_macro = macro_f1_fast(
        y,
        dapt_pred,
        n_classes,
    )

    observed_delta = (
        dapt_macro
        - base_macro
    )

    sk_base = f1_score(
        y,
        base_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    sk_dapt = f1_score(
        y,
        dapt_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    if not (
        math.isclose(
            base_macro,
            sk_base,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and
        math.isclose(
            dapt_macro,
            sk_dapt,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            f"{task}: Macro-F1 implementation mismatch."
        )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    rng_boot = np.random.default_rng(
        RNG_SEED
        + task_i * 1000
        + 101
    )

    class_indices = [
        np.flatnonzero(
            y == cls
        )
        for cls in labels
    ]

    if any(
        len(x) == 0
        for x in class_indices
    ):
        raise RuntimeError(
            f"{task}: zero-support class."
        )

    boot_deltas = np.empty(
        BOOTSTRAP_B,
        dtype=float,
    )

    for b in range(BOOTSTRAP_B):

        idx = np.concatenate(
            [
                rng_boot.choice(
                    arr,
                    size=len(arr),
                    replace=True,
                )
                for arr in class_indices
            ]
        )

        boot_deltas[b] = (
            macro_f1_fast(
                y[idx],
                dapt_pred[idx],
                n_classes,
            )
            -
            macro_f1_fast(
                y[idx],
                base_pred[idx],
                n_classes,
            )
        )

    ci_low, ci_high = np.percentile(
        boot_deltas,
        [2.5, 97.5],
    )

    for b, val in enumerate(
        boot_deltas,
        start=1,
    ):
        bootstrap_rows.append(
            {
                "task":
                    task,

                "replicate":
                    b,

                "delta_macro_f1_dapt_minus_base":
                    float(val),
            }
        )

    # --------------------------------------------------------
    # Randomization
    # --------------------------------------------------------

    rng_rand = np.random.default_rng(
        RNG_SEED
        + task_i * 1000
        + 707
    )

    null_deltas = np.empty(
        RANDOMIZATION_B,
        dtype=float,
    )

    for b in range(RANDOMIZATION_B):

        swap = rng_rand.integers(
            0,
            2,
            size=len(y),
            dtype=np.int8,
        ).astype(bool)

        randomized_base_prob = np.where(
            swap[:, None],
            dapt_prob,
            base_prob,
        )

        randomized_dapt_prob = np.where(
            swap[:, None],
            base_prob,
            dapt_prob,
        )

        randomized_base_pred = np.argmax(
            randomized_base_prob,
            axis=1,
        ).astype(int)

        randomized_dapt_pred = np.argmax(
            randomized_dapt_prob,
            axis=1,
        ).astype(int)

        null_deltas[b] = (
            macro_f1_fast(
                y,
                randomized_dapt_pred,
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
            np.abs(null_deltas)
            >= abs(observed_delta)
        )
    )

    p_randomization = (
        1 + exceed
    ) / (
        RANDOMIZATION_B + 1
    )

    for b, val in enumerate(
        null_deltas,
        start=1,
    ):
        randomization_rows.append(
            {
                "task":
                    task,

                "replicate":
                    b,

                "null_delta_macro_f1_dapt_minus_base":
                    float(val),
            }
        )

    paired_rows.append(
        {
            "task":
                task,

            "n_docs":
                len(y),

            "n_classes":
                n_classes,

            "base_macro_f1":
                float(base_macro),

            "dapt_macro_f1":
                float(dapt_macro),

            "observed_delta_dapt_minus_base":
                float(observed_delta),

            "bootstrap_mean_delta":
                float(boot_deltas.mean()),

            "bootstrap_ci95_low":
                float(ci_low),

            "bootstrap_ci95_high":
                float(ci_high),

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
                float(p_randomization),
        }
    )


paired = pd.DataFrame(
    paired_rows
)

paired["bh_q"] = bh_fdr(
    paired[
        "randomization_p_two_sided"
    ].to_numpy()
)


paired.to_csv(
    OUTDIR / "paired_inference_v3.csv",
    index=False,
)


pd.DataFrame(
    bootstrap_rows
).to_csv(
    OUTDIR / "bootstrap_deltas_v3.csv",
    index=False,
)


pd.DataFrame(
    randomization_rows
).to_csv(
    OUTDIR / "randomization_null_deltas_v3.csv",
    index=False,
)


# ============================================================
# 7. Final provenance
# ============================================================

summary = {
    "analysis_status":
        "COMPLETE",

    "expected_runs":
        120,

    "validated_prediction_files":
        len(input_hash_rows),

    "tasks":
        EXPECTED_TASKS,

    "initializations":
        EXPECTED_INITS,

    "seeds":
        EXPECTED_SEEDS,

    "n_docs_per_task":
        EXPECTED_N_DOCS,

    "bootstrap_replicates_per_task":
        BOOTSTRAP_B,

    "randomization_replicates_per_task":
        RANDOMIZATION_B,

    "rng_seed":
        RNG_SEED,

    "statistical_plan_sha256":
        sha256(PLAN),

    "implementation_lock_sha256":
        sha256(LOCK),

    "manifest_sha256":
        sha256(MANIFEST),

    "full_finetune_audit_sha256":
        sha256(FULL_AUDIT),

    "analysis_script_sha256":
        sha256(Path(__file__)),

    "eventgold_status":
        "SEALED_NOT_ACCESSED",
}


summary_path = (
    OUTDIR
    / "analysis_summary_v3.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)


result_files = sorted(
    p
    for p in OUTDIR.iterdir()
    if (
        p.is_file()
        and
        p.name
        != "STATISTICAL_RESULTS_SHA256SUMS_v3.txt"
    )
)


hash_file = (
    OUTDIR
    / "STATISTICAL_RESULTS_SHA256SUMS_v3.txt"
)

hash_file.write_text(
    "".join(
        f"{sha256(p)}  {p.name}\n"
        for p in result_files
    )
)


print("=" * 88)
print(
    "SUPERVISED TRANSFORMER OOF STATISTICAL ANALYSIS — COMPLETE"
)
print("=" * 88)

print()
print(
    "TASK-LEVEL FIVE-SEED OOF ENSEMBLE METRICS"
)

print(
    task_metrics.to_string(
        index=False
    )
)

print()
print(
    "PAIRED BASE vs DAPT INFERENCE"
)

print(
    paired.to_string(
        index=False
    )
)

print()
print(
    "RESULT HASHES"
)

print(
    hash_file.read_text()
)

print(
    "✅ EventGold-35 was not accessed."
)
