from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

# ============================================================
# CONFIG
# ============================================================

SEED = 42
N_SPLITS = 4

ROOT = Path.cwd()

DATA = (
    ROOT
    / "01_event_aware_v3/03_scalegold_active_learning/"
      "LegacyAux_leakage_free_v3.csv"
)

OUTDIR = (
    ROOT
    / "01_event_aware_v3/04_classical_baselines/"
      "legacyaux199_tfidf_v3"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

FOLDS_PATH = (
    ROOT
    / "01_event_aware_v3/04_classical_baselines/"
      "frozen_folds_v3/"
      "LegacyAux199_FROZEN_FOLDS_v3.csv"
)

TASKS = {
    "primary_frame":
        "final_primary_frame_validated",

    "stance":
        "final_stance_validated",

    "misinformation_relation":
        "final_misinformation_relation_validated",
}

# ============================================================
# DATA
# ============================================================

df = pd.read_csv(DATA, low_memory=False)

if len(df) != 199:
    raise RuntimeError(
        f"Expected LegacyAux-199, observed {len(df)} rows."
    )

required = [
    "doc_id",
    "dapt_text",
    *TASKS.values(),
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

df["dapt_text"] = (
    df["dapt_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

if df["dapt_text"].eq("").any():
    raise RuntimeError("Empty text detected.")

frozen_folds = pd.read_csv(
    FOLDS_PATH,
    low_memory=False,
)

if len(frozen_folds) != 597:
    raise RuntimeError(
        f"Expected 597 frozen-fold rows; "
        f"observed {len(frozen_folds)}."
    )

df["doc_id"] = (
    df["doc_id"]
    .astype(str)
    .str.strip()
)

frozen_folds["doc_id"] = (
    frozen_folds["doc_id"]
    .astype(str)
    .str.strip()
)

print("Frozen folds:", FOLDS_PATH)
print("Frozen-fold rows:", len(frozen_folds))

print("==============================================")
print("CLASSICAL TF-IDF BENCHMARKS v3")
print("==============================================")
print("Dataset: LegacyAux-199")
print("N:", len(df))
print("CV:", N_SPLITS, "folds")
print("Primary metric: Macro-F1")
print("")


# ============================================================
# FEATURE BUILDERS
# ============================================================

def word_features():
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=30000,
    )


def word_char_features():
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=30000,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=30000,
            ),
        ),
    ])


MODELS = {
    "TFIDF_Word_LogReg": Pipeline([
        ("features", word_features()),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    ]),

    "TFIDF_Word_LinearSVM": Pipeline([
        ("features", word_features()),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=SEED,
                tol=1e-6,
                max_iter=20000,
            ),
        ),
    ]),

    "TFIDF_WordChar_LinearSVM": Pipeline([
        ("features", word_char_features()),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=SEED,
                tol=1e-6,
                max_iter=20000,
            ),
        ),
    ]),
}


# ============================================================
# STORAGE
# ============================================================

fold_results = []
summary_results = []
oof_results = []
class_results = []


# ============================================================
# BENCHMARK
# ============================================================

X = df["dapt_text"].to_numpy()

for task_name, target_col in TASKS.items():

    print("\n")
    print("=" * 70)
    print("TASK:", task_name)
    print("=" * 70)

    y = (
        df[target_col]
        .astype(str)
        .to_numpy()
    )

    counts = (
        pd.Series(y)
        .value_counts()
    )

    print(counts.to_string())

    if counts.min() < N_SPLITS:
        raise RuntimeError(
            f"{task_name}: smallest class has only "
            f"{counts.min()} observations."
        )

    labels = sorted(
        np.unique(y).tolist()
    )

    task_folds = (
        frozen_folds.loc[
            frozen_folds["task"] == task_name,
            ["doc_id", "fold", "gold"]
        ]
        .copy()
    )

    if len(task_folds) != len(df):
        raise RuntimeError(
            f"{task_name}: expected {len(df)} frozen "
            f"assignments; observed {len(task_folds)}."
        )

    if task_folds["doc_id"].duplicated().any():
        raise RuntimeError(
            f"{task_name}: duplicate doc_id in frozen folds."
        )

    fold_lookup = (
        task_folds
        .set_index("doc_id")["fold"]
    )

    gold_lookup = (
        task_folds
        .set_index("doc_id")["gold"]
    )

    fold_assignments = (
        df["doc_id"]
        .map(fold_lookup)
    )

    frozen_gold = (
        df["doc_id"]
        .map(gold_lookup)
        .astype(str)
        .to_numpy()
    )

    if fold_assignments.isna().any():
        missing_ids = (
            df.loc[
                fold_assignments.isna(),
                "doc_id"
            ]
            .tolist()
        )

        raise RuntimeError(
            f"{task_name}: missing frozen assignments "
            f"for {len(missing_ids)} documents."
        )

    if not np.array_equal(
        frozen_gold,
        y.astype(str),
    ):
        raise RuntimeError(
            f"{task_name}: frozen gold labels do not "
            f"match current LegacyAux labels."
        )

    fold_assignments = (
        fold_assignments
        .astype(int)
        .to_numpy()
    )

    observed_folds = sorted(
        np.unique(fold_assignments).tolist()
    )

    if observed_folds != [1, 2, 3, 4]:
        raise RuntimeError(
            f"{task_name}: unexpected folds "
            f"{observed_folds}."
        )

    print(
        "Frozen fold sizes:",
        {
            int(f): int(
                np.sum(fold_assignments == f)
            )
            for f in observed_folds
        }
    )

    for model_name, model_template in MODELS.items():

        print("\n----------------------------------------------")
        print(model_name)
        print("----------------------------------------------")

        oof_pred = np.empty(
            len(df),
            dtype=object,
        )

        for fold in range(1, N_SPLITS + 1):

            test_idx = np.where(
                fold_assignments == fold
            )[0]

            train_idx = np.where(
                fold_assignments != fold
            )[0]

            if len(test_idx) == 0:
                raise RuntimeError(
                    f"{task_name}: frozen fold "
                    f"{fold} is empty."
                )

            if len(
                np.intersect1d(
                    train_idx,
                    test_idx
                )
            ) != 0:
                raise RuntimeError(
                    f"{task_name}: train/test leakage "
                    f"in fold {fold}."
                )

            model = clone(
                model_template
            )

            model.fit(
                X[train_idx],
                y[train_idx],
            )

            pred = model.predict(
                X[test_idx]
            )

            oof_pred[test_idx] = pred

            fold_macro = f1_score(
                y[test_idx],
                pred,
                average="macro",
                zero_division=0,
            )

            fold_ba = balanced_accuracy_score(
                y[test_idx],
                pred,
            )

            fold_acc = accuracy_score(
                y[test_idx],
                pred,
            )

            fold_weighted = f1_score(
                y[test_idx],
                pred,
                average="weighted",
                zero_division=0,
            )

            fold_results.append({
                "task": task_name,
                "model": model_name,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "macro_f1": fold_macro,
                "balanced_accuracy": fold_ba,
                "accuracy": fold_acc,
                "weighted_f1": fold_weighted,
            })

            print(
                f"Fold {fold}: "
                f"Macro-F1={fold_macro:.4f} | "
                f"BA={fold_ba:.4f}"
            )

        # ====================================================
        # OOF METRICS
        # ====================================================

        macro = f1_score(
            y,
            oof_pred,
            average="macro",
            zero_division=0,
        )

        ba = balanced_accuracy_score(
            y,
            oof_pred,
        )

        weighted = f1_score(
            y,
            oof_pred,
            average="weighted",
            zero_division=0,
        )

        acc = accuracy_score(
            y,
            oof_pred,
        )

        summary_results.append({
            "task": task_name,
            "model": model_name,
            "n": len(y),
            "n_classes": len(labels),
            "macro_f1_oof": macro,
            "balanced_accuracy_oof": ba,
            "weighted_f1_oof": weighted,
            "accuracy_oof": acc,
        })

        print("")
        print(
            f"OOF Macro-F1={macro:.4f} | "
            f"BA={ba:.4f} | "
            f"Weighted-F1={weighted:.4f} | "
            f"Accuracy={acc:.4f}"
        )

        # ====================================================
        # PER-CLASS RESULTS
        # ====================================================

        precision, recall, f1s, support = (
            precision_recall_fscore_support(
                y,
                oof_pred,
                labels=labels,
                zero_division=0,
            )
        )

        for label, p, r, f, s in zip(
            labels,
            precision,
            recall,
            f1s,
            support,
        ):
            class_results.append({
                "task": task_name,
                "model": model_name,
                "class": label,
                "precision": p,
                "recall": r,
                "f1": f,
                "support": int(s),
            })

        # ====================================================
        # OOF PREDICTIONS
        # ====================================================

        for i in range(len(df)):
            oof_results.append({
                "doc_id": df.iloc[i]["doc_id"],
                "task": task_name,
                "model": model_name,
                "gold": y[i],
                "prediction": oof_pred[i],
                "correct": bool(
                    y[i] == oof_pred[i]
                ),
            })

        # ====================================================
        # FINAL MODEL ON ALL AUXILIARY DATA
        # ====================================================

        final_model = clone(
            model_template
        )

        final_model.fit(
            X,
            y,
        )

        joblib.dump(
            final_model,
            OUTDIR
            / f"{task_name}__{model_name}.joblib"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)

summary_df = (
    pd.DataFrame(summary_results)
    .sort_values(
        [
            "task",
            "macro_f1_oof",
        ],
        ascending=[
            True,
            False,
        ],
    )
)

oof_df = pd.DataFrame(
    oof_results
)

class_df = pd.DataFrame(
    class_results
)

fold_df.to_csv(
    OUTDIR / "fold_metrics_v3.csv",
    index=False,
)

summary_df.to_csv(
    OUTDIR / "benchmark_summary_v3.csv",
    index=False,
)

oof_df.to_csv(
    OUTDIR / "oof_predictions_v3.csv",
    index=False,
)

class_df.to_csv(
    OUTDIR / "per_class_metrics_v3.csv",
    index=False,
)


meta = {
    "dataset": "LegacyAux-199",
    "n": 199,
    "cv": "4-fold stratified cross-validation",
    "seed": SEED,
    "primary_metric": "macro_f1",
    "secondary_metrics": [
        "balanced_accuracy",
        "weighted_f1",
        "accuracy",
    ],
    "models": list(MODELS.keys()),
    "tasks": TASKS,
    "note": (
        "LegacyAux-199 is an auxiliary historical-label "
        "benchmark and not the final EventGold structural test."
    ),
}

with open(
    OUTDIR / "benchmark_QC_v3.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        meta,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 78)
print("CLASSICAL BASELINES COMPLETE")
print("=" * 78)

print(
    summary_df[
        [
            "task",
            "model",
            "macro_f1_oof",
            "balanced_accuracy_oof",
            "weighted_f1_oof",
            "accuracy_oof",
        ]
    ].to_string(
        index=False
    )
)

print("\nOutputs:")
print(OUTDIR)
