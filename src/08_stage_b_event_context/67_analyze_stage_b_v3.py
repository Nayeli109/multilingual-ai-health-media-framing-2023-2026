from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

DEST = (
    ROOT
    / "01_event_aware_v3/08_eve_frame_stage_b_event_context"
)

PLAN = (
    DEST
    / "STAGE_B_COMPARISON_ANALYSIS_PLAN_v3.txt"
)

LOCK = (
    DEST
    / "STAGE_B_STATISTICAL_IMPLEMENTATION_LOCK_v3.txt"
)

INPUT_MANIFEST = (
    DEST
    / "STAGE_B_ANALYSIS_INPUT_PREDICTION_HASHES_v3.csv"
)

OUT = (
    DEST
    / "comparison_results_v3"
)

TMP = (
    DEST
    / "comparison_results_v3_tmp"
)


# ============================================================
# FROZEN HASHES
# ============================================================

EXPECTED_PLAN_SHA = (
    "2c0dd392647994298d8888e17d8ad853"
    "03612987ce0a694f302d202ebba400e5"
)

EXPECTED_LOCK_SHA = (
    "92c734c2b63a06b5b34cb9f20ca9170"
    "04c868c9b4a9d6e8404f0f766f87beadb"
)

EXPECTED_INPUT_MANIFEST_SHA = (
    "298760125cf13293db0ebf9330f82cfe"
    "f0448cfcef84806be950696680407c9c"
)


# ============================================================
# FROZEN DESIGN
# ============================================================

TASKS = [
    "primary_frame",
    "stance",
    "misinformation_relation",
]

SYSTEMS = [
    "STAGE_A_NO_EVENT",
    "VERIFIED_CONTEXT",
    "PERMUTED_CONTEXT",
]

SYSTEM_ARCHITECTURES = {
    "STAGE_A_NO_EVENT":
        "EVE_FRAME_STAGE_A_NO_EVENT",

    "VERIFIED_CONTEXT":
        "EVE_FRAME_STAGE_B_VERIFIED_CONTEXT",

    "PERMUTED_CONTEXT":
        "EVE_FRAME_STAGE_B_PERMUTED_CONTEXT",
}

SEEDS = [
    11,
    29,
    47,
    83,
    131,
]

FOLDS = [
    1,
    2,
    3,
    4,
]

N_ARTICLES = 199
N_ENSEMBLE_SEEDS = 5
N_BOOTSTRAP = 10000
N_RANDOMIZATION = 10000

ROW_SUM_ATOL = 1e-5

PROB_RE = re.compile(
    r"^prob_class_([0-9]+)$"
)


# ------------------------------------------------------------
# RNG seeds frozen in implementation lock
# ------------------------------------------------------------

BOOTSTRAP_SEEDS = {
    (
        "PRIMARY",
        "primary_frame",
    ): 2026092401,

    (
        "PRIMARY",
        "stance",
    ): 2026092402,

    (
        "PRIMARY",
        "misinformation_relation",
    ): 2026092403,

    (
        "CONTEXT_SPECIFICITY",
        "primary_frame",
    ): 2026092421,

    (
        "CONTEXT_SPECIFICITY",
        "stance",
    ): 2026092422,

    (
        "CONTEXT_SPECIFICITY",
        "misinformation_relation",
    ): 2026092423,
}


RANDOMIZATION_SEEDS = {
    (
        "PRIMARY",
        "primary_frame",
    ): 2026092411,

    (
        "PRIMARY",
        "stance",
    ): 2026092412,

    (
        "PRIMARY",
        "misinformation_relation",
    ): 2026092413,

    (
        "CONTEXT_SPECIFICITY",
        "primary_frame",
    ): 2026092431,

    (
        "CONTEXT_SPECIFICITY",
        "stance",
    ): 2026092432,

    (
        "CONTEXT_SPECIFICITY",
        "misinformation_relation",
    ): 2026092433,
}


FAMILIES = {
    "PRIMARY": (
        "VERIFIED_CONTEXT",
        "STAGE_A_NO_EVENT",
    ),

    "CONTEXT_SPECIFICITY": (
        "VERIFIED_CONTEXT",
        "PERMUTED_CONTEXT",
    ),
}


# ============================================================
# UTILITIES
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def require_hash(path, expected, label):

    if not Path(path).exists():
        raise RuntimeError(
            f"{label} missing: {path}"
        )

    observed = sha256(path)

    if observed != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {observed}"
        )


def macro_f1(y_true, y_pred, k):

    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(k)),
            average="macro",
            zero_division=0,
        )
    )


def bh_adjust(pvalues):

    p = np.asarray(
        pvalues,
        dtype=np.float64,
    )

    m = len(p)

    if m != 3:
        raise RuntimeError(
            f"BH family must contain exactly 3 p-values; got {m}."
        )

    if (
        (~np.isfinite(p)).any()
        or (p < 0).any()
        or (p > 1).any()
    ):
        raise RuntimeError(
            "Invalid p-value supplied to BH."
        )

    order = np.argsort(
        p,
        kind="mergesort",
    )

    sorted_p = p[order]

    raw = (
        sorted_p
        * float(m)
        / np.arange(
            1,
            m + 1,
            dtype=np.float64,
        )
    )

    adjusted_sorted = np.minimum.accumulate(
        raw[::-1]
    )[::-1]

    adjusted_sorted = np.minimum(
        adjusted_sorted,
        1.0,
    )

    adjusted = np.empty(
        m,
        dtype=np.float64,
    )

    adjusted[order] = adjusted_sorted

    return adjusted


def extract_prob_columns(df):

    pairs = []

    for col in df.columns:

        m = PROB_RE.fullmatch(
            str(col)
        )

        if m is not None:
            pairs.append(
                (
                    int(m.group(1)),
                    str(col),
                )
            )

    if not pairs:
        raise RuntimeError(
            "No probability columns detected."
        )

    pairs.sort(
        key=lambda x: x[0]
    )

    ids = [
        x[0]
        for x in pairs
    ]

    expected = list(
        range(
            len(ids)
        )
    )

    if ids != expected:
        raise RuntimeError(
            "Probability class IDs are not contiguous from zero. "
            f"Observed: {ids}"
        )

    return [
        x[1]
        for x in pairs
    ]


def validate_probability_matrix(
    prob,
    path,
):

    prob = np.asarray(
        prob,
        dtype=np.float64,
    )

    if prob.ndim != 2:
        raise RuntimeError(
            f"Probability matrix is not 2-D: {path}"
        )

    if not np.isfinite(
        prob
    ).all():
        raise RuntimeError(
            f"Non-finite probabilities: {path}"
        )

    # No clipping / no renormalization.
    if (
        (prob < 0.0).any()
        or
        (prob > 1.0).any()
    ):
        raise RuntimeError(
            f"Probability outside [0,1]: {path}"
        )

    sums = prob.sum(
        axis=1,
        dtype=np.float64,
    )

    if not np.allclose(
        sums,
        1.0,
        rtol=0.0,
        atol=ROW_SUM_ATOL,
    ):
        bad = np.max(
            np.abs(
                sums
                - 1.0
            )
        )

        raise RuntimeError(
            f"Probability-row sum violation in {path}; "
            f"maximum absolute deviation={bad}"
        )


# ============================================================
# LOAD + VERIFY FROZEN INPUT MANIFEST
# ============================================================

def load_inputs():

    require_hash(
        PLAN,
        EXPECTED_PLAN_SHA,
        "Analysis plan",
    )

    require_hash(
        LOCK,
        EXPECTED_LOCK_SHA,
        "Statistical implementation lock",
    )

    require_hash(
        INPUT_MANIFEST,
        EXPECTED_INPUT_MANIFEST_SHA,
        "Prediction-input manifest",
    )

    manifest = pd.read_csv(
        INPUT_MANIFEST
    )

    required = {
        "system",
        "task",
        "outer_fold",
        "seed",
        "relative_path",
        "sha256",
    }

    missing = (
        required
        - set(
            manifest.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Input manifest missing columns: {sorted(missing)}"
        )

    if len(manifest) != 180:
        raise RuntimeError(
            f"Expected 180 input files, got {len(manifest)}."
        )

    observed_systems = set(
        manifest["system"]
        .astype(str)
    )

    if observed_systems != set(
        SYSTEMS
    ):
        raise RuntimeError(
            f"Unexpected systems: {observed_systems}"
        )

    verification_rows = []
    raw = {}

    for r in manifest.itertuples(
        index=False
    ):

        system = str(
            r.system
        )

        task = str(
            r.task
        )

        fold = int(
            r.outer_fold
        )

        seed = int(
            r.seed
        )

        if task not in TASKS:
            raise RuntimeError(
                f"Unexpected task: {task}"
            )

        if fold not in FOLDS:
            raise RuntimeError(
                f"Unexpected fold: {fold}"
            )

        if seed not in SEEDS:
            raise RuntimeError(
                f"Unexpected seed: {seed}"
            )

        path = (
            ROOT
            / str(
                r.relative_path
            )
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing prediction input: {path}"
            )

        observed_sha = sha256(
            path
        )

        expected_sha = str(
            r.sha256
        )

        if (
            observed_sha
            != expected_sha
        ):
            raise RuntimeError(
                f"Prediction-input SHA mismatch:\n"
                f"{path}\n"
                f"Expected: {expected_sha}\n"
                f"Observed: {observed_sha}"
            )

        df = pd.read_csv(
            path,
            dtype={
                "doc_id": str,
            },
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
        }

        miss = (
            required_cols
            - set(
                df.columns
            )
        )

        if miss:
            raise RuntimeError(
                f"{path} missing required columns: "
                f"{sorted(miss)}"
            )

        prob_cols = (
            extract_prob_columns(
                df
            )
        )

        prob = (
            df[
                prob_cols
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        validate_probability_matrix(
            prob,
            path,
        )


        # ----------------------------------------------------
        # Provenance fields
        # ----------------------------------------------------

        if set(
            df["task"]
            .astype(str)
        ) != {
            task
        }:
            raise RuntimeError(
                f"Task metadata mismatch: {path}"
            )

        if set(
            pd.to_numeric(
                df["outer_fold"],
                errors="raise",
            ).astype(int)
        ) != {
            fold
        }:
            raise RuntimeError(
                f"Fold metadata mismatch: {path}"
            )

        if set(
            pd.to_numeric(
                df["seed"],
                errors="raise",
            ).astype(int)
        ) != {
            seed
        }:
            raise RuntimeError(
                f"Seed metadata mismatch: {path}"
            )

        expected_arch = (
            SYSTEM_ARCHITECTURES[
                system
            ]
        )

        if set(
            df["architecture"]
            .astype(str)
        ) != {
            expected_arch
        }:
            raise RuntimeError(
                f"Architecture metadata mismatch: {path}"
            )

        if set(
            df["initialization"]
            .astype(str)
        ) != {
            "XLMR_BASE"
        }:
            raise RuntimeError(
                f"Initialization metadata mismatch: {path}"
            )


        # ----------------------------------------------------
        # Gold / predicted IDs
        # ----------------------------------------------------

        gold_float = pd.to_numeric(
            df["gold_id"],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        gold_int = gold_float.astype(
            np.int64
        )

        if not np.array_equal(
            gold_float,
            gold_int.astype(
                np.float64
            ),
        ):
            raise RuntimeError(
                f"Non-integer gold_id: {path}"
            )

        pred_float = pd.to_numeric(
            df["prediction_id"],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        pred_int = pred_float.astype(
            np.int64
        )

        if not np.array_equal(
            pred_float,
            pred_int.astype(
                np.float64
            ),
        ):
            raise RuntimeError(
                f"Non-integer prediction_id: {path}"
            )

        k = len(
            prob_cols
        )

        if (
            (gold_int < 0).any()
            or
            (gold_int >= k).any()
        ):
            raise RuntimeError(
                f"gold_id outside declared class space: {path}"
            )

        if (
            (pred_int < 0).any()
            or
            (pred_int >= k).any()
        ):
            raise RuntimeError(
                f"prediction_id outside declared class space: {path}"
            )

        argmax_pred = np.argmax(
            prob,
            axis=1,
        )

        if not np.array_equal(
            pred_int,
            argmax_pred,
        ):
            raise RuntimeError(
                f"Stored prediction_id disagrees with "
                f"probability argmax: {path}"
            )


        # ----------------------------------------------------
        # Within-file document uniqueness
        # ----------------------------------------------------

        if df[
            "doc_id"
        ].duplicated().any():
            raise RuntimeError(
                f"Duplicate doc_id within run: {path}"
            )


        key = (
            system,
            task,
            fold,
            seed,
        )

        if key in raw:
            raise RuntimeError(
                f"Duplicate run key: {key}"
            )

        raw[key] = {
            "df":
                df.copy(),

            "prob_cols":
                tuple(
                    prob_cols
                ),

            "k":
                k,

            "path":
                path,

            "sha256":
                observed_sha,
        }

        verification_rows.append({
            "system":
                system,

            "task":
                task,

            "outer_fold":
                fold,

            "seed":
                seed,

            "relative_path":
                str(
                    path.relative_to(
                        ROOT
                    )
                ),

            "expected_sha256":
                expected_sha,

            "observed_sha256":
                observed_sha,

            "sha_match":
                True,

            "n_rows":
                int(
                    len(df)
                ),

            "n_classes":
                int(
                    k
                ),
        })


    expected_keys = {
        (
            system,
            task,
            fold,
            seed,
        )
        for system in SYSTEMS
        for task in TASKS
        for fold in FOLDS
        for seed in SEEDS
    }

    if set(
        raw.keys()
    ) != expected_keys:

        missing = sorted(
            expected_keys
            - set(
                raw.keys()
            )
        )

        extra = sorted(
            set(
                raw.keys()
            )
            - expected_keys
        )

        raise RuntimeError(
            "Input run grid mismatch.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    return (
        raw,
        pd.DataFrame(
            verification_rows
        ),
    )


# ============================================================
# LABEL MAP
# ============================================================

def build_label_map(
    raw,
    system,
    task,
):

    mapping = {}

    for fold in FOLDS:
        for seed in SEEDS:

            df = raw[
                (
                    system,
                    task,
                    fold,
                    seed,
                )
            ][
                "df"
            ]

            for (
                id_col,
                label_col,
            ) in [
                (
                    "gold_id",
                    "gold_label",
                ),
                (
                    "prediction_id",
                    "prediction_label",
                ),
            ]:

                ids = pd.to_numeric(
                    df[id_col],
                    errors="raise",
                ).astype(int)

                labels = (
                    df[label_col]
                    .astype(str)
                )

                for cid, label in zip(
                    ids,
                    labels,
                ):

                    cid = int(
                        cid
                    )

                    if (
                        cid in mapping
                        and mapping[cid]
                        != label
                    ):
                        raise RuntimeError(
                            f"Inconsistent label mapping for "
                            f"{system}/{task}/class {cid}: "
                            f"{mapping[cid]!r} vs {label!r}"
                        )

                    mapping[cid] = (
                        label
                    )

    return mapping


# ============================================================
# FIVE-SEED OOF ASSEMBLY
# ============================================================

def assemble_oof(
    raw,
    system,
    task,
):

    task_entries = [
        raw[
            (
                system,
                task,
                fold,
                seed,
            )
        ]
        for fold in FOLDS
        for seed in SEEDS
    ]

    prob_schemas = {
        entry[
            "prob_cols"
        ]
        for entry
        in task_entries
    }

    if len(
        prob_schemas
    ) != 1:
        raise RuntimeError(
            f"Inconsistent probability schema: {system}/{task}"
        )

    prob_cols = list(
        next(
            iter(
                prob_schemas
            )
        )
    )

    k = len(
        prob_cols
    )

    parts = []

    for fold in FOLDS:
        for seed in SEEDS:

            entry = raw[
                (
                    system,
                    task,
                    fold,
                    seed,
                )
            ]

            df = entry[
                "df"
            ].copy()

            keep = [
                "doc_id",
                "outer_fold",
                "seed",
                "gold_id",
                "gold_label",
                *prob_cols,
            ]

            parts.append(
                df[
                    keep
                ]
            )

    long = pd.concat(
        parts,
        ignore_index=True,
    )

    long[
        "outer_fold"
    ] = pd.to_numeric(
        long[
            "outer_fold"
        ],
        errors="raise",
    ).astype(int)

    long[
        "seed"
    ] = pd.to_numeric(
        long[
            "seed"
        ],
        errors="raise",
    ).astype(int)

    long[
        "gold_id"
    ] = pd.to_numeric(
        long[
            "gold_id"
        ],
        errors="raise",
    ).astype(int)


    # --------------------------------------------------------
    # Exactly five rows per article
    # --------------------------------------------------------

    counts = (
        long
        .groupby(
            "doc_id",
            sort=False,
        )
        .size()
    )

    if not (
        counts
        == N_ENSEMBLE_SEEDS
    ).all():
        bad = counts[
            counts
            != N_ENSEMBLE_SEEDS
        ]

        raise RuntimeError(
            f"{system}/{task}: articles without exactly "
            f"five seed rows:\n{bad}"
        )

    if len(
        counts
    ) != N_ARTICLES:
        raise RuntimeError(
            f"{system}/{task}: expected 199 unique articles, "
            f"got {len(counts)}."
        )


    # --------------------------------------------------------
    # Audit each article
    # --------------------------------------------------------

    rows = []

    for doc_id, g in long.groupby(
        "doc_id",
        sort=False,
    ):

        seeds = sorted(
            g[
                "seed"
            ].astype(
                int
            ).tolist()
        )

        if seeds != SEEDS:
            raise RuntimeError(
                f"{system}/{task}/{doc_id}: seed set mismatch "
                f"{seeds}"
            )

        folds = sorted(
            g[
                "outer_fold"
            ].astype(
                int
            ).unique()
            .tolist()
        )

        if len(
            folds
        ) != 1:
            raise RuntimeError(
                f"{system}/{task}/{doc_id}: "
                f"article occurs in multiple folds: {folds}"
            )

        gold_ids = sorted(
            g[
                "gold_id"
            ].astype(
                int
            ).unique()
            .tolist()
        )

        if len(
            gold_ids
        ) != 1:
            raise RuntimeError(
                f"{system}/{task}/{doc_id}: "
                f"inconsistent gold_id: {gold_ids}"
            )

        gold_labels = sorted(
            g[
                "gold_label"
            ].astype(str)
            .unique()
            .tolist()
        )

        if len(
            gold_labels
        ) != 1:
            raise RuntimeError(
                f"{system}/{task}/{doc_id}: "
                f"inconsistent gold_label: {gold_labels}"
            )

        prob = (
            g[
                prob_cols
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        mean_prob = np.mean(
            prob,
            axis=0,
            dtype=np.float64,
        )

        if not np.isfinite(
            mean_prob
        ).all():
            raise RuntimeError(
                "Non-finite ensemble probabilities."
            )

        if not np.isclose(
            mean_prob.sum(
                dtype=np.float64
            ),
            1.0,
            rtol=0.0,
            atol=ROW_SUM_ATOL,
        ):
            raise RuntimeError(
                f"Ensemble probability sum violation: "
                f"{system}/{task}/{doc_id}"
            )

        pred = int(
            np.argmax(
                mean_prob
            )
        )

        row = {
            "system":
                system,

            "task":
                task,

            "doc_id":
                str(
                    doc_id
                ),

            "outer_fold":
                int(
                    folds[0]
                ),

            "gold_id":
                int(
                    gold_ids[0]
                ),

            "gold_label":
                gold_labels[0],

            "n_seed_models":
                N_ENSEMBLE_SEEDS,

            "ensemble_seed_set":
                "11|29|47|83|131",

            "prediction_id":
                pred,

            "n_classes":
                k,
        }

        for j, value in enumerate(
            mean_prob
        ):
            row[
                f"prob_class_{j}"
            ] = float(
                value
            )

        rows.append(
            row
        )


    out = pd.DataFrame(
        rows
    ).sort_values(
        "doc_id",
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    if len(
        out
    ) != N_ARTICLES:
        raise RuntimeError(
            "OOF ensemble article count mismatch."
        )

    if out[
        "doc_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate OOF ensemble doc_id."
        )

    return (
        out,
        prob_cols,
    )


# ============================================================
# FINAL ENSEMBLE METRICS
# ============================================================

def ensemble_metrics(
    oof,
    k,
):

    y = (
        oof[
            "gold_id"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    pred = (
        oof[
            "prediction_id"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    return {
        "macro_f1":
            macro_f1(
                y,
                pred,
                k,
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y,
                    pred,
                    labels=list(
                        range(k)
                    ),
                    average="weighted",
                    zero_division=0,
                )
            ),
    }


# ============================================================
# PAIRED CONTRAST PREPARATION
# ============================================================

def paired_arrays(
    ensemble_map,
    system_a,
    system_b,
    task,
):

    a = (
        ensemble_map[
            (
                system_a,
                task,
            )
        ]
        .copy()
    )

    b = (
        ensemble_map[
            (
                system_b,
                task,
            )
        ]
        .copy()
    )

    if len(a) != N_ARTICLES:
        raise RuntimeError(
            "System A does not contain 199 OOF articles."
        )

    if len(b) != N_ARTICLES:
        raise RuntimeError(
            "System B does not contain 199 OOF articles."
        )

    a_ids = set(
        a[
            "doc_id"
        ].astype(str)
    )

    b_ids = set(
        b[
            "doc_id"
        ].astype(str)
    )

    if a_ids != b_ids:
        raise RuntimeError(
            f"Paired doc_id mismatch: "
            f"{system_a} vs {system_b}, {task}"
        )

    a = (
        a
        .sort_values(
            "doc_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    b = (
        b
        .sort_values(
            "doc_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    if not np.array_equal(
        a[
            "doc_id"
        ].astype(str)
        .to_numpy(),
        b[
            "doc_id"
        ].astype(str)
        .to_numpy(),
    ):
        raise RuntimeError(
            "Paired doc order mismatch after explicit sort."
        )

    ya = (
        a[
            "gold_id"
        ].to_numpy(
            dtype=np.int64
        )
    )

    yb = (
        b[
            "gold_id"
        ].to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        ya,
        yb,
    ):
        raise RuntimeError(
            f"Paired gold_id mismatch: "
            f"{system_a} vs {system_b}, {task}"
        )

    if not np.array_equal(
        a[
            "gold_label"
        ].astype(str)
        .to_numpy(),
        b[
            "gold_label"
        ].astype(str)
        .to_numpy(),
    ):
        raise RuntimeError(
            f"Paired gold_label mismatch: "
            f"{system_a} vs {system_b}, {task}"
        )


    ka = int(
        a[
            "n_classes"
        ].iloc[0]
    )

    kb = int(
        b[
            "n_classes"
        ].iloc[0]
    )

    if ka != kb:
        raise RuntimeError(
            "Paired class-space size mismatch."
        )

    k = ka

    prob_cols = [
        f"prob_class_{j}"
        for j in range(
            k
        )
    ]

    for col in prob_cols:

        if (
            col not in a.columns
            or col not in b.columns
        ):
            raise RuntimeError(
                f"Missing paired probability column: {col}"
            )

    pa = (
        a[
            prob_cols
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    pb = (
        b[
            prob_cols
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    validate_probability_matrix(
        pa,
        f"{system_a}/{task}/OOF",
    )

    validate_probability_matrix(
        pb,
        f"{system_b}/{task}/OOF",
    )

    return (
        a[
            "doc_id"
        ].astype(str)
        .to_numpy(),
        ya,
        pa,
        pb,
        k,
    )


# ============================================================
# BOOTSTRAP
# ============================================================

def paired_stratified_bootstrap(
    y,
    prob_a,
    prob_b,
    k,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    strata = []

    for class_id in range(
        k
    ):

        idx = np.flatnonzero(
            y
            == class_id
        )

        if len(
            idx
        ) == 0:
            raise RuntimeError(
                f"Declared class {class_id} has no true-label "
                f"articles; stratified bootstrap cannot proceed."
            )

        strata.append(
            idx
        )

    deltas = np.empty(
        N_BOOTSTRAP,
        dtype=np.float64,
    )

    for b in range(
        N_BOOTSTRAP
    ):

        sampled = []

        for idx in strata:

            sampled.append(
                rng.choice(
                    idx,
                    size=len(
                        idx
                    ),
                    replace=True,
                )
            )

        sample_idx = np.concatenate(
            sampled
        )

        yb = y[
            sample_idx
        ]

        pa = prob_a[
            sample_idx
        ]

        pb = prob_b[
            sample_idx
        ]

        pred_a = np.argmax(
            pa,
            axis=1,
        )

        pred_b = np.argmax(
            pb,
            axis=1,
        )

        deltas[b] = (
            macro_f1(
                yb,
                pred_a,
                k,
            )
            -
            macro_f1(
                yb,
                pred_b,
                k,
            )
        )

    ci = np.quantile(
        deltas,
        [
            0.025,
            0.975,
        ],
    )

    return (
        deltas,
        float(
            ci[0]
        ),
        float(
            ci[1]
        ),
        float(
            np.mean(
                deltas
                > 0.0
            )
        ),
    )


# ============================================================
# PAIRED PROBABILITY-VECTOR RANDOMIZATION
# ============================================================

def paired_randomization(
    y,
    prob_a,
    prob_b,
    k,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    observed = (
        macro_f1(
            y,
            np.argmax(
                prob_a,
                axis=1,
            ),
            k,
        )
        -
        macro_f1(
            y,
            np.argmax(
                prob_b,
                axis=1,
            ),
            k,
        )
    )

    null = np.empty(
        N_RANDOMIZATION,
        dtype=np.float64,
    )

    n = len(
        y
    )

    for r in range(
        N_RANDOMIZATION
    ):

        swap = (
            rng.integers(
                0,
                2,
                size=n,
                dtype=np.int8,
            )
            .astype(
                bool
            )
        )

        swap2 = swap[
            :,
            None,
        ]

        randomized_a = np.where(
            swap2,
            prob_b,
            prob_a,
        )

        randomized_b = np.where(
            swap2,
            prob_a,
            prob_b,
        )

        pred_a = np.argmax(
            randomized_a,
            axis=1,
        )

        pred_b = np.argmax(
            randomized_b,
            axis=1,
        )

        null[r] = (
            macro_f1(
                y,
                pred_a,
                k,
            )
            -
            macro_f1(
                y,
                pred_b,
                k,
            )
        )

    extreme = int(
        np.sum(
            np.abs(
                null
            )
            >=
            abs(
                observed
            )
        )
    )

    p = (
        1.0
        + float(
            extreme
        )
    ) / (
        float(
            N_RANDOMIZATION
        )
        + 1.0
    )

    return (
        float(
            observed
        ),
        null,
        float(
            p
        ),
        extreme,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    args = parser.parse_args()


    # --------------------------------------------------------
    # Load + hard-verify every frozen input
    # --------------------------------------------------------

    raw, hash_audit = (
        load_inputs()
    )


    # --------------------------------------------------------
    # Assemble all nine system-task OOF ensembles
    # --------------------------------------------------------

    ensemble_map = {}
    schema_map = {}
    label_maps = {}

    for system in SYSTEMS:
        for task in TASKS:

            label_maps[
                (
                    system,
                    task,
                )
            ] = build_label_map(
                raw,
                system,
                task,
            )

            oof, prob_cols = (
                assemble_oof(
                    raw,
                    system,
                    task,
                )
            )

            ensemble_map[
                (
                    system,
                    task,
                )
            ] = oof

            schema_map[
                (
                    system,
                    task,
                )
            ] = tuple(
                prob_cols
            )


    # --------------------------------------------------------
    # Cross-system schema + gold pairing checks
    # BEFORE any performance inference
    # --------------------------------------------------------

    for task in TASKS:

        schemas = {
            schema_map[
                (
                    system,
                    task,
                )
            ]
            for system in SYSTEMS
        }

        if len(
            schemas
        ) != 1:
            raise RuntimeError(
                f"Cross-system probability schema mismatch: {task}"
            )

        reference = (
            ensemble_map[
                (
                    "STAGE_A_NO_EVENT",
                    task,
                )
            ]
            .sort_values(
                "doc_id",
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        for system in [
            "VERIFIED_CONTEXT",
            "PERMUTED_CONTEXT",
        ]:

            current = (
                ensemble_map[
                    (
                        system,
                        task,
                    )
                ]
                .sort_values(
                    "doc_id",
                    kind="mergesort",
                )
                .reset_index(
                    drop=True
                )
            )

            if not np.array_equal(
                reference[
                    "doc_id"
                ].astype(str)
                .to_numpy(),
                current[
                    "doc_id"
                ].astype(str)
                .to_numpy(),
            ):
                raise RuntimeError(
                    f"Cross-system article mismatch: {task}"
                )

            if not np.array_equal(
                reference[
                    "gold_id"
                ].to_numpy(
                    dtype=np.int64
                ),
                current[
                    "gold_id"
                ].to_numpy(
                    dtype=np.int64
                ),
            ):
                raise RuntimeError(
                    f"Cross-system true-label mismatch: {task}"
                )


    if args.preflight_only:

        print("=" * 88)
        print("STAGE-B STATISTICAL ANALYSIS PREFLIGHT OK")
        print("=" * 88)
        print("Frozen raw inputs verified : 180 / 180")
        print("Systems                    : 3")
        print("Tasks                      : 3")
        print("OOF ensembles verified     : 9")
        print("OOF articles per ensemble  : 199")
        print("Seeds per article          : 5")
        print("Plan SHA                    :", sha256(PLAN))
        print("Implementation lock SHA     :", sha256(LOCK))
        print("Input manifest SHA          :", sha256(INPUT_MANIFEST))
        print()
        print("No task-level performance metrics computed.")
        print("No bootstrap executed.")
        print("No randomization executed.")
        print("EventGold remains SEALED_NOT_ACCESSED.")
        return


    # --------------------------------------------------------
    # Full analysis output must be new
    # --------------------------------------------------------

    if (
        OUT.exists()
        or TMP.exists()
    ):
        raise RuntimeError(
            "Stage-B comparison output already exists. "
            "Refusing overwrite."
        )

    TMP.mkdir(
        parents=True,
        exist_ok=False,
    )


    # --------------------------------------------------------
    # Final OOF ensemble artifact
    # --------------------------------------------------------

    ensemble_rows = []

    for system in SYSTEMS:
        for task in TASKS:

            current = (
                ensemble_map[
                    (
                        system,
                        task,
                    )
                ]
                .copy()
            )

            label_map = (
                label_maps[
                    (
                        system,
                        task,
                    )
                ]
            )

            current[
                "prediction_label"
            ] = (
                current[
                    "prediction_id"
                ]
                .map(
                    lambda x:
                        label_map.get(
                            int(
                                x
                            ),
                            f"class_{int(x)}",
                        )
                )
            )

            ensemble_rows.append(
                current
            )

    ensemble_all = pd.concat(
        ensemble_rows,
        ignore_index=True,
        sort=False,
    )

    ensemble_all.to_csv(
        TMP
        / "stage_b_oof_ensemble_predictions_v3.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Task-level descriptive metrics
    # --------------------------------------------------------

    task_metric_rows = []
    per_class_rows = []
    confusion_rows = []

    for system in SYSTEMS:
        for task in TASKS:

            oof = ensemble_map[
                (
                    system,
                    task,
                )
            ]

            k = int(
                oof[
                    "n_classes"
                ].iloc[0]
            )

            metrics = ensemble_metrics(
                oof,
                k,
            )

            row = {
                "system":
                    system,

                "task":
                    task,

                "n_articles":
                    len(
                        oof
                    ),

                "n_classes":
                    k,
            }

            row.update(
                metrics
            )

            task_metric_rows.append(
                row
            )


            y = (
                oof[
                    "gold_id"
                ]
                .to_numpy(
                    dtype=np.int64
                )
            )

            pred = (
                oof[
                    "prediction_id"
                ]
                .to_numpy(
                    dtype=np.int64
                )
            )

            precision, recall, f1, support = (
                precision_recall_fscore_support(
                    y,
                    pred,
                    labels=list(
                        range(k)
                    ),
                    zero_division=0,
                )
            )

            label_map = label_maps[
                (
                    system,
                    task,
                )
            ]

            for class_id in range(
                k
            ):

                per_class_rows.append({
                    "system":
                        system,

                    "task":
                        task,

                    "class_id":
                        class_id,

                    "class_label":
                        label_map.get(
                            class_id,
                            f"class_{class_id}",
                        ),

                    "support":
                        int(
                            support[
                                class_id
                            ]
                        ),

                    "precision":
                        float(
                            precision[
                                class_id
                            ]
                        ),

                    "recall":
                        float(
                            recall[
                                class_id
                            ]
                        ),

                    "f1":
                        float(
                            f1[
                                class_id
                            ]
                        ),
                })


            cm = confusion_matrix(
                y,
                pred,
                labels=list(
                    range(k)
                ),
            )

            for true_id in range(
                k
            ):
                for pred_id in range(
                    k
                ):

                    confusion_rows.append({
                        "system":
                            system,

                        "task":
                            task,

                        "true_class_id":
                            true_id,

                        "predicted_class_id":
                            pred_id,

                        "count":
                            int(
                                cm[
                                    true_id,
                                    pred_id,
                                ]
                            ),
                    })


    task_metrics_df = pd.DataFrame(
        task_metric_rows
    )

    task_metrics_df.to_csv(
        TMP
        / "task_metrics_stage_b_v3.csv",
        index=False,
    )

    pd.DataFrame(
        per_class_rows
    ).to_csv(
        TMP
        / "per_class_metrics_stage_b_v3.csv",
        index=False,
    )

    pd.DataFrame(
        confusion_rows
    ).to_csv(
        TMP
        / "confusion_matrices_stage_b_v3.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Descriptive seed stability
    # --------------------------------------------------------

    seed_rows = []

    for system in SYSTEMS:
        for task in TASKS:
            for seed in SEEDS:

                pieces = []

                for fold in FOLDS:

                    entry = raw[
                        (
                            system,
                            task,
                            fold,
                            seed,
                        )
                    ]

                    df = entry[
                        "df"
                    ]

                    prob_cols = list(
                        entry[
                            "prob_cols"
                        ]
                    )

                    piece = df[
                        [
                            "doc_id",
                            "gold_id",
                            *prob_cols,
                        ]
                    ].copy()

                    pieces.append(
                        piece
                    )

                seed_df = pd.concat(
                    pieces,
                    ignore_index=True,
                )

                if len(
                    seed_df
                ) != N_ARTICLES:
                    raise RuntimeError(
                        f"Seed-level OOF row count !=199: "
                        f"{system}/{task}/{seed}"
                    )

                if seed_df[
                    "doc_id"
                ].duplicated().any():
                    raise RuntimeError(
                        f"Seed-level duplicate article: "
                        f"{system}/{task}/{seed}"
                    )

                prob_cols = (
                    extract_prob_columns(
                        seed_df
                    )
                )

                k = len(
                    prob_cols
                )

                prob = seed_df[
                    prob_cols
                ].to_numpy(
                    dtype=np.float64
                )

                validate_probability_matrix(
                    prob,
                    f"{system}/{task}/seed{seed}",
                )

                y = pd.to_numeric(
                    seed_df[
                        "gold_id"
                    ],
                    errors="raise",
                ).to_numpy(
                    dtype=np.int64
                )

                pred = np.argmax(
                    prob,
                    axis=1,
                )

                seed_rows.append({
                    "system":
                        system,

                    "task":
                        task,

                    "seed":
                        seed,

                    "n_articles":
                        len(
                            seed_df
                        ),

                    "macro_f1":
                        macro_f1(
                            y,
                            pred,
                            k,
                        ),

                    "balanced_accuracy":
                        float(
                            balanced_accuracy_score(
                                y,
                                pred,
                            )
                        ),

                    "accuracy":
                        float(
                            accuracy_score(
                                y,
                                pred,
                            )
                        ),

                    "weighted_f1":
                        float(
                            f1_score(
                                y,
                                pred,
                                labels=list(
                                    range(k)
                                ),
                                average="weighted",
                                zero_division=0,
                            )
                        ),
                })

    pd.DataFrame(
        seed_rows
    ).to_csv(
        TMP
        / "seed_stability_stage_b_v3.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Confirmatory paired inference
    # --------------------------------------------------------

    inference_rows = []
    bootstrap_rows = []
    randomization_rows = []

    for family_name, (
        system_a,
        system_b,
    ) in FAMILIES.items():

        family_indices = []

        for task in TASKS:

            (
                doc_ids,
                y,
                prob_a,
                prob_b,
                k,
            ) = paired_arrays(
                ensemble_map,
                system_a,
                system_b,
                task,
            )


            observed_delta = (
                macro_f1(
                    y,
                    np.argmax(
                        prob_a,
                        axis=1,
                    ),
                    k,
                )
                -
                macro_f1(
                    y,
                    np.argmax(
                        prob_b,
                        axis=1,
                    ),
                    k,
                )
            )


            # ------------------------------------------------
            # Bootstrap
            # ------------------------------------------------

            boot_seed = BOOTSTRAP_SEEDS[
                (
                    family_name,
                    task,
                )
            ]

            (
                boot,
                ci_low,
                ci_high,
                boot_prob_gt_zero,
            ) = paired_stratified_bootstrap(
                y=y,
                prob_a=prob_a,
                prob_b=prob_b,
                k=k,
                seed=boot_seed,
            )

            for index, value in enumerate(
                boot
            ):

                bootstrap_rows.append({
                    "family":
                        family_name,

                    "system_a":
                        system_a,

                    "system_b":
                        system_b,

                    "task":
                        task,

                    "replicate":
                        index,

                    "rng_seed":
                        boot_seed,

                    "delta_macro_f1_A_minus_B":
                        float(
                            value
                        ),
                })


            # ------------------------------------------------
            # Randomization
            # ------------------------------------------------

            rand_seed = RANDOMIZATION_SEEDS[
                (
                    family_name,
                    task,
                )
            ]

            (
                randomized_observed,
                null,
                pvalue,
                extreme_count,
            ) = paired_randomization(
                y=y,
                prob_a=prob_a,
                prob_b=prob_b,
                k=k,
                seed=rand_seed,
            )

            if not np.isclose(
                randomized_observed,
                observed_delta,
                rtol=0.0,
                atol=1e-15,
            ):
                raise RuntimeError(
                    "Observed contrast changed between "
                    "point-estimate and randomization paths."
                )

            for index, value in enumerate(
                null
            ):

                randomization_rows.append({
                    "family":
                        family_name,

                    "system_a":
                        system_a,

                    "system_b":
                        system_b,

                    "task":
                        task,

                    "replicate":
                        index,

                    "rng_seed":
                        rand_seed,

                    "null_delta_macro_f1_A_minus_B":
                        float(
                            value
                        ),
                })


            metrics_a = task_metrics_df[
                (
                    task_metrics_df[
                        "system"
                    ]
                    == system_a
                )
                &
                (
                    task_metrics_df[
                        "task"
                    ]
                    == task
                )
            ].iloc[
                0
            ]

            metrics_b = task_metrics_df[
                (
                    task_metrics_df[
                        "system"
                    ]
                    == system_b
                )
                &
                (
                    task_metrics_df[
                        "task"
                    ]
                    == task
                )
            ].iloc[
                0
            ]


            inference_rows.append({
                "family":
                    family_name,

                "system_a":
                    system_a,

                "system_b":
                    system_b,

                "task":
                    task,

                "n_articles":
                    len(
                        y
                    ),

                "n_classes":
                    k,

                "macro_f1_system_a":
                    float(
                        metrics_a[
                            "macro_f1"
                        ]
                    ),

                "macro_f1_system_b":
                    float(
                        metrics_b[
                            "macro_f1"
                        ]
                    ),

                "delta_macro_f1_A_minus_B":
                    float(
                        observed_delta
                    ),

                "bootstrap_ci95_low":
                    ci_low,

                "bootstrap_ci95_high":
                    ci_high,

                "bootstrap_fraction_delta_gt_0":
                    boot_prob_gt_zero,

                "bootstrap_replicates":
                    N_BOOTSTRAP,

                "bootstrap_seed":
                    boot_seed,

                "randomization_p_two_sided":
                    pvalue,

                "randomization_extreme_count":
                    extreme_count,

                "randomization_replicates":
                    N_RANDOMIZATION,

                "randomization_seed":
                    rand_seed,
            })

            family_indices.append(
                len(
                    inference_rows
                )
                - 1
            )


        # ----------------------------------------------------
        # Separate 3-task BH family
        # ----------------------------------------------------

        family_p = [
            inference_rows[i][
                "randomization_p_two_sided"
            ]
            for i in family_indices
        ]

        if len(family_p) != 3:
            raise RuntimeError(
                f"Expected exactly 3 p-values in {family_name}; "
                f"got {len(family_p)}."
            )

        family_q = bh_adjust(
            family_p
        )

        for i, q in zip(
            family_indices,
            family_q,
        ):
            inference_rows[i][
                "bh_q_within_family"
            ] = float(q)
    inference_df = pd.DataFrame(
        inference_rows
    )

    inference_df.to_csv(
        TMP
        / "paired_inference_stage_b_v3.csv",
        index=False,
    )

    pd.DataFrame(
        bootstrap_rows
    ).to_csv(
        TMP
        / "bootstrap_deltas_stage_b_v3.csv",
        index=False,
    )

    pd.DataFrame(
        randomization_rows
    ).to_csv(
        TMP
        / "randomization_null_deltas_stage_b_v3.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Save raw-input hash verification
    # --------------------------------------------------------

    hash_audit.to_csv(
        TMP
        / "stage_b_analysis_input_hash_verification_v3.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    primary_rows = (
        inference_df[
            inference_df[
                "family"
            ]
            == "PRIMARY"
        ]
        .set_index(
            "task"
        )
    )

    control_rows = (
        inference_df[
            inference_df[
                "family"
            ]
            == "CONTEXT_SPECIFICITY"
        ]
        .set_index(
            "task"
        )
    )


    summary = {
        "status":
            "STAGE_B_COMPARISON_ANALYSIS_COMPLETE",

        "analysis_design": {
            "primary_family":
                "VERIFIED_CONTEXT_vs_STAGE_A_NO_EVENT",

            "context_specificity_family":
                "VERIFIED_CONTEXT_vs_PERMUTED_CONTEXT",

            "primary_metric":
                "five-seed probability-ensemble OOF Macro-F1",

            "inferential_unit":
                "article",

            "n_articles_per_task":
                199,

            "training_seeds":
                SEEDS,

            "bootstrap_replicates":
                N_BOOTSTRAP,

            "randomization_replicates":
                N_RANDOMIZATION,

            "multiple_testing":
                (
                    "Benjamini-Hochberg separately across "
                    "3 tasks within each family"
                ),
        },

        "frozen_inputs": {
            "analysis_plan_sha256":
                sha256(PLAN),

            "statistical_implementation_lock_sha256":
                sha256(LOCK),

            "prediction_input_manifest_sha256":
                sha256(INPUT_MANIFEST),

            "analysis_script_sha256":
                sha256(Path(__file__)),
        },

        "primary_family": {},

        "context_specificity_family": {},

        "interpretation_constraints": {
            "legacyaux_status":
                "DEVELOPMENTAL_EVIDENCE",

            "null_result_is_equivalence":
                False,

            "bootstrap_fraction_is_bayesian_posterior":
                False,

            "seeds_are_inferential_observations":
                False,

            "folds_are_inferential_observations":
                False,

            "causal_claim_from_permuted_control":
                False,

            "eventgold_status":
                "SEALED_NOT_ACCESSED",
        },
    }

    for task in TASKS:

        r = primary_rows.loc[task]

        summary["primary_family"][task] = {
            "macro_f1_verified_context":
                float(r["macro_f1_system_a"]),

            "macro_f1_stage_a_no_event":
                float(r["macro_f1_system_b"]),

            "delta_macro_f1":
                float(r["delta_macro_f1_A_minus_B"]),

            "bootstrap_ci95": [
                float(r["bootstrap_ci95_low"]),
                float(r["bootstrap_ci95_high"]),
            ],

            "randomization_p_two_sided":
                float(r["randomization_p_two_sided"]),

            "bh_q_within_family":
                float(r["bh_q_within_family"]),
        }

        r = control_rows.loc[task]

        summary["context_specificity_family"][task] = {
            "macro_f1_verified_context":
                float(r["macro_f1_system_a"]),

            "macro_f1_permuted_context":
                float(r["macro_f1_system_b"]),

            "delta_macro_f1":
                float(r["delta_macro_f1_A_minus_B"]),

            "bootstrap_ci95": [
                float(r["bootstrap_ci95_low"]),
                float(r["bootstrap_ci95_high"]),
            ],

            "randomization_p_two_sided":
                float(r["randomization_p_two_sided"]),

            "bh_q_within_family":
                float(r["bh_q_within_family"]),
        }
    (
        TMP
        / "analysis_summary_v3.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # Hash every output except checksum file itself
    # --------------------------------------------------------

    output_files = sorted(
        [
            p
            for p in TMP.iterdir()
            if p.is_file()
        ],
        key=lambda p:
            p.name,
    )

    checksum_path = (
        TMP
        / "STAGE_B_COMPARISON_RESULTS_SHA256SUMS_v3.txt"
    )

    checksum_path.write_text(
        "".join(
            f"{sha256(p)}  {p.name}\n"
            for p
            in output_files
        ),
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # Atomic finalization
    # --------------------------------------------------------

    TMP.rename(
        OUT
    )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------


    print("=" * 100)
    print("STAGE-B PREDECLARED COMPARISON ANALYSIS COMPLETE")
    print("=" * 100)

    report_columns = [
        "task",
        "macro_f1_system_a",
        "macro_f1_system_b",
        "delta_macro_f1_A_minus_B",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "randomization_p_two_sided",
        "bh_q_within_family",
    ]

    print()
    print("Primary family:")

    primary_report = inference_df.loc[
        inference_df["family"].eq("PRIMARY"),
        report_columns,
    ]

    print(
        primary_report.to_string(
            index=False
        )
    )

    print()
    print("Context-specificity family:")

    control_report = inference_df.loc[
        inference_df["family"].eq("CONTEXT_SPECIFICITY"),
        report_columns,
    ]

    print(
        control_report.to_string(
            index=False
        )
    )

    print()
    print(
        "Results directory:",
        OUT,
    )

    print()
    print(
        "EventGold remains SEALED_NOT_ACCESSED."
    )


if __name__ == "__main__":
    main()
