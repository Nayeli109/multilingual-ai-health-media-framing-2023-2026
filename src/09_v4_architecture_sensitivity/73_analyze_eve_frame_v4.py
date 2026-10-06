#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

V4 = (
    ROOT
    / "01_event_aware_v3/09_eve_frame_v4_architecture_sensitivity"
)

STAGEB_ANALYSIS = (
    ROOT
    / "01_event_aware_v3/08_eve_frame_stage_b_event_context/"
      "67_analyze_stage_b_v3.py"
)

PROTOCOL = (
    V4
    / "EVE_FRAME_V4_ARCHITECTURE_SENSITIVITY_PROTOCOL_v4.txt"
)

STAT_LOCK = (
    V4
    / "V4_STATISTICAL_IMPLEMENTATION_LOCK_v4.txt"
)

RNG_PATH = (
    V4
    / "V4_INFERENCE_RNG_v4.json"
)

STAT_FREEZE = (
    V4
    / "FROZEN_V4_STATISTICAL_IMPLEMENTATION_SHA256SUMS_v4.txt"
)

INPUT_MANIFEST = (
    V4
    / "V4_ANALYSIS_INPUT_PREDICTION_HASHES_v4.csv"
)

INPUT_FREEZE = (
    V4
    / "FROZEN_V4_ANALYSIS_INPUTS_SHA256SUMS_v4.txt"
)

OUT = (
    V4
    / "v4_confirmatory_analysis_v4"
)

TMP = (
    V4
    / "v4_confirmatory_analysis_v4_tmp"
)


# ============================================================
# FROZEN HASHES
# ============================================================

EXPECTED_STAGEB_ANALYSIS_SHA = (
    "76bf72bafc71c5140845270cbe527d91506dc265232813fd6d0c7bc4a2d34b3a"
)

EXPECTED_PROTOCOL_SHA = (
    "39459e4d72bfc14a2ee1cd02d2414f32989788a2de32a431c68d16f7913c6dbe"
)

EXPECTED_STAT_LOCK_SHA = (
    "1a0432d770438a2698166bda7e66d6a22a3b6684c67b4af2f6d8e9242a873ba8"
)

EXPECTED_RNG_SHA = (
    "ec71f0a3cb06e83c3f4ef2516e60f3212b652e1e403c65d8cd0128fa1a6c915e"
)

EXPECTED_STAT_FREEZE_SHA = (
    "fa8eb4b455af5b5982b41965afc3307dd80deecbc05d605bf082bd59633ded66"
)

EXPECTED_INPUT_MANIFEST_SHA = (
    "491834fc5ca1d3ba8c9ef20e5535dc12f8549f0547a32894b88eb992d168e250"
)

EXPECTED_INPUT_FREEZE_SHA = (
    "983f5d5eb19c33c838b0bc30966a913aad30332fd693267919767aa75c8efc01"
)


# ============================================================
# FROZEN DESIGN
# ============================================================

TASKS = [
    "primary_frame",
    "stance",
    "misinformation_relation",
]

FOLDS = [
    1,
    2,
    3,
    4,
]

SEEDS = [
    11,
    29,
    47,
    83,
    131,
]

ARCHITECTURES = [
    "V4_FILM_R32",
    "V4_LR_BILINEAR_R32",
]

SYSTEMS = [
    "STAGE_A_NO_EVENT",
    "V4_FILM_R32_VERIFIED_CONTEXT",
    "V4_FILM_R32_PERMUTED_CONTEXT",
    "V4_LR_BILINEAR_R32_VERIFIED_CONTEXT",
    "V4_LR_BILINEAR_R32_PERMUTED_CONTEXT",
]

SYSTEM_ARCHITECTURES = {
    "STAGE_A_NO_EVENT":
        "EVE_FRAME_STAGE_A_NO_EVENT",

    "V4_FILM_R32_VERIFIED_CONTEXT":
        "EVE_FRAME_V4_FILM_R32_VERIFIED_CONTEXT",

    "V4_FILM_R32_PERMUTED_CONTEXT":
        "EVE_FRAME_V4_FILM_R32_PERMUTED_CONTEXT",

    "V4_LR_BILINEAR_R32_VERIFIED_CONTEXT":
        "EVE_FRAME_V4_LR_BILINEAR_R32_VERIFIED_CONTEXT",

    "V4_LR_BILINEAR_R32_PERMUTED_CONTEXT":
        "EVE_FRAME_V4_LR_BILINEAR_R32_PERMUTED_CONTEXT",
}

N_ARTICLES = 199
N_ENSEMBLE_SEEDS = 5
N_BOOTSTRAP = 10000
N_RANDOMIZATION = 10000

PROB_RE = re.compile(
    r"^prob_class_([0-9]+)$"
)


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

    path = Path(path)

    if not path.is_file():
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


def load_stageb():

    spec = importlib.util.spec_from_file_location(
        "stageb_frozen_v3",
        STAGEB_ANALYSIS,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not import frozen Stage-B analysis."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    required = [
        "macro_f1",
        "extract_prob_columns",
        "validate_probability_matrix",
        "paired_stratified_bootstrap",
        "paired_randomization",
    ]

    missing = [
        name
        for name in required
        if not hasattr(
            module,
            name,
        )
    ]

    if missing:
        raise RuntimeError(
            "Frozen Stage-B implementation missing "
            f"required functions: {missing}"
        )

    if int(
        module.N_BOOTSTRAP
    ) != N_BOOTSTRAP:
        raise RuntimeError(
            "Stage-B bootstrap replicate mismatch."
        )

    if int(
        module.N_RANDOMIZATION
    ) != N_RANDOMIZATION:
        raise RuntimeError(
            "Stage-B randomization replicate mismatch."
        )

    return module


def load_rng():

    rng = json.loads(
        RNG_PATH.read_text(
            encoding="utf-8"
        )
    )

    if int(
        rng["n_bootstrap"]
    ) != N_BOOTSTRAP:
        raise RuntimeError(
            "Frozen bootstrap count mismatch."
        )

    if int(
        rng["n_randomization"]
    ) != N_RANDOMIZATION:
        raise RuntimeError(
            "Frozen randomization count mismatch."
        )

    if list(
        rng["training_seeds"]
    ) != SEEDS:
        raise RuntimeError(
            "Frozen training seed set mismatch."
        )

    expected = {
        f"{family}|{arch}|{task}"
        for family in [
            "INCREMENTAL",
            "CONTEXT_SPECIFICITY",
        ]
        for arch in ARCHITECTURES
        for task in TASKS
    }

    if set(
        rng["bootstrap_seeds"]
    ) != expected:
        raise RuntimeError(
            "Frozen bootstrap key set mismatch."
        )

    if set(
        rng["randomization_seeds"]
    ) != expected:
        raise RuntimeError(
            "Frozen randomization key set mismatch."
        )

    return rng


def bh_adjust6(pvalues):

    p = np.asarray(
        pvalues,
        dtype=np.float64,
    )

    m = len(p)

    if m != 6:
        raise RuntimeError(
            f"BH family must contain exactly "
            f"6 p-values; got {m}."
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

    sorted_p = p[
        order
    ]

    raw = (
        sorted_p
        * float(m)
        / np.arange(
            1,
            m + 1,
            dtype=np.float64,
        )
    )

    adjusted_sorted = (
        np.minimum.accumulate(
            raw[::-1]
        )[::-1]
    )

    adjusted_sorted = np.minimum(
        adjusted_sorted,
        1.0,
    )

    adjusted = np.empty(
        m,
        dtype=np.float64,
    )

    adjusted[
        order
    ] = adjusted_sorted

    return adjusted


def verify_frozen_references():

    require_hash(
        STAGEB_ANALYSIS,
        EXPECTED_STAGEB_ANALYSIS_SHA,
        "Stage-B statistical implementation",
    )

    require_hash(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA,
        "V4 protocol",
    )

    require_hash(
        STAT_LOCK,
        EXPECTED_STAT_LOCK_SHA,
        "V4 statistical lock",
    )

    require_hash(
        RNG_PATH,
        EXPECTED_RNG_SHA,
        "V4 RNG map",
    )

    require_hash(
        STAT_FREEZE,
        EXPECTED_STAT_FREEZE_SHA,
        "V4 statistical freeze",
    )

    require_hash(
        INPUT_MANIFEST,
        EXPECTED_INPUT_MANIFEST_SHA,
        "V4 analysis-input manifest",
    )

    require_hash(
        INPUT_FREEZE,
        EXPECTED_INPUT_FREEZE_SHA,
        "V4 analysis-input freeze",
    )


# ============================================================
# LOAD + VERIFY THE 300 FROZEN PREDICTION FILES
# ============================================================

def load_inputs(stageb):

    manifest = pd.read_csv(
        INPUT_MANIFEST
    )

    required_manifest = {
        "system",
        "task",
        "outer_fold",
        "seed",
        "relative_path",
        "sha256",
    }

    missing = (
        required_manifest
        - set(
            manifest.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Input manifest missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        manifest
    ) != 300:
        raise RuntimeError(
            f"Expected exactly 300 prediction "
            f"inputs; got {len(manifest)}."
        )

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

    raw = {}
    audit_rows = []

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

        key = (
            system,
            task,
            fold,
            seed,
        )

        if key not in expected_keys:
            raise RuntimeError(
                f"Unexpected input configuration: {key}"
            )

        if key in raw:
            raise RuntimeError(
                f"Duplicate input configuration: {key}"
            )

        path = (
            ROOT
            / str(
                r.relative_path
            )
        )

        if not path.is_file():
            raise RuntimeError(
                f"Missing prediction input: {path}"
            )

        observed_sha = sha256(
            path
        )

        expected_sha = str(
            r.sha256
        )

        if observed_sha != expected_sha:
            raise RuntimeError(
                f"Prediction SHA mismatch: {path}"
            )

        df = pd.read_csv(
            path
        )

        required_df = {
            "doc_id",
            "task",
            "outer_fold",
            "architecture",
            "seed",
            "gold_id",
            "gold_label",
        }

        missing_df = (
            required_df
            - set(
                df.columns
            )
        )

        if missing_df:
            raise RuntimeError(
                f"Prediction file missing columns "
                f"{sorted(missing_df)}: {path}"
            )

        if len(
            df
        ) == 0:
            raise RuntimeError(
                f"Prediction file empty: {path}"
            )

        if df[
            "doc_id"
        ].astype(str).duplicated().any():
            raise RuntimeError(
                f"Duplicate article within prediction "
                f"file: {path}"
            )

        if set(
            df[
                "task"
            ].astype(str)
        ) != {
            task
        }:
            raise RuntimeError(
                f"Task metadata mismatch: {path}"
            )

        if set(
            pd.to_numeric(
                df[
                    "outer_fold"
                ],
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
                df[
                    "seed"
                ],
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
            df[
                "architecture"
            ].astype(str)
        ) != {
            expected_arch
        }:
            raise RuntimeError(
                "Architecture metadata mismatch.\n"
                f"System: {system}\n"
                f"Expected: {expected_arch}\n"
                f"Observed: "
                f"{sorted(set(df['architecture'].astype(str)))}\n"
                f"Path: {path}"
            )

        prob_cols = list(
            stageb.extract_prob_columns(
                df
            )
        )

        if not prob_cols:
            raise RuntimeError(
                f"No probability columns: {path}"
            )

        expected_prob_cols = [
            f"prob_class_{i}"
            for i in range(
                len(
                    prob_cols
                )
            )
        ]

        if prob_cols != expected_prob_cols:
            raise RuntimeError(
                "Probability classes are not "
                f"contiguous from zero: {path}"
            )

        prob = df[
            prob_cols
        ].to_numpy(
            dtype=np.float64
        )

        stageb.validate_probability_matrix(
            prob,
            f"{system}/{task}/fold{fold}/seed{seed}",
        )

        gold = pd.to_numeric(
            df[
                "gold_id"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.int64
        )

        k = len(
            prob_cols
        )

        if (
            (gold < 0).any()
            or (gold >= k).any()
        ):
            raise RuntimeError(
                f"Gold class ID outside range: {path}"
            )

        if "prediction_id" in df.columns:

            pred = pd.to_numeric(
                df[
                    "prediction_id"
                ],
                errors="raise",
            ).to_numpy(
                dtype=np.int64
            )

            expected_pred = np.argmax(
                prob,
                axis=1,
            )

            if not np.array_equal(
                pred,
                expected_pred,
            ):
                raise RuntimeError(
                    f"Stored prediction_id does not "
                    f"match argmax(prob): {path}"
                )

        df = df.copy()

        df[
            "doc_id"
        ] = df[
            "doc_id"
        ].astype(str)

        df[
            "gold_id"
        ] = pd.to_numeric(
            df[
                "gold_id"
            ],
            errors="raise",
        ).astype(int)

        df[
            "outer_fold"
        ] = pd.to_numeric(
            df[
                "outer_fold"
            ],
            errors="raise",
        ).astype(int)

        df[
            "seed"
        ] = pd.to_numeric(
            df[
                "seed"
            ],
            errors="raise",
        ).astype(int)

        raw[
            key
        ] = {
            "df":
                df,

            "prob_cols":
                tuple(
                    prob_cols
                ),

            "path":
                path,

            "sha256":
                observed_sha,
        }

        audit_rows.append({
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

            "hash_match":
                True,
        })

    if set(
        raw.keys()
    ) != expected_keys:
        raise RuntimeError(
            "Loaded input Cartesian design "
            "does not match frozen design."
        )

    return (
        raw,
        pd.DataFrame(
            audit_rows
        ),
    )


# ============================================================
# FIVE-SEED OOF PROBABILITY ENSEMBLE
# ============================================================

def assemble_oof(
    stageb,
    raw,
    system,
    task,
):

    seed_frames = []
    canonical_prob_cols = None

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

            if canonical_prob_cols is None:
                canonical_prob_cols = (
                    prob_cols
                )

            elif prob_cols != canonical_prob_cols:
                raise RuntimeError(
                    f"Probability schema changed "
                    f"within {system}/{task}."
                )

            keep = [
                "doc_id",
                "outer_fold",
                "seed",
                "gold_id",
                "gold_label",
                *prob_cols,
            ]

            pieces.append(
                df[
                    keep
                ].copy()
            )

        seed_df = pd.concat(
            pieces,
            ignore_index=True,
            sort=False,
        )

        if len(
            seed_df
        ) != N_ARTICLES:
            raise RuntimeError(
                f"Seed-level OOF row count != "
                f"{N_ARTICLES}: "
                f"{system}/{task}/{seed}"
            )

        if seed_df[
            "doc_id"
        ].duplicated().any():
            raise RuntimeError(
                f"Seed-level duplicate article: "
                f"{system}/{task}/{seed}"
            )

        seed_df = (
            seed_df
            .sort_values(
                "doc_id",
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        seed_frames.append(
            seed_df
        )

    reference = (
        seed_frames[
            0
        ]
    )

    for current in seed_frames[
        1:
    ]:

        if not np.array_equal(
            reference[
                "doc_id"
            ].to_numpy(),
            current[
                "doc_id"
            ].to_numpy(),
        ):
            raise RuntimeError(
                f"Cross-seed article mismatch: "
                f"{system}/{task}"
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
                f"Cross-seed gold mismatch: "
                f"{system}/{task}"
            )

        if not np.array_equal(
            reference[
                "outer_fold"
            ].to_numpy(
                dtype=np.int64
            ),
            current[
                "outer_fold"
            ].to_numpy(
                dtype=np.int64
            ),
        ):
            raise RuntimeError(
                f"Cross-seed outer-fold mismatch: "
                f"{system}/{task}"
            )

        if not np.array_equal(
            reference[
                "gold_label"
            ].astype(str).to_numpy(),
            current[
                "gold_label"
            ].astype(str).to_numpy(),
        ):
            raise RuntimeError(
                f"Cross-seed gold-label mismatch: "
                f"{system}/{task}"
            )

    stack = np.stack(
        [
            current[
                canonical_prob_cols
            ].to_numpy(
                dtype=np.float64
            )
            for current
            in seed_frames
        ],
        axis=0,
    )

    if stack.shape[
        0
    ] != N_ENSEMBLE_SEEDS:
        raise RuntimeError(
            "Unexpected ensemble seed count."
        )

    mean_prob = np.mean(
        stack,
        axis=0,
        dtype=np.float64,
    )

    stageb.validate_probability_matrix(
        mean_prob,
        f"{system}/{task}/five_seed_ensemble",
    )

    pred = np.argmax(
        mean_prob,
        axis=1,
    )

    out = pd.DataFrame({
        "system":
            system,

        "task":
            task,

        "doc_id":
            reference[
                "doc_id"
            ].astype(str).to_numpy(),

        "outer_fold":
            reference[
                "outer_fold"
            ].to_numpy(
                dtype=np.int64
            ),

        "gold_id":
            reference[
                "gold_id"
            ].to_numpy(
                dtype=np.int64
            ),

        "gold_label":
            reference[
                "gold_label"
            ].astype(str).to_numpy(),

        "prediction_id":
            pred.astype(
                np.int64
            ),

        "n_classes":
            len(
                canonical_prob_cols
            ),

        "n_seed_models":
            N_ENSEMBLE_SEEDS,

        "ensemble_seed_set":
            "11|29|47|83|131",
    })

    for j, col in enumerate(
        canonical_prob_cols
    ):
        out[
            col
        ] = mean_prob[
            :,
            j
        ]

    if len(
        out
    ) != N_ARTICLES:
        raise RuntimeError(
            "OOF ensemble article count mismatch."
        )

    return (
        out,
        tuple(
            canonical_prob_cols
        ),
    )


def build_all_ensembles(
    stageb,
    raw,
):

    ensemble_map = {}
    schema_map = {}

    for system in SYSTEMS:

        for task in TASKS:

            oof, schema = assemble_oof(
                stageb,
                raw,
                system,
                task,
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
            ] = schema

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
                f"Cross-system probability "
                f"schema mismatch: {task}"
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

        for system in SYSTEMS[
            1:
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
                ].to_numpy(),
                current[
                    "doc_id"
                ].to_numpy(),
            ):
                raise RuntimeError(
                    f"Cross-system article mismatch: "
                    f"{task}"
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
                    f"Cross-system true-label mismatch: "
                    f"{task}"
                )

    return (
        ensemble_map,
        schema_map,
    )


# ============================================================
# METRICS + PAIRED ARRAYS
# ============================================================

def task_metrics(
    stageb,
    oof,
):

    y = oof[
        "gold_id"
    ].to_numpy(
        dtype=np.int64
    )

    pred = oof[
        "prediction_id"
    ].to_numpy(
        dtype=np.int64
    )

    k = int(
        oof[
            "n_classes"
        ].iloc[
            0
        ]
    )

    return {
        "macro_f1":
            float(
                stageb.macro_f1(
                    y,
                    pred,
                    k,
                )
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
                        range(
                            k
                        )
                    ),
                    average="weighted",
                    zero_division=0,
                )
            ),
    }


def paired_arrays(
    ensemble_map,
    schema_map,
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
        .sort_values(
            "doc_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    b = (
        ensemble_map[
            (
                system_b,
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
        a[
            "doc_id"
        ].to_numpy(),
        b[
            "doc_id"
        ].to_numpy(),
    ):
        raise RuntimeError(
            f"Paired article mismatch: "
            f"{system_a} vs {system_b}/{task}"
        )

    y_a = a[
        "gold_id"
    ].to_numpy(
        dtype=np.int64
    )

    y_b = b[
        "gold_id"
    ].to_numpy(
        dtype=np.int64
    )

    if not np.array_equal(
        y_a,
        y_b,
    ):
        raise RuntimeError(
            f"Paired gold mismatch: "
            f"{system_a} vs {system_b}/{task}"
        )

    schema_a = list(
        schema_map[
            (
                system_a,
                task,
            )
        ]
    )

    schema_b = list(
        schema_map[
            (
                system_b,
                task,
            )
        ]
    )

    if schema_a != schema_b:
        raise RuntimeError(
            f"Paired probability schema mismatch: "
            f"{system_a} vs {system_b}/{task}"
        )

    prob_a = a[
        schema_a
    ].to_numpy(
        dtype=np.float64
    )

    prob_b = b[
        schema_b
    ].to_numpy(
        dtype=np.float64
    )

    return (
        y_a,
        prob_a,
        prob_b,
        len(
            schema_a
        ),
    )


# ============================================================
# CONFIRMATORY ANALYSIS
# ============================================================

def run_confirmatory(
    stageb,
    rng,
    ensemble_map,
    schema_map,
):

    metric_rows = []

    for system in SYSTEMS:

        for task in TASKS:

            oof = ensemble_map[
                (
                    system,
                    task,
                )
            ]

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
                    int(
                        oof[
                            "n_classes"
                        ].iloc[
                            0
                        ]
                    ),
            }

            row.update(
                task_metrics(
                    stageb,
                    oof,
                )
            )

            metric_rows.append(
                row
            )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    inference_rows = []
    bootstrap_rows = []
    randomization_rows = []

    for architecture in ARCHITECTURES:

        verified = (
            f"{architecture}"
            "_VERIFIED_CONTEXT"
        )

        permuted = (
            f"{architecture}"
            "_PERMUTED_CONTEXT"
        )

        contrasts = [
            (
                "INCREMENTAL",
                verified,
                "STAGE_A_NO_EVENT",
            ),
            (
                "CONTEXT_SPECIFICITY",
                verified,
                permuted,
            ),
        ]

        for family, system_a, system_b in contrasts:

            for task in TASKS:

                y, prob_a, prob_b, k = (
                    paired_arrays(
                        ensemble_map,
                        schema_map,
                        system_a,
                        system_b,
                        task,
                    )
                )

                pred_a = np.argmax(
                    prob_a,
                    axis=1,
                )

                pred_b = np.argmax(
                    prob_b,
                    axis=1,
                )

                metric_a = float(
                    stageb.macro_f1(
                        y,
                        pred_a,
                        k,
                    )
                )

                metric_b = float(
                    stageb.macro_f1(
                        y,
                        pred_b,
                        k,
                    )
                )

                observed_delta = (
                    metric_a
                    - metric_b
                )

                rng_key = (
                    f"{family}|"
                    f"{architecture}|"
                    f"{task}"
                )

                boot_seed = int(
                    rng[
                        "bootstrap_seeds"
                    ][
                        rng_key
                    ]
                )

                rand_seed = int(
                    rng[
                        "randomization_seeds"
                    ][
                        rng_key
                    ]
                )

                (
                    boot,
                    ci_low,
                    ci_high,
                    boot_prob_gt_zero,
                ) = (
                    stageb.paired_stratified_bootstrap(
                        y=y,
                        prob_a=prob_a,
                        prob_b=prob_b,
                        k=k,
                        seed=boot_seed,
                    )
                )

                if len(
                    boot
                ) != N_BOOTSTRAP:
                    raise RuntimeError(
                        "Bootstrap replicate count mismatch."
                    )

                (
                    randomized_observed,
                    null,
                    pvalue,
                    extreme_count,
                ) = (
                    stageb.paired_randomization(
                        y=y,
                        prob_a=prob_a,
                        prob_b=prob_b,
                        k=k,
                        seed=rand_seed,
                    )
                )

                if len(
                    null
                ) != N_RANDOMIZATION:
                    raise RuntimeError(
                        "Randomization replicate count mismatch."
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

                inference_rows.append({
                    "family":
                        family,

                    "architecture":
                        architecture,

                    "task":
                        task,

                    "system_a":
                        system_a,

                    "system_b":
                        system_b,

                    "macro_f1_system_a":
                        metric_a,

                    "macro_f1_system_b":
                        metric_b,

                    "delta_macro_f1_A_minus_B":
                        float(
                            observed_delta
                        ),

                    "bootstrap_ci95_low":
                        float(
                            ci_low
                        ),

                    "bootstrap_ci95_high":
                        float(
                            ci_high
                        ),

                    "bootstrap_fraction_delta_gt_0":
                        float(
                            boot_prob_gt_zero
                        ),

                    "bootstrap_replicates":
                        N_BOOTSTRAP,

                    "bootstrap_seed":
                        boot_seed,

                    "randomization_p_two_sided":
                        float(
                            pvalue
                        ),

                    "randomization_extreme_count":
                        int(
                            extreme_count
                        ),

                    "randomization_replicates":
                        N_RANDOMIZATION,

                    "randomization_seed":
                        rand_seed,
                })

                for i, value in enumerate(
                    boot
                ):

                    bootstrap_rows.append({
                        "family":
                            family,

                        "architecture":
                            architecture,

                        "task":
                            task,

                        "replicate":
                            i,

                        "rng_seed":
                            boot_seed,

                        "delta_macro_f1_A_minus_B":
                            float(
                                value
                            ),
                    })

                for i, value in enumerate(
                    null
                ):

                    randomization_rows.append({
                        "family":
                            family,

                        "architecture":
                            architecture,

                        "task":
                            task,

                        "replicate":
                            i,

                        "rng_seed":
                            rand_seed,

                        "null_delta_macro_f1_A_minus_B":
                            float(
                                value
                            ),
                    })

    if len(
        inference_rows
    ) != 12:
        raise RuntimeError(
            "Expected exactly 12 confirmatory contrasts."
        )

    for family in [
        "INCREMENTAL",
        "CONTEXT_SPECIFICITY",
    ]:

        indices = [
            i
            for i, row
            in enumerate(
                inference_rows
            )
            if row[
                "family"
            ] == family
        ]

        if len(
            indices
        ) != 6:
            raise RuntimeError(
                f"Expected exactly 6 p-values "
                f"in {family}; got {len(indices)}."
            )

        family_p = [
            inference_rows[
                i
            ][
                "randomization_p_two_sided"
            ]
            for i in indices
        ]

        family_q = bh_adjust6(
            family_p
        )

        for i, q in zip(
            indices,
            family_q,
        ):

            inference_rows[
                i
            ][
                "bh_q_within_family"
            ] = float(
                q
            )

    return (
        metrics_df,
        pd.DataFrame(
            inference_rows
        ),
        pd.DataFrame(
            bootstrap_rows
        ),
        pd.DataFrame(
            randomization_rows
        ),
    )


# ============================================================
# FROZEN ARCHITECTURE-SELECTION RULE
# ============================================================

def select_architecture(
    metrics_df,
    inference_df,
):

    eligibility = {}

    for architecture in ARCHITECTURES:

        eligible_tasks = []

        for task in TASKS:

            inc = inference_df[
                (
                    inference_df[
                        "family"
                    ] == "INCREMENTAL"
                )
                &
                (
                    inference_df[
                        "architecture"
                    ] == architecture
                )
                &
                (
                    inference_df[
                        "task"
                    ] == task
                )
            ]

            spec = inference_df[
                (
                    inference_df[
                        "family"
                    ] == "CONTEXT_SPECIFICITY"
                )
                &
                (
                    inference_df[
                        "architecture"
                    ] == architecture
                )
                &
                (
                    inference_df[
                        "task"
                    ] == task
                )
            ]

            if len(
                inc
            ) != 1 or len(
                spec
            ) != 1:
                raise RuntimeError(
                    "Selection contrast missing or duplicated."
                )

            inc = inc.iloc[
                0
            ]

            spec = spec.iloc[
                0
            ]

            criterion_a = (
                float(
                    inc[
                        "delta_macro_f1_A_minus_B"
                    ]
                )
                > 0.0
                and
                float(
                    inc[
                        "bh_q_within_family"
                    ]
                )
                < 0.05
            )

            criterion_b = (
                float(
                    spec[
                        "delta_macro_f1_A_minus_B"
                    ]
                )
                > 0.0
                and
                float(
                    spec[
                        "bh_q_within_family"
                    ]
                )
                < 0.05
            )

            if (
                criterion_a
                and criterion_b
            ):
                eligible_tasks.append(
                    task
                )

        eligibility[
            architecture
        ] = {
            "eligible":
                bool(
                    eligible_tasks
                ),

            "same_task_confirmatory_hits":
                eligible_tasks,
        }

    eligible_architectures = [
        architecture
        for architecture in ARCHITECTURES
        if eligibility[
            architecture
        ][
            "eligible"
        ]
    ]

    verified_means = {}

    for architecture in ARCHITECTURES:

        system = (
            f"{architecture}"
            "_VERIFIED_CONTEXT"
        )

        rows = metrics_df[
            metrics_df[
                "system"
            ] == system
        ]

        if set(
            rows[
                "task"
            ]
        ) != set(
            TASKS
        ):
            raise RuntimeError(
                f"Missing verified task metric "
                f"for {architecture}."
            )

        verified_means[
            architecture
        ] = float(
            rows[
                "macro_f1"
            ].mean()
        )

    if len(
        eligible_architectures
    ) == 0:

        selected = (
            "STAGE_A_NO_EVENT"
        )

        reason = (
            "NO_V4_ARCHITECTURE_ELIGIBLE"
        )

    elif len(
        eligible_architectures
    ) == 1:

        selected = (
            eligible_architectures[
                0
            ]
        )

        reason = (
            "EXACTLY_ONE_V4_ARCHITECTURE_ELIGIBLE"
        )

    elif len(
        eligible_architectures
    ) == 2:

        film = verified_means[
            "V4_FILM_R32"
        ]

        bilinear = verified_means[
            "V4_LR_BILINEAR_R32"
        ]

        if round(
            film,
            12,
        ) == round(
            bilinear,
            12,
        ):

            selected = (
                "V4_FILM_R32"
            )

            reason = (
                "BOTH_ELIGIBLE_MEAN_TIE_12DP_"
                "PRESPECIFIED_FILM_TIEBREAK"
            )

        elif film > bilinear:

            selected = (
                "V4_FILM_R32"
            )

            reason = (
                "BOTH_ELIGIBLE_FILM_LARGER_"
                "THREE_TASK_MEAN"
            )

        else:

            selected = (
                "V4_LR_BILINEAR_R32"
            )

            reason = (
                "BOTH_ELIGIBLE_BILINEAR_LARGER_"
                "THREE_TASK_MEAN"
            )

    else:
        raise RuntimeError(
            "Impossible eligibility state."
        )

    if selected == "STAGE_A_NO_EVENT":

        selected_system = (
            "STAGE_A_NO_EVENT"
        )

    else:

        selected_system = (
            f"{selected}"
            "_VERIFIED_CONTEXT"
        )

    return {
        "selected_architecture":
            selected,

        "selected_system":
            selected_system,

        "selection_reason":
            reason,

        "eligibility":
            eligibility,

        "verified_equal_weight_mean_macro_f1":
            verified_means,

        "tie_precision_decimal_places":
            12,

        "tie_precedence":
            [
                "V4_FILM_R32",
                "V4_LR_BILINEAR_R32",
            ],

        "eventgold_status":
            "SEALED_NOT_ACCESSED",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-check-only",
        action="store_true",
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.self_check_only
        and args.preflight_only
    ):
        raise RuntimeError(
            "Choose only one diagnostic mode."
        )

    verify_frozen_references()

    stageb = load_stageb()

    rng = load_rng()

    if args.self_check_only:

        print("=" * 88)
        print(
            "V4 ANALYSIS SOURCE SELF-CHECK OK"
        )
        print("=" * 88)
        print(
            "Frozen Stage-B inference imported : YES"
        )
        print(
            "Bootstrap implementation inherited : YES"
        )
        print(
            "Randomization implementation inherited : YES"
        )
        print(
            "BH family size                    : 6"
        )
        print(
            "Confirmatory families              : 2"
        )
        print(
            "Confirmatory contrasts             : 12"
        )
        print(
            "Prediction CSV content parsed      : NO"
        )
        print(
            "Performance metrics computed       : NO"
        )
        print(
            "EventGold remains SEALED_NOT_ACCESSED."
        )
        return

    raw, hash_audit = load_inputs(
        stageb
    )

    ensemble_map, schema_map = (
        build_all_ensembles(
            stageb,
            raw,
        )
    )

    if args.preflight_only:

        print("=" * 88)
        print(
            "V4 STATISTICAL ANALYSIS PREFLIGHT OK"
        )
        print("=" * 88)
        print(
            "Frozen raw inputs verified : 300 / 300"
        )
        print(
            "Systems                    : 5"
        )
        print(
            "Tasks                      : 3"
        )
        print(
            "OOF ensembles verified     : 15"
        )
        print(
            "OOF articles per ensemble  : 199"
        )
        print(
            "Seeds per article          : 5"
        )
        print(
            "Confirmatory families      : 2 x 6"
        )
        print(
            "Task-level performance metrics computed : NO"
        )
        print(
            "Bootstrap executed                   : NO"
        )
        print(
            "Randomization executed               : NO"
        )
        print(
            "EventGold remains SEALED_NOT_ACCESSED."
        )
        return

    if (
        OUT.exists()
        or TMP.exists()
    ):
        raise RuntimeError(
            "V4 confirmatory output already exists. "
            "Refusing overwrite."
        )

    TMP.mkdir(
        parents=True,
        exist_ok=False,
    )

    ensemble_rows = []

    for system in SYSTEMS:

        for task in TASKS:

            ensemble_rows.append(
                ensemble_map[
                    (
                        system,
                        task,
                    )
                ].copy()
            )

    ensemble_all = pd.concat(
        ensemble_rows,
        ignore_index=True,
        sort=False,
    )

    (
        metrics_df,
        inference_df,
        bootstrap_df,
        randomization_df,
    ) = run_confirmatory(
        stageb,
        rng,
        ensemble_map,
        schema_map,
    )

    selection = select_architecture(
        metrics_df,
        inference_df,
    )

    ensemble_all.to_csv(
        TMP
        / "v4_oof_ensemble_predictions_v4.csv",
        index=False,
    )

    metrics_df.to_csv(
        TMP
        / "task_metrics_v4.csv",
        index=False,
    )

    inference_df.to_csv(
        TMP
        / "paired_inference_v4.csv",
        index=False,
    )

    bootstrap_df.to_csv(
        TMP
        / "bootstrap_deltas_v4.csv",
        index=False,
    )

    randomization_df.to_csv(
        TMP
        / "randomization_null_deltas_v4.csv",
        index=False,
    )

    hash_audit.to_csv(
        TMP
        / "v4_analysis_input_hash_verification.csv",
        index=False,
    )

    (
        TMP
        / "architecture_selection_v4.json"
    ).write_text(
        json.dumps(
            selection,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "status":
            "V4_CONFIRMATORY_ANALYSIS_COMPLETE",

        "analysis_design": {
            "primary_metric":
                "five-seed probability-ensemble OOF Macro-F1",

            "inferential_unit":
                "article",

            "n_articles_per_task":
                N_ARTICLES,

            "training_seeds":
                SEEDS,

            "bootstrap_replicates":
                N_BOOTSTRAP,

            "randomization_replicates":
                N_RANDOMIZATION,

            "incremental_family_tests":
                6,

            "context_specificity_family_tests":
                6,

            "multiple_testing":
                (
                    "Benjamini-Hochberg separately "
                    "across 6 predeclared contrasts "
                    "within each of two families"
                ),
        },

        "frozen_inputs": {
            "protocol_sha256":
                sha256(
                    PROTOCOL
                ),

            "statistical_lock_sha256":
                sha256(
                    STAT_LOCK
                ),

            "rng_sha256":
                sha256(
                    RNG_PATH
                ),

            "input_manifest_sha256":
                sha256(
                    INPUT_MANIFEST
                ),

            "stage_b_analysis_sha256":
                sha256(
                    STAGEB_ANALYSIS
                ),

            "analysis_script_sha256":
                sha256(
                    Path(
                        __file__
                    )
                ),
        },

        "selection":
            selection,

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

            "bootstrap_can_override_selection":
                False,

            "descriptive_metrics_can_override_selection":
                False,

            "eventgold_status":
                "SEALED_NOT_ACCESSED",
        },
    }

    (
        TMP
        / "analysis_summary_v4.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

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
        / "V4_CONFIRMATORY_RESULTS_SHA256SUMS_v4.txt"
    )

    checksum_path.write_text(
        "".join(
            f"{sha256(p)}  {p.name}\n"
            for p in output_files
        ),
        encoding="utf-8",
    )

    TMP.rename(
        OUT
    )

    print("=" * 100)
    print(
        "V4 PREDECLARED CONFIRMATORY ANALYSIS COMPLETE"
    )
    print("=" * 100)

    report_columns = [
        "family",
        "architecture",
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

    print(
        inference_df[
            report_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        json.dumps(
            selection,
            indent=2,
            ensure_ascii=False,
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
