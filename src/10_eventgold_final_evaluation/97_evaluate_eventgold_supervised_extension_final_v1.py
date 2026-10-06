#!/usr/bin/env python3

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)


ROOT = Path.cwd().resolve()

FINAL = (
    ROOT
    / "01_event_aware_v3"
    / "10_eventgold_final_evaluation"
)

BENCH = (
    ROOT
    / "01_event_aware_v3"
    / "00_frozen_benchmark"
)

STAGE_A = (
    ROOT
    / "01_event_aware_v3"
    / "07_eve_frame_stage_a_no_event"
)

SOURCE_TRAINER = (
    STAGE_A
    / "37_train_eve_frame_stage_a_v3.py"
)

CHECKPOINT_MANIFEST = (
    FINAL
    / "FINAL_REFIT_CHECKPOINT_MANIFEST_v1.csv"
)

CHECKPOINT_CLOSURE = (
    FINAL
    / "FINAL_REFIT_CHECKPOINT_CLOSURE_v1.txt"
)

CHECKPOINT_FREEZE = (
    FINAL
    / "FROZEN_FINAL_REFIT_CHECKPOINTS_SHA256SUMS_v1.txt"
)

ORIGINAL_PROTOCOL = (
    FINAL
    / "EVENTGOLD_ONE_SHOT_EVALUATION_PROTOCOL_v1.txt"
)

RNG_FILE = (
    FINAL
    / "EVENTGOLD_BOOTSTRAP_RNG_v1.json"
)

ORIGINAL_PROTOCOL_FREEZE = (
    FINAL
    / "FROZEN_EVENTGOLD_EVALUATION_PROTOCOL_SHA256SUMS_v1.txt"
)

ORIGINAL_IMPLEMENTATION_LOCK = (
    FINAL
    / "EVENTGOLD_EVALUATION_IMPLEMENTATION_LOCK_v1.txt"
)

ORIGINAL_EXECUTION_LOCK = (
    FINAL
    / "EVENTGOLD_FINAL_EXECUTION_LOCK_v1.txt"
)

SUPERVISED_EXTENSION_PROTOCOL = (
    FINAL
    / "EventGold35_SUPERVISED_EXTENSION_PROTOCOL_v1.txt"
)

FINAL_EVAL_PROTOCOL = (
    FINAL
    / "EVENTGOLD35_SUPERVISED_EXTENSION_FINAL_EVALUATION_PROTOCOL_v1.txt"
)

FINAL_EVAL_PROTOCOL_FREEZE = (
    FINAL
    / "FROZEN_EVENTGOLD35_SUPERVISED_EXTENSION_FINAL_EVALUATION_PROTOCOL_SHA256SUMS_v1.txt"
)

FINAL_GOLD_DIR = (
    FINAL
    / "eventgold35_supervised_extension_v1"
)

EVENTGOLD = (
    FINAL_GOLD_DIR
    / "EventGold35_SUPERVISED_EXTENSION_FINAL_GOLD_v1.csv"
)

FINAL_GOLD_FREEZE = (
    FINAL
    / "FROZEN_EVENTGOLD35_SUPERVISED_EXTENSION_FINAL_GOLD_SHA256SUMS_v1.txt"
)

OUTDIR = (
    FINAL
    / "eventgold35_supervised_extension_final_results_v1"
)

TMPDIR = (
    FINAL
    / "eventgold35_supervised_extension_final_results_v1_tmp"
)


TASKS = [
    "primary_frame",
    "stance",
    "misinformation_relation",
]

LABEL_COLUMNS = {
    "primary_frame":
        "final_primary_frame_validated",

    "stance":
        "final_stance_validated",

    "misinformation_relation":
        "final_misinformation_relation_validated",
}

SEEDS = [
    11,
    29,
    47,
    83,
    131,
]


EXPECTED = {
    "stage_a_trainer":
        "62f5f55fcdeaf4e5c947313bcc326daa"
        "8ceb225e2ab954ffa7b24430e1e760b9",

    "checkpoint_manifest":
        "05d8cbfb7aba539d149af8925dda435de"
        "58cdfa4f5cd28e6d8b6344671da860a",

    "checkpoint_closure":
        "8a2a42f4bc7838670dfb039e1ddcf065"
        "edbfab9512614ea6307109ed47464e79",

    "checkpoint_freeze":
        "946413f086bdb70aca8e905d5e95e646"
        "f54fd91e28929a6fbe2e4dfd2fb17cb2",

    "original_protocol":
        "4346903b989a9fd0bca289f779c6378b"
        "75a141cc62f5253d16433ed1f8d63a2c",

    "rng":
        "4c15cd3357629e5a0e9fc48c08e80ec"
        "7dcbf2a81e4800ad5308b64ea1b82405d",

    "original_protocol_freeze":
        "3a41cd4c3e3bcafcd6d82003a39dcd9c"
        "1eda30faaf954a838723ac6eb8fd0607",

    "original_implementation_lock":
        "7ea030a134e38589be58216a6a868496"
        "b390443416d850deadaf37fabf56c7a3",

    "original_execution_lock":
        "6fb0ecd657aa069b093b00098863dfb6"
        "79252f352d7be673772f062241d761ff",

    "supervised_extension_protocol":
        "ff3d4f2361aa4fe6e435348a7b69b46"
        "bc9bca79245ffdc5d9e8da1c27ff46ff4",

    "final_gold_freeze":
        "9b067f2a5567fbcdbe4659a092272579"
        "3de2746cd3f9ea9698c6d50eb26231cd",

    "final_eval_protocol":
        "7dd54e5fe8ac6ccadeb21f882c446f0f"
        "d7e4860a38e06ecd34e57c5c6ca3c405",

    "final_eval_protocol_freeze":
        "c4dba0d887b0e04cf8c8e233d5be35"
        "eeab134ad71de19803900a2d16294dcf09",
}


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def assert_hash(
    path,
    expected,
    name,
):

    observed = sha256(path)

    if observed != expected:
        raise RuntimeError(
            f"{name} hash mismatch: "
            f"{observed} != {expected}"
        )


def verify_small_frozen_references():

    checks = [
        (
            SOURCE_TRAINER,
            EXPECTED["stage_a_trainer"],
            "Stage-A trainer",
        ),
        (
            CHECKPOINT_MANIFEST,
            EXPECTED["checkpoint_manifest"],
            "checkpoint manifest",
        ),
        (
            CHECKPOINT_CLOSURE,
            EXPECTED["checkpoint_closure"],
            "checkpoint closure",
        ),
        (
            CHECKPOINT_FREEZE,
            EXPECTED["checkpoint_freeze"],
            "checkpoint freeze",
        ),
        (
            ORIGINAL_PROTOCOL,
            EXPECTED["original_protocol"],
            "original EventGold protocol",
        ),
        (
            RNG_FILE,
            EXPECTED["rng"],
            "EventGold bootstrap RNG",
        ),
        (
            ORIGINAL_PROTOCOL_FREEZE,
            EXPECTED["original_protocol_freeze"],
            "original EventGold protocol freeze",
        ),
        (
            ORIGINAL_IMPLEMENTATION_LOCK,
            EXPECTED["original_implementation_lock"],
            "original implementation lock",
        ),
        (
            ORIGINAL_EXECUTION_LOCK,
            EXPECTED["original_execution_lock"],
            "original execution lock",
        ),
        (
            SUPERVISED_EXTENSION_PROTOCOL,
            EXPECTED["supervised_extension_protocol"],
            "supervised-extension protocol",
        ),
        (
            FINAL_GOLD_FREEZE,
            EXPECTED["final_gold_freeze"],
            "final-gold freeze manifest",
        ),
        (
            FINAL_EVAL_PROTOCOL,
            EXPECTED["final_eval_protocol"],
            "final-evaluation protocol",
        ),
        (
            FINAL_EVAL_PROTOCOL_FREEZE,
            EXPECTED["final_eval_protocol_freeze"],
            "final-evaluation protocol freeze",
        ),
    ]

    for path, expected, name in checks:

        assert_hash(
            path,
            expected,
            name,
        )


def load_stage_a():

    spec = (
        importlib.util
        .spec_from_file_location(
            "frozen_stage_a_v3",
            SOURCE_TRAINER,
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):
        raise RuntimeError(
            "Could not import frozen "
            "Stage-A trainer."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    checks = {
        "MODEL_NAME":
            "FacebookAI/xlm-roberta-base",
        "MAX_LENGTH":
            256,
        "STRIDE":
            32,
        "MAX_CHUNKS":
            8,
    }

    for k, expected in checks.items():

        observed = getattr(
            module,
            k,
        )

        if observed != expected:
            raise RuntimeError(
                f"Frozen Stage-A constant "
                f"mismatch: {k}"
            )

    if list(
        module.TASKS
    ) != TASKS:
        raise RuntimeError(
            "Frozen Stage-A task order "
            "mismatch."
        )

    return module


def load_rng():

    x = json.loads(
        RNG_FILE.read_text(
            encoding="utf-8"
        )
    )

    if (
        int(
            x[
                "bootstrap_replicates"
            ]
        )
        != 10000
    ):
        raise RuntimeError(
            "Unexpected bootstrap count."
        )

    expected_seeds = {
        "primary_frame":
            2026092701,
        "stance":
            2026092702,
        "misinformation_relation":
            2026092703,
    }

    if (
        x["seeds"]
        != expected_seeds
    ):
        raise RuntimeError(
            "Frozen bootstrap RNG "
            "mapping mismatch."
        )

    return x


def load_checkpoint_manifest():

    df = pd.read_csv(
        CHECKPOINT_MANIFEST
    )

    required = {
        "run_name",
        "task",
        "seed",
        "final_refit_epoch",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "summary_sha256",
        "label_maps_sha256",
        "document_chunking_sha256",
    }

    if (
        set(df.columns)
        != required
    ):
        raise RuntimeError(
            "Unexpected checkpoint "
            "manifest schema."
        )

    if len(df) != 15:
        raise RuntimeError(
            "Expected 15 checkpoints."
        )

    if (
        df[
            ["task", "seed"]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate task/seed checkpoint."
        )

    if set(
        df["task"]
    ) != set(TASKS):
        raise RuntimeError(
            "Checkpoint task set mismatch."
        )

    if set(
        df["seed"].astype(int)
    ) != set(SEEDS):
        raise RuntimeError(
            "Checkpoint seed set mismatch."
        )

    for task in TASKS:

        x = df[
            df["task"]
            == task
        ]

        if set(
            x["seed"].astype(int)
        ) != set(SEEDS):
            raise RuntimeError(
                f"Missing seed for task {task}"
            )

    return df


def expected_eventgold_sha():

    lines = (
        FINAL_GOLD_FREEZE
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    target = (
        "EventGold35_SUPERVISED_EXTENSION_FINAL_GOLD_v1.csv"
    )

    matches = []

    for raw in lines:

        line = raw.strip()

        if not line:
            continue

        parts = line.split(
            maxsplit=1
        )

        if len(parts) != 2:
            continue

        digest = parts[0].strip()

        name = (
            parts[1]
            .strip()
            .lstrip("*")
        )

        if Path(name).name == target:

            matches.append(
                digest
            )

    if len(matches) != 1:

        raise RuntimeError(
            "Could not uniquely resolve "
            "final-gold SHA256 from frozen "
            "final-gold manifest."
        )

    return matches[0]


def normalize_label_maps(
    maps,
):

    out = {}

    for task in TASKS:

        if task not in maps:
            raise RuntimeError(
                f"Checkpoint missing label map "
                f"for {task}"
            )

        out[task] = {
            int(k): str(v)
            for k, v
            in maps[
                task
            ].items()
        }

        expected_ids = list(
            range(
                len(
                    out[
                        task
                    ]
                )
            )
        )

        if sorted(
            out[
                task
            ].keys()
        ) != expected_ids:
            raise RuntimeError(
                f"Non-contiguous label IDs "
                f"for {task}"
            )

    return out


def validate_eventgold(
    df,
    frozen_maps,
):

    if len(df) != 35:
        raise RuntimeError(
            f"Expected 35 EventGold rows; "
            f"got {len(df)}"
        )

    required = [
        "doc_id",
        "dapt_text",
        *LABEL_COLUMNS.values(),
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing EventGold columns: "
            f"{missing}"
        )

    if df[
        "doc_id"
    ].isna().any():
        raise RuntimeError(
            "Missing EventGold doc_id."
        )

    df = df.copy()

    df["doc_id"] = (
        df[
            "doc_id"
        ]
        .astype(str)
    )

    if df[
        "doc_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate EventGold doc_id."
        )

    df["dapt_text"] = (
        df[
            "dapt_text"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if (
        df[
            "dapt_text"
        ]
        == ""
    ).any():
        raise RuntimeError(
            "Empty EventGold dapt_text."
        )

    gold_ids = {}

    for task in TASKS:

        col = (
            LABEL_COLUMNS[
                task
            ]
        )

        if df[
            col
        ].isna().any():
            raise RuntimeError(
                f"Missing EventGold labels "
                f"for {task}"
            )

        labels = (
            df[
                col
            ]
            .astype(str)
        )

        label_to_id = {
            label: idx
            for idx, label
            in frozen_maps[
                task
            ].items()
        }

        unknown = sorted(
            set(labels)
            - set(
                label_to_id
            )
        )

        if unknown:
            raise RuntimeError(
                f"EventGold contains labels "
                f"outside frozen map for "
                f"{task}: {unknown}"
            )

        gold_ids[task] = np.asarray(
            [
                label_to_id[x]
                for x in labels
            ],
            dtype=int,
        )

    return (
        df,
        gold_ids,
    )


def bootstrap_macro_f1(
    stagea,
    y_true,
    y_pred,
    label_ids,
    seed,
    n_boot=10000,
):

    rng = (
        np.random
        .default_rng(
            int(seed)
        )
    )

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    labels = list(
        label_ids
    )

    classes = sorted(
        np.unique(
            y_true
        ).tolist()
    )

    by_class = {
        c:
            np.flatnonzero(
                y_true == c
            )
        for c in classes
    }

    reps = np.empty(
        n_boot,
        dtype=float,
    )

    for b in range(
        n_boot
    ):

        pieces = []

        for c in classes:

            idx = (
                by_class[
                    c
                ]
            )

            sampled = rng.choice(
                idx,
                size=len(idx),
                replace=True,
            )

            pieces.append(
                sampled
            )

        take = np.concatenate(
            pieces
        )

        m = stagea.metric_dict(
            y_true[
                take
            ],
            y_pred[
                take
            ],
            labels,
        )

        reps[b] = float(
            m[
                "macro_f1"
            ]
        )

    ci = np.quantile(
        reps,
        [
            0.025,
            0.975,
        ],
        method="linear",
    )

    return (
        reps,
        float(ci[0]),
        float(ci[1]),
    )


def self_check():

    verify_small_frozen_references()

    load_rng()

    manifest = (
        load_checkpoint_manifest()
    )

    expected_gold_sha = (
        expected_eventgold_sha()
    )

    observed_gold_sha = sha256(
        EVENTGOLD
    )

    if observed_gold_sha != expected_gold_sha:

        raise RuntimeError(
            "Final-gold SHA256 mismatch "
            "during self-check."
        )

    if len(manifest) != 15:
        raise RuntimeError(
            "Manifest self-check failed."
        )

    if (
        OUTDIR.exists()
        or
        TMPDIR.exists()
    ):
        raise RuntimeError(
            "EventGold output directory "
            "already exists."
        )

    print(
        "=" * 80
    )

    print(
        "EVENTGOLD SUPERVISED-EXTENSION EVALUATOR "
        "SELF-CHECK OK"
    )

    print(
        "=" * 80
    )

    print(
        "Frozen checkpoint manifest : 15"
    )

    print(
        "Frozen tasks               : 3"
    )

    print(
        "Frozen seeds per task      : 5"
    )

    print(
        "Bootstrap replicates       : 10000"
    )

    print(
        "EventGold CSV parsed       : NO"
    )

    print(
        "EventGold performance      : NO"
    )

    print(
        "EVENTGOLD_STATUS="
        "SEALED_NOT_ACCESSED"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-check-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_check_only:

        self_check()

        return

    verify_small_frozen_references()

    rng_config = load_rng()

    checkpoint_manifest = (
        load_checkpoint_manifest()
    )

    if (
        OUTDIR.exists()
        or
        TMPDIR.exists()
    ):
        raise RuntimeError(
            "EventGold result directory "
            "already exists."
        )

    #
    # FIRST BYTE-LEVEL ACCESS TO EVENTGOLD CSV.
    # Hash verification occurs before CSV parsing.
    #
    expected_sha = (
        expected_eventgold_sha()
    )

    observed_sha = sha256(
        EVENTGOLD
    )

    if (
        observed_sha
        != expected_sha
    ):
        raise RuntimeError(
            "EventGold CSV SHA256 mismatch."
        )

    #
    # FIRST SEMANTIC ACCESS TO EVENTGOLD.
    #
    raw = pd.read_csv(
        EVENTGOLD,
        low_memory=False,
    )

    stagea = load_stage_a()

    stagea.configure_determinism()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for frozen "
            "EventGold inference."
        )

    device = torch.device(
        "cuda"
    )

    tokenizer = (
        stagea
        .AutoTokenizer
        .from_pretrained(
            stagea.MODEL_NAME,
            local_files_only=True,
            use_fast=True,
        )
    )

    #
    # Establish frozen label maps from first
    # checkpoint without using EventGold labels.
    #
    first_row = (
        checkpoint_manifest
        .iloc[0]
    )

    first_checkpoint = Path(
        first_row[
            "checkpoint_path"
        ]
    )

    if not first_checkpoint.is_absolute():
        first_checkpoint = (
            ROOT
            / first_checkpoint
        )

    if (
        sha256(
            first_checkpoint
        )
        != first_row[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "First checkpoint hash mismatch."
        )

    first_bundle = torch.load(
        first_checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    frozen_maps = (
        normalize_label_maps(
            first_bundle[
                "label_maps"
            ]
        )
    )

    frozen_n_classes = {
        task:
            int(
                first_bundle[
                    "n_classes"
                ][task]
            )
        for task in TASKS
    }

    del first_bundle

    (
        df,
        gold_ids,
    ) = validate_eventgold(
        raw,
        frozen_maps,
    )

    doc_ids = (
        df[
            "doc_id"
        ]
        .astype(str)
        .tolist()
    )

    token_store, chunking = (
        stagea
        .tokenize_documents(
            tokenizer,
            df,
            "dapt_text",
        )
    )

    TMPDIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    chunking.to_csv(
        TMPDIR
        / "eventgold35_supervised_extension_document_chunking_v1.csv",
        index=False,
    )

    (
        TMPDIR
        / "eventgold35_supervised_extension_label_maps_v1.json"
    ).write_text(
        json.dumps(
            {
                task: {
                    str(k): v
                    for k, v
                    in frozen_maps[
                        task
                    ].items()
                }
                for task in TASKS
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    seed_prediction_rows = []

    ensemble_rows = []

    metric_rows = []

    per_class_rows = []

    confusion_rows = []

    bootstrap_rows = []

    hash_audit_rows = [
        {
            "artifact_type":
                "EVENTGOLD_CSV",
            "name":
                "EventGold35_SUPERVISED_EXTENSION_FINAL_GOLD_v1.csv",
            "sha256":
                observed_sha,
            "status":
                "VERIFIED",
        }
    ]

    for task in TASKS:

        task_rows = (
            checkpoint_manifest[
                checkpoint_manifest[
                    "task"
                ]
                == task
            ]
            .copy()
        )

        task_rows[
            "seed"
        ] = (
            task_rows[
                "seed"
            ]
            .astype(int)
        )

        task_rows = (
            task_rows
            .sort_values(
                "seed"
            )
        )

        if (
            task_rows[
                "seed"
            ].tolist()
            != SEEDS
        ):
            raise RuntimeError(
                f"Seed order/set mismatch "
                f"for {task}"
            )

        seed_probs = []

        reference_maps = None
        reference_n_classes = None

        for r in task_rows.itertuples(
            index=False
        ):

            checkpoint = Path(
                r.checkpoint_path
            )

            if not checkpoint.is_absolute():
                checkpoint = (
                    ROOT
                    / checkpoint
                )

            observed_checkpoint_sha = (
                sha256(
                    checkpoint
                )
            )

            if (
                observed_checkpoint_sha
                != r.checkpoint_sha256
            ):
                raise RuntimeError(
                    f"Checkpoint hash mismatch: "
                    f"{r.run_name}"
                )

            hash_audit_rows.append({
                "artifact_type":
                    "CHECKPOINT",
                "name":
                    str(r.run_name),
                "sha256":
                    observed_checkpoint_sha,
                "status":
                    "VERIFIED",
            })

            bundle = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )

            if (
                bundle[
                    "architecture"
                ]
                != "EVE_FRAME_STAGE_A_NO_EVENT"
            ):
                raise RuntimeError(
                    "Checkpoint architecture "
                    "mismatch."
                )

            if (
                bundle[
                    "stage"
                ]
                != "FINAL_REFIT_PRE_EVENTGOLD"
            ):
                raise RuntimeError(
                    "Checkpoint stage mismatch."
                )

            if (
                bundle[
                    "target_task"
                ]
                != task
            ):
                raise RuntimeError(
                    "Checkpoint target-task "
                    "mismatch."
                )

            if (
                int(
                    bundle[
                        "seed"
                    ]
                )
                != int(
                    r.seed
                )
            ):
                raise RuntimeError(
                    "Checkpoint seed mismatch."
                )

            maps = (
                normalize_label_maps(
                    bundle[
                        "label_maps"
                    ]
                )
            )

            n_classes = {
                t:
                    int(
                        bundle[
                            "n_classes"
                        ][t]
                    )
                for t in TASKS
            }

            if reference_maps is None:

                reference_maps = maps

                reference_n_classes = (
                    n_classes
                )

            else:

                if maps != reference_maps:
                    raise RuntimeError(
                        "Label-map mismatch "
                        "across seeds."
                    )

                if (
                    n_classes
                    != reference_n_classes
                ):
                    raise RuntimeError(
                        "n_classes mismatch "
                        "across seeds."
                    )

            if maps != frozen_maps:
                raise RuntimeError(
                    "Checkpoint label maps "
                    "differ from frozen reference."
                )

            if (
                n_classes
                != frozen_n_classes
            ):
                raise RuntimeError(
                    "Checkpoint n_classes "
                    "differ from frozen reference."
                )

            model = (
                stagea
                .initialize_model(
                    int(
                        r.seed
                    ),
                    n_classes,
                    device,
                )
            )

            missing, unexpected = (
                model.load_state_dict(
                    bundle[
                        "state_dict"
                    ],
                    strict=True,
                )
            )

            if missing or unexpected:
                raise RuntimeError(
                    "Unexpected state_dict "
                    "load result."
                )

            model.eval()

            probs_for_seed = []

            with torch.inference_mode():

                for doc_id in doc_ids:

                    logits = (
                        stagea
                        .forward_document(
                            model,
                            tokenizer,
                            token_store[
                                doc_id
                            ],
                            device,
                        )
                    )

                    p = (
                        torch.softmax(
                            logits[
                                task
                            ].float(),
                            dim=-1,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(float)
                    )

                    if (
                        p.shape
                        != (
                            n_classes[
                                task
                            ],
                        )
                    ):
                        raise RuntimeError(
                            "Probability shape "
                            "mismatch."
                        )

                    if not np.all(
                        np.isfinite(
                            p
                        )
                    ):
                        raise RuntimeError(
                            "Non-finite "
                            "probability."
                        )

                    if not np.isclose(
                        p.sum(),
                        1.0,
                        atol=1e-6,
                    ):
                        raise RuntimeError(
                            "Probability sum "
                            "mismatch."
                        )

                    probs_for_seed.append(
                        p
                    )

            probs_for_seed = np.vstack(
                probs_for_seed
            )

            seed_probs.append(
                probs_for_seed
            )

            pred_ids = (
                probs_for_seed
                .argmax(
                    axis=1
                )
            )

            for i, doc_id in enumerate(
                doc_ids
            ):

                row = {
                    "doc_id":
                        doc_id,
                    "task":
                        task,
                    "seed":
                        int(r.seed),
                    "gold_id":
                        int(
                            gold_ids[
                                task
                            ][i]
                        ),
                    "gold_label":
                        frozen_maps[
                            task
                        ][
                            int(
                                gold_ids[
                                    task
                                ][i]
                            )
                        ],
                    "prediction_id":
                        int(
                            pred_ids[
                                i
                            ]
                        ),
                    "prediction_label":
                        frozen_maps[
                            task
                        ][
                            int(
                                pred_ids[
                                    i
                                ]
                            )
                        ],
                }

                for class_id in range(
                    n_classes[
                        task
                    ]
                ):

                    row[
                        f"prob_{class_id}"
                    ] = float(
                        probs_for_seed[
                            i,
                            class_id,
                        ]
                    )

                seed_prediction_rows.append(
                    row
                )

            del model
            del bundle

            gc.collect()
            torch.cuda.empty_cache()

        stack = np.stack(
            seed_probs,
            axis=0,
        )

        if (
            stack.shape[0]
            != 5
        ):
            raise RuntimeError(
                "Expected five seed "
                "probability matrices."
            )

        ensemble_prob = (
            stack.mean(
                axis=0
            )
        )

        if not np.allclose(
            ensemble_prob.sum(
                axis=1
            ),
            1.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Ensemble probability "
                "normalization failure."
            )

        ensemble_pred = (
            ensemble_prob
            .argmax(
                axis=1
            )
        )

        for i, doc_id in enumerate(
            doc_ids
        ):

            row = {
                "doc_id":
                    doc_id,
                "task":
                    task,
                "gold_id":
                    int(
                        gold_ids[
                            task
                        ][i]
                    ),
                "gold_label":
                    frozen_maps[
                        task
                    ][
                        int(
                            gold_ids[
                                task
                            ][i]
                        )
                    ],
                "prediction_id":
                    int(
                        ensemble_pred[
                            i
                        ]
                    ),
                "prediction_label":
                    frozen_maps[
                        task
                    ][
                        int(
                            ensemble_pred[
                                i
                            ]
                        )
                    ],
            }

            for class_id in range(
                frozen_n_classes[
                    task
                ]
            ):

                row[
                    f"prob_{class_id}"
                ] = float(
                    ensemble_prob[
                        i,
                        class_id,
                    ]
                )

            ensemble_rows.append(
                row
            )

        labels = list(
            range(
                frozen_n_classes[
                    task
                ]
            )
        )

        metrics = (
            stagea.metric_dict(
                gold_ids[
                    task
                ],
                ensemble_pred,
                labels,
            )
        )

        (
            class_precision,
            class_recall,
            class_f1,
            class_support,
        ) = precision_recall_fscore_support(
            gold_ids[
                task
            ],
            ensemble_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        for class_id in labels:

            per_class_rows.append({
                "task":
                    task,
                "class_id":
                    int(class_id),
                "class_label":
                    frozen_maps[
                        task
                    ][
                        int(class_id)
                    ],
                "support":
                    int(
                        class_support[
                            class_id
                        ]
                    ),
                "precision":
                    float(
                        class_precision[
                            class_id
                        ]
                    ),
                "recall":
                    float(
                        class_recall[
                            class_id
                        ]
                    ),
                "f1":
                    float(
                        class_f1[
                            class_id
                        ]
                    ),
            })

        cm = confusion_matrix(
            gold_ids[
                task
            ],
            ensemble_pred,
            labels=labels,
        )

        for true_id in labels:

            for pred_id in labels:

                confusion_rows.append({
                    "task":
                        task,
                    "true_class_id":
                        int(true_id),
                    "true_class_label":
                        frozen_maps[
                            task
                        ][
                            int(true_id)
                        ],
                    "predicted_class_id":
                        int(pred_id),
                    "predicted_class_label":
                        frozen_maps[
                            task
                        ][
                            int(pred_id)
                        ],
                    "count":
                        int(
                            cm[
                                true_id,
                                pred_id,
                            ]
                        ),
                })

        (
            reps,
            ci_low,
            ci_high,
        ) = bootstrap_macro_f1(
            stagea,
            gold_ids[
                task
            ],
            ensemble_pred,
            labels,
            rng_config[
                "seeds"
            ][task],
            n_boot=10000,
        )

        metric_rows.append({
            "task":
                task,
            "n_articles":
                35,
            "n_classes":
                frozen_n_classes[
                    task
                ],
            "macro_f1":
                float(
                    metrics[
                        "macro_f1"
                    ]
                ),
            "macro_f1_ci95_low":
                ci_low,
            "macro_f1_ci95_high":
                ci_high,
            "balanced_accuracy":
                float(
                    metrics[
                        "balanced_accuracy"
                    ]
                ),
            "accuracy":
                float(
                    metrics[
                        "accuracy"
                    ]
                ),
            "weighted_f1":
                float(
                    metrics[
                        "weighted_f1"
                    ]
                ),
            "bootstrap_replicates":
                10000,
            "bootstrap_rng_seed":
                int(
                    rng_config[
                        "seeds"
                    ][task]
                ),
        })

        for b, value in enumerate(
            reps
        ):

            bootstrap_rows.append({
                "task":
                    task,
                "replicate":
                    b,
                "macro_f1":
                    float(
                        value
                    ),
            })

    seed_df = pd.DataFrame(
        seed_prediction_rows
    )

    ensemble_df = pd.DataFrame(
        ensemble_rows
    )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    confusion_df = pd.DataFrame(
        confusion_rows
    )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    hash_df = pd.DataFrame(
        hash_audit_rows
    )

    if len(seed_df) != (
        35
        * 3
        * 5
    ):
        raise RuntimeError(
            "Unexpected seed-prediction "
            "row count."
        )

    if len(
        ensemble_df
    ) != (
        35
        * 3
    ):
        raise RuntimeError(
            "Unexpected ensemble "
            "prediction row count."
        )

    if len(
        metrics_df
    ) != 3:
        raise RuntimeError(
            "Unexpected metrics row count."
        )

    if len(
        per_class_df
    ) != 18:

        raise RuntimeError(
            "Unexpected per-class metric "
            "row count."
        )

    if len(
        confusion_df
    ) != 122:

        raise RuntimeError(
            "Unexpected confusion-matrix "
            "row count."
        )

    if len(
        bootstrap_df
    ) != 30000:
        raise RuntimeError(
            "Unexpected bootstrap "
            "row count."
        )

    seed_df.to_csv(
        TMPDIR
        / "eventgold35_supervised_extension_seed_predictions_v1.csv",
        index=False,
    )

    ensemble_df.to_csv(
        TMPDIR
        / "eventgold35_supervised_extension_ensemble_predictions_v1.csv",
        index=False,
    )

    metrics_df.to_csv(
        TMPDIR
        / "eventgold35_supervised_extension_metrics_v1.csv",
        index=False,
    )

    per_class_df.to_csv(
        TMPDIR
        / "eventgold35_supervised_extension_per_class_metrics_v1.csv",
        index=False,
    )

    confusion_df.to_csv(
        TMPDIR
        / "eventgold35_supervised_extension_confusion_matrix_long_v1.csv",
        index=False,
    )

    bootstrap_df.to_csv(
        TMPDIR
        / "eventgold35_supervised_extension_bootstrap_macro_f1_v1.csv",
        index=False,
    )

    hash_df.to_csv(
        TMPDIR
        / "eventgold35_supervised_extension_input_hash_audit_v1.csv",
        index=False,
    )

    summary = {
        "status":
            "COMPLETE",

        "evaluation":
            "EVENTGOLD35_SUPERVISED_EXTENSION_FINAL",

        "architecture":
            "STAGE_A_NO_EVENT",

        "n_articles":
            35,

        "tasks":
            TASKS,

        "seeds":
            SEEDS,

        "ensemble":
            "equal_mean_probability_5_seeds_per_task",

        "primary_metric":
            "macro_f1",

        "secondary_metrics": [
            "balanced_accuracy",
            "accuracy",
            "weighted_f1",
        ],

        "additional_descriptive_outputs": [
            "per_class_precision_recall_f1_support",
            "full_label_universe_confusion_matrix",
        ],

        "bootstrap":
            {
                "replicates":
                    10000,
                "stratified_by":
                    "gold_label_within_task",
                "ci":
                    "percentile_2.5_97.5",
                "quantile_method":
                    "linear",
                "rng_seeds":
                    rng_config[
                        "seeds"
                    ],
            },

        "hypothesis_testing":
            "NONE",

        "final_gold_sha256":
            observed_sha,

        "checkpoint_manifest_sha256":
            EXPECTED[
                "checkpoint_manifest"
            ],

        "checkpoint_freeze_sha256":
            EXPECTED[
                "checkpoint_freeze"
            ],

        "evaluation_protocol_sha256":
            EXPECTED[
                "final_eval_protocol"
            ],

        "evaluation_rng_sha256":
            EXPECTED[
                "rng"
            ],

        "final_gold_status":
            "ACCESSED_SUPERVISED_EXTENSION_FINAL",
    }

    (
        TMPDIR
        / "eventgold35_supervised_extension_summary_v1.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result_files = [
        "eventgold35_supervised_extension_seed_predictions_v1.csv",
        "eventgold35_supervised_extension_ensemble_predictions_v1.csv",
        "eventgold35_supervised_extension_metrics_v1.csv",
        "eventgold35_supervised_extension_per_class_metrics_v1.csv",
        "eventgold35_supervised_extension_confusion_matrix_long_v1.csv",
        "eventgold35_supervised_extension_bootstrap_macro_f1_v1.csv",
        "eventgold35_supervised_extension_input_hash_audit_v1.csv",
        "eventgold35_supervised_extension_label_maps_v1.json",
        "eventgold35_supervised_extension_document_chunking_v1.csv",
        "eventgold35_supervised_extension_summary_v1.json",
    ]

    checksum_path = (
        TMPDIR
        / "EVENTGOLD35_SUPERVISED_EXTENSION_RESULTS_SHA256SUMS_v1.txt"
    )

    with checksum_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for name in result_files:

            p = (
                TMPDIR
                / name
            )

            f.write(
                f"{sha256(p)}  "
                f"{name}\n"
            )

    TMPDIR.rename(
        OUTDIR
    )

    print(
        "=" * 88
    )

    print(
        "EVENTGOLD-35 SUPERVISED EXTENSION "
        "FINAL EVALUATION COMPLETE"
    )

    print(
        "=" * 88
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print()

    print(
        "SEED_PREDICTION_ROWS="
        + str(
            len(
                seed_df
            )
        )
    )

    print(
        "ENSEMBLE_PREDICTION_ROWS="
        + str(
            len(
                ensemble_df
            )
        )
    )

    print(
        "BOOTSTRAP_ROWS="
        + str(
            len(
                bootstrap_df
            )
        )
    )

    print(
        "EVENTGOLD_STATUS="
        "ACCESSED_SUPERVISED_EXTENSION_FINAL"
    )


if __name__ == "__main__":
    main()
