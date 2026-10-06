#!/usr/bin/env python3

import os

# Required before CUDA deterministic GEMM execution.
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
import shutil
import time
from pathlib import Path

import pandas as pd
import torch


ROOT = Path.cwd().resolve()

A_ROOT = (
    ROOT
    / "01_event_aware_v3"
    / "07_eve_frame_stage_a_no_event"
)

FINAL_ROOT = (
    ROOT
    / "01_event_aware_v3"
    / "10_eventgold_final_evaluation"
)

SOURCE_TRAINER = (
    A_ROOT
    / "37_train_eve_frame_stage_a_v3.py"
)

DATA_PATH = (
    ROOT
    / "01_event_aware_v3"
    / "03_scalegold_active_learning"
    / "LegacyAux_leakage_free_v3.csv"
)

EPOCH_TABLE = (
    FINAL_ROOT
    / "FINAL_REFIT_EPOCHS_v1.csv"
)

EPOCH_RULE = (
    FINAL_ROOT
    / "FINAL_REFIT_EPOCH_RULE_v1.txt"
)

RUNROOT = (
    FINAL_ROOT
    / "final_refit_runs_v1"
)


EXPECTED_SOURCE_SHA = (
    "62f5f55fcdeaf4e5c947313bcc326daa"
    "8ceb225e2ab954ffa7b24430e1e760b9"
)

EXPECTED_EPOCH_TABLE_SHA = (
    "885c9f2ddb9c50ffc0c224ab037184b42"
    "d69422e7d34cde6b9a376c2b35e247e"
)

EXPECTED_EPOCH_RULE_SHA = (
    "edfe2007ee176921a6b0c94fd08d4887"
    "8820d5bd9707c8e3ff29840573e2a168"
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


def file_sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def verify_frozen_inputs():

    if file_sha256(SOURCE_TRAINER) != EXPECTED_SOURCE_SHA:
        raise RuntimeError(
            "Frozen Stage-A trainer hash mismatch."
        )

    if file_sha256(EPOCH_TABLE) != EXPECTED_EPOCH_TABLE_SHA:
        raise RuntimeError(
            "Frozen final-refit epoch table hash mismatch."
        )

    if file_sha256(EPOCH_RULE) != EXPECTED_EPOCH_RULE_SHA:
        raise RuntimeError(
            "Frozen final-refit epoch rule hash mismatch."
        )

    if not DATA_PATH.is_file():
        raise RuntimeError(
            f"Missing LegacyAux data: {DATA_PATH}"
        )


def load_frozen_stage_a():

    spec = importlib.util.spec_from_file_location(
        "frozen_stage_a_v3",
        SOURCE_TRAINER,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load frozen Stage-A trainer."
        )

    module = importlib.util.module_from_spec(
        spec
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
        "DROPOUT":
            0.10,
        "LR":
            1e-5,
        "WEIGHT_DECAY":
            0.01,
        "WARMUP_FRACTION":
            0.10,
        "GRAD_CLIP":
            1.0,
        "GRAD_ACCUM":
            8,
        "MAX_SELECTION_EPOCHS":
            8,
    }

    for name, expected in checks.items():

        observed = getattr(
            module,
            name,
        )

        if observed != expected:
            raise RuntimeError(
                f"Frozen Stage-A constant mismatch: "
                f"{name}: "
                f"observed={observed!r}, "
                f"expected={expected!r}"
            )

    if list(module.TASKS) != TASKS:
        raise RuntimeError(
            "Frozen Stage-A task order mismatch."
        )

    return module


def load_epoch_table():

    df = pd.read_csv(
        EPOCH_TABLE
    )

    required = [
        "task",
        "seed",
        "fold1_selected_epoch",
        "fold2_selected_epoch",
        "fold3_selected_epoch",
        "fold4_selected_epoch",
        "median_selected_epoch",
        "final_refit_epoch",
        "scheduler_horizon_epochs",
    ]

    if list(df.columns) != required:
        raise RuntimeError(
            "Unexpected final-refit epoch table schema."
        )

    if len(df) != 15:
        raise RuntimeError(
            f"Expected 15 epoch configurations; got {len(df)}"
        )

    if df[
        ["task", "seed"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate task/seed in epoch table."
        )

    if set(df["task"]) != set(TASKS):
        raise RuntimeError(
            "Unexpected task set in epoch table."
        )

    if set(
        df["seed"].astype(int)
    ) != set(SEEDS):
        raise RuntimeError(
            "Unexpected seed set in epoch table."
        )

    if not (
        df["final_refit_epoch"]
        .astype(int)
        .between(1, 8)
        .all()
    ):
        raise RuntimeError(
            "Invalid final-refit epoch."
        )

    if not (
        df["scheduler_horizon_epochs"]
        .astype(int)
        == 8
    ).all():
        raise RuntimeError(
            "Scheduler horizon mismatch."
        )

    return df


def get_config(
    epoch_df,
    target_task,
    seed,
):

    x = epoch_df[
        (epoch_df["task"] == target_task)
        &
        (
            epoch_df["seed"].astype(int)
            == int(seed)
        )
    ]

    if len(x) != 1:
        raise RuntimeError(
            "Expected exactly one task/seed "
            "epoch configuration."
        )

    row = x.iloc[0]

    return {
        "target_task":
            target_task,
        "seed":
            int(seed),
        "final_refit_epoch":
            int(
                row[
                    "final_refit_epoch"
                ]
            ),
        "scheduler_horizon_epochs":
            int(
                row[
                    "scheduler_horizon_epochs"
                ]
            ),
        "source_fold_epochs": [
            int(
                row[
                    f"fold{i}_selected_epoch"
                ]
            )
            for i in [1, 2, 3, 4]
        ],
        "median_selected_epoch":
            float(
                row[
                    "median_selected_epoch"
                ]
            ),
    }


def prepare_legacyaux(
    stagea,
):

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    if len(df) != 199:
        raise RuntimeError(
            f"Expected 199 LegacyAux rows; got {len(df)}"
        )

    required = [
        "doc_id",
        "dapt_text",
        *stagea.LABEL_COLUMNS.values(),
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing LegacyAux columns: {missing}"
        )

    if df["doc_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate doc_id in LegacyAux."
        )

    df["doc_id"] = (
        df["doc_id"]
        .astype(str)
    )

    df["dapt_text"] = (
        df["dapt_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if (
        df["dapt_text"] == ""
    ).any():
        raise RuntimeError(
            "Empty dapt_text detected."
        )

    label_maps = {}
    label_to_id = {}
    label_ids = {}
    n_classes = {}

    for task in TASKS:

        col = (
            stagea
            .LABEL_COLUMNS[
                task
            ]
        )

        if df[col].isna().any():
            raise RuntimeError(
                f"Missing labels for {task}"
            )

        values = (
            df[col]
            .astype(str)
        )

        classes = sorted(
            values.unique()
        )

        label_maps[task] = {
            i: label
            for i, label
            in enumerate(classes)
        }

        label_to_id[task] = {
            label: i
            for i, label
            in label_maps[
                task
            ].items()
        }

        label_ids[task] = {
            str(doc_id): int(
                label_to_id[
                    task
                ][label]
            )
            for doc_id, label
            in zip(
                df["doc_id"],
                values,
            )
        }

        n_classes[task] = len(
            classes
        )

    doc_ids = (
        df["doc_id"]
        .astype(str)
        .tolist()
    )

    if len(doc_ids) != 199:
        raise RuntimeError(
            "Unexpected document count."
        )

    if len(set(doc_ids)) != 199:
        raise RuntimeError(
            "LegacyAux doc IDs are not unique."
        )

    return (
        df,
        doc_ids,
        label_maps,
        label_ids,
        n_classes,
    )


def validate_existing_run(
    outdir,
    config,
):

    summary_path = (
        outdir
        / "run_summary_final_refit_v1.json"
    )

    checkpoint_path = (
        outdir
        / "model_state_dict_v1.pt"
    )

    if (
        not summary_path.is_file()
        or
        not checkpoint_path.is_file()
    ):
        return False

    try:
        x = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    required = {
        "status":
            "COMPLETE",
        "architecture":
            "EVE_FRAME_STAGE_A_NO_EVENT",
        "stage":
            "FINAL_REFIT_PRE_EVENTGOLD",
        "target_task":
            config[
                "target_task"
            ],
        "seed":
            config[
                "seed"
            ],
        "final_refit_epoch":
            config[
                "final_refit_epoch"
            ],
        "scheduler_horizon_epochs":
            8,
        "n_train_documents":
            199,
        "source_stage_a_trainer_sha256":
            EXPECTED_SOURCE_SHA,
        "epoch_table_sha256":
            EXPECTED_EPOCH_TABLE_SHA,
        "epoch_rule_sha256":
            EXPECTED_EPOCH_RULE_SHA,
        "eventgold_status":
            "SEALED_NOT_ACCESSED",
    }

    for k, expected in required.items():
        if x.get(k) != expected:
            return False

    if (
        x.get("checkpoint_sha256")
        != file_sha256(
            checkpoint_path
        )
    ):
        return False

    if (
        x.get(
            "final_refit_trainer_sha256"
        )
        != file_sha256(
            Path(__file__)
        )
    ):
        return False

    if (
        x.get("data_sha256")
        != file_sha256(
            DATA_PATH
        )
    ):
        return False

    return True


def run_preflight(
    stagea,
    epoch_df,
    target_task,
    seed,
):

    config = get_config(
        epoch_df,
        target_task,
        seed,
    )

    (
        df,
        doc_ids,
        label_maps,
        label_ids,
        n_classes,
    ) = prepare_legacyaux(
        stagea
    )

    del df
    del label_maps
    del label_ids

    print(
        "FINAL REFIT PREFLIGHT OK | "
        f"task={target_task} | "
        f"seed={seed} | "
        f"epoch={config['final_refit_epoch']} | "
        f"scheduler=8 | "
        f"n={len(doc_ids)} | "
        f"classes={n_classes}"
    )

    print(
        "MODEL_TRAINING_EXECUTED=NO"
    )

    print(
        "EVENTGOLD_STATUS=SEALED_NOT_ACCESSED"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-task",
        required=True,
        choices=TASKS,
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=SEEDS,
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    parser.add_argument(
        "--skip-if-valid",
        action="store_true",
    )

    args = parser.parse_args()

    verify_frozen_inputs()

    stagea = load_frozen_stage_a()

    epoch_df = load_epoch_table()

    if args.preflight_only:

        run_preflight(
            stagea,
            epoch_df,
            args.target_task,
            args.seed,
        )

        return

    config = get_config(
        epoch_df,
        args.target_task,
        args.seed,
    )

    run_name = (
        f"{args.target_task}"
        f"__seed{args.seed}"
    )

    outdir = (
        RUNROOT
        / run_name
    )

    tmpdir = (
        RUNROOT
        / f"{run_name}__tmp"
    )

    if outdir.exists():

        if (
            args.skip_if_valid
            and
            validate_existing_run(
                outdir,
                config,
            )
        ):

            print(
                "VALID EXISTING FINAL REFIT "
                "— SKIP:",
                run_name,
            )

            return

        raise RuntimeError(
            f"Final run directory already exists: "
            f"{outdir}"
        )

    if tmpdir.exists():
        raise RuntimeError(
            f"Temporary final run directory "
            f"already exists: {tmpdir}"
        )

    RUNROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmpdir.mkdir()

    start = time.time()

    try:

        stagea.configure_determinism()

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for frozen "
                "Stage-A final refitting."
            )

        torch.cuda.reset_peak_memory_stats()

        device = torch.device(
            "cuda"
        )

        (
            df,
            doc_ids,
            label_maps,
            label_ids,
            n_classes,
        ) = prepare_legacyaux(
            stagea
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

        token_store, chunking = (
            stagea.tokenize_documents(
                tokenizer,
                df,
                "dapt_text",
            )
        )

        chunking_path = (
            tmpdir
            / "document_chunking_v1.csv"
        )

        chunking.to_csv(
            chunking_path,
            index=False,
        )

        label_maps_path = (
            tmpdir
            / "label_maps_v1.json"
        )

        label_maps_path.write_text(
            json.dumps(
                {
                    task: {
                        str(k): v
                        for k, v
                        in label_maps[
                            task
                        ].items()
                    }
                    for task
                    in TASKS
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        class_weights = {}

        for task in TASKS:

            (
                class_weights[
                    task
                ],
                _,
            ) = (
                stagea
                .make_class_weights(
                    doc_ids,
                    label_ids[
                        task
                    ],
                    n_classes[
                        task
                    ],
                    device,
                )
            )

        model = (
            stagea
            .initialize_model(
                args.seed,
                n_classes,
                device,
            )
        )

        stagea.train_epochs(
            model,
            tokenizer,
            token_store,
            doc_ids,
            label_ids,
            class_weights,
            device,
            epochs=
                config[
                    "final_refit_epoch"
                ],
            scheduler_epochs=8,
            seed=args.seed,
        )

        checkpoint_path = (
            tmpdir
            / "model_state_dict_v1.pt"
        )

        state_dict_cpu = {
            k:
                v.detach().cpu()
            for k, v
            in model.state_dict().items()
        }

        checkpoint_bundle = {
            "architecture":
                "EVE_FRAME_STAGE_A_NO_EVENT",
            "stage":
                "FINAL_REFIT_PRE_EVENTGOLD",
            "target_task":
                args.target_task,
            "seed":
                int(args.seed),
            "final_refit_epoch":
                int(
                    config[
                        "final_refit_epoch"
                    ]
                ),
            "scheduler_horizon_epochs":
                8,
            "model_name":
                stagea.MODEL_NAME,
            "tasks":
                TASKS,
            "n_classes":
                {
                    k: int(v)
                    for k, v
                    in n_classes.items()
                },
            "label_maps":
                label_maps,
            "state_dict":
                state_dict_cpu,
        }

        torch.save(
            checkpoint_bundle,
            checkpoint_path,
        )

        elapsed = (
            time.time()
            - start
        )

        peak_gb = (
            torch.cuda
            .max_memory_allocated()
            / (1024 ** 3)
        )

        summary = {
            "status":
                "COMPLETE",

            "architecture":
                "EVE_FRAME_STAGE_A_NO_EVENT",

            "stage":
                "FINAL_REFIT_PRE_EVENTGOLD",

            "target_task":
                args.target_task,

            "seed":
                int(args.seed),

            "n_train_documents":
                199,

            "source_fold_epochs":
                config[
                    "source_fold_epochs"
                ],

            "median_selected_epoch":
                config[
                    "median_selected_epoch"
                ],

            "final_refit_epoch":
                int(
                    config[
                        "final_refit_epoch"
                    ]
                ),

            "scheduler_horizon_epochs":
                8,

            "joint_loss":
                "equal_mean_three_class_weighted_CE",

            "model_name":
                stagea.MODEL_NAME,

            "max_length":
                256,

            "stride":
                32,

            "max_chunks":
                8,

            "dropout":
                0.10,

            "learning_rate":
                1e-5,

            "weight_decay":
                0.01,

            "warmup_fraction":
                0.10,

            "gradient_clip":
                1.0,

            "gradient_accumulation":
                8,

            "strict_deterministic_algorithms":
                True,

            "attention_implementation":
                "eager",

            "bf16":
                True,

            "data_sha256":
                file_sha256(
                    DATA_PATH
                ),

            "source_stage_a_trainer_sha256":
                EXPECTED_SOURCE_SHA,

            "epoch_table_sha256":
                EXPECTED_EPOCH_TABLE_SHA,

            "epoch_rule_sha256":
                EXPECTED_EPOCH_RULE_SHA,

            "final_refit_trainer_sha256":
                file_sha256(
                    Path(__file__)
                ),

            "document_chunking_sha256":
                file_sha256(
                    chunking_path
                ),

            "label_maps_sha256":
                file_sha256(
                    label_maps_path
                ),

            "checkpoint_sha256":
                file_sha256(
                    checkpoint_path
                ),

            "elapsed_seconds":
                float(
                    elapsed
                ),

            "peak_gpu_memory_gb":
                float(
                    peak_gb
                ),

            "eventgold_status":
                "SEALED_NOT_ACCESSED",
        }

        summary_path = (
            tmpdir
            / "run_summary_final_refit_v1.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        del model
        del state_dict_cpu
        del checkpoint_bundle

        gc.collect()
        torch.cuda.empty_cache()

        tmpdir.rename(
            outdir
        )

        print("=" * 88)
        print(
            "FINAL STAGE-A REFIT COMPLETE"
        )
        print("=" * 88)

        print(
            "Run:",
            run_name,
        )

        print(
            "Target task:",
            args.target_task,
        )

        print(
            "Seed:",
            args.seed,
        )

        print(
            "Epochs:",
            config[
                "final_refit_epoch"
            ],
        )

        print(
            "Training documents:",
            199,
        )

        print(
            "Checkpoint:",
            outdir
            / "model_state_dict_v1.pt",
        )

        print(
            "EventGold:",
            "SEALED_NOT_ACCESSED",
        )

    except Exception:

        # Preserve failed partial work for diagnosis.
        # Never silently overwrite or delete it.
        raise


if __name__ == "__main__":
    main()
