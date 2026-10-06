from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


ROOT = Path.cwd()

DATA_PATH = (
    ROOT
    / "01_event_aware_v3/03_scalegold_active_learning/"
      "LegacyAux_leakage_free_v3.csv"
)

SPLITS_PATH = (
    ROOT
    / "01_event_aware_v3/05_supervised_transformers/"
      "frozen_finetune_design_v3/"
      "finetune_nested_splits_v3.csv"
)

OUTROOT = (
    ROOT
    / "01_event_aware_v3/07_eve_frame_stage_a_no_event/"
      "runs_v3"
)

MODEL_NAME = "FacebookAI/xlm-roberta-base"

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

MAX_LENGTH = 256
STRIDE = 32
MAX_CHUNKS = 8

DROPOUT = 0.10
LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
GRAD_CLIP = 1.0
GRAD_ACCUM = 8
MAX_SELECTION_EPOCHS = 8


def file_sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16-capable GPU required."
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    if hasattr(
        torch.backends.cuda,
        "enable_cudnn_sdp",
    ):
        torch.backends.cuda.enable_cudnn_sdp(
            False
        )

    torch.use_deterministic_algorithms(
        True,
        warn_only=False,
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


def make_class_weights(
    doc_ids,
    label_ids,
    n_classes,
    device,
):
    vals = np.asarray(
        [
            label_ids[d]
            for d in doc_ids
        ],
        dtype=int,
    )

    counts = np.bincount(
        vals,
        minlength=n_classes,
    )

    weights = np.zeros(
        n_classes,
        dtype=np.float32,
    )

    present = counts > 0

    weights[present] = (
        len(vals)
        /
        (
            present.sum()
            * counts[present]
        )
    )

    return (
        torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        ),
        counts,
    )


def tokenize_documents(
    tokenizer,
    df,
    text_col,
):
    store = {}
    rows = []

    for row in df.itertuples(
        index=False
    ):

        doc_id = str(
            getattr(
                row,
                "doc_id",
            )
        )

        text = str(
            getattr(
                row,
                text_col,
            )
        )

        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            padding=False,
        )

        total_chunks = len(
            enc["input_ids"]
        )

        used_chunks = min(
            total_chunks,
            MAX_CHUNKS,
        )

        chunks = []

        for i in range(
            used_chunks
        ):
            chunks.append({
                "input_ids":
                    enc["input_ids"][i],

                "attention_mask":
                    enc["attention_mask"][i],

                "offset_mapping":
                    enc["offset_mapping"][i],

                "special_tokens_mask":
                    enc[
                        "special_tokens_mask"
                    ][i],
            })

        if not chunks:
            raise RuntimeError(
                f"No token chunks for {doc_id}"
            )

        store[doc_id] = chunks

        rows.append({
            "doc_id":
                doc_id,

            "text_characters":
                len(text),

            "total_chunks":
                total_chunks,

            "used_chunks":
                used_chunks,

            "truncated_by_chunk_cap":
                bool(
                    total_chunks
                    > MAX_CHUNKS
                ),
        })

    return (
        store,
        pd.DataFrame(rows),
    )


class StageAModel(
    nn.Module
):

    def __init__(
        self,
        n_classes,
    ):
        super().__init__()

        self.encoder = (
            AutoModel.from_pretrained(
                MODEL_NAME,
                local_files_only=True,
                add_pooling_layer=False,
                attn_implementation="eager",
            )
        )

        hidden = int(
            self.encoder.config.hidden_size
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        self.heads = nn.ModuleDict({
            task:
                nn.Linear(
                    hidden,
                    n_classes[task],
                )
            for task in TASKS
        })

    def classify(
        self,
        pooled,
    ):
        x = self.dropout(
            pooled
        )

        return {
            task:
                self.heads[
                    task
                ](x)
            for task in TASKS
        }


def encode_document(
    model,
    tokenizer,
    chunks,
    device,
):
    batch = tokenizer.pad(
        {
            "input_ids": [
                x["input_ids"]
                for x in chunks
            ],

            "attention_mask": [
                x["attention_mask"]
                for x in chunks
            ],
        },
        padding=True,
        return_tensors="pt",
    )

    input_ids = (
        batch["input_ids"]
        .to(device)
    )

    attention_mask = (
        batch["attention_mask"]
        .to(device)
    )

    out = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    hidden = out.last_hidden_state

    vectors = {}

    for ci, chunk in enumerate(
        chunks
    ):

        ids = chunk[
            "input_ids"
        ]

        offsets = chunk[
            "offset_mapping"
        ]

        specials = chunk[
            "special_tokens_mask"
        ]

        for ti in range(
            len(ids)
        ):

            if specials[ti]:
                continue

            start, end = offsets[ti]

            if (
                int(start) == 0
                and
                int(end) == 0
            ):
                continue

            key = (
                int(start),
                int(end),
                int(ids[ti]),
            )

            if key not in vectors:
                vectors[key] = []

            vectors[key].append(
                hidden[
                    ci,
                    ti,
                    :
                ]
            )

    if not vectors:
        raise RuntimeError(
            "Document produced no unique "
            "non-special tokens."
        )

    unique_vectors = []

    for vals in vectors.values():
        unique_vectors.append(
            torch.stack(
                vals,
                dim=0,
            ).mean(
                dim=0
            )
        )

    pooled = torch.stack(
        unique_vectors,
        dim=0,
    ).mean(
        dim=0
    )

    return pooled


def forward_document(
    model,
    tokenizer,
    chunks,
    device,
):
    pooled = encode_document(
        model,
        tokenizer,
        chunks,
        device,
    )

    return model.classify(
        pooled
    )


def build_optimizer_scheduler(
    model,
    n_train,
    scheduler_epochs,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = math.ceil(
        n_train
        / GRAD_ACCUM
    )

    total_steps = (
        steps_per_epoch
        * scheduler_epochs
    )

    warmup_steps = int(
        round(
            WARMUP_FRACTION
            * total_steps
        )
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=
                warmup_steps,
            num_training_steps=
                total_steps,
        )
    )

    return (
        optimizer,
        scheduler,
    )


def train_epochs(
    model,
    tokenizer,
    token_store,
    train_ids,
    label_ids,
    class_weights,
    device,
    epochs,
    scheduler_epochs,
    seed,
):
    optimizer, scheduler = (
        build_optimizer_scheduler(
            model,
            len(train_ids),
            scheduler_epochs,
        )
    )

    model.train()

    for epoch in range(
        1,
        epochs + 1,
    ):

        rng = np.random.default_rng(
            seed
            + epoch * 100003
        )

        order = np.asarray(
            train_ids,
            dtype=object,
        )

        order = order[
            rng.permutation(
                len(order)
            )
        ].tolist()

        optimizer.zero_grad(
            set_to_none=True
        )

        for pos, doc_id in enumerate(
            order,
            start=1,
        ):

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                logits = (
                    forward_document(
                        model,
                        tokenizer,
                        token_store[
                            doc_id
                        ],
                        device,
                    )
                )

            losses = []

            for task in TASKS:

                target = torch.tensor(
                    [
                        label_ids[
                            task
                        ][doc_id]
                    ],
                    dtype=torch.long,
                    device=device,
                )

                loss = F.cross_entropy(
                    logits[
                        task
                    ].float().unsqueeze(
                        0
                    ),
                    target,
                    weight=
                        class_weights[
                            task
                        ],
                    reduction="sum",
                )

                losses.append(
                    loss
                )

            total_loss = (
                torch.stack(
                    losses
                ).mean()
                / GRAD_ACCUM
            )

            total_loss.backward()

            should_step = (
                pos % GRAD_ACCUM == 0
                or
                pos == len(order)
            )

            if should_step:

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    token_store,
    doc_ids,
    label_ids,
    label_maps,
    class_weights,
    device,
    target_task,
    collect_target=False,
):
    model.eval()

    gold = {
        t: []
        for t in TASKS
    }

    pred = {
        t: []
        for t in TASKS
    }

    target_losses = []
    target_records = []

    for doc_id in doc_ids:

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            logits = (
                forward_document(
                    model,
                    tokenizer,
                    token_store[
                        doc_id
                    ],
                    device,
                )
            )

        for task in TASKS:

            logit = logits[
                task
            ].float()

            probability = torch.softmax(
                logit,
                dim=-1,
            )

            gold_id = int(
                label_ids[
                    task
                ][doc_id]
            )

            pred_id = int(
                torch.argmax(
                    probability
                ).item()
            )

            gold[
                task
            ].append(
                gold_id
            )

            pred[
                task
            ].append(
                pred_id
            )

            if task == target_task:

                target = torch.tensor(
                    [gold_id],
                    dtype=torch.long,
                    device=device,
                )

                loss = F.cross_entropy(
                    logit.unsqueeze(
                        0
                    ),
                    target,
                    weight=
                        class_weights[
                            task
                        ],
                    reduction="sum",
                )

                target_losses.append(
                    float(
                        loss.item()
                    )
                )

                if collect_target:

                    rec = {
                        "doc_id":
                            doc_id,

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
                    }

                    logit_np = (
                        logit.detach()
                        .cpu()
                        .numpy()
                    )

                    prob_np = (
                        probability.detach()
                        .cpu()
                        .numpy()
                    )

                    for i in range(
                        len(
                            label_maps[
                                task
                            ]
                        )
                    ):
                        rec[
                            f"logit_class_{i}"
                        ] = float(
                            logit_np[i]
                        )

                        rec[
                            f"prob_class_{i}"
                        ] = float(
                            prob_np[i]
                        )

                    target_records.append(
                        rec
                    )

    metrics = {}

    for task in TASKS:

        labels = list(
            range(
                len(
                    label_maps[
                        task
                    ]
                )
            )
        )

        metrics[
            task
        ] = metric_dict(
            np.asarray(
                gold[task],
                dtype=int,
            ),
            np.asarray(
                pred[task],
                dtype=int,
            ),
            labels,
        )

    model.train()

    return {
        "metrics":
            metrics,

        "target_loss":
            float(
                np.mean(
                    target_losses
                )
            ),

        "target_records":
            target_records,
    }


def initialize_model(
    seed,
    n_classes,
    device,
):
    set_seed(seed)

    model = StageAModel(
        n_classes
    )

    model.to(
        device
    )

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-task",
        required=True,
        choices=TASKS,
    )

    parser.add_argument(
        "--outer-fold",
        required=True,
        type=int,
        choices=[
            1,
            2,
            3,
            4,
        ],
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[
            11,
            29,
            47,
            83,
            131,
        ],
    )

    parser.add_argument(
        "--skip-if-valid",
        action="store_true",
    )

    args = parser.parse_args()

    configure_determinism()

    target_task = (
        args.target_task
    )

    fold = int(
        args.outer_fold
    )

    seed = int(
        args.seed
    )

    run_name = (
        f"{target_task}"
        f"__fold{fold}"
        f"__seed{seed}"
    )

    outdir = (
        OUTROOT
        / run_name
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        outdir
        / "run_summary_v3.json"
    )

    if (
        args.skip_if_valid
        and
        summary_path.exists()
    ):
        old = json.loads(
            summary_path.read_text()
        )

        valid = (
            old.get("status")
            == "COMPLETE"
            and
            old.get("target_task")
            == target_task
            and
            int(
                old.get(
                    "outer_fold",
                    -1,
                )
            ) == fold
            and
            int(
                old.get(
                    "seed",
                    -1,
                )
            ) == seed
            and
            old.get("architecture")
            == "EVE_FRAME_STAGE_A_NO_EVENT"
            and
            old.get("event_context")
            == "DISABLED"
            and
            old.get("trainer_sha256")
            == file_sha256(Path(__file__))
            and
            old.get("data_sha256")
            == file_sha256(DATA_PATH)
            and
            old.get("splits_sha256")
            == file_sha256(SPLITS_PATH)
        )

        if valid:
            print(
                "VALID EXISTING RUN — SKIP:",
                run_name,
            )
            return

    start_time = time.time()

    torch.cuda.reset_peak_memory_stats()

    device = torch.device(
        "cuda"
    )

    print("=" * 88)
    print(
        "EVE-FRAME STAGE A — NO_EVENT"
    )
    print("=" * 88)
    print("Run:", run_name)
    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    if len(df) != 199:
        raise RuntimeError(
            f"Expected 199 LegacyAux rows; "
            f"got {len(df)}"
        )

    if df[
        "doc_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate doc_id in LegacyAux."
        )

    required = [
        "doc_id",
        "dapt_text",
        *LABEL_COLUMNS.values(),
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing data columns: {missing}"
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

        col = LABEL_COLUMNS[
            task
        ]

        if df[
            col
        ].isna().any():
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

        label_maps[
            task
        ] = {
            i: label
            for i, label
            in enumerate(
                classes
            )
        }

        label_to_id[
            task
        ] = {
            label: i
            for i, label
            in label_maps[
                task
            ].items()
        }

        label_ids[
            task
        ] = {
            str(doc_id):
                int(
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

        n_classes[
            task
        ] = len(
            classes
        )

    splits = pd.read_csv(
        SPLITS_PATH
    )

    split = splits[
        (
            splits["task"]
            == target_task
        )
        &
        (
            splits["outer_fold"]
            == fold
        )
    ].copy()

    if len(split) != 199:
        raise RuntimeError(
            "Expected 199 split rows."
        )

    split[
        "doc_id"
    ] = split[
        "doc_id"
    ].astype(str)

    if set(
        split["doc_id"]
    ) != set(
        df["doc_id"]
    ):
        raise RuntimeError(
            "Split/data doc_id mismatch."
        )

    train_ids = (
        split[
            split["role"]
            == "inner_train"
        ][
            "doc_id"
        ].tolist()
    )

    dev_ids = (
        split[
            split["role"]
            == "inner_dev"
        ][
            "doc_id"
        ].tolist()
    )

    test_ids = (
        split[
            split["role"]
            == "outer_test"
        ][
            "doc_id"
        ].tolist()
    )

    if (
        set(train_ids)
        & set(dev_ids)
    ):
        raise RuntimeError(
            "Train/dev overlap."
        )

    if (
        set(train_ids)
        & set(test_ids)
    ):
        raise RuntimeError(
            "Train/test overlap."
        )

    if (
        set(dev_ids)
        & set(test_ids)
    ):
        raise RuntimeError(
            "Dev/test overlap."
        )

    outer_train_ids = (
        train_ids
        + dev_ids
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            use_fast=True,
        )
    )

    token_store, chunking = (
        tokenize_documents(
            tokenizer,
            df,
            "dapt_text",
        )
    )

    chunking.to_csv(
        outdir
        / "document_chunking_v3.csv",
        index=False,
    )

    (
        outdir
        / "label_maps_v3.json"
    ).write_text(
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
        )
    )

    selection_weights = {}
    selection_counts = {}

    for task in TASKS:

        (
            selection_weights[
                task
            ],
            selection_counts[
                task
            ],
        ) = make_class_weights(
            train_ids,
            label_ids[
                task
            ],
            n_classes[
                task
            ],
            device,
        )

    model = initialize_model(
        seed,
        n_classes,
        device,
    )

    optimizer, scheduler = (
        build_optimizer_scheduler(
            model,
            len(
                train_ids
            ),
            MAX_SELECTION_EPOCHS,
        )
    )

    history = []

    best_epoch = None
    best_score = -np.inf
    best_loss = np.inf

    model.train()

    for epoch in range(
        1,
        MAX_SELECTION_EPOCHS + 1,
    ):

        rng = np.random.default_rng(
            seed
            + epoch * 100003
        )

        order = np.asarray(
            train_ids,
            dtype=object,
        )

        order = order[
            rng.permutation(
                len(order)
            )
        ].tolist()

        optimizer.zero_grad(
            set_to_none=True
        )

        for pos, doc_id in enumerate(
            order,
            start=1,
        ):

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                logits = (
                    forward_document(
                        model,
                        tokenizer,
                        token_store[
                            doc_id
                        ],
                        device,
                    )
                )

            losses = []

            for task in TASKS:

                target = torch.tensor(
                    [
                        label_ids[
                            task
                        ][doc_id]
                    ],
                    dtype=torch.long,
                    device=device,
                )

                loss = F.cross_entropy(
                    logits[
                        task
                    ].float().unsqueeze(
                        0
                    ),
                    target,
                    weight=
                        selection_weights[
                            task
                        ],
                    reduction="sum",
                )

                losses.append(
                    loss
                )

            total_loss = (
                torch.stack(
                    losses
                ).mean()
                / GRAD_ACCUM
            )

            total_loss.backward()

            if (
                pos % GRAD_ACCUM == 0
                or
                pos == len(order)
            ):

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

        dev = evaluate(
            model,
            tokenizer,
            token_store,
            dev_ids,
            label_ids,
            label_maps,
            selection_weights,
            device,
            target_task,
            collect_target=False,
        )

        score = (
            dev["metrics"]
            [target_task]
            ["macro_f1"]
        )

        val_loss = (
            dev[
                "target_loss"
            ]
        )

        history.append({
            "epoch":
                epoch,

            "target_task":
                target_task,

            "target_dev_macro_f1":
                score,

            "target_dev_loss":
                val_loss,

            "primary_frame_dev_macro_f1":
                dev["metrics"]
                ["primary_frame"]
                ["macro_f1"],

            "stance_dev_macro_f1":
                dev["metrics"]
                ["stance"]
                ["macro_f1"],

            "misinformation_relation_dev_macro_f1":
                dev["metrics"]
                ["misinformation_relation"]
                ["macro_f1"],
        })

        improved = (
            score
            > best_score + 1e-12
        )

        tied_better_loss = (
            abs(
                score - best_score
            ) <= 1e-12
            and
            val_loss
            < best_loss
        )

        if (
            improved
            or tied_better_loss
        ):
            best_score = score
            best_loss = val_loss
            best_epoch = epoch

        print(
            f"epoch={epoch} "
            f"target={target_task} "
            f"dev_macro_f1={score:.6f} "
            f"dev_loss={val_loss:.6f}"
        )

    pd.DataFrame(
        history
    ).to_csv(
        outdir
        / "inner_selection_history_v3.csv",
        index=False,
    )

    if best_epoch is None:
        raise RuntimeError(
            "No selected epoch."
        )

    del model
    del optimizer
    del scheduler

    gc.collect()
    torch.cuda.empty_cache()

    refit_weights = {}
    refit_counts = {}

    for task in TASKS:

        (
            refit_weights[
                task
            ],
            refit_counts[
                task
            ],
        ) = make_class_weights(
            outer_train_ids,
            label_ids[
                task
            ],
            n_classes[
                task
            ],
            device,
        )

    model = initialize_model(
        seed,
        n_classes,
        device,
    )

    train_epochs(
        model,
        tokenizer,
        token_store,
        outer_train_ids,
        label_ids,
        refit_weights,
        device,
        epochs=best_epoch,
        scheduler_epochs=
            MAX_SELECTION_EPOCHS,
        seed=seed,
    )

    test = evaluate(
        model,
        tokenizer,
        token_store,
        test_ids,
        label_ids,
        label_maps,
        refit_weights,
        device,
        target_task,
        collect_target=True,
    )

    records = (
        test[
            "target_records"
        ]
    )

    prediction_df = pd.DataFrame(
        records
    )

    prediction_df.insert(
        1,
        "task",
        target_task,
    )

    prediction_df.insert(
        2,
        "outer_fold",
        fold,
    )

    prediction_df.insert(
        3,
        "architecture",
        "EVE_FRAME_STAGE_A_NO_EVENT",
    )

    prediction_df.insert(
        4,
        "initialization",
        "XLMR_BASE",
    )

    prediction_df.insert(
        5,
        "seed",
        seed,
    )

    prediction_df.to_csv(
        outdir
        / "outer_test_predictions_v3.csv",
        index=False,
    )

    outer = (
        test["metrics"]
        [target_task]
    )

    elapsed = (
        time.time()
        - start_time
    )

    peak_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    summary = {
        "status":
            "COMPLETE",

        "architecture":
            "EVE_FRAME_STAGE_A_NO_EVENT",

        "stage":
            "A",

        "event_context":
            "DISABLED",

        "event_mapping_overlap":
            "2_of_199",

        "target_task":
            target_task,

        "outer_fold":
            fold,

        "seed":
            seed,

        "initialization":
            "XLMR_BASE",

        "n_documents":
            199,

        "n_inner_train":
            len(
                train_ids
            ),

        "n_inner_dev":
            len(
                dev_ids
            ),

        "n_outer_train_refit":
            len(
                outer_train_ids
            ),

        "n_outer_test":
            len(
                test_ids
            ),

        "selected_epoch":
            int(
                best_epoch
            ),

        "selected_dev_macro_f1":
            float(
                best_score
            ),

        "outer_test_macro_f1":
            outer[
                "macro_f1"
            ],

        "outer_test_balanced_accuracy":
            outer[
                "balanced_accuracy"
            ],

        "outer_test_accuracy":
            outer[
                "accuracy"
            ],

        "outer_test_weighted_f1":
            outer[
                "weighted_f1"
            ],

        "all_task_outer_test_metrics":
            test[
                "metrics"
            ],

        "joint_loss":
            "equal_mean_three_class_weighted_CE",

        "max_length":
            MAX_LENGTH,

        "stride":
            STRIDE,

        "max_chunks":
            MAX_CHUNKS,

        "learning_rate":
            LR,

        "weight_decay":
            WEIGHT_DECAY,

        "warmup_fraction":
            WARMUP_FRACTION,

        "gradient_accumulation":
            GRAD_ACCUM,

        "max_selection_epochs":
            MAX_SELECTION_EPOCHS,

        "refit_scheduler_horizon_epochs":
            MAX_SELECTION_EPOCHS,

        "strict_deterministic_algorithms":
            True,

        "attention_implementation":
            "eager",

        "bf16":
            True,

        "elapsed_seconds":
            float(
                elapsed
            ),

        "peak_gpu_memory_gb":
            float(
                peak_gb
            ),

        "data_sha256":
            file_sha256(
                DATA_PATH
            ),

        "splits_sha256":
            file_sha256(
                SPLITS_PATH
            ),

        "trainer_sha256":
            file_sha256(
                Path(__file__)
            ),

        "eventgold_status":
            "SEALED_NOT_ACCESSED",
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("=" * 88)
    print("RUN COMPLETE")
    print("=" * 88)
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
