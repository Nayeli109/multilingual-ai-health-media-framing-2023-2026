from pathlib import Path
import json
import math
import random
import shutil
import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

MODEL_NAME = "FacebookAI/xlm-roberta-base"

MAX_LENGTH = 256
STRIDE = 32
MAX_CHUNKS_PER_ARTICLE = 8

MLM_PROBABILITY = 0.15

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

GRAD_ACCUM = 4

EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

VAL_FRACTION = 0.10

NUM_WORKERS = 2


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

DATA_PATH = (
    ROOT
    / "01_event_aware_v3/01_data_audit/"
      "DAPT_INDUCTIVE_LEGACYAUX_HOLDOUT_v3.csv"
)

OUTDIR = (
    ROOT
    / "01_event_aware_v3/02_domain_adaptation/"
      "xlmr_health_ai_dapt_inductive_lr2e5_v3"
)

BEST_DIR = OUTDIR / "best_model"
FINAL_DIR = OUTDIR / "final_model"

HISTORY_OUT = OUTDIR / "training_history.csv"
METRICS_OUT = OUTDIR / "dapt_metrics.json"

TRAIN_DOCS_OUT = OUTDIR / "train_documents.csv"
VAL_DOCS_OUT = OUTDIR / "validation_documents.csv"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not visible. "
        "Run this script inside a SLURM GPU allocation."
    )

device = torch.device("cuda")

gpu_name = torch.cuda.get_device_name(0)

bf16 = torch.cuda.is_bf16_supported()

amp_dtype = (
    torch.bfloat16
    if bf16
    else torch.float16
)

print("==============================================")
print("XLM-R DOMAIN-ADAPTIVE PRETRAINING")
print("==============================================")
print("GPU            :", gpu_name)
print("BF16           :", bf16)
print("Model          :", MODEL_NAME)
print("Max length     :", MAX_LENGTH)
print("Epochs         :", EPOCHS)
print("Train batch    :", TRAIN_BATCH_SIZE)
print("Grad accum     :", GRAD_ACCUM)
print(
    "Effective batch:",
    TRAIN_BATCH_SIZE * GRAD_ACCUM
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

required = [
    "doc_id",
    "dapt_text",
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

df = df[
    df["dapt_text"] != ""
].copy()

print("\nStrict DAPT documents:", len(df))


# ============================================================
# ARTICLE-LEVEL TRAIN / VALIDATION SPLIT
#
# Split BEFORE token chunking, so chunks from the same article
# can never appear in both train and validation.
# ============================================================

stratify = None

if (
    "language" in df.columns
    and df["language"].nunique() > 1
):
    stratify = df["language"]

train_df, val_df = train_test_split(
    df,
    test_size=VAL_FRACTION,
    random_state=SEED,
    stratify=stratify,
)

train_df = (
    train_df
    .sort_values("doc_id")
    .reset_index(drop=True)
)

val_df = (
    val_df
    .sort_values("doc_id")
    .reset_index(drop=True)
)

assert (
    set(train_df["doc_id"])
    & set(val_df["doc_id"])
) == set()

train_df.to_csv(
    TRAIN_DOCS_OUT,
    index=False
)

val_df.to_csv(
    VAL_DOCS_OUT,
    index=False
)

print(
    "Train documents      :",
    len(train_df)
)

print(
    "Validation documents :",
    len(val_df)
)


# ============================================================
# MODEL / TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

model = AutoModelForMaskedLM.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

model.to(device)


# ============================================================
# TOKENIZATION INTO ARTICLE-CONTAINED WINDOWS
# ============================================================

def tokenize_documents(frame, split_name):

    examples = []

    lengths = []

    print(
        f"\nTokenizing {split_name}: "
        f"{len(frame)} documents"
    )

    for row_idx, row in frame.iterrows():

        encoded = tokenizer(
            row["dapt_text"],
            truncation=True,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            add_special_tokens=True,
        )

        ids_list = encoded["input_ids"]
        masks_list = encoded["attention_mask"]

        n_chunks = min(
            len(ids_list),
            MAX_CHUNKS_PER_ARTICLE
        )

        for j in range(n_chunks):

            examples.append({
                "input_ids": ids_list[j],
                "attention_mask": masks_list[j],
            })

            lengths.append(
                len(ids_list[j])
            )

    print(
        f"{split_name} chunks:",
        len(examples)
    )

    print(
        f"{split_name} mean tokens:",
        round(float(np.mean(lengths)), 1)
    )

    return examples


train_examples = tokenize_documents(
    train_df,
    "TRAIN"
)

val_examples = tokenize_documents(
    val_df,
    "VALIDATION"
)


class ChunkDataset(Dataset):

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


train_ds = ChunkDataset(
    train_examples
)

val_ds = ChunkDataset(
    val_examples
)


# ============================================================
# DYNAMIC MLM MASKING
# ============================================================

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROBABILITY,
)


train_loader = DataLoader(
    train_ds,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
    num_workers=0,
    pin_memory=True,
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

steps_per_epoch = math.ceil(
    len(train_loader)
    / GRAD_ACCUM
)

total_steps = (
    steps_per_epoch
    * EPOCHS
)

warmup_steps = int(
    total_steps
    * WARMUP_RATIO
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


# ============================================================
# AMP
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(not bf16)
)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    # Same masking sequence every evaluation:
    # makes epoch-to-epoch validation directly comparable.
    torch.manual_seed(20260917)
    torch.cuda.manual_seed_all(20260917)

    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:

        batch = {
            k: v.to(
                device,
                non_blocking=True
            )
            for k, v in batch.items()
        }

        with torch.amp.autocast(
            "cuda",
            dtype=amp_dtype
        ):

            outputs = model(**batch)

        total_loss += float(
            outputs.loss.detach().cpu()
        )

        total_batches += 1

    mean_loss = (
        total_loss
        / max(total_batches, 1)
    )

    perplexity = math.exp(
        min(mean_loss, 20)
    )

    model.train()

    return mean_loss, perplexity


# ============================================================
# BASELINE BEFORE DAPT
# ============================================================

print("\n==============================================")
print("PRE-DAPT VALIDATION")
print("==============================================")

baseline_val_loss, baseline_ppl = evaluate()

print(
    "Validation MLM loss:",
    round(baseline_val_loss, 6)
)

print(
    "Validation perplexity:",
    round(baseline_ppl, 4)
)


history = [{
    "epoch": 0,
    "train_loss": np.nan,
    "val_loss": baseline_val_loss,
    "val_perplexity": baseline_ppl,
}]


# ============================================================
# TRAIN
# ============================================================

# Best TRAINED checkpoint is selected only among DAPT epochs.
# Epoch 0 remains the untouched pre-DAPT reference.
best_val_loss = float("inf")
best_epoch = None

global_step = 0

start_time = time.time()

optimizer.zero_grad(
    set_to_none=True
)

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0
    seen_batches = 0

    print("\n==============================================")
    print(f"EPOCH {epoch}/{EPOCHS}")
    print("==============================================")

    for batch_idx, batch in enumerate(
        train_loader,
        start=1
    ):

        batch = {
            k: v.to(
                device,
                non_blocking=True
            )
            for k, v in batch.items()
        }

        with torch.amp.autocast(
            "cuda",
            dtype=amp_dtype
        ):

            outputs = model(**batch)

            loss = (
                outputs.loss
                / GRAD_ACCUM
            )

        running_loss += (
            float(
                outputs.loss
                .detach()
                .cpu()
            )
        )

        seen_batches += 1

        if scaler.is_enabled():

            scaler.scale(
                loss
            ).backward()

        else:

            loss.backward()

        do_step = (
            batch_idx % GRAD_ACCUM == 0
            or batch_idx == len(train_loader)
        )

        if do_step:

            if scaler.is_enabled():

                scaler.unscale_(
                    optimizer
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            if scaler.is_enabled():

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()

            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            global_step += 1

        if (
            batch_idx % 50 == 0
            or batch_idx == len(train_loader)
        ):

            avg_so_far = (
                running_loss
                / seen_batches
            )

            print(
                f"epoch={epoch} "
                f"batch={batch_idx}/{len(train_loader)} "
                f"loss={avg_so_far:.5f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    train_loss = (
        running_loss
        / max(seen_batches, 1)
    )

    val_loss, val_ppl = evaluate()

    print(
        f"\nEpoch {epoch} train loss : "
        f"{train_loss:.6f}"
    )

    print(
        f"Epoch {epoch} val loss   : "
        f"{val_loss:.6f}"
    )

    print(
        f"Epoch {epoch} val ppl    : "
        f"{val_ppl:.4f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_perplexity": val_ppl,
    })

    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_OUT,
        index=False
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch

        if BEST_DIR.exists():
            shutil.rmtree(
                BEST_DIR
            )

        BEST_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        model.save_pretrained(
            BEST_DIR,
            safe_serialization=True
        )

        tokenizer.save_pretrained(
            BEST_DIR
        )

        print(
            "✅ New best model saved."
        )


# ============================================================
# FINAL MODEL
# ============================================================

if FINAL_DIR.exists():
    shutil.rmtree(
        FINAL_DIR
    )

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model.save_pretrained(
    FINAL_DIR,
    safe_serialization=True
)

tokenizer.save_pretrained(
    FINAL_DIR
)


# ============================================================
# METRICS
# ============================================================

elapsed_sec = (
    time.time()
    - start_time
)

history_df = pd.DataFrame(
    history
)

trained_history_df = history_df.loc[
    history_df["epoch"] > 0
].copy()

best_trained_row = trained_history_df.loc[
    trained_history_df["val_loss"].idxmin()
]

overall_best_row = history_df.loc[
    history_df["val_loss"].idxmin()
]

metrics = {
    "model": MODEL_NAME,
    "gpu": gpu_name,
    "bf16": bool(bf16),

    "strict_dapt_documents": int(
        len(df)
    ),

    "train_documents": int(
        len(train_df)
    ),

    "validation_documents": int(
        len(val_df)
    ),

    "train_chunks": int(
        len(train_examples)
    ),

    "validation_chunks": int(
        len(val_examples)
    ),

    "max_length": MAX_LENGTH,
    "stride": STRIDE,

    "mlm_probability": MLM_PROBABILITY,

    "epochs": EPOCHS,

    "learning_rate": LR,

    "effective_batch_size": (
        TRAIN_BATCH_SIZE
        * GRAD_ACCUM
    ),

    "baseline_val_loss": float(
        baseline_val_loss
    ),

    "baseline_val_perplexity": float(
        baseline_ppl
    ),

    "best_dapt_epoch": int(
        best_trained_row["epoch"]
    ),

    "best_dapt_val_loss": float(
        best_trained_row["val_loss"]
    ),

    "best_dapt_val_perplexity": float(
        best_trained_row["val_perplexity"]
    ),

    "dapt_val_loss_improvement": float(
        baseline_val_loss
        - best_trained_row["val_loss"]
    ),

    "dapt_improved_over_baseline": bool(
        best_trained_row["val_loss"]
        < baseline_val_loss
    ),

    "overall_best_epoch_including_baseline": int(
        overall_best_row["epoch"]
    ),

    "training_seconds": float(
        elapsed_sec
    ),

    "seed": SEED,
}

METRICS_OUT.write_text(
    json.dumps(
        metrics,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print("\n==============================================")
print("DAPT COMPLETE")
print("==============================================")

print(
    "Baseline val loss:",
    round(
        baseline_val_loss,
        6
    )
)

print(
    "Best val loss    :",
    round(
        metrics["best_dapt_val_loss"],
        6
    )
)

print(
    "Loss improvement :",
    round(
        metrics["dapt_val_loss_improvement"],
        6
    )
)

print(
    "Best epoch       :",
    metrics["best_dapt_epoch"]
)

print(
    "Elapsed minutes  :",
    round(
        elapsed_sec / 60,
        2
    )
)

print("\nOutputs:")
print(BEST_DIR)
print(FINAL_DIR)
print(HISTORY_OUT)
print(METRICS_OUT)
