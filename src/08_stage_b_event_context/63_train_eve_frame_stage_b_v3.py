from pathlib import Path
import argparse
import hashlib
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel


ROOT = Path.cwd()

DEST = (
    ROOT
    / "01_event_aware_v3/08_eve_frame_stage_b_event_context"
)

BASE = DEST / "62_stage_b_base_from_stage_a_v3.py"

PROTOTYPES = (
    DEST
    / "event_prototypes_v3/"
      "verified_event_prototypes_xlmr_v3.npy"
)

AVAILABILITY = (
    DEST
    / "event_prototypes_v3/"
      "context_availability_legacyaux199_v3.csv"
)

CONTROL_MAP = (
    DEST
    / "control_maps_v3/"
      "permuted_context_mapping_v3.csv"
)

ARCH_PROTOCOL = (
    DEST
    / "EVE_FRAME_STAGE_B_ARCHITECTURE_PROTOCOL_v3.txt"
)

PROTO_CLOSURE = (
    DEST
    / "EVENT_PROTOTYPE_FINAL_CLOSURE_v3.txt"
)

CONTROL_CLOSURE = (
    DEST
    / "PERMUTED_CONTEXT_MAPPING_FINAL_CLOSURE_v3.txt"
)

SPLITS = (
    ROOT
    / "01_event_aware_v3/05_supervised_transformers/"
      "frozen_finetune_design_v3/"
      "finetune_nested_splits_v3.csv"
)

ORIGINAL_STAGE_A = (
    ROOT
    / "01_event_aware_v3/07_eve_frame_stage_a_no_event/"
      "37_train_eve_frame_stage_a_v3.py"
)


EXPECTED = {
    "base":
        "4665373fb9fc1189e1fce6fd54bb91028102633e7515a673b7df36ade4f5c3dd",

    "prototype_matrix":
        "0f7e9956210c7dbaf460867ce2a71164403a429ed44b600dfc10b43425ecaa4f",

    "availability":
        "b2c976c5d2ab5c578774ae5203c5006328618a8ed6d4f4a6aef6d1bb8833bf74",

    "control_map":
        "1a0a64813648be7e24ddf781ab265c080657c4a11f45b2a1a7d11d185cb0b51d",

    "architecture_protocol":
        "076ca9b3a99997b83d1ce43ca75fa59cc151f968d6c34c9fae92ef1ddf96c93f",

    "prototype_closure":
        "32afbc814bf2e7fc5b1e18df6725d623cbee2ab58452942105272b9fcc850fe3",

    "control_closure":
        "dea64f3cd61b9cab141ec281a71810c809a78b2904941d7226f1f009c777c54b",

    "splits":
        "3aa88ac76bf63dd2e39f3a2943ba46bca6b528a427678b49ef1174f4f8cd0584",

    "original_stage_a":
        "62f5f55fcdeaf4e5c947313bcc326daa8ceb225e2ab954ffa7b24430e1e760b9",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def check(path, expected, label):
    got = sha256(path)
    if got != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {got}"
        )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--arm",
    required=True,
    choices=[
        "VERIFIED_CONTEXT",
        "PERMUTED_CONTEXT",
    ],
)

parser.add_argument(
    "--target-task",
    required=True,
    choices=[
        "primary_frame",
        "stance",
        "misinformation_relation",
    ],
)

parser.add_argument(
    "--outer-fold",
    required=True,
    type=int,
    choices=[1, 2, 3, 4],
)

parser.add_argument(
    "--seed",
    required=True,
    type=int,
)

parser.add_argument(
    "--skip-if-valid",
    action="store_true",
)

parser.add_argument(
    "--preflight-only",
    action="store_true",
)

args = parser.parse_args()


# ------------------------------------------------------------
# Frozen-input integrity
# ------------------------------------------------------------

check(BASE, EXPECTED["base"], "Stage-B base")
check(
    PROTOTYPES,
    EXPECTED["prototype_matrix"],
    "Prototype matrix",
)
check(
    AVAILABILITY,
    EXPECTED["availability"],
    "Context availability",
)
check(
    CONTROL_MAP,
    EXPECTED["control_map"],
    "Permuted-context map",
)
check(
    ARCH_PROTOCOL,
    EXPECTED["architecture_protocol"],
    "Architecture protocol",
)
check(
    PROTO_CLOSURE,
    EXPECTED["prototype_closure"],
    "Prototype closure",
)
check(
    CONTROL_CLOSURE,
    EXPECTED["control_closure"],
    "Control-map closure",
)
check(
    SPLITS,
    EXPECTED["splits"],
    "Frozen nested splits",
)
check(
    ORIGINAL_STAGE_A,
    EXPECTED["original_stage_a"],
    "Original frozen Stage-A trainer",
)


# ------------------------------------------------------------
# Arm must be set before importing the patched Stage-A base.
# ------------------------------------------------------------

os.environ["EVE_STAGEB_ARM"] = args.arm

spec = importlib.util.spec_from_file_location(
    "eve_stage_b_base_v3",
    BASE,
)

stagea = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    stagea
)

if getattr(stagea, "EVENT_CONTEXT", None) != args.arm:
    raise RuntimeError(
        "Stage-B base EVENT_CONTEXT does not match requested arm."
    )

expected_architecture = (
    f"EVE_FRAME_STAGE_B_{args.arm}"
)

if getattr(
    stagea,
    "STAGE_B_ARCHITECTURE",
    None,
) != expected_architecture:
    raise RuntimeError(
        "Stage-B architecture metadata mismatch."
    )


# ------------------------------------------------------------
# Load frozen prototypes and masks
# ------------------------------------------------------------

prototype_matrix = np.load(
    PROTOTYPES,
    allow_pickle=False,
)

if prototype_matrix.shape != (51, 768):
    raise RuntimeError(
        f"Unexpected prototype shape: "
        f"{prototype_matrix.shape}"
    )

if prototype_matrix.dtype != np.float32:
    raise RuntimeError(
        f"Unexpected prototype dtype: "
        f"{prototype_matrix.dtype}"
    )

availability_df = pd.read_csv(
    AVAILABILITY,
    dtype={
        "legacy_doc_id": str,
    },
)

availability_df["context_available"] = pd.to_numeric(
    availability_df["context_available"],
    errors="raise",
).astype(int)

availability_df["prototype_row_index"] = pd.to_numeric(
    availability_df["prototype_row_index"],
    errors="coerce",
).astype("Int64")

if len(availability_df) != 199:
    raise RuntimeError("Expected 199 availability rows.")

if int(
    availability_df["context_available"].sum()
) != 51:
    raise RuntimeError(
        "Expected exactly 51 context-available docs."
    )

TRUE_ROW = {}

for r in availability_df.itertuples(index=False):
    doc = str(r.legacy_doc_id)

    if int(r.context_available) == 1:
        if pd.isna(r.prototype_row_index):
            raise RuntimeError(
                f"Missing true prototype row for {doc}"
            )
        TRUE_ROW[doc] = int(r.prototype_row_index)
    else:
        TRUE_ROW[doc] = None


# ------------------------------------------------------------
# Load frozen permuted-control mapping
# ------------------------------------------------------------

control_df = pd.read_csv(
    CONTROL_MAP,
    dtype={
        "legacy_doc_id": str,
        "assigned_source_legacy_doc_id": str,
    },
    low_memory=False,
)

control_df["outer_fold"] = pd.to_numeric(
    control_df["outer_fold"],
    errors="raise",
).astype(int)

control_df["context_available"] = pd.to_numeric(
    control_df["context_available"],
    errors="raise",
).astype(int)

control_df["assigned_prototype_row_index"] = pd.to_numeric(
    control_df["assigned_prototype_row_index"],
    errors="coerce",
).astype("Int64")

if control_df.duplicated(
    [
        "target_task",
        "outer_fold",
        "phase",
        "legacy_doc_id",
    ]
).any():
    raise RuntimeError(
        "Duplicate permuted-context mapping key."
    )

CONTROL_ROW = {}

for r in control_df.itertuples(index=False):
    key = (
        str(r.target_task),
        int(r.outer_fold),
        str(r.phase),
        str(r.legacy_doc_id),
    )

    if int(r.context_available) == 1:
        if pd.isna(r.assigned_prototype_row_index):
            raise RuntimeError(
                f"Missing assigned prototype row: {key}"
            )
        CONTROL_ROW[key] = int(
            r.assigned_prototype_row_index
        )
    else:
        CONTROL_ROW[key] = None


# ------------------------------------------------------------
# Preserve Stage-A tokenization exactly while attaching doc_id.
# ------------------------------------------------------------

class DocumentChunks(list):
    def __init__(self, values, doc_id):
        super().__init__(values)
        self.doc_id = str(doc_id)


ORIGINAL_TOKENIZE = stagea.tokenize_documents


def tokenize_documents(
    tokenizer,
    df,
    text_col,
):
    store, audit = ORIGINAL_TOKENIZE(
        tokenizer,
        df,
        text_col,
    )

    for doc_id in list(store):
        store[doc_id] = DocumentChunks(
            store[doc_id],
            doc_id,
        )

    return store, audit


# ------------------------------------------------------------
# Stage-B model.
#
# IMPORTANT:
# The encoder, dropout and three heads are instantiated in the
# exact Stage-A order BEFORE the new context modules.
# Therefore for the same seed the Stage-A core initialization
# is preserved.
# ------------------------------------------------------------

class StageBModel(nn.Module):

    def __init__(
        self,
        n_classes,
    ):
        super().__init__()

        self.encoder = (
            AutoModel.from_pretrained(
                stagea.MODEL_NAME,
                local_files_only=True,
                add_pooling_layer=False,
                attn_implementation="eager",
            )
        )

        hidden = int(
            self.encoder.config.hidden_size
        )

        if hidden != 768:
            raise RuntimeError(
                f"Unexpected hidden size: {hidden}"
            )

        # Exact Stage-A core initialization order.
        self.dropout = nn.Dropout(
            stagea.DROPOUT
        )

        self.heads = nn.ModuleDict({
            task:
                nn.Linear(
                    hidden,
                    n_classes[task],
                )
            for task in stagea.TASKS
        })

        # New Stage-B context branch comes AFTER the frozen
        # conceptual Stage-A core initialization.
        self.context_projection = nn.Linear(
            hidden,
            hidden,
        )

        self.context_layernorm = nn.LayerNorm(
            hidden
        )

        self.context_gate = nn.Linear(
            hidden * 2,
            1,
        )

        self.context_phase = "inner_train"

    def classify(
        self,
        pooled,
    ):
        x = self.dropout(
            pooled
        )

        return {
            task:
                self.heads[task](x)
            for task in stagea.TASKS
        }


# ------------------------------------------------------------
# Context lookup
# ------------------------------------------------------------

def context_row_for(
    model,
    doc_id,
):
    doc_id = str(doc_id)

    if args.arm == "VERIFIED_CONTEXT":
        if doc_id not in TRUE_ROW:
            raise RuntimeError(
                f"Unknown LegacyAux doc_id: {doc_id}"
            )
        return TRUE_ROW[doc_id]

    key = (
        args.target_task,
        int(args.outer_fold),
        str(model.context_phase),
        doc_id,
    )

    if key not in CONTROL_ROW:
        raise RuntimeError(
            f"Missing PERMUTED_CONTEXT mapping: {key}"
        )

    return CONTROL_ROW[key]


# ------------------------------------------------------------
# Stage-B forward.
# ------------------------------------------------------------

def forward_document(
    model,
    tokenizer,
    chunks,
    device,
):
    if not hasattr(
        chunks,
        "doc_id",
    ):
        raise RuntimeError(
            "DocumentChunks lost doc_id metadata."
        )

    z = stagea.encode_document(
        model,
        tokenizer,
        chunks,
        device,
    )

    row = context_row_for(
        model,
        chunks.doc_id,
    )

    # Exact NO_CONTEXT invariant.
    if row is None:
        fused = z

    else:
        if row < 0 or row >= len(prototype_matrix):
            raise RuntimeError(
                f"Invalid prototype row: {row}"
            )

        e = torch.as_tensor(
            prototype_matrix[row],
            dtype=torch.float32,
            device=device,
        )

        c = model.context_layernorm(
            model.context_projection(
                e
            )
        )

        gate_input = torch.cat(
            [
                z,
                c,
            ],
            dim=-1,
        )

        g = torch.sigmoid(
            model.context_gate(
                gate_input
            )
        ).squeeze(-1)

        fused = (
            z
            + g * c
        )

    return model.classify(
        fused
    )


# ------------------------------------------------------------
# Phase control for the PERMUTED_CONTEXT arm.
# ------------------------------------------------------------

ORIGINAL_EVALUATE = stagea.evaluate
ORIGINAL_TRAIN_EPOCHS = stagea.train_epochs


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
    old_phase = model.context_phase

    model.context_phase = (
        "outer_test"
        if collect_target
        else "inner_dev"
    )

    try:
        return ORIGINAL_EVALUATE(
            model,
            tokenizer,
            token_store,
            doc_ids,
            label_ids,
            label_maps,
            class_weights,
            device,
            target_task,
            collect_target=collect_target,
        )
    finally:
        model.context_phase = old_phase


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
    model.context_phase = "outer_train"

    return ORIGINAL_TRAIN_EPOCHS(
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
    )


# ------------------------------------------------------------
# Monkeypatch ONLY the architecture/context-specific pieces.
# All Stage-A split, class-weight, selection, refit, evaluation
# and output machinery remains inherited from the frozen code.
# ------------------------------------------------------------

stagea.StageAModel = StageBModel
stagea.tokenize_documents = tokenize_documents
stagea.forward_document = forward_document
stagea.evaluate = evaluate
stagea.train_epochs = train_epochs

stagea.OUTROOT = (
    DEST
    / "stage_b_runs_v3"
    / args.arm.lower()
)


# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------

current = control_df[
    control_df["target_task"].eq(
        args.target_task
    )
    & control_df["outer_fold"].eq(
        args.outer_fold
    )
]

for phase in [
    "inner_train",
    "inner_dev",
    "outer_train",
    "outer_test",
]:
    if not current["phase"].eq(phase).any():
        raise RuntimeError(
            f"Missing control phase: {phase}"
        )

if args.preflight_only:
    print("=" * 80)
    print("STAGE-B PREFLIGHT OK")
    print("=" * 80)
    print("Arm:", args.arm)
    print("Target task:", args.target_task)
    print("Outer fold:", args.outer_fold)
    print("Seed:", args.seed)
    print("Prototype shape:", prototype_matrix.shape)
    print(
        "Context available:",
        sum(v is not None for v in TRUE_ROW.values()),
    )
    print("Control rows for task/fold:", len(current))
    print("Output root:", stagea.OUTROOT)
    print("✅ Frozen Stage-A machinery imported.")
    print("✅ Stage-B context branch patched.")
    print("✅ NO_CONTEXT invariant implemented.")
    print("✅ PERMUTED_CONTEXT mappings available.")
    print("✅ EventGold not accessed.")
    raise SystemExit(0)


# ------------------------------------------------------------
# Delegate the full nested training procedure to the frozen
# Stage-A machinery.
# ------------------------------------------------------------

stagea_args = [
    str(BASE),
    "--target-task",
    args.target_task,
    "--outer-fold",
    str(args.outer_fold),
    "--seed",
    str(args.seed),
]

if args.skip_if_valid:
    stagea_args.append(
        "--skip-if-valid"
    )

sys.argv = stagea_args

stagea.main()
