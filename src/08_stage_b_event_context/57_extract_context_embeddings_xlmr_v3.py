from pathlib import Path

import gc
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoModel,
    AutoTokenizer,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

STAGEB = (
    ROOT
    / "01_event_aware_v3/08_eve_frame_stage_b_event_context"
)

PAIRFILE = (
    STAGEB
    / "verified_context_v3/"
      "verified_event_neighbor_pairs_v3.csv"
)

CORPUS = (
    ROOT
    / "01_event_aware_v3/01_data_audit/"
      "Corpus1502_document_holdout_v3.csv"
)

TEXT_FREEZE = (
    STAGEB
    / "EVE_FRAME_STAGE_B_CONTEXT_TEXT_SOURCE_FREEZE_v3.txt"
)

EMBED_PROTOCOL = (
    STAGEB
    / "EVE_FRAME_STAGE_B_CONTEXT_EMBEDDING_PROTOCOL_v3.txt"
)

STAGE_A_TRAINER = (
    ROOT
    / "01_event_aware_v3/07_eve_frame_stage_a_no_event/"
      "37_train_eve_frame_stage_a_v3.py"
)

OUT = (
    STAGEB
    / "context_embeddings_v3"
)

TMP = (
    STAGEB
    / "context_embeddings_v3_tmp"
)


# ============================================================
# FROZEN CONSTANTS
# ============================================================

MODEL_NAME = "FacebookAI/xlm-roberta-base"

TEXT_COL = "dapt_text"

MAX_LENGTH = 256

STRIDE = 32

MAX_CHUNKS = 8

EXPECTED_N_PAIRS = 112

EXPECTED_N_DOCS = 81

EXPECTED_HIDDEN = 768


EXPECTED_SHA256 = {
    "pairfile":
        "43f4585b9c2114ef7f06f5f10aaf8cf1c35f3211f8a459301278b4e823d58cc6",

    "corpus":
        "8998f17cdb16550fb581f9dc21dbae8e064839cb6dbfcffc1445f2b433bc49f7",

    "text_freeze":
        "d5117af6c59d3274085c2203f047a571c4456ab6c115469da0c81b5bc4ef8d8d",

    "embedding_protocol":
        "c708a5a56ebe2effa725ed692f9c3db6f28080cc1bf19b3a521dcb82a4436dbc",

    "stage_a_trainer":
        "62f5f55fcdeaf4e5c947313bcc326daa8ceb225e2ab954ffa7b24430e1e760b9",
}


# ============================================================
# UTILITIES
# ============================================================

def file_sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def require_hash(
    path,
    expected,
    label,
):
    observed = file_sha256(path)

    if observed != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {observed}"
        )


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

    torch.backends.cuda.enable_flash_sdp(
        False
    )

    torch.backends.cuda.enable_mem_efficient_sdp(
        False
    )

    torch.backends.cuda.enable_math_sdp(
        True
    )

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


# ============================================================
# STAGE-A-COMPATIBLE TOKENIZATION
# ============================================================

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

        store[
            doc_id
        ] = chunks

        rows.append({
            "doc_id":
                doc_id,

            "dapt_text_length":
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
        pd.DataFrame(
            rows
        ),
    )


# ============================================================
# FROZEN CONTEXT MODEL WRAPPER
# ============================================================

class ContextModel(
    torch.nn.Module
):
    def __init__(
        self,
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

        for param in (
            self.encoder.parameters()
        ):
            param.requires_grad = False


# ============================================================
# STAGE-A-COMPATIBLE OVERLAP-AWARE POOLING
# ============================================================

@torch.no_grad()
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
        batch[
            "input_ids"
        ]
        .to(
            device
        )
    )

    attention_mask = (
        batch[
            "attention_mask"
        ]
        .to(
            device
        )
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        out = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

    hidden = (
        out.last_hidden_state
    )

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
            if specials[
                ti
            ]:
                continue

            start, end = offsets[
                ti
            ]

            if (
                int(start) == 0
                and
                int(end) == 0
            ):
                continue

            key = (
                int(start),
                int(end),
                int(
                    ids[
                        ti
                    ]
                ),
            )

            if key not in vectors:
                vectors[
                    key
                ] = []

            vectors[
                key
            ].append(
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

    for vals in (
        vectors.values()
    ):
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

    return (
        pooled
        .detach()
        .float()
        .cpu()
        .numpy()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # ONE-TIME OUTPUT GUARD
    # --------------------------------------------------------

    if OUT.exists():
        raise RuntimeError(
            f"Final output already exists: {OUT}"
        )

    if TMP.exists():
        raise RuntimeError(
            f"Temporary output already exists: {TMP}"
        )


    # --------------------------------------------------------
    # VERIFY FROZEN INPUTS
    # --------------------------------------------------------

    require_hash(
        PAIRFILE,
        EXPECTED_SHA256[
            "pairfile"
        ],
        "Verified event-neighbor pair file",
    )

    require_hash(
        CORPUS,
        EXPECTED_SHA256[
            "corpus"
        ],
        "Canonical Clean1502 corpus",
    )

    require_hash(
        TEXT_FREEZE,
        EXPECTED_SHA256[
            "text_freeze"
        ],
        "Context text-source freeze",
    )

    require_hash(
        EMBED_PROTOCOL,
        EXPECTED_SHA256[
            "embedding_protocol"
        ],
        "Context embedding protocol",
    )

    require_hash(
        STAGE_A_TRAINER,
        EXPECTED_SHA256[
            "stage_a_trainer"
        ],
        "Frozen Stage-A trainer",
    )


    # --------------------------------------------------------
    # DETERMINISM
    # --------------------------------------------------------

    configure_determinism()

    torch.manual_seed(
        20260922
    )

    torch.cuda.manual_seed_all(
        20260922
    )

    np.random.seed(
        20260922
    )


    # --------------------------------------------------------
    # LOAD VERIFIED RELATIONS
    # --------------------------------------------------------

    pairs = pd.read_csv(
        PAIRFILE,
        dtype=str,
        low_memory=False,
    )

    if len(
        pairs
    ) != EXPECTED_N_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_PAIRS} "
            f"verified pairs, got {len(pairs)}."
        )

    required_columns = {
        "legacy_doc_id",
        "background_doc_id",
        "verified_event_neighbor",
    }

    missing_columns = (
        required_columns
        - set(
            pairs.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required pair columns: "
            f"{sorted(missing_columns)}"
        )

    verified_flag = (
        pd.to_numeric(
            pairs[
                "verified_event_neighbor"
            ],
            errors="raise",
        )
    )

    if not (
        verified_flag
        .eq(
            1
        )
        .all()
    ):
        raise RuntimeError(
            "Pair table contains a non-verified row."
        )

    background_ids = sorted(
        set(
            pairs[
                "background_doc_id"
            ]
            .astype(str)
        )
    )

    if len(
        background_ids
    ) != EXPECTED_N_DOCS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_DOCS} "
            f"unique background docs, "
            f"got {len(background_ids)}."
        )


    # --------------------------------------------------------
    # LOAD CANONICAL TEXT
    # --------------------------------------------------------

    corpus = pd.read_csv(
        CORPUS,
        dtype={
            "doc_id": str,
        },
        low_memory=False,
    )

    if len(
        corpus
    ) != 1502:
        raise RuntimeError(
            f"Expected 1502 canonical rows, "
            f"got {len(corpus)}."
        )

    if corpus[
        "doc_id"
    ].nunique() != 1502:
        raise RuntimeError(
            "Canonical corpus doc_id is not unique."
        )

    if TEXT_COL not in (
        corpus.columns
    ):
        raise RuntimeError(
            f"Missing canonical text field: {TEXT_COL}"
        )

    subset = (
        corpus[
            corpus[
                "doc_id"
            ]
            .isin(
                background_ids
            )
        ][
            [
                "doc_id",
                TEXT_COL,
            ]
        ]
        .copy()
    )

    if len(
        subset
    ) != EXPECTED_N_DOCS:
        raise RuntimeError(
            f"Recovered {len(subset)} "
            f"of {EXPECTED_N_DOCS} "
            "required background documents."
        )

    subset[
        "doc_id"
    ] = (
        subset[
            "doc_id"
        ]
        .astype(str)
    )

    subset[
        TEXT_COL
    ] = (
        subset[
            TEXT_COL
        ]
        .fillna("")
        .astype(str)
    )

    if (
        subset[
            TEXT_COL
        ]
        .str.strip()
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "At least one verified background "
            "document has empty dapt_text."
        )

    subset = (
        subset
        .set_index(
            "doc_id"
        )
        .loc[
            background_ids
        ]
        .reset_index()
    )

    if (
        subset[
            "doc_id"
        ]
        .tolist()
        != background_ids
    ):
        raise RuntimeError(
            "Lexicographic document ordering failed."
        )


    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )
    )

    token_store, chunking = (
        tokenize_documents(
            tokenizer,
            subset,
            TEXT_COL,
        )
    )

    if len(
        chunking
    ) != EXPECTED_N_DOCS:
        raise RuntimeError(
            "Unexpected chunking row count."
        )

    if (
        chunking[
            "doc_id"
        ]
        .tolist()
        != background_ids
    ):
        raise RuntimeError(
            "Chunking order differs from "
            "frozen doc_id order."
        )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
    )

    model = ContextModel()

    hidden_size = int(
        model.encoder.config.hidden_size
    )

    if hidden_size != EXPECTED_HIDDEN:
        raise RuntimeError(
            f"Expected hidden size "
            f"{EXPECTED_HIDDEN}, "
            f"got {hidden_size}."
        )

    model.to(
        device
    )

    model.eval()

    if model.training:
        raise RuntimeError(
            "Context model unexpectedly "
            "remains in training mode."
        )

    if any(
        p.requires_grad
        for p in model.parameters()
    ):
        raise RuntimeError(
            "Context model contains "
            "trainable parameters."
        )


    # --------------------------------------------------------
    # ENCODE 81 DOCUMENTS
    # --------------------------------------------------------

    matrix = np.empty(
        (
            EXPECTED_N_DOCS,
            EXPECTED_HIDDEN,
        ),
        dtype=np.float32,
    )

    for row_index, doc_id in enumerate(
        background_ids
    ):
        vector = encode_document(
            model,
            tokenizer,
            token_store[
                doc_id
            ],
            device,
        )

        if vector.shape != (
            EXPECTED_HIDDEN,
        ):
            raise RuntimeError(
                f"Unexpected vector shape for "
                f"{doc_id}: {vector.shape}"
            )

        if vector.dtype != np.float32:
            raise RuntimeError(
                f"Unexpected vector dtype for "
                f"{doc_id}: {vector.dtype}"
            )

        if not np.isfinite(
            vector
        ).all():
            raise RuntimeError(
                f"Non-finite vector for {doc_id}."
            )

        norm = float(
            np.linalg.norm(
                vector
            )
        )

        if not np.isfinite(
            norm
        ) or norm <= 0:
            raise RuntimeError(
                f"Invalid vector norm for {doc_id}."
            )

        matrix[
            row_index
        ] = vector

        print(
            f"[{row_index + 1:02d}/"
            f"{EXPECTED_N_DOCS}] "
            f"{doc_id} "
            f"norm={norm:.6f}",
            flush=True,
        )


    # --------------------------------------------------------
    # FINAL MATRIX QC
    # --------------------------------------------------------

    if matrix.shape != (
        EXPECTED_N_DOCS,
        EXPECTED_HIDDEN,
    ):
        raise RuntimeError(
            f"Final matrix shape mismatch: "
            f"{matrix.shape}"
        )

    if matrix.dtype != np.float32:
        raise RuntimeError(
            f"Final matrix dtype mismatch: "
            f"{matrix.dtype}"
        )

    if not np.isfinite(
        matrix
    ).all():
        raise RuntimeError(
            "Final matrix contains "
            "non-finite values."
        )

    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    if (
        (~np.isfinite(norms)).any()
        or
        (norms <= 0).any()
    ):
        raise RuntimeError(
            "Invalid final vector norm detected."
        )


    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    manifest = (
        chunking
        .copy()
    )

    manifest.insert(
        0,
        "row_index",
        np.arange(
            EXPECTED_N_DOCS,
            dtype=int,
        ),
    )

    if manifest[
        "doc_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate doc_id in output manifest."
        )

    if manifest[
        "row_index"
    ].tolist() != list(
        range(
            EXPECTED_N_DOCS
        )
    ):
        raise RuntimeError(
            "Manifest row_index mismatch."
        )

    if manifest[
        "doc_id"
    ].tolist() != background_ids:
        raise RuntimeError(
            "Manifest row order differs "
            "from embedding matrix row order."
        )


    # --------------------------------------------------------
    # WRITE ATOMIC OUTPUT
    # --------------------------------------------------------

    TMP.mkdir(
        parents=True,
        exist_ok=False,
    )

    embedding_path = (
        TMP
        / "context_background_embeddings_xlmr_v3.npy"
    )

    manifest_path = (
        TMP
        / "context_background_embedding_manifest_v3.csv"
    )

    chunking_path = (
        TMP
        / "context_background_chunking_v3.csv"
    )

    results_path = (
        TMP
        / "EVE_FRAME_STAGE_B_CONTEXT_EMBEDDING_RESULTS_v3.json"
    )

    np.save(
        embedding_path,
        matrix,
        allow_pickle=False,
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    chunking.to_csv(
        chunking_path,
        index=False,
    )


    # --------------------------------------------------------
    # RELOAD STORED MATRIX
    # --------------------------------------------------------

    reloaded = np.load(
        embedding_path,
        allow_pickle=False,
    )

    if reloaded.shape != (
        EXPECTED_N_DOCS,
        EXPECTED_HIDDEN,
    ):
        raise RuntimeError(
            "Stored matrix shape mismatch."
        )

    if reloaded.dtype != np.float32:
        raise RuntimeError(
            "Stored matrix dtype mismatch."
        )

    if not np.array_equal(
        matrix,
        reloaded,
    ):
        raise RuntimeError(
            "Stored matrix differs "
            "from in-memory matrix."
        )


    # --------------------------------------------------------
    # RESULTS JSON
    # --------------------------------------------------------

    results = {
        "status":
            "CONTEXT_EMBEDDING_EXTRACTION_COMPLETE",

        "model_name":
            MODEL_NAME,

        "context_encoder_trainable":
            False,

        "context_encoder_eval_mode":
            True,

        "text_field":
            TEXT_COL,

        "max_length":
            MAX_LENGTH,

        "stride":
            STRIDE,

        "max_chunks":
            MAX_CHUNKS,

        "requested_verified_pairs":
            EXPECTED_N_PAIRS,

        "requested_unique_background_documents":
            EXPECTED_N_DOCS,

        "encoded_documents":
            int(
                matrix.shape[
                    0
                ]
            ),

        "embedding_shape":
            [
                int(x)
                for x
                in matrix.shape
            ],

        "embedding_dtype":
            str(
                matrix.dtype
            ),

        "all_values_finite":
            bool(
                np.isfinite(
                    matrix
                ).all()
            ),

        "all_vector_norms_positive":
            bool(
                (
                    norms
                    > 0
                ).all()
            ),

        "vector_norm_min":
            float(
                norms.min()
            ),

        "vector_norm_median":
            float(
                np.median(
                    norms
                )
            ),

        "vector_norm_mean":
            float(
                norms.mean()
            ),

        "vector_norm_max":
            float(
                norms.max()
            ),

        "truncated_by_chunk_cap_documents":
            int(
                chunking[
                    "truncated_by_chunk_cap"
                ]
                .astype(bool)
                .sum()
            ),

        "bf16_autocast":
            True,

        "stored_float32":
            True,

        "deterministic_algorithms":
            True,

        "tf32":
            False,

        "flash_sdp":
            False,

        "memory_efficient_sdp":
            False,

        "main_task_labels_used":
            False,

        "topic_used":
            False,

        "language_used_as_feature":
            False,

        "country_used_as_feature":
            False,

        "region_used_as_feature":
            False,

        "source_domain_used_as_feature":
            False,

        "eventgold_status":
            "SEALED_NOT_ACCESSED",

        "input_sha256": {
            "verified_event_neighbor_pairs":
                file_sha256(
                    PAIRFILE
                ),

            "canonical_corpus":
                file_sha256(
                    CORPUS
                ),

            "context_text_source_freeze":
                file_sha256(
                    TEXT_FREEZE
                ),

            "context_embedding_protocol":
                file_sha256(
                    EMBED_PROTOCOL
                ),

            "stage_a_trainer":
                file_sha256(
                    STAGE_A_TRAINER
                ),
        },
    }

    results_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # OUTPUT CHECKSUMS
    # --------------------------------------------------------

    files_to_hash = sorted(
        [
            p
            for p in TMP.iterdir()
            if p.is_file()
        ],
        key=lambda p: p.name,
    )

    checksum_path = (
        TMP
        / "EVE_FRAME_STAGE_B_CONTEXT_EMBEDDING_SHA256SUMS_v3.txt"
    )

    checksum_path.write_text(
        "".join(
            f"{file_sha256(p)}  {p.name}\n"
            for p in files_to_hash
        ),
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # ATOMIC FINALIZATION
    # --------------------------------------------------------

    del model

    gc.collect()

    torch.cuda.empty_cache()

    TMP.rename(
        OUT
    )


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    final_embedding = (
        OUT
        / "context_background_embeddings_xlmr_v3.npy"
    )

    final_manifest = (
        OUT
        / "context_background_embedding_manifest_v3.csv"
    )

    final_results = (
        OUT
        / "EVE_FRAME_STAGE_B_CONTEXT_EMBEDDING_RESULTS_v3.json"
    )

    print()
    print("=" * 100)
    print(
        "EVE-FRAME STAGE B CONTEXT EMBEDDING "
        "EXTRACTION COMPLETE"
    )
    print("=" * 100)

    print(
        "Encoded documents:",
        EXPECTED_N_DOCS,
    )

    print(
        "Embedding shape:",
        matrix.shape,
    )

    print(
        "Embedding dtype:",
        matrix.dtype,
    )

    print(
        "Vector norm min:",
        float(
            norms.min()
        ),
    )

    print(
        "Vector norm median:",
        float(
            np.median(
                norms
            )
        ),
    )

    print(
        "Vector norm mean:",
        float(
            norms.mean()
        ),
    )

    print(
        "Vector norm max:",
        float(
            norms.max()
        ),
    )

    print(
        "Documents truncated by chunk cap:",
        int(
            chunking[
                "truncated_by_chunk_cap"
            ]
            .astype(bool)
            .sum()
        ),
    )

    print()
    print(
        "Embedding SHA256:",
        file_sha256(
            final_embedding
        ),
    )

    print(
        "Manifest SHA256:",
        file_sha256(
            final_manifest
        ),
    )

    print(
        "Results SHA256:",
        file_sha256(
            final_results
        ),
    )

    print()
    print(
        "✅ Frozen XLM-R base used."
    )

    print(
        "✅ Context encoder remained frozen."
    )

    print(
        "✅ Stage-A overlap-aware pooling reproduced."
    )

    print(
        "✅ 81/81 verified background documents encoded."
    )

    print(
        "✅ Main-task labels not used."
    )

    print(
        "✅ EventGold not accessed."
    )


if __name__ == "__main__":
    main()
