from pathlib import Path

import hashlib
import json

import numpy as np
import pandas as pd


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

BYDOC = (
    STAGEB
    / "verified_context_v3/"
      "verified_context_by_legacyaux_doc_v3.csv"
)

EMBED_MATRIX = (
    STAGEB
    / "context_embeddings_v3/"
      "context_background_embeddings_xlmr_v3.npy"
)

EMBED_MANIFEST = (
    STAGEB
    / "context_embeddings_v3/"
      "context_background_embedding_manifest_v3.csv"
)

CONTEXT_EMBED_CLOSURE = (
    STAGEB
    / "CONTEXT_EMBEDDING_FINAL_CLOSURE_v3.txt"
)

PROTOCOL = (
    STAGEB
    / "EVE_FRAME_STAGE_B_EVENT_PROTOTYPE_PROTOCOL_v3.txt"
)

ARCH_PROTOCOL = (
    STAGEB
    / "EVE_FRAME_STAGE_B_ARCHITECTURE_PROTOCOL_v3.txt"
)

OUT = (
    STAGEB
    / "event_prototypes_v3"
)

TMP = (
    STAGEB
    / "event_prototypes_v3_tmp"
)


# ============================================================
# FROZEN CONSTANTS
# ============================================================

EXPECTED_PAIRS = 112

EXPECTED_BACKGROUND_DOCS = 81

EXPECTED_TARGETS_WITH_CONTEXT = 51

EXPECTED_TARGETS_NO_CONTEXT = 148

EXPECTED_TARGETS_TOTAL = 199

EXPECTED_DIM = 768


EXPECTED_SHA256 = {
    "pairfile":
        "43f4585b9c2114ef7f06f5f10aaf8cf1c35f3211f8a459301278b4e823d58cc6",

    "bydoc":
        "c0102a3e0c02ec042b3eb4e4f4cd01c155227042148e3559f79d0f31e24007fa",

    "embedding_matrix":
        "3cf6e72a6da04a49fa032cb97984d4e851111cd306f0e28d65141212cb0d15d4",

    "embedding_manifest":
        "6dcd4722cc3354a56d678eb8b46eac71adec8ef4705e35eb01221a645d2534a2",

    "context_embedding_closure":
        "ce1d26936cee5ca201f6c97740b658771105a886ca5621ae37e1f3541dce8e73",

    "prototype_protocol":
        "33ed50d7bf221ed3e20728b111d6a35d85763812fdc5b8d128b15044f8c5700e",

    "architecture_protocol":
        "076ca9b3a99997b83d1ce43ca75fa59cc151f968d6c34c9fae92ef1ddf96c93f",
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


# ============================================================
# OUTPUT GUARD
# ============================================================

if OUT.exists():
    raise RuntimeError(
        f"Final prototype output already exists: {OUT}"
    )

if TMP.exists():
    raise RuntimeError(
        f"Temporary prototype output already exists: {TMP}"
    )


# ============================================================
# VERIFY FROZEN INPUTS
# ============================================================

require_hash(
    PAIRFILE,
    EXPECTED_SHA256["pairfile"],
    "Verified pair table",
)

require_hash(
    BYDOC,
    EXPECTED_SHA256["bydoc"],
    "Frozen article-level context table",
)

require_hash(
    EMBED_MATRIX,
    EXPECTED_SHA256["embedding_matrix"],
    "Frozen context embedding matrix",
)

require_hash(
    EMBED_MANIFEST,
    EXPECTED_SHA256["embedding_manifest"],
    "Frozen context embedding manifest",
)

require_hash(
    CONTEXT_EMBED_CLOSURE,
    EXPECTED_SHA256["context_embedding_closure"],
    "Context embedding final closure",
)

require_hash(
    PROTOCOL,
    EXPECTED_SHA256["prototype_protocol"],
    "Event prototype protocol",
)

require_hash(
    ARCH_PROTOCOL,
    EXPECTED_SHA256["architecture_protocol"],
    "Stage-B architecture protocol",
)


# ============================================================
# LOAD INPUTS
# ============================================================

pairs = pd.read_csv(
    PAIRFILE,
    dtype=str,
    low_memory=False,
)

bydoc = pd.read_csv(
    BYDOC,
    dtype={
        "legacy_doc_id": str,
    },
    low_memory=False,
)

embedding_manifest = pd.read_csv(
    EMBED_MANIFEST,
    dtype={
        "doc_id": str,
    },
    low_memory=False,
)

embedding_matrix = np.load(
    EMBED_MATRIX,
    allow_pickle=False,
)


# ============================================================
# PAIR-TABLE QC
# ============================================================

if len(pairs) != EXPECTED_PAIRS:
    raise RuntimeError(
        f"Expected {EXPECTED_PAIRS} verified pairs, "
        f"got {len(pairs)}."
    )

required_pair_columns = {
    "legacy_doc_id",
    "background_doc_id",
    "verified_event_neighbor",
}

missing_pair_columns = (
    required_pair_columns
    - set(pairs.columns)
)

if missing_pair_columns:
    raise RuntimeError(
        "Missing required pair columns: "
        f"{sorted(missing_pair_columns)}"
    )

pairs["legacy_doc_id"] = (
    pairs["legacy_doc_id"]
    .astype(str)
)

pairs["background_doc_id"] = (
    pairs["background_doc_id"]
    .astype(str)
)

verified_flag = pd.to_numeric(
    pairs["verified_event_neighbor"],
    errors="raise",
)

if not verified_flag.eq(1).all():
    raise RuntimeError(
        "Verified-pair file contains a non-verified row."
    )

if pairs[
    [
        "legacy_doc_id",
        "background_doc_id",
    ]
].duplicated().any():
    raise RuntimeError(
        "Duplicate target/background pair detected."
    )

unique_background = set(
    pairs["background_doc_id"]
)

unique_targets = set(
    pairs["legacy_doc_id"]
)

if len(unique_background) != EXPECTED_BACKGROUND_DOCS:
    raise RuntimeError(
        f"Expected {EXPECTED_BACKGROUND_DOCS} unique "
        f"background docs, got {len(unique_background)}."
    )

if len(unique_targets) != EXPECTED_TARGETS_WITH_CONTEXT:
    raise RuntimeError(
        f"Expected {EXPECTED_TARGETS_WITH_CONTEXT} unique "
        f"context targets, got {len(unique_targets)}."
    )


# ============================================================
# EMBEDDING QC
# ============================================================

if embedding_matrix.shape != (
    EXPECTED_BACKGROUND_DOCS,
    EXPECTED_DIM,
):
    raise RuntimeError(
        f"Unexpected embedding shape: "
        f"{embedding_matrix.shape}"
    )

if embedding_matrix.dtype != np.float32:
    raise RuntimeError(
        f"Unexpected embedding dtype: "
        f"{embedding_matrix.dtype}"
    )

if not np.isfinite(
    embedding_matrix
).all():
    raise RuntimeError(
        "Embedding matrix contains non-finite values."
    )

if len(
    embedding_manifest
) != EXPECTED_BACKGROUND_DOCS:
    raise RuntimeError(
        "Unexpected embedding-manifest row count."
    )

if embedding_manifest[
    "doc_id"
].nunique() != EXPECTED_BACKGROUND_DOCS:
    raise RuntimeError(
        "Embedding-manifest doc_id is not unique."
    )

if "row_index" not in embedding_manifest.columns:
    raise RuntimeError(
        "Embedding manifest lacks row_index."
    )

embedding_manifest[
    "row_index"
] = pd.to_numeric(
    embedding_manifest[
        "row_index"
    ],
    errors="raise",
).astype(int)

if embedding_manifest[
    "row_index"
].tolist() != list(
    range(
        EXPECTED_BACKGROUND_DOCS
    )
):
    raise RuntimeError(
        "Embedding manifest row_index is invalid."
    )

if set(
    embedding_manifest["doc_id"]
) != unique_background:
    raise RuntimeError(
        "Pair-table background IDs do not exactly "
        "match embedding-manifest IDs."
    )

embedding_row_for_doc = {
    str(row.doc_id):
        int(row.row_index)

    for row in embedding_manifest.itertuples(
        index=False
    )
}


# ============================================================
# ARTICLE-LEVEL AVAILABILITY QC
# ============================================================

if len(bydoc) != EXPECTED_TARGETS_TOTAL:
    raise RuntimeError(
        f"Expected {EXPECTED_TARGETS_TOTAL} article rows, "
        f"got {len(bydoc)}."
    )

if bydoc[
    "legacy_doc_id"
].nunique() != EXPECTED_TARGETS_TOTAL:
    raise RuntimeError(
        "Article-level legacy_doc_id is not unique."
    )

required_bydoc_columns = {
    "legacy_doc_id",
    "context_available",
    "n_verified_event_neighbors",
}

missing_bydoc_columns = (
    required_bydoc_columns
    - set(bydoc.columns)
)

if missing_bydoc_columns:
    raise RuntimeError(
        "Missing article-level columns: "
        f"{sorted(missing_bydoc_columns)}"
    )

bydoc[
    "context_available"
] = pd.to_numeric(
    bydoc[
        "context_available"
    ],
    errors="raise",
).astype(int)

bydoc[
    "n_verified_event_neighbors"
] = pd.to_numeric(
    bydoc[
        "n_verified_event_neighbors"
    ],
    errors="raise",
).astype(int)

if not set(
    bydoc[
        "context_available"
    ].unique()
).issubset(
    {0, 1}
):
    raise RuntimeError(
        "context_available contains values outside {0,1}."
    )

n_context = int(
    bydoc[
        "context_available"
    ].sum()
)

n_no_context = int(
    (
        bydoc[
            "context_available"
        ]
        == 0
    ).sum()
)

if n_context != EXPECTED_TARGETS_WITH_CONTEXT:
    raise RuntimeError(
        f"Expected {EXPECTED_TARGETS_WITH_CONTEXT} "
        f"context-available articles, got {n_context}."
    )

if n_no_context != EXPECTED_TARGETS_NO_CONTEXT:
    raise RuntimeError(
        f"Expected {EXPECTED_TARGETS_NO_CONTEXT} "
        f"NO_CONTEXT articles, got {n_no_context}."
    )

context_ids_from_bydoc = set(
    bydoc.loc[
        bydoc[
            "context_available"
        ].eq(1),
        "legacy_doc_id",
    ]
)

if context_ids_from_bydoc != unique_targets:
    raise RuntimeError(
        "Pair-derived context targets do not exactly "
        "match article-level context_available=1 targets."
    )


# ============================================================
# NEIGHBOR-COUNT CROSS-CHECK
# ============================================================

pair_counts = (
    pairs.groupby(
        "legacy_doc_id"
    )
    .size()
    .to_dict()
)

for row in bydoc.itertuples(
    index=False
):
    doc_id = str(
        row.legacy_doc_id
    )

    expected_count = int(
        pair_counts.get(
            doc_id,
            0,
        )
    )

    frozen_count = int(
        row.n_verified_event_neighbors
    )

    if expected_count != frozen_count:
        raise RuntimeError(
            f"Neighbor-count mismatch for {doc_id}: "
            f"pair table={expected_count}, "
            f"article table={frozen_count}"
        )

    expected_available = int(
        expected_count > 0
    )

    if int(
        row.context_available
    ) != expected_available:
        raise RuntimeError(
            f"context_available mismatch for {doc_id}."
        )


# ============================================================
# BUILD PROTOTYPES
# ============================================================

prototype_target_ids = sorted(
    unique_targets
)

prototype_matrix = np.empty(
    (
        EXPECTED_TARGETS_WITH_CONTEXT,
        EXPECTED_DIM,
    ),
    dtype=np.float32,
)

prototype_manifest_rows = []

membership_rows = []

prototype_row_for_target = {}


for prototype_row_index, target_id in enumerate(
    prototype_target_ids
):

    g = pairs[
        pairs[
            "legacy_doc_id"
        ].eq(
            target_id
        )
    ].copy()

    background_ids = sorted(
        g[
            "background_doc_id"
        ].astype(str).tolist()
    )

    if not background_ids:
        raise RuntimeError(
            f"No verified neighbors for {target_id}."
        )

    emb_rows = [
        embedding_row_for_doc[
            bg
        ]
        for bg in background_ids
    ]

    selected = (
        embedding_matrix[
            emb_rows
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    prototype64 = selected.mean(
        axis=0,
        dtype=np.float64,
    )

    if prototype64.shape != (
        EXPECTED_DIM,
    ):
        raise RuntimeError(
            f"Unexpected prototype shape for {target_id}."
        )

    if not np.isfinite(
        prototype64
    ).all():
        raise RuntimeError(
            f"Non-finite prototype for {target_id}."
        )

    prototype32 = prototype64.astype(
        np.float32
    )

    prototype_norm = float(
        np.linalg.norm(
            prototype32
        )
    )

    if (
        not np.isfinite(
            prototype_norm
        )
        or prototype_norm <= 0
    ):
        raise RuntimeError(
            f"Invalid prototype norm for {target_id}."
        )

    prototype_matrix[
        prototype_row_index
    ] = prototype32

    prototype_row_for_target[
        target_id
    ] = prototype_row_index

    prototype_manifest_rows.append({
        "row_index":
            prototype_row_index,

        "legacy_doc_id":
            target_id,

        "n_verified_neighbors":
            len(background_ids),

        "prototype_norm":
            prototype_norm,
    })

    for bg in background_ids:
        membership_rows.append({
            "legacy_doc_id":
                target_id,

            "background_doc_id":
                bg,

            "prototype_row_index":
                prototype_row_index,

            "background_embedding_row_index":
                embedding_row_for_doc[
                    bg
                ],
        })


# ============================================================
# FINAL PROTOTYPE QC
# ============================================================

if prototype_matrix.shape != (
    EXPECTED_TARGETS_WITH_CONTEXT,
    EXPECTED_DIM,
):
    raise RuntimeError(
        "Final prototype matrix shape mismatch."
    )

if prototype_matrix.dtype != np.float32:
    raise RuntimeError(
        "Final prototype matrix dtype mismatch."
    )

if not np.isfinite(
    prototype_matrix
).all():
    raise RuntimeError(
        "Prototype matrix contains non-finite values."
    )

prototype_norms = np.linalg.norm(
    prototype_matrix,
    axis=1,
)

if (
    (~np.isfinite(
        prototype_norms
    )).any()
    or
    (prototype_norms <= 0).any()
):
    raise RuntimeError(
        "Invalid prototype norm detected."
    )


prototype_manifest = pd.DataFrame(
    prototype_manifest_rows
)

membership = pd.DataFrame(
    membership_rows
)


if len(
    prototype_manifest
) != EXPECTED_TARGETS_WITH_CONTEXT:
    raise RuntimeError(
        "Prototype manifest row count mismatch."
    )

if prototype_manifest[
    "legacy_doc_id"
].nunique() != EXPECTED_TARGETS_WITH_CONTEXT:
    raise RuntimeError(
        "Duplicate prototype target ID."
    )

if prototype_manifest[
    "row_index"
].tolist() != list(
    range(
        EXPECTED_TARGETS_WITH_CONTEXT
    )
):
    raise RuntimeError(
        "Prototype manifest row order mismatch."
    )

if prototype_manifest[
    "legacy_doc_id"
].tolist() != prototype_target_ids:
    raise RuntimeError(
        "Prototype target ordering mismatch."
    )

if len(
    membership
) != EXPECTED_PAIRS:
    raise RuntimeError(
        "Prototype membership row count mismatch."
    )

if membership[
    [
        "legacy_doc_id",
        "background_doc_id",
    ]
].duplicated().any():
    raise RuntimeError(
        "Duplicate membership pair detected."
    )


# ============================================================
# BUILD 199-ARTICLE AVAILABILITY RESOURCE
# ============================================================

availability = (
    bydoc[
        [
            "legacy_doc_id",
            "context_available",
            "n_verified_event_neighbors",
        ]
    ]
    .copy()
    .sort_values(
        "legacy_doc_id"
    )
    .reset_index(
        drop=True
    )
)

availability[
    "prototype_row_index"
] = availability[
    "legacy_doc_id"
].map(
    prototype_row_for_target
)

availability[
    "prototype_row_index"
] = availability[
    "prototype_row_index"
].astype(
    "Int64"
)


context_rows = availability[
    availability[
        "context_available"
    ].eq(1)
]

no_context_rows = availability[
    availability[
        "context_available"
    ].eq(0)
]


if len(
    context_rows
) != EXPECTED_TARGETS_WITH_CONTEXT:
    raise RuntimeError(
        "Context-available row count mismatch."
    )

if len(
    no_context_rows
) != EXPECTED_TARGETS_NO_CONTEXT:
    raise RuntimeError(
        "NO_CONTEXT row count mismatch."
    )

if context_rows[
    "prototype_row_index"
].isna().any():
    raise RuntimeError(
        "A context-available article lacks prototype_row_index."
    )

if no_context_rows[
    "prototype_row_index"
].notna().any():
    raise RuntimeError(
        "A NO_CONTEXT article received a prototype_row_index."
    )

valid_rows = set(
    range(
        EXPECTED_TARGETS_WITH_CONTEXT
    )
)

observed_rows = set(
    context_rows[
        "prototype_row_index"
    ]
    .astype(int)
)

if observed_rows != valid_rows:
    raise RuntimeError(
        "Prototype-row mapping is incomplete."
    )


# ============================================================
# ATOMIC WRITE
# ============================================================

TMP.mkdir(
    parents=True,
    exist_ok=False,
)


prototype_path = (
    TMP
    / "verified_event_prototypes_xlmr_v3.npy"
)

manifest_path = (
    TMP
    / "verified_event_prototype_manifest_v3.csv"
)

membership_path = (
    TMP
    / "verified_event_prototype_membership_v3.csv"
)

availability_path = (
    TMP
    / "context_availability_legacyaux199_v3.csv"
)

results_path = (
    TMP
    / "EVE_FRAME_STAGE_B_EVENT_PROTOTYPE_RESULTS_v3.json"
)


np.save(
    prototype_path,
    prototype_matrix,
    allow_pickle=False,
)

prototype_manifest.to_csv(
    manifest_path,
    index=False,
)

membership.to_csv(
    membership_path,
    index=False,
)

availability.to_csv(
    availability_path,
    index=False,
)


# ============================================================
# STORED-MATRIX RELOAD CHECK
# ============================================================

reloaded = np.load(
    prototype_path,
    allow_pickle=False,
)

if reloaded.shape != (
    EXPECTED_TARGETS_WITH_CONTEXT,
    EXPECTED_DIM,
):
    raise RuntimeError(
        "Stored prototype shape mismatch."
    )

if reloaded.dtype != np.float32:
    raise RuntimeError(
        "Stored prototype dtype mismatch."
    )

if not np.array_equal(
    prototype_matrix,
    reloaded,
):
    raise RuntimeError(
        "Stored prototype matrix differs "
        "from in-memory matrix."
    )


# ============================================================
# RESULTS JSON
# ============================================================

neighbor_counts = (
    prototype_manifest[
        "n_verified_neighbors"
    ]
    .astype(int)
    .to_numpy()
)


results = {
    "status":
        "VERIFIED_EVENT_PROTOTYPE_CONSTRUCTION_COMPLETE",

    "aggregation":
        "uniform_arithmetic_mean_all_verified_neighbors",

    "input_accumulation_dtype":
        "float64",

    "stored_prototype_dtype":
        "float32",

    "prototype_shape":
        [
            int(x)
            for x in prototype_matrix.shape
        ],

    "verified_pairs":
        EXPECTED_PAIRS,

    "unique_background_documents":
        EXPECTED_BACKGROUND_DOCS,

    "context_available_articles":
        EXPECTED_TARGETS_WITH_CONTEXT,

    "no_context_articles":
        EXPECTED_TARGETS_NO_CONTEXT,

    "total_legacyaux_articles":
        EXPECTED_TARGETS_TOTAL,

    "prototype_norm_min":
        float(
            prototype_norms.min()
        ),

    "prototype_norm_median":
        float(
            np.median(
                prototype_norms
            )
        ),

    "prototype_norm_mean":
        float(
            prototype_norms.mean()
        ),

    "prototype_norm_max":
        float(
            prototype_norms.max()
        ),

    "verified_neighbors_min":
        int(
            neighbor_counts.min()
        ),

    "verified_neighbors_median":
        float(
            np.median(
                neighbor_counts
            )
        ),

    "verified_neighbors_mean":
        float(
            neighbor_counts.mean()
        ),

    "verified_neighbors_max":
        int(
            neighbor_counts.max()
        ),

    "verifier_probability_weighting":
        False,

    "cosine_weighting":
        False,

    "temporal_weighting":
        False,

    "top_k":
        None,

    "learned_attention":
        False,

    "nearest_neighbor_fallback":
        False,

    "same_topic_fallback":
        False,

    "main_task_labels_used":
        False,

    "eventgold_status":
        "SEALED_NOT_ACCESSED",

    "input_sha256": {
        "verified_event_neighbor_pairs":
            file_sha256(
                PAIRFILE
            ),

        "verified_context_by_doc":
            file_sha256(
                BYDOC
            ),

        "context_embedding_matrix":
            file_sha256(
                EMBED_MATRIX
            ),

        "context_embedding_manifest":
            file_sha256(
                EMBED_MANIFEST
            ),

        "context_embedding_closure":
            file_sha256(
                CONTEXT_EMBED_CLOSURE
            ),

        "prototype_protocol":
            file_sha256(
                PROTOCOL
            ),

        "architecture_protocol":
            file_sha256(
                ARCH_PROTOCOL
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


# ============================================================
# CHECKSUM MANIFEST
# ============================================================

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
    / "EVE_FRAME_STAGE_B_EVENT_PROTOTYPE_SHA256SUMS_v3.txt"
)

checksum_path.write_text(
    "".join(
        f"{file_sha256(p)}  {p.name}\n"
        for p in files_to_hash
    ),
    encoding="utf-8",
)


# ============================================================
# FINALIZE
# ============================================================

TMP.rename(
    OUT
)


# ============================================================
# REPORT
# ============================================================

print("=" * 100)
print(
    "EVE-FRAME STAGE B VERIFIED EVENT PROTOTYPES COMPLETE"
)
print("=" * 100)

print()
print(
    "Verified pairs:",
    EXPECTED_PAIRS,
)

print(
    "Unique background embeddings:",
    EXPECTED_BACKGROUND_DOCS,
)

print(
    "Context-available LegacyAux:",
    EXPECTED_TARGETS_WITH_CONTEXT,
)

print(
    "NO_CONTEXT LegacyAux:",
    EXPECTED_TARGETS_NO_CONTEXT,
)

print()
print(
    "Prototype matrix shape:",
    prototype_matrix.shape,
)

print(
    "Prototype dtype:",
    prototype_matrix.dtype,
)

print()
print(
    "Prototype norm min:",
    float(
        prototype_norms.min()
    ),
)

print(
    "Prototype norm median:",
    float(
        np.median(
            prototype_norms
        )
    ),
)

print(
    "Prototype norm mean:",
    float(
        prototype_norms.mean()
    ),
)

print(
    "Prototype norm max:",
    float(
        prototype_norms.max()
    ),
)

print()
print(
    "Verified neighbors min:",
    int(
        neighbor_counts.min()
    ),
)

print(
    "Verified neighbors median:",
    float(
        np.median(
            neighbor_counts
        )
    ),
)

print(
    "Verified neighbors mean:",
    float(
        neighbor_counts.mean()
    ),
)

print(
    "Verified neighbors max:",
    int(
        neighbor_counts.max()
    ),
)

print()
print(
    "Prototype SHA256:",
    file_sha256(
        OUT
        / "verified_event_prototypes_xlmr_v3.npy"
    ),
)

print(
    "Manifest SHA256:",
    file_sha256(
        OUT
        / "verified_event_prototype_manifest_v3.csv"
    ),
)

print(
    "Availability SHA256:",
    file_sha256(
        OUT
        / "context_availability_legacyaux199_v3.csv"
    ),
)

print(
    "Results SHA256:",
    file_sha256(
        OUT
        / "EVE_FRAME_STAGE_B_EVENT_PROTOTYPE_RESULTS_v3.json"
    ),
)

print()
print(
    "✅ Uniform aggregation only."
)

print(
    "✅ No verifier-probability weighting."
)

print(
    "✅ No top-k truncation."
)

print(
    "✅ 51 context prototypes constructed."
)

print(
    "✅ 148 articles preserved as NO_CONTEXT."
)

print(
    "✅ Main-task labels not used."
)

print(
    "✅ EventGold not accessed."
)
