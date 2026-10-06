from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path.cwd()

STAGEB = (
    ROOT
    / "01_event_aware_v3/08_eve_frame_stage_b_event_context"
)

VERROOT = (
    STAGEB
    / "event_verifier_v3"
)

COVERAGE = (
    STAGEB
    / "coverage_v3"
)

PAIRS = (
    COVERAGE
    / "event_context_neighbor_pairs_v3.csv"
)

COVERAGE_HASHES = (
    COVERAGE
    / "EVENT_CONTEXT_COVERAGE_SHA256SUMS_v3.txt"
)

LEGACY = (
    ROOT
    / "01_event_aware_v3/03_scalegold_active_learning/"
      "LegacyAux_leakage_free_v3.csv"
)

MANIFEST = (
    ROOT
    / "00_redesign_v2/01_manifest/"
      "document_manifest_v2_draft.csv"
)

EMB = (
    ROOT
    / "models/embeddings/"
      "full_paraphrase_multilingual_mpnet_embeddings_v1.npy"
)

MODEL = (
    VERROOT
    / "model_dev_v3/cv_v3/"
      "event_verifier_DEV80_frozen_model_v3.joblib"
)

VERIFIER_CLOSURE = (
    VERROOT
    / "EVENT_VERIFIER_FINAL_CLOSURE_v3.txt"
)

PROTOCOL = (
    STAGEB
    / "EVE_FRAME_STAGE_B_VERIFIED_CONTEXT_PROTOCOL_v3.txt"
)

OUT = (
    STAGEB
    / "verified_context_v3"
)

TMP = (
    STAGEB
    / "verified_context_v3_tmp"
)


EXPECTED = {
    "protocol":
        "2791d8699e2b2157b584c7b883768ff7bb7f96b0cff1716dc5b6372bf19d96c3",

    "verifier_closure":
        "d19165aa7dfdbe62744504800ff34c2a6aa44851fc311ba010bc34bb467f984d",

    "verifier_model":
        "7c4e162234d48dfd40293116ca7af59de7e41a8d56982cb700643e1ccff9066a",

    "embedding":
        "b02eff9b802bdfd59f1a719b0023f9d05b576805819e235d0071d783621c8bed",
}


THRESHOLD = 0.5


def sha256(path):
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
    observed = sha256(path)

    if observed != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {observed}"
        )


def checksum_from_manifest(
    checksum_file,
    filename,
):
    for line in Path(
        checksum_file
    ).read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split(
            None,
            1,
        )

        if len(parts) != 2:
            continue

        digest, name = parts

        name = name.strip()

        if name == filename:
            return digest

    raise RuntimeError(
        f"Could not find {filename} "
        f"in checksum manifest."
    )


def norm_string(x):
    if pd.isna(x):
        return ""

    return str(x).strip()


# ============================================================
# OUTPUT GUARD
# ============================================================

if OUT.exists():
    raise RuntimeError(
        f"Final verified-context output already exists: {OUT}"
    )

if TMP.exists():
    raise RuntimeError(
        f"Temporary verified-context output already exists: {TMP}"
    )


# ============================================================
# VERIFY FROZEN INPUTS
# ============================================================

require_hash(
    PROTOCOL,
    EXPECTED["protocol"],
    "Verified-context protocol",
)

require_hash(
    VERIFIER_CLOSURE,
    EXPECTED["verifier_closure"],
    "Event-verifier closure",
)

require_hash(
    MODEL,
    EXPECTED["verifier_model"],
    "Frozen event verifier",
)

require_hash(
    EMB,
    EXPECTED["embedding"],
    "MPNet embeddings",
)


expected_pairs_sha = checksum_from_manifest(
    COVERAGE_HASHES,
    PAIRS.name,
)

require_hash(
    PAIRS,
    expected_pairs_sha,
    "Frozen coverage neighbor-pair table",
)


# ============================================================
# LOAD DATA
# ============================================================

pairs = pd.read_csv(
    PAIRS,
    low_memory=False,
)

legacy = pd.read_csv(
    LEGACY,
    low_memory=False,
)

manifest = pd.read_csv(
    MANIFEST,
    low_memory=False,
)

emb = np.load(
    EMB,
    mmap_mode="r",
)

payload = joblib.load(
    MODEL
)


# ============================================================
# VERIFY VERIFIER PAYLOAD
# ============================================================

if (
    payload["family"]
    != "SYMMETRIC_MPNET"
):
    raise RuntimeError(
        "Frozen verifier family mismatch."
    )

if not np.isclose(
    float(
        payload["C"]
    ),
    0.0001,
):
    raise RuntimeError(
        "Frozen verifier C mismatch."
    )

if not np.isclose(
    float(
        payload["threshold"]
    ),
    THRESHOLD,
):
    raise RuntimeError(
        "Frozen verifier threshold mismatch."
    )

if sha256(
    MANIFEST
) != payload[
    "manifest_sha256"
]:
    raise RuntimeError(
        "Manifest differs from the manifest "
        "used by the frozen event verifier."
    )


pipe = payload[
    "pipeline"
]


# ============================================================
# BASIC INPUT QC
# ============================================================

for df in [
    pairs,
    legacy,
    manifest,
]:
    if "doc_id" in df.columns:
        df["doc_id"] = (
            df["doc_id"]
            .astype(str)
        )


pairs[
    "legacy_doc_id"
] = (
    pairs[
        "legacy_doc_id"
    ]
    .astype(str)
)

pairs[
    "background_doc_id"
] = (
    pairs[
        "background_doc_id"
    ]
    .astype(str)
)


if len(
    legacy
) != 199:
    raise RuntimeError(
        f"Expected LegacyAux199, got {len(legacy)}."
    )

if legacy[
    "doc_id"
].nunique() != 199:
    raise RuntimeError(
        "LegacyAux doc_id not unique."
    )

if (
    len(manifest) != 1537
    or emb.shape != (
        1537,
        768,
    )
):
    raise RuntimeError(
        "Unexpected manifest/embedding dimensions."
    )


broad = pairs[
    pairs[
        "regime"
    ]
    .astype(str)
    .eq(
        "BROAD"
    )
].copy()


if len(
    broad
) != 454:
    raise RuntimeError(
        f"Expected exactly 454 BROAD candidate pairs, "
        f"got {len(broad)}."
    )


if broad[
    [
        "legacy_doc_id",
        "background_doc_id",
    ]
].duplicated().any():
    raise RuntimeError(
        "Duplicate BROAD article pair detected."
    )


legacy_ids = set(
    legacy[
        "doc_id"
    ].astype(str)
)


if not set(
    broad[
        "legacy_doc_id"
    ]
).issubset(
    legacy_ids
):
    raise RuntimeError(
        "A BROAD target is not in LegacyAux."
    )


if broad[
    "legacy_doc_id"
].nunique() != 103:
    raise RuntimeError(
        "Expected 103 LegacyAux articles "
        "with BROAD candidates."
    )


# ============================================================
# RECOMPUTE FROZEN VERIFIER FEATURES
# ============================================================

manifest = manifest.reset_index(
    drop=True
)

row_for_doc = {
    d: i
    for i, d
    in enumerate(
        manifest[
            "doc_id"
        ]
    )
}


all_needed = set(
    broad[
        "legacy_doc_id"
    ]
).union(
    set(
        broad[
            "background_doc_id"
        ]
    )
)


missing = (
    all_needed
    - set(
        row_for_doc
    )
)

if missing:
    raise RuntimeError(
        f"{len(missing)} pair documents "
        "missing from embedding manifest."
    )


a_rows = np.asarray(
    [
        row_for_doc[
            x
        ]
        for x
        in broad[
            "legacy_doc_id"
        ]
    ],
    dtype=int,
)

b_rows = np.asarray(
    [
        row_for_doc[
            x
        ]
        for x
        in broad[
            "background_doc_id"
        ]
    ],
    dtype=int,
)


u = np.asarray(
    emb[
        a_rows
    ],
    dtype=np.float64,
)

v = np.asarray(
    emb[
        b_rows
    ],
    dtype=np.float64,
)


u_norm = np.linalg.norm(
    u,
    axis=1,
    keepdims=True,
)

v_norm = np.linalg.norm(
    v,
    axis=1,
    keepdims=True,
)


if (
    (u_norm <= 0).any()
    or
    (v_norm <= 0).any()
):
    raise RuntimeError(
        "Zero-norm embedding detected."
    )


u = (
    u
    / u_norm
)

v = (
    v
    / v_norm
)


cosine = (
    u
    * v
).sum(
    axis=1
)


stored_cosine = (
    broad[
        "cosine_similarity"
    ]
    .astype(float)
    .to_numpy()
)


max_cosine_error = float(
    np.max(
        np.abs(
            cosine
            - stored_cosine
        )
    )
)


if max_cosine_error > 1e-5:
    raise RuntimeError(
        "Recomputed cosine does not match "
        "the frozen BROAD candidate table. "
        f"Max absolute error={max_cosine_error}"
    )


delta_days = (
    broad[
        "delta_days"
    ]
    .astype(float)
    .to_numpy()
)


if (
    (delta_days < 0).any()
    or
    (delta_days > 14).any()
):
    raise RuntimeError(
        "BROAD temporal constraint violated."
    )


X = np.column_stack(
    [
        np.abs(
            u - v
        ),
        u * v,
        cosine,
        np.log1p(
            delta_days
        ),
    ]
)


if X.shape != (
    454,
    1538,
):
    raise RuntimeError(
        f"Unexpected verifier feature shape: {X.shape}"
    )


if not np.isfinite(
    X
).all():
    raise RuntimeError(
        "Non-finite verifier feature detected."
    )


# ============================================================
# APPLY FROZEN VERIFIER
# ============================================================

prob = (
    pipe
    .predict_proba(
        X
    )[
        :,
        1
    ]
)


verified = (
    prob
    >= THRESHOLD
)


scored = broad.copy()

scored[
    "verifier_prob_same_event"
] = prob

scored[
    "verified_event_neighbor"
] = verified.astype(
    int
)


scored = scored.sort_values(
    [
        "legacy_doc_id",
        "verified_event_neighbor",
        "verifier_prob_same_event",
        "delta_days",
        "background_doc_id",
    ],
    ascending=[
        True,
        False,
        False,
        True,
        True,
    ],
).reset_index(
    drop=True
)


verified_pairs = scored[
    scored[
        "verified_event_neighbor"
    ]
    .eq(
        1
    )
].copy()


# ============================================================
# ARTICLE-LEVEL VERIFIED CONTEXT
# ============================================================

doc_rows = []


legacy_order = (
    legacy[
        "doc_id"
    ]
    .astype(str)
    .tolist()
)


for doc_id in legacy_order:

    g = scored[
        scored[
            "legacy_doc_id"
        ]
        .eq(
            doc_id
        )
    ]

    vg = g[
        g[
            "verified_event_neighbor"
        ]
        .eq(
            1
        )
    ]


    n_candidates = int(
        len(
            g
        )
    )

    n_verified = int(
        len(
            vg
        )
    )


    if n_candidates:

        max_candidate_prob = float(
            g[
                "verifier_prob_same_event"
            ].max()
        )

    else:

        max_candidate_prob = np.nan


    if n_verified:

        verified_probs = (
            vg[
                "verifier_prob_same_event"
            ]
            .astype(float)
        )

        max_verified_prob = float(
            verified_probs.max()
        )

        mean_verified_prob = float(
            verified_probs.mean()
        )

        median_verified_prob = float(
            verified_probs.median()
        )

        sources = {
            norm_string(x)
            for x in vg[
                "background_source_domain"
            ]
            if norm_string(x)
        }

        countries = {
            norm_string(x)
            for x in vg[
                "background_country"
            ]
            if norm_string(x)
        }

        languages = {
            norm_string(x)
            for x in vg[
                "background_language"
            ]
            if norm_string(x)
        }


        cross_source = (
            vg.apply(
                lambda r:
                    norm_string(
                        r[
                            "legacy_source_domain"
                        ]
                    )
                    !=
                    norm_string(
                        r[
                            "background_source_domain"
                        ]
                    ),
                axis=1,
            )
        )

        cross_country = (
            vg.apply(
                lambda r:
                    norm_string(
                        r[
                            "legacy_country"
                        ]
                    )
                    !=
                    norm_string(
                        r[
                            "background_country"
                        ]
                    ),
                axis=1,
            )
        )

        cross_language = (
            vg.apply(
                lambda r:
                    norm_string(
                        r[
                            "legacy_language"
                        ]
                    )
                    !=
                    norm_string(
                        r[
                            "background_language"
                        ]
                    ),
                axis=1,
            )
        )


        n_cross_source = int(
            cross_source.sum()
        )

        n_cross_country = int(
            cross_country.sum()
        )

        n_cross_language = int(
            cross_language.sum()
        )

        n_unique_sources = len(
            sources
        )

        n_unique_countries = len(
            countries
        )

        n_unique_languages = len(
            languages
        )

    else:

        max_verified_prob = np.nan
        mean_verified_prob = np.nan
        median_verified_prob = np.nan

        n_cross_source = 0
        n_cross_country = 0
        n_cross_language = 0

        n_unique_sources = 0
        n_unique_countries = 0
        n_unique_languages = 0


    doc_rows.append({
        "legacy_doc_id":
            doc_id,

        "n_broad_candidates":
            n_candidates,

        "n_verified_event_neighbors":
            n_verified,

        "context_available":
            int(
                n_verified > 0
            ),

        "max_candidate_verifier_probability":
            max_candidate_prob,

        "max_verified_probability":
            max_verified_prob,

        "mean_verified_probability":
            mean_verified_prob,

        "median_verified_probability":
            median_verified_prob,

        "n_unique_verified_sources":
            n_unique_sources,

        "n_unique_verified_countries":
            n_unique_countries,

        "n_unique_verified_languages":
            n_unique_languages,

        "n_verified_cross_source_neighbors":
            n_cross_source,

        "n_verified_cross_country_neighbors":
            n_cross_country,

        "n_verified_cross_language_neighbors":
            n_cross_language,
    })


by_doc = pd.DataFrame(
    doc_rows
)


if len(
    by_doc
) != 199:
    raise RuntimeError(
        "Expected exactly 199 article-level rows."
    )


# ============================================================
# AGGREGATE SUMMARY
# ============================================================

n_verified_pairs = int(
    len(
        verified_pairs
    )
)


n_ge1 = int(
    (
        by_doc[
            "n_verified_event_neighbors"
        ]
        >= 1
    ).sum()
)

n_ge2 = int(
    (
        by_doc[
            "n_verified_event_neighbors"
        ]
        >= 2
    ).sum()
)

n_ge3 = int(
    (
        by_doc[
            "n_verified_event_neighbors"
        ]
        >= 3
    ).sum()
)

n_ge5 = int(
    (
        by_doc[
            "n_verified_event_neighbors"
        ]
        >= 5
    ).sum()
)

n_no_context = int(
    (
        by_doc[
            "context_available"
        ]
        == 0
    ).sum()
)


cross_source_articles = int(
    (
        by_doc[
            "n_verified_cross_source_neighbors"
        ]
        > 0
    ).sum()
)

cross_country_articles = int(
    (
        by_doc[
            "n_verified_cross_country_neighbors"
        ]
        > 0
    ).sum()
)

cross_language_articles = int(
    (
        by_doc[
            "n_verified_cross_language_neighbors"
        ]
        > 0
    ).sum()
)


summary_row = {
    "n_legacy_articles":
        199,

    "broad_candidate_pairs":
        454,

    "legacy_articles_with_broad_candidates":
        103,

    "verified_candidate_pairs":
        n_verified_pairs,

    "candidate_acceptance_rate":
        float(
            n_verified_pairs
            / 454
        ),

    "articles_ge1_verified_neighbor":
        n_ge1,

    "verified_context_coverage_ge1":
        float(
            n_ge1
            / 199
        ),

    "articles_ge2_verified_neighbors":
        n_ge2,

    "verified_context_coverage_ge2":
        float(
            n_ge2
            / 199
        ),

    "articles_ge3_verified_neighbors":
        n_ge3,

    "verified_context_coverage_ge3":
        float(
            n_ge3
            / 199
        ),

    "articles_ge5_verified_neighbors":
        n_ge5,

    "verified_context_coverage_ge5":
        float(
            n_ge5
            / 199
        ),

    "articles_no_context":
        n_no_context,

    "no_context_fraction":
        float(
            n_no_context
            / 199
        ),

    "articles_with_verified_cross_source_neighbor":
        cross_source_articles,

    "verified_cross_source_coverage":
        float(
            cross_source_articles
            / 199
        ),

    "articles_with_verified_cross_country_neighbor":
        cross_country_articles,

    "verified_cross_country_coverage":
        float(
            cross_country_articles
            / 199
        ),

    "articles_with_verified_cross_language_neighbor":
        cross_language_articles,

    "verified_cross_language_coverage":
        float(
            cross_language_articles
            / 199
        ),

    "verified_neighbor_count_mean_all_articles":
        float(
            by_doc[
                "n_verified_event_neighbors"
            ].mean()
        ),

    "verified_neighbor_count_median_all_articles":
        float(
            by_doc[
                "n_verified_event_neighbors"
            ].median()
        ),

    "verified_neighbor_count_max":
        int(
            by_doc[
                "n_verified_event_neighbors"
            ].max()
        ),

    "candidate_probability_mean":
        float(
            scored[
                "verifier_prob_same_event"
            ].mean()
        ),

    "candidate_probability_median":
        float(
            scored[
                "verifier_prob_same_event"
            ].median()
        ),

    "candidate_probability_min":
        float(
            scored[
                "verifier_prob_same_event"
            ].min()
        ),

    "candidate_probability_max":
        float(
            scored[
                "verifier_prob_same_event"
            ].max()
        ),

    "verified_probability_mean":
        (
            float(
                verified_pairs[
                    "verifier_prob_same_event"
                ].mean()
            )
            if n_verified_pairs
            else np.nan
        ),

    "verified_probability_median":
        (
            float(
                verified_pairs[
                    "verifier_prob_same_event"
                ].median()
            )
            if n_verified_pairs
            else np.nan
        ),

    "max_recomputed_vs_stored_cosine_error":
        max_cosine_error,
}


summary_df = pd.DataFrame(
    [
        summary_row
    ]
)


# ============================================================
# SAVE ATOMICALLY
# ============================================================

TMP.mkdir(
    parents=True,
    exist_ok=False,
)


scored.to_csv(
    TMP
    / "verified_context_all_broad_candidates_v3.csv",
    index=False,
)


verified_pairs.to_csv(
    TMP
    / "verified_event_neighbor_pairs_v3.csv",
    index=False,
)


by_doc.to_csv(
    TMP
    / "verified_context_by_legacyaux_doc_v3.csv",
    index=False,
)


summary_df.to_csv(
    TMP
    / "verified_context_summary_v3.csv",
    index=False,
)


summary_json = {
    "status":
        "VERIFIED_CONTEXT_AUDIT_COMPLETE",

    "event_verifier": {
        "family":
            "SYMMETRIC_MPNET",

        "C":
            0.0001,

        "threshold":
            THRESHOLD,

        "model_sha256":
            sha256(
                MODEL
            ),

        "closure_sha256":
            sha256(
                VERIFIER_CLOSURE
            ),
    },

    "retrieval": {
        "rule":
            "cosine >= 0.68 AND absolute delta_days <= 14",

        "candidate_pairs":
            454,

        "candidate_target_articles":
            103,

        "frozen_pair_table_sha256":
            sha256(
                PAIRS
            ),

        "coverage_checksum_manifest_sha256":
            sha256(
                COVERAGE_HASHES
            ),
    },

    "legacyaux": {
        "n":
            199,

        "sha256":
            sha256(
                LEGACY
            ),
    },

    "verified_context":
        summary_row,

    "main_task_labels_used":
        False,

    "topic_used":
        False,

    "eventgold_status":
        "SEALED_NOT_ACCESSED",

    "protocol_sha256":
        sha256(
            PROTOCOL
        ),

    "embedding_sha256":
        sha256(
            EMB
        ),

    "manifest_sha256":
        sha256(
            MANIFEST
        ),
}


(
    TMP
    / "EVE_FRAME_STAGE_B_VERIFIED_CONTEXT_RESULTS_v3.json"
).write_text(
    json.dumps(
        summary_json,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


files = sorted(
    p
    for p in TMP.iterdir()
    if p.is_file()
)


hash_file = (
    TMP
    / "EVE_FRAME_STAGE_B_VERIFIED_CONTEXT_SHA256SUMS_v3.txt"
)


hash_file.write_text(
    "".join(
        f"{sha256(p)}  {p.name}\n"
        for p in files
    ),
    encoding="utf-8",
)


TMP.rename(
    OUT
)


# ============================================================
# REPORT
# ============================================================

print("=" * 100)
print(
    "EVE-FRAME STAGE B VERIFIED-CONTEXT AUDIT COMPLETE"
)
print("=" * 100)

print()
print(
    "LegacyAux articles:",
    199,
)

print(
    "BROAD candidate pairs:",
    454,
)

print(
    "LegacyAux with BROAD candidates:",
    103,
)

print()
print(
    "Verified event-neighbor pairs:",
    n_verified_pairs,
)

print(
    "Candidate acceptance rate:",
    summary_row[
        "candidate_acceptance_rate"
    ],
)

print()
print(
    "Articles >=1 verified neighbor:",
    n_ge1,
    "/ 199 =",
    summary_row[
        "verified_context_coverage_ge1"
    ],
)

print(
    "Articles >=2 verified neighbors:",
    n_ge2,
    "/ 199 =",
    summary_row[
        "verified_context_coverage_ge2"
    ],
)

print(
    "Articles >=3 verified neighbors:",
    n_ge3,
    "/ 199 =",
    summary_row[
        "verified_context_coverage_ge3"
    ],
)

print(
    "Articles >=5 verified neighbors:",
    n_ge5,
    "/ 199 =",
    summary_row[
        "verified_context_coverage_ge5"
    ],
)

print()
print(
    "NO_CONTEXT:",
    n_no_context,
    "/ 199 =",
    summary_row[
        "no_context_fraction"
    ],
)

print()
print(
    "Verified cross-source coverage:",
    cross_source_articles,
    "/ 199 =",
    summary_row[
        "verified_cross_source_coverage"
    ],
)

print(
    "Verified cross-country coverage:",
    cross_country_articles,
    "/ 199 =",
    summary_row[
        "verified_cross_country_coverage"
    ],
)

print(
    "Verified cross-language coverage:",
    cross_language_articles,
    "/ 199 =",
    summary_row[
        "verified_cross_language_coverage"
    ],
)

print()
print(
    "Candidate probability median:",
    summary_row[
        "candidate_probability_median"
    ],
)

print(
    "Verified probability median:",
    summary_row[
        "verified_probability_median"
    ],
)

print(
    "Max embedding cosine reconstruction error:",
    max_cosine_error,
)

print()
print(
    "Results SHA256:",
    sha256(
        OUT
        / "EVE_FRAME_STAGE_B_VERIFIED_CONTEXT_RESULTS_v3.json"
    ),
)

print(
    "By-document SHA256:",
    sha256(
        OUT
        / "verified_context_by_legacyaux_doc_v3.csv"
    ),
)

print()
print(
    "✅ Frozen event verifier used unchanged."
)

print(
    "✅ Threshold remained 0.5."
)

print(
    "✅ No main-task labels used."
)

print(
    "✅ No topic labels used."
)

print(
    "✅ No unverified neighbor forced into context."
)

print(
    "✅ EventGold not accessed."
)
