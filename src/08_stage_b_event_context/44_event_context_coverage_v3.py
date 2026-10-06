from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path.cwd()

CLEAN = (
    ROOT
    / "01_event_aware_v3/01_data_audit/"
      "Corpus1502_document_holdout_v3.csv"
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

PROTOCOL = (
    ROOT
    / "01_event_aware_v3/08_eve_frame_stage_b_event_context/"
      "EVENT_CONTEXT_COVERAGE_PROTOCOL_v3.txt"
)

OUTDIR = (
    ROOT
    / "01_event_aware_v3/08_eve_frame_stage_b_event_context/"
      "coverage_v3"
)

TMPDIR = (
    ROOT
    / "01_event_aware_v3/08_eve_frame_stage_b_event_context/"
      "coverage_v3_tmp"
)

REGIMES = {
    "BROAD": {
        "max_days": 14,
        "min_similarity": 0.68,
    },
    "STRICT": {
        "max_days": 7,
        "min_similarity": 0.75,
    },
}

TOP_K_INSPECTION = 10


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def normalize_text(s):
    return (
        s.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )


if OUTDIR.exists():
    raise RuntimeError(
        f"Final output directory already exists: {OUTDIR}"
    )

if TMPDIR.exists():
    raise RuntimeError(
        f"Temporary output directory already exists: {TMPDIR}"
    )

TMPDIR.mkdir(
    parents=True,
    exist_ok=False,
)


# ============================================================
# LOAD INPUTS
# ============================================================

clean = pd.read_csv(
    CLEAN,
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


for df in [
    clean,
    legacy,
    manifest,
]:
    df["doc_id"] = (
        df["doc_id"]
        .astype(str)
    )


if len(clean) != 1502:
    raise RuntimeError(
        f"Expected Clean1502, got {len(clean)}"
    )

if clean["doc_id"].nunique() != 1502:
    raise RuntimeError(
        "Clean1502 doc_id not unique."
    )

if len(legacy) != 199:
    raise RuntimeError(
        f"Expected LegacyAux199, got {len(legacy)}"
    )

if legacy["doc_id"].nunique() != 199:
    raise RuntimeError(
        "LegacyAux doc_id not unique."
    )

if len(manifest) != 1537:
    raise RuntimeError(
        f"Expected manifest1537, got {len(manifest)}"
    )

if emb.shape != (1537, 768):
    raise RuntimeError(
        f"Unexpected embedding shape: {emb.shape}"
    )

if len(manifest) != emb.shape[0]:
    raise RuntimeError(
        "Manifest / embedding row-count mismatch."
    )

if (
    clean["doc_id"].isin(
        manifest["doc_id"]
    ).sum()
    != 1502
):
    raise RuntimeError(
        "Clean corpus not fully represented in manifest."
    )

if (
    legacy["doc_id"].isin(
        clean["doc_id"]
    ).sum()
    != 199
):
    raise RuntimeError(
        "LegacyAux is not a subset of Clean1502."
    )


# The 1537 -> 1502 difference must remain untouched.
if len(manifest) - len(clean) != 35:
    raise RuntimeError(
        "Expected 35-document holdout difference."
    )


# ============================================================
# PUBLICATION DATES
# ============================================================

clean["_date"] = pd.to_datetime(
    clean["publication_date"],
    errors="coerce",
    utc=True,
)

if clean["_date"].isna().any():
    raise RuntimeError(
        "Unparseable publication date."
    )


# ============================================================
# BUILD CLEAN BACKGROUND
# ============================================================

legacy_ids = set(
    legacy["doc_id"]
)

background = clean[
    ~clean["doc_id"].isin(
        legacy_ids
    )
].copy()

if len(background) != 1303:
    raise RuntimeError(
        f"Expected raw background1303, "
        f"got {len(background)}"
    )


legacy_clean = clean[
    clean["doc_id"].isin(
        legacy_ids
    )
].copy()

if len(legacy_clean) != 199:
    raise RuntimeError(
        "Could not recover 199 LegacyAux docs "
        "inside Clean1502."
    )


title_col = (
    "title_clean"
    if "title_clean" in clean.columns
    else "title"
)

text_col = (
    "text_clean"
    if "text_clean" in clean.columns
    else "text"
)


background["_title_norm"] = normalize_text(
    background[title_col]
)

legacy_clean["_title_norm"] = normalize_text(
    legacy_clean[title_col]
)

background["_text_norm"] = normalize_text(
    background[text_col]
)

legacy_clean["_text_norm"] = normalize_text(
    legacy_clean[text_col]
)


legacy_titles = set(
    x
    for x in legacy_clean["_title_norm"]
    if x
)

legacy_texts = set(
    x
    for x in legacy_clean["_text_norm"]
    if x
)


background[
    "_duplicate_legacy"
] = (
    background["_title_norm"].isin(
        legacy_titles
    )
    |
    background["_text_norm"].isin(
        legacy_texts
    )
)


n_duplicate_excluded = int(
    background[
        "_duplicate_legacy"
    ].sum()
)

background = background[
    ~background[
        "_duplicate_legacy"
    ]
].copy()


if len(background) != 1254:
    raise RuntimeError(
        f"Expected final background1254, "
        f"got {len(background)}"
    )


# ============================================================
# EMBEDDING ROW ALIGNMENT
# ============================================================

manifest = manifest.reset_index(
    drop=True
)

row_for_doc = {
    doc_id: i
    for i, doc_id
    in enumerate(
        manifest["doc_id"]
    )
}


legacy_rows = np.asarray(
    [
        row_for_doc[x]
        for x
        in legacy_clean["doc_id"]
    ],
    dtype=int,
)

background_rows = np.asarray(
    [
        row_for_doc[x]
        for x
        in background["doc_id"]
    ],
    dtype=int,
)


# Materialize only needed vectors.
legacy_vec = np.asarray(
    emb[
        legacy_rows
    ],
    dtype=np.float32,
)

background_vec = np.asarray(
    emb[
        background_rows
    ],
    dtype=np.float32,
)


legacy_norm = np.linalg.norm(
    legacy_vec,
    axis=1,
    keepdims=True,
)

background_norm = np.linalg.norm(
    background_vec,
    axis=1,
    keepdims=True,
)


if (
    (legacy_norm <= 0).any()
    or
    (background_norm <= 0).any()
):
    raise RuntimeError(
        "Zero-norm embedding detected."
    )


legacy_unit = (
    legacy_vec
    / legacy_norm
)

background_unit = (
    background_vec
    / background_norm
)


# 199 x 1254 cosine matrix.
similarity = (
    legacy_unit
    @ background_unit.T
)


if similarity.shape != (
    199,
    1254,
):
    raise RuntimeError(
        f"Unexpected similarity shape: "
        f"{similarity.shape}"
    )


if not np.isfinite(
    similarity
).all():
    raise RuntimeError(
        "Non-finite cosine similarity."
    )


# ============================================================
# METADATA ARRAYS
# ============================================================

legacy_clean = legacy_clean.reset_index(
    drop=True
)

background = background.reset_index(
    drop=True
)


legacy_dates = (
    legacy_clean["_date"]
    .dt.normalize()
)

background_dates = (
    background["_date"]
    .dt.normalize()
)


legacy_days = (
    legacy_dates.astype(
        "int64"
    )
    // 86_400_000_000_000
).to_numpy()

background_days = (
    background_dates.astype(
        "int64"
    )
    // 86_400_000_000_000
).to_numpy()


delta_days = np.abs(
    legacy_days[
        :,
        None
    ]
    -
    background_days[
        None,
        :
    ]
)


if delta_days.shape != (
    199,
    1254,
):
    raise RuntimeError(
        "Unexpected temporal-distance matrix."
    )


# ============================================================
# COVERAGE
# ============================================================

doc_rows = []
neighbor_rows = []
top_rows = []


for i in range(
    len(
        legacy_clean
    )
):

    q = legacy_clean.iloc[
        i
    ]

    sim_i = similarity[
        i
    ]

    days_i = delta_days[
        i
    ]

    # Top inspection records constrained only by broad
    # temporal horizon, without semantic threshold.
    temporal_candidates = np.flatnonzero(
        days_i <= 14
    )

    if len(
        temporal_candidates
    ):
        order = temporal_candidates[
            np.argsort(
                -sim_i[
                    temporal_candidates
                ]
            )
        ][:TOP_K_INSPECTION]

        for rank, j in enumerate(
            order,
            start=1,
        ):

            b = background.iloc[
                int(j)
            ]

            top_rows.append({
                "legacy_doc_id":
                    q["doc_id"],

                "legacy_title":
                    q["title"],

                "legacy_publication_date":
                    str(
                        q[
                            "publication_date"
                        ]
                    ),

                "rank_within_14d":
                    rank,

                "background_doc_id":
                    b["doc_id"],

                "background_title":
                    b["title"],

                "background_publication_date":
                    str(
                        b[
                            "publication_date"
                        ]
                    ),

                "cosine_similarity":
                    float(
                        sim_i[j]
                    ),

                "delta_days":
                    int(
                        days_i[j]
                    ),

                "legacy_source_domain":
                    q[
                        "source_domain"
                    ],

                "background_source_domain":
                    b[
                        "source_domain"
                    ],

                "legacy_country":
                    q[
                        "country"
                    ],

                "background_country":
                    b[
                        "country"
                    ],

                "legacy_language":
                    q[
                        "language"
                    ],

                "background_language":
                    b[
                        "language"
                    ],
            })


    for regime, config in (
        REGIMES.items()
    ):

        mask = (
            (
                days_i
                <= config[
                    "max_days"
                ]
            )
            &
            (
                sim_i
                >= config[
                    "min_similarity"
                ]
            )
        )

        idx = np.flatnonzero(
            mask
        )

        sims = sim_i[
            idx
        ]

        days = days_i[
            idx
        ]

        if len(idx):

            sources = (
                background.iloc[
                    idx
                ][
                    "source_domain"
                ]
                .fillna("")
                .astype(str)
                .to_numpy()
            )

            countries = (
                background.iloc[
                    idx
                ][
                    "country"
                ]
                .fillna("")
                .astype(str)
                .to_numpy()
            )

            languages = (
                background.iloc[
                    idx
                ][
                    "language"
                ]
                .fillna("")
                .astype(str)
                .to_numpy()
            )

            q_source = str(
                q[
                    "source_domain"
                ]
            )

            q_country = str(
                q[
                    "country"
                ]
            )

            q_language = str(
                q[
                    "language"
                ]
            )

            cross_source = (
                sources
                != q_source
            )

            cross_country = (
                countries
                != q_country
            )

            cross_language = (
                languages
                != q_language
            )

            max_sim = float(
                sims.max()
            )

            mean_sim = float(
                sims.mean()
            )

            median_sim = float(
                np.median(
                    sims
                )
            )

            min_days = int(
                days.min()
            )

            median_days = float(
                np.median(
                    days
                )
            )

            n_sources = int(
                len(
                    set(
                        sources
                    )
                )
            )

            n_countries = int(
                len(
                    set(
                        countries
                    )
                )
            )

            n_languages = int(
                len(
                    set(
                        languages
                    )
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

        else:

            max_sim = np.nan
            mean_sim = np.nan
            median_sim = np.nan

            min_days = np.nan
            median_days = np.nan

            n_sources = 0
            n_countries = 0
            n_languages = 0

            n_cross_source = 0
            n_cross_country = 0
            n_cross_language = 0


        doc_rows.append({
            "regime":
                regime,

            "legacy_doc_id":
                q[
                    "doc_id"
                ],

            "legacy_publication_date":
                str(
                    q[
                        "publication_date"
                    ]
                ),

            "legacy_source_domain":
                q[
                    "source_domain"
                ],

            "legacy_country":
                q[
                    "country"
                ],

            "legacy_language":
                q[
                    "language"
                ],

            "n_neighbors":
                int(
                    len(idx)
                ),

            "max_similarity":
                max_sim,

            "mean_similarity":
                mean_sim,

            "median_similarity":
                median_sim,

            "min_delta_days":
                min_days,

            "median_delta_days":
                median_days,

            "n_unique_sources":
                n_sources,

            "n_unique_countries":
                n_countries,

            "n_unique_languages":
                n_languages,

            "n_cross_source_neighbors":
                n_cross_source,

            "n_cross_country_neighbors":
                n_cross_country,

            "n_cross_language_neighbors":
                n_cross_language,
        })


        for j in idx:

            b = background.iloc[
                int(j)
            ]

            neighbor_rows.append({
                "regime":
                    regime,

                "legacy_doc_id":
                    q[
                        "doc_id"
                    ],

                "background_doc_id":
                    b[
                        "doc_id"
                    ],

                "cosine_similarity":
                    float(
                        sim_i[j]
                    ),

                "delta_days":
                    int(
                        days_i[j]
                    ),

                "legacy_source_domain":
                    q[
                        "source_domain"
                    ],

                "background_source_domain":
                    b[
                        "source_domain"
                    ],

                "same_source":
                    bool(
                        str(
                            q[
                                "source_domain"
                            ]
                        )
                        ==
                        str(
                            b[
                                "source_domain"
                            ]
                        )
                    ),

                "legacy_country":
                    q[
                        "country"
                    ],

                "background_country":
                    b[
                        "country"
                    ],

                "same_country":
                    bool(
                        str(
                            q[
                                "country"
                            ]
                        )
                        ==
                        str(
                            b[
                                "country"
                            ]
                        )
                    ),

                "legacy_language":
                    q[
                        "language"
                    ],

                "background_language":
                    b[
                        "language"
                    ],

                "same_language":
                    bool(
                        str(
                            q[
                                "language"
                            ]
                        )
                        ==
                        str(
                            b[
                                "language"
                            ]
                        )
                    ),
            })


coverage = pd.DataFrame(
    doc_rows
)

neighbors = pd.DataFrame(
    neighbor_rows
)

top = pd.DataFrame(
    top_rows
)


if len(
    coverage
) != (
    199
    * len(
        REGIMES
    )
):
    raise RuntimeError(
        "Coverage row-count mismatch."
    )


coverage.to_csv(
    TMPDIR
    / "event_context_coverage_by_doc_v3.csv",
    index=False,
)

neighbors.to_csv(
    TMPDIR
    / "event_context_neighbor_pairs_v3.csv",
    index=False,
)

top.to_csv(
    TMPDIR
    / "event_context_top10_within14d_v3.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for regime in REGIMES:

    g = coverage[
        coverage[
            "regime"
        ]
        == regime
    ].copy()

    n = len(g)

    positive = g[
        g[
            "n_neighbors"
        ]
        > 0
    ]

    summary_rows.append({
        "regime":
            regime,

        "n_legacy_articles":
            n,

        "articles_ge1_neighbor":
            int(
                (
                    g[
                        "n_neighbors"
                    ]
                    >= 1
                ).sum()
            ),

        "coverage_ge1":
            float(
                (
                    g[
                        "n_neighbors"
                    ]
                    >= 1
                ).mean()
            ),

        "articles_ge2_neighbors":
            int(
                (
                    g[
                        "n_neighbors"
                    ]
                    >= 2
                ).sum()
            ),

        "coverage_ge2":
            float(
                (
                    g[
                        "n_neighbors"
                    ]
                    >= 2
                ).mean()
            ),

        "articles_ge3_neighbors":
            int(
                (
                    g[
                        "n_neighbors"
                    ]
                    >= 3
                ).sum()
            ),

        "coverage_ge3":
            float(
                (
                    g[
                        "n_neighbors"
                    ]
                    >= 3
                ).mean()
            ),

        "articles_ge5_neighbors":
            int(
                (
                    g[
                        "n_neighbors"
                    ]
                    >= 5
                ).sum()
            ),

        "coverage_ge5":
            float(
                (
                    g[
                        "n_neighbors"
                    ]
                    >= 5
                ).mean()
            ),

        "articles_with_cross_source_neighbor":
            int(
                (
                    g[
                        "n_cross_source_neighbors"
                    ]
                    > 0
                ).sum()
            ),

        "coverage_cross_source":
            float(
                (
                    g[
                        "n_cross_source_neighbors"
                    ]
                    > 0
                ).mean()
            ),

        "articles_with_cross_country_neighbor":
            int(
                (
                    g[
                        "n_cross_country_neighbors"
                    ]
                    > 0
                ).sum()
            ),

        "coverage_cross_country":
            float(
                (
                    g[
                        "n_cross_country_neighbors"
                    ]
                    > 0
                ).mean()
            ),

        "articles_with_cross_language_neighbor":
            int(
                (
                    g[
                        "n_cross_language_neighbors"
                    ]
                    > 0
                ).sum()
            ),

        "coverage_cross_language":
            float(
                (
                    g[
                        "n_cross_language_neighbors"
                    ]
                    > 0
                ).mean()
            ),

        "neighbor_count_mean":
            float(
                g[
                    "n_neighbors"
                ].mean()
            ),

        "neighbor_count_median":
            float(
                g[
                    "n_neighbors"
                ].median()
            ),

        "neighbor_count_max":
            int(
                g[
                    "n_neighbors"
                ].max()
            ),

        "covered_top1_similarity_mean":
            (
                float(
                    positive[
                        "max_similarity"
                    ].mean()
                )
                if len(
                    positive
                )
                else None
            ),

        "covered_top1_similarity_median":
            (
                float(
                    positive[
                        "max_similarity"
                    ].median()
                )
                if len(
                    positive
                )
                else None
            ),

        "covered_min_delta_days_median":
            (
                float(
                    positive[
                        "min_delta_days"
                    ].median()
                )
                if len(
                    positive
                )
                else None
            ),
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    TMPDIR
    / "event_context_coverage_summary_v3.csv",
    index=False,
)


json_summary = {
    "status":
        "COMPLETE",

    "purpose":
        "unsupervised_event_context_coverage_only",

    "clean_corpus_n":
        int(
            len(
                clean
            )
        ),

    "legacyaux_n":
        int(
            len(
                legacy_clean
            )
        ),

    "raw_background_n":
        1303,

    "exact_duplicate_background_excluded":
        n_duplicate_excluded,

    "final_clean_background_n":
        int(
            len(
                background
            )
        ),

    "embedding_shape":
        list(
            emb.shape
        ),

    "embedding_sha256":
        sha256(
            EMB
        ),

    "clean_corpus_sha256":
        sha256(
            CLEAN
        ),

    "legacyaux_sha256":
        sha256(
            LEGACY
        ),

    "manifest_sha256":
        sha256(
            MANIFEST
        ),

    "protocol_sha256":
        sha256(
            PROTOCOL
        ),

    "regimes":
        REGIMES,

    "labels_used_for_retrieval":
        False,

    "topic_used_for_retrieval":
        False,

    "eventgold_status":
        "SEALED_NOT_ACCESSED",
}


(
    TMPDIR
    / "event_context_coverage_summary_v3.json"
).write_text(
    json.dumps(
        json_summary,
        indent=2,
        ensure_ascii=False,
    )
)


# ============================================================
# HASH RESULTS
# ============================================================

files = sorted(
    p
    for p in TMPDIR.iterdir()
    if p.is_file()
)

hash_file = (
    TMPDIR
    / "EVENT_CONTEXT_COVERAGE_SHA256SUMS_v3.txt"
)

hash_file.write_text(
    "".join(
        f"{sha256(p)}  {p.name}\n"
        for p in files
    )
)


TMPDIR.rename(
    OUTDIR
)


print("=" * 100)
print(
    "EVENT-CONTEXT COVERAGE AUDIT COMPLETE"
)
print("=" * 100)

print()
print(
    summary_df.to_string(
        index=False
    )
)

print()
print(
    "Final clean background:",
    len(
        background
    )
)

print(
    "Exact duplicate exclusions:",
    n_duplicate_excluded
)

print(
    "Neighbor-pair rows:",
    len(
        neighbors
    )
)

print()
print(
    "✅ No supervised target labels used."
)

print(
    "✅ No topic labels used."
)

print(
    "✅ EventGold-35 not accessed."
)
