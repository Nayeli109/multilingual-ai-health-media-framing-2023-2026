from pathlib import Path
import hashlib
import json
import re

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

CORPUS_PATH = (
    ROOT
    / "results/tables/topics_full_v1/document_topics_full_v1.csv"
)

GOLD_PATH = (
    ROOT
    / "01_event_aware_v3/00_frozen_benchmark/EventGold35_FINAL.csv"
)

OUTDIR = (
    ROOT
    / "01_event_aware_v3/01_data_audit"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

POOL_OUT = OUTDIR / "Corpus1502_document_holdout_v3.csv"
DAPT_OUT = OUTDIR / "Corpus1502_DAPT_text_v3.csv"
SUMMARY_OUT = OUTDIR / "Corpus1502_document_holdout_QC_v3.json"
AUDIT_OUT = OUTDIR / "Corpus1502_leakage_audit_v3.csv"
HASH_OUT = OUTDIR / "Corpus1502_SHA256SUMS.txt"


# ============================================================
# EXPECTED FROZEN COUNTS
# ============================================================

EXPECTED_CORPUS = 1537
EXPECTED_GOLD = 35
EXPECTED_EVENTS = 10
EXPECTED_POOL = 1502


# ============================================================
# HELPERS
# ============================================================

def fail(msg):
    raise RuntimeError(
        "\nCORPUS1502 VALIDATION FAILED\n" + str(msg)
    )


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def text_hash(x):
    x = normalize_text(x)

    if not x:
        return ""

    return hashlib.sha256(
        x.encode("utf-8")
    ).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def build_text(df):
    """
    Build one text field without modifying original columns.
    Priority:
      document_for_modeling
      text
      full_text
      content
      article_text
      title
    """

    priority = [
        "document_for_modeling",
        "text",
        "full_text",
        "content",
        "article_text",
        "title",
    ]

    available = [
        c for c in priority
        if c in df.columns
    ]

    if not available:
        fail(
            "No usable article-text column was found.\n"
            f"Available columns: {list(df.columns)}"
        )

    values = []
    sources = []

    for _, row in df.iterrows():

        selected = ""
        source = ""

        for col in available:

            candidate = normalize_text(
                row[col]
            )

            if candidate:
                selected = candidate
                source = col
                break

        values.append(selected)
        sources.append(source)

    return (
        pd.Series(values, index=df.index),
        pd.Series(sources, index=df.index),
        available,
    )


# ============================================================
# LOAD
# ============================================================

if not CORPUS_PATH.exists():
    fail(
        f"Corpus file not found:\n{CORPUS_PATH}"
    )

if not GOLD_PATH.exists():
    fail(
        f"EventGold35 file not found:\n{GOLD_PATH}"
    )

corpus = pd.read_csv(CORPUS_PATH)
gold = pd.read_csv(GOLD_PATH)

print("Corpus rows:", len(corpus))
print("Gold rows  :", len(gold))


# ============================================================
# BASIC VALIDATION
# ============================================================

if len(corpus) != EXPECTED_CORPUS:
    fail(
        f"Expected {EXPECTED_CORPUS} corpus rows; "
        f"observed {len(corpus)}."
    )

if len(gold) != EXPECTED_GOLD:
    fail(
        f"Expected {EXPECTED_GOLD} gold rows; "
        f"observed {len(gold)}."
    )

if "doc_id" not in corpus.columns:
    fail("Corpus does not contain doc_id.")

if "doc_id" not in gold.columns:
    fail("Gold file does not contain doc_id.")

corpus["doc_id"] = (
    corpus["doc_id"]
    .astype(str)
    .str.strip()
)

gold["doc_id"] = (
    gold["doc_id"]
    .astype(str)
    .str.strip()
)

if corpus["doc_id"].duplicated().any():

    duplicated = (
        corpus.loc[
            corpus["doc_id"].duplicated(False),
            "doc_id"
        ]
        .tolist()
    )

    fail(
        f"Corpus contains duplicate doc_id values: "
        f"{duplicated[:10]}"
    )

if gold["doc_id"].duplicated().any():

    fail(
        "EventGold35 contains duplicate doc_id values."
    )


# ============================================================
# EVENT CHECK
# ============================================================

event_col = None

for c in [
    "event_id_final_v2",
    "event_id",
]:
    if c in gold.columns:
        event_col = c
        break

gold_events = (
    gold[event_col].nunique()
    if event_col
    else None
)

if gold_events is not None:
    if gold_events != EXPECTED_EVENTS:
        fail(
            f"Expected {EXPECTED_EVENTS} gold events; "
            f"observed {gold_events}."
        )


# ============================================================
# GOLD MUST BE CONTAINED IN CORPUS
# ============================================================

corpus_ids = set(corpus["doc_id"])
gold_ids = set(gold["doc_id"])

missing_gold_ids = sorted(
    gold_ids - corpus_ids
)

if missing_gold_ids:

    fail(
        "Some EventGold35 documents are absent from the "
        "1537-document corpus:\n"
        + "\n".join(missing_gold_ids)
    )


# ============================================================
# BUILD 1502 DOCUMENT-HOLDOUT POOL
# ============================================================

pool = (
    corpus.loc[
        ~corpus["doc_id"].isin(gold_ids)
    ]
    .copy()
    .reset_index(drop=True)
)

if len(pool) != EXPECTED_POOL:

    fail(
        f"Expected {EXPECTED_POOL} holdout-pool rows; "
        f"observed {len(pool)}."
    )

overlap = (
    set(pool["doc_id"])
    & gold_ids
)

if overlap:

    fail(
        f"Gold/pool doc_id leakage detected: "
        f"{sorted(overlap)}"
    )


# ============================================================
# BUILD TEXT FOR POOL AND GOLD
# ============================================================

pool_text, pool_source, pool_available = (
    build_text(pool)
)

gold_text, gold_source, gold_available = (
    build_text(gold)
)

pool["dapt_text"] = pool_text
pool["dapt_text_source"] = pool_source

gold = gold.copy()
gold["_audit_text"] = gold_text
gold["_audit_text_source"] = gold_source


# ============================================================
# EXACT-TEXT LEAKAGE AUDIT
# ============================================================

pool["_normalized_text_hash"] = (
    pool["dapt_text"]
    .map(text_hash)
)

gold["_normalized_text_hash"] = (
    gold["_audit_text"]
    .map(text_hash)
)

gold_text_hashes = set(
    gold.loc[
        gold["_normalized_text_hash"] != "",
        "_normalized_text_hash"
    ]
)

pool["exact_gold_text_duplicate"] = (
    pool["_normalized_text_hash"]
    .isin(gold_text_hashes)
)

exact_duplicate_rows = int(
    pool["exact_gold_text_duplicate"].sum()
)


# ============================================================
# TITLE-LEVEL AUDIT
# ============================================================

if (
    "title" in pool.columns
    and "title" in gold.columns
):

    normalized_gold_titles = set(
        gold["title"]
        .map(normalize_text)
        .str.lower()
    )

    normalized_gold_titles.discard("")

    pool["_normalized_title"] = (
        pool["title"]
        .map(normalize_text)
        .str.lower()
    )

    pool["exact_gold_title_duplicate"] = (
        pool["_normalized_title"]
        .isin(normalized_gold_titles)
    )

else:

    pool["exact_gold_title_duplicate"] = False


title_duplicate_rows = int(
    pool["exact_gold_title_duplicate"].sum()
)


# ============================================================
# STRICT DAPT SET
#
# EventGold documents are removed.
# Exact full-text duplicates of EventGold are ALSO removed.
#
# Title matches are retained but flagged because syndicated
# coverage may share headlines without being exact text copies.
# ============================================================

dapt = (
    pool.loc[
        ~pool["exact_gold_text_duplicate"]
    ]
    .copy()
)

dapt["dapt_text_length"] = (
    dapt["dapt_text"]
    .astype(str)
    .str.len()
)

empty_text_rows = int(
    (dapt["dapt_text_length"] == 0).sum()
)

if empty_text_rows > 0:

    fail(
        f"{empty_text_rows} DAPT rows have no usable text."
    )


# ============================================================
# AUDIT TABLE
# ============================================================

audit_cols = [
    "doc_id",
    "dapt_text_source",
    "exact_gold_text_duplicate",
    "exact_gold_title_duplicate",
]

for optional in [
    "title",
    "country",
    "region",
    "language",
    "year",
    "published_date",
]:
    if optional in pool.columns:
        audit_cols.append(optional)

audit = pool[
    audit_cols
].copy()


# ============================================================
# SAVE
# ============================================================

# Full 1502 document-holdout pool.
pool_export = pool.drop(
    columns=[
        c for c in [
            "_normalized_text_hash",
            "_normalized_title",
        ]
        if c in pool.columns
    ]
)

pool_export.to_csv(
    POOL_OUT,
    index=False,
)

# DAPT-ready text file: only columns needed downstream.
dapt_cols = [
    "doc_id",
    "dapt_text",
    "dapt_text_source",
    "dapt_text_length",
]

for optional in [
    "title",
    "country",
    "region",
    "language",
    "year",
    "published_date",
    "topic_v1",
    "topic_label_v1",
]:
    if optional in dapt.columns:
        dapt_cols.append(optional)

dapt[
    dapt_cols
].to_csv(
    DAPT_OUT,
    index=False,
)

audit.to_csv(
    AUDIT_OUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "source_corpus_rows": int(len(corpus)),
    "eventgold35_rows": int(len(gold)),
    "eventgold35_events": (
        int(gold_events)
        if gold_events is not None
        else None
    ),
    "document_holdout_pool_rows": int(len(pool)),
    "doc_id_overlap_with_eventgold35": int(
        len(overlap)
    ),
    "exact_text_duplicates_of_eventgold35": (
        exact_duplicate_rows
    ),
    "exact_title_matches_to_eventgold35": (
        title_duplicate_rows
    ),
    "dapt_ready_rows_after_exact_text_exclusion": int(
        len(dapt)
    ),
    "dapt_empty_text_rows": empty_text_rows,
    "pool_text_columns_available": (
        pool_available
    ),
    "gold_text_columns_available": (
        gold_available
    ),
    "text_source_distribution": (
        dapt["dapt_text_source"]
        .value_counts(dropna=False)
        .to_dict()
    ),
    "language_distribution": (
        dapt["language"]
        .value_counts(dropna=False)
        .to_dict()
        if "language" in dapt.columns
        else None
    ),
    "region_distribution": (
        dapt["region"]
        .value_counts(dropna=False)
        .to_dict()
        if "region" in dapt.columns
        else None
    ),
    "outputs": {
        "document_holdout_pool": str(POOL_OUT),
        "dapt_ready_text": str(DAPT_OUT),
        "leakage_audit": str(AUDIT_OUT),
    },
}

SUMMARY_OUT.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================
# SHA256
# ============================================================

files_to_hash = [
    POOL_OUT,
    DAPT_OUT,
    AUDIT_OUT,
    SUMMARY_OUT,
]

with HASH_OUT.open(
    "w",
    encoding="utf-8",
) as f:

    for path in files_to_hash:

        f.write(
            f"{file_sha256(path)}  "
            f"{path.relative_to(ROOT)}\n"
        )


# ============================================================
# CONSOLE
# ============================================================

print("\n==============================================")
print("CORPUS1502 DOCUMENT-HOLDOUT QC: PASSED")
print("==============================================")

print(
    "Source corpus                  :",
    len(corpus)
)

print(
    "Frozen EventGold35             :",
    len(gold)
)

print(
    "Gold events                    :",
    gold_events
)

print(
    "Document-holdout pool          :",
    len(pool)
)

print(
    "doc_id overlap                 :",
    len(overlap)
)

print(
    "Exact gold-text duplicates     :",
    exact_duplicate_rows
)

print(
    "Exact gold-title matches       :",
    title_duplicate_rows
)

print(
    "DAPT-ready rows                :",
    len(dapt)
)

print(
    "Empty DAPT texts               :",
    empty_text_rows
)

print("\nText-source distribution:")
print(
    dapt["dapt_text_source"]
    .value_counts(dropna=False)
    .to_string()
)

if "language" in dapt.columns:

    print("\nLanguage distribution:")
    print(
        dapt["language"]
        .value_counts(dropna=False)
        .to_string()
    )

if "region" in dapt.columns:

    print("\nRegion distribution:")
    print(
        dapt["region"]
        .value_counts(dropna=False)
        .to_string()
    )

print("\nOutputs:")
print(POOL_OUT)
print(DAPT_OUT)
print(AUDIT_OUT)
print(SUMMARY_OUT)
print(HASH_OUT)
