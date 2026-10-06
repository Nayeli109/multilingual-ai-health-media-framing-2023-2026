from pathlib import Path
from collections import defaultdict, deque
import hashlib
import json
import pandas as pd


ROOT = Path.cwd()

POOL_PATH = ROOT / (
    "01_event_aware_v3/01_data_audit/"
    "Corpus1502_DAPT_text_v3.csv"
)

AUDIT_PATH = ROOT / (
    "01_event_aware_v3/01_data_audit/"
    "Corpus1502_leakage_audit_v3.csv"
)

GOLD_PATH = ROOT / (
    "01_event_aware_v3/00_frozen_benchmark/"
    "EventGold35_FINAL.csv"
)

PAIR_FILES = [
    ROOT / (
        "00_redesign_v2/02_event_matching/outputs/"
        "same_event_candidate_pairs_v2.csv"
    ),
    ROOT / (
        "00_redesign_v2/05_event_matching_robustness/outputs/"
        "event_matching_unreviewed_expansion_pairs_v2.csv"
    ),
]

OUTDIR = ROOT / "01_event_aware_v3/01_data_audit"

STRICT_OUT = OUTDIR / (
    "DAPT_STRICT_EVENT_HOLDOUT_v3.csv"
)

EXCLUSION_OUT = OUTDIR / (
    "DAPT_STRICT_EVENT_HOLDOUT_exclusions_v3.csv"
)

QC_OUT = OUTDIR / (
    "DAPT_STRICT_EVENT_HOLDOUT_QC_v3.json"
)


def clean_id(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def detect_pair_columns(df, path):
    patterns = [
        ("doc_id_a", "doc_id_b"),
        ("doc_id_1", "doc_id_2"),
        ("doc_id_x", "doc_id_y"),
        ("left_doc_id", "right_doc_id"),
        ("source_doc_id", "target_doc_id"),
        ("doc1_id", "doc2_id"),
        ("article_id_a", "article_id_b"),
        ("article_id_1", "article_id_2"),
    ]

    for a, b in patterns:
        if a in df.columns and b in df.columns:
            return a, b

    possible = [
        c for c in df.columns
        if (
            ("doc" in c.lower() or "article" in c.lower())
            and "id" in c.lower()
        )
    ]

    if len(possible) == 2:
        return possible[0], possible[1]

    raise RuntimeError(
        f"\nCould not detect pair ID columns in:\n{path}\n"
        f"Columns:\n{list(df.columns)}\n"
    )


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ============================================================
# LOAD
# ============================================================

pool = pd.read_csv(POOL_PATH)
audit = pd.read_csv(AUDIT_PATH)
gold = pd.read_csv(GOLD_PATH)

pool["doc_id"] = pool["doc_id"].map(clean_id)
gold["doc_id"] = gold["doc_id"].map(clean_id)
audit["doc_id"] = audit["doc_id"].map(clean_id)

gold_ids = set(gold["doc_id"])
pool_ids = set(pool["doc_id"])

assert len(gold) == 35
assert len(pool) == 1502
assert len(gold_ids & pool_ids) == 0


# ============================================================
# EXACT-TITLE HOLDOUT
# ============================================================

title_holdout = set(
    audit.loc[
        audit["exact_gold_title_duplicate"] == True,
        "doc_id"
    ].map(clean_id)
)

print("Exact-title holdout:", len(title_holdout))


# ============================================================
# BUILD EVENT-CANDIDATE GRAPH
# ============================================================

graph = defaultdict(set)
pair_reports = []

for path in PAIR_FILES:

    if not path.exists():
        print("WARNING: pair file missing:", path)
        continue

    df = pd.read_csv(path)

    a_col, b_col = detect_pair_columns(df, path)

    valid_edges = 0

    for a, b in zip(df[a_col], df[b_col]):

        a = clean_id(a)
        b = clean_id(b)

        if not a or not b or a == b:
            continue

        graph[a].add(b)
        graph[b].add(a)
        valid_edges += 1

    pair_reports.append({
        "file": str(path.relative_to(ROOT)),
        "rows": int(len(df)),
        "id_column_a": a_col,
        "id_column_b": b_col,
        "valid_edges": int(valid_edges),
    })

    print(
        f"{path.name}: "
        f"{len(df)} rows | "
        f"{a_col} <-> {b_col}"
    )


# ============================================================
# ONE-HOP GOLD NEIGHBORS
# ============================================================

one_hop = set()

for gid in gold_ids:
    one_hop.update(graph.get(gid, set()))

one_hop -= gold_ids


# ============================================================
# CONNECTED-COMPONENT CLOSURE FROM GOLD
#
# Conservative leakage guard:
# any article connected through the frozen candidate-event
# graph to EventGold is removed from DAPT.
# ============================================================

visited = set(gold_ids)
queue = deque(gold_ids)

while queue:

    current = queue.popleft()

    for neighbor in graph.get(current, set()):

        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)

graph_closure_neighbors = visited - gold_ids


# Only IDs actually present in the 1502 development pool
one_hop_pool = one_hop & pool_ids
closure_pool = graph_closure_neighbors & pool_ids
title_holdout_pool = title_holdout & pool_ids


# ============================================================
# STRICT EXCLUSION SET
# ============================================================

strict_exclude = (
    closure_pool
    | title_holdout_pool
)

strict = (
    pool.loc[
        ~pool["doc_id"].isin(strict_exclude)
    ]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# EXCLUSION AUDIT
# ============================================================

rows = []

for doc_id in sorted(strict_exclude):

    reasons = []

    if doc_id in title_holdout_pool:
        reasons.append("exact_gold_title")

    if doc_id in one_hop_pool:
        reasons.append("direct_gold_event_candidate_neighbor")

    if (
        doc_id in closure_pool
        and doc_id not in one_hop_pool
    ):
        reasons.append("transitive_gold_event_candidate_neighbor")

    source_row = pool.loc[
        pool["doc_id"] == doc_id
    ]

    title = (
        source_row["title"].iloc[0]
        if (
            len(source_row)
            and "title" in source_row.columns
        )
        else ""
    )

    language = (
        source_row["language"].iloc[0]
        if (
            len(source_row)
            and "language" in source_row.columns
        )
        else ""
    )

    region = (
        source_row["region"].iloc[0]
        if (
            len(source_row)
            and "region" in source_row.columns
        )
        else ""
    )

    rows.append({
        "doc_id": doc_id,
        "exclusion_reason": " | ".join(reasons),
        "title": title,
        "language": language,
        "region": region,
    })

exclusions = pd.DataFrame(rows)


# ============================================================
# FAIL-CLOSED QC
# ============================================================

gold_overlap = set(strict["doc_id"]) & gold_ids
closure_overlap = set(strict["doc_id"]) & closure_pool
title_overlap = set(strict["doc_id"]) & title_holdout_pool

if gold_overlap:
    raise RuntimeError(
        f"Gold document leakage: {sorted(gold_overlap)}"
    )

if closure_overlap:
    raise RuntimeError(
        "Gold candidate-event graph leakage remained."
    )

if title_overlap:
    raise RuntimeError(
        "Exact-title leakage remained."
    )

if strict["dapt_text"].isna().any():
    raise RuntimeError(
        "Missing DAPT text detected."
    )

if (
    strict["dapt_text"]
    .astype(str)
    .str.strip()
    .eq("")
    .any()
):
    raise RuntimeError(
        "Empty DAPT text detected."
    )


# ============================================================
# SAVE
# ============================================================

strict.to_csv(
    STRICT_OUT,
    index=False
)

exclusions.to_csv(
    EXCLUSION_OUT,
    index=False
)

qc = {
    "original_corpus": 1537,
    "eventgold_articles": int(len(gold)),
    "document_holdout_pool": int(len(pool)),
    "exact_gold_title_holdout": int(
        len(title_holdout_pool)
    ),
    "direct_gold_candidate_neighbors": int(
        len(one_hop_pool)
    ),
    "candidate_graph_closure_neighbors": int(
        len(closure_pool)
    ),
    "total_additional_exclusions": int(
        len(strict_exclude)
    ),
    "strict_dapt_articles": int(len(strict)),
    "gold_doc_overlap_final": int(
        len(gold_overlap)
    ),
    "gold_candidate_graph_overlap_final": int(
        len(closure_overlap)
    ),
    "exact_title_overlap_final": int(
        len(title_overlap)
    ),
    "language_distribution": (
        strict["language"]
        .value_counts()
        .to_dict()
        if "language" in strict.columns
        else None
    ),
    "region_distribution": (
        strict["region"]
        .value_counts()
        .to_dict()
        if "region" in strict.columns
        else None
    ),
    "pair_files": pair_reports,
    "strict_dataset_sha256": sha256(
        STRICT_OUT
    ),
}

QC_OUT.write_text(
    json.dumps(
        qc,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# REPORT
# ============================================================

print("\n==============================================")
print("STRICT EVENT-HOLDOUT DAPT QC: PASSED")
print("==============================================")

print(
    "Document-level pool              :",
    len(pool)
)

print(
    "Exact-title exclusions           :",
    len(title_holdout_pool)
)

print(
    "Direct Gold candidate neighbors  :",
    len(one_hop_pool)
)

print(
    "Graph-closure neighbors          :",
    len(closure_pool)
)

print(
    "Total additional exclusions      :",
    len(strict_exclude)
)

print(
    "STRICT DAPT articles             :",
    len(strict)
)

print(
    "Gold doc overlap                 :",
    len(gold_overlap)
)

print(
    "Gold-event candidate overlap     :",
    len(closure_overlap)
)

print(
    "Exact-title overlap              :",
    len(title_overlap)
)

if "language" in strict.columns:
    print("\nLanguage:")
    print(
        strict["language"]
        .value_counts()
        .to_string()
    )

print("\nSaved:")
print(STRICT_OUT)
print(EXCLUSION_OUT)
print(QC_OUT)
