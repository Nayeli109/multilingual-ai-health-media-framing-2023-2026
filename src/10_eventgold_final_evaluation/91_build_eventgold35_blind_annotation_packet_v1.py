from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path.cwd()

BENCH = (
    ROOT
    / "01_event_aware_v3"
    / "00_frozen_benchmark"
)

FINAL = (
    ROOT
    / "01_event_aware_v3"
    / "10_eventgold_final_evaluation"
)

SOURCE = BENCH / "EventGold35_FINAL.csv"
SOURCE_SHA_FILE = BENCH / "EventGold35_SHA256SUMS.txt"

PROTOCOL = (
    FINAL
    / "EventGold35_SUPERVISED_EXTENSION_PROTOCOL_v1.txt"
)

OUTDIR = (
    FINAL
    / "eventgold35_supervised_extension_v1"
)

PRIVATE_MASTER = (
    OUTDIR
    / "EventGold35_SUPERVISED_PRIVATE_MASTER_v1.csv"
)

PRIVATE_KEY = (
    OUTDIR
    / "EventGold35_ANNOTATION_PRIVATE_KEY_v1.csv"
)

BLIND_MASTER = (
    OUTDIR
    / "EventGold35_BLIND_MASTER_v1.csv"
)

ANNOTATOR_A = (
    OUTDIR
    / "EventGold35_ANNOTATOR_A_BLIND_v1.csv"
)

ANNOTATOR_B = (
    OUTDIR
    / "EventGold35_ANNOTATOR_B_BLIND_v1.csv"
)

QC_PATH = (
    OUTDIR
    / "EventGold35_BLIND_PACKET_QC_v1.json"
)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x)

    x = x.replace("\r", " ")
    x = x.replace("\n", " ")

    x = " ".join(x.split())

    return x.strip()


def expected_source_sha():
    lines = SOURCE_SHA_FILE.read_text(
        encoding="utf-8"
    ).splitlines()

    hits = []

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 2:
            continue

        digest = parts[0]
        filename = parts[-1]

        if filename.endswith(
            "EventGold35_FINAL.csv"
        ):
            hits.append(digest)

    if len(hits) != 1:
        raise RuntimeError(
            "Could not uniquely recover frozen "
            "EventGold35 CSV SHA256."
        )

    return hits[0]


if not SOURCE.exists():
    raise RuntimeError(
        f"Missing source EventGold file: {SOURCE}"
    )

if not PROTOCOL.exists():
    raise RuntimeError(
        f"Missing frozen supervised-extension "
        f"protocol: {PROTOCOL}"
    )

observed_sha = sha256(SOURCE)
expected_sha = expected_source_sha()

if observed_sha != expected_sha:
    raise RuntimeError(
        "EventGold source SHA256 mismatch."
    )


df = pd.read_csv(SOURCE)

if len(df) != 35:
    raise RuntimeError(
        f"Expected 35 EventGold rows; "
        f"observed {len(df)}."
    )


required = [
    "doc_id",
    "title",
    "article_excerpt_1800",
    "article_excerpt_3500",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required source columns: {missing}"
    )


df["doc_id"] = (
    df["doc_id"]
    .map(normalize_text)
)

if df["doc_id"].eq("").any():
    raise RuntimeError(
        "Empty doc_id detected."
    )

if df["doc_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate doc_id detected."
    )


# ------------------------------------------------------------
# Prospectively frozen compatibility text rule.
# ------------------------------------------------------------

rows = []

for _, row in df.iterrows():

    title = normalize_text(
        row["title"]
    )

    if not title:
        raise RuntimeError(
            "Empty title detected."
        )

    selected = ""
    source = ""

    for col in [
        "article_excerpt_3500",
        "article_excerpt_1800",
        "title",
    ]:

        candidate = normalize_text(
            row[col]
        )

        if candidate:
            selected = candidate
            source = col
            break

    if not selected:
        raise RuntimeError(
            "No usable prospective dapt_text "
            f"for doc_id={row['doc_id']}"
        )

    doc_id = row["doc_id"]

    order_key = hashlib.sha256(
        (
            "EVENTGOLD35_SUPERVISED_EXTENSION_V1|"
            + doc_id
        ).encode("utf-8")
    ).hexdigest()

    rows.append({
        "doc_id": doc_id,
        "title": title,
        "dapt_text": selected,
        "dapt_text_source": source,
        "annotation_excerpt": selected[:1400],
        "blind_order_sha256": order_key,
    })


private = pd.DataFrame(rows)

private = (
    private
    .sort_values(
        [
            "blind_order_sha256",
            "doc_id",
        ]
    )
    .reset_index(drop=True)
)

private.insert(
    0,
    "annotation_id",
    [
        f"EG35_ANN_{i:03d}"
        for i in range(
            1,
            len(private) + 1,
        )
    ],
)


if private["annotation_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate annotation_id detected."
    )

if private["annotation_excerpt"].eq("").any():
    raise RuntimeError(
        "Empty annotation excerpt detected."
    )

if (
    private["annotation_excerpt"]
    .str.len()
    .gt(1400)
    .any()
):
    raise RuntimeError(
        "Annotation excerpt exceeds "
        "1400 characters."
    )


# ------------------------------------------------------------
# Private master:
# retained for final gold/model merge;
# never shown to annotators.
# ------------------------------------------------------------

private.to_csv(
    PRIVATE_MASTER,
    index=False,
    lineterminator="\n",
)


# ------------------------------------------------------------
# Private annotation key:
# maps blinded annotation IDs to document IDs only.
# ------------------------------------------------------------

private[
    [
        "annotation_id",
        "doc_id",
        "blind_order_sha256",
    ]
].to_csv(
    PRIVATE_KEY,
    index=False,
    lineterminator="\n",
)


# ------------------------------------------------------------
# Blind master:
# ONLY the information permitted during annotation.
# ------------------------------------------------------------

blind = private[
    [
        "annotation_id",
        "title",
        "annotation_excerpt",
    ]
].copy()

blind.to_csv(
    BLIND_MASTER,
    index=False,
    lineterminator="\n",
)


# ------------------------------------------------------------
# Independent annotator A file.
# ------------------------------------------------------------

a = blind.copy()

a["human_primary_frame_A"] = ""
a["human_stance_A"] = ""
a["human_misinformation_relation_A"] = ""
a["human_notes_A"] = ""

a.to_csv(
    ANNOTATOR_A,
    index=False,
    lineterminator="\n",
)


# ------------------------------------------------------------
# Independent annotator B file.
# ------------------------------------------------------------

b = blind.copy()

b["human_primary_frame_B"] = ""
b["human_stance_B"] = ""
b["human_misinformation_relation_B"] = ""
b["human_notes_B"] = ""

b.to_csv(
    ANNOTATOR_B,
    index=False,
    lineterminator="\n",
)


# ------------------------------------------------------------
# Fail-closed blindness checks.
# ------------------------------------------------------------

expected_blind_cols = [
    "annotation_id",
    "title",
    "annotation_excerpt",
]

if list(blind.columns) != expected_blind_cols:
    raise RuntimeError(
        "Blind master schema mismatch."
    )

expected_a = (
    expected_blind_cols
    + [
        "human_primary_frame_A",
        "human_stance_A",
        "human_misinformation_relation_A",
        "human_notes_A",
    ]
)

expected_b = (
    expected_blind_cols
    + [
        "human_primary_frame_B",
        "human_stance_B",
        "human_misinformation_relation_B",
        "human_notes_B",
    ]
)

if list(a.columns) != expected_a:
    raise RuntimeError(
        "Annotator A schema mismatch."
    )

if list(b.columns) != expected_b:
    raise RuntimeError(
        "Annotator B schema mismatch."
    )


for c in expected_a[3:]:
    if a[c].astype(str).str.strip().ne("").any():
        raise RuntimeError(
            "Annotator A label field is not blank."
        )

for c in expected_b[3:]:
    if b[c].astype(str).str.strip().ne("").any():
        raise RuntimeError(
            "Annotator B label field is not blank."
        )


prohibited_blind_columns = {
    "doc_id",
    "country",
    "region",
    "language",
    "year",
    "publication_date",
    "outlet",
    "source_domain",
    "url",
    "problem_definition",
    "problem_definition_span",
    "causal_actor",
    "causal_actor_type",
    "consequence_type",
    "consequence_description",
    "evaluation_tone",
    "remedy_present",
    "remedy_description",
    "remedy_actor",
    "responsibility_actor",
    "responsibility_actor_type",
    "epistemic_certainty",
    "epistemic_cues",
    "evidence_source_type",
    "evidence_source_actor",
    "evidence_span",
    "health_claim_present",
    "health_claim_text",
    "ai_system_or_company",
    "coder_confidence",
    "coder_notes",
}

if prohibited_blind_columns & set(a.columns):
    raise RuntimeError(
        "Prohibited EventGold field leaked "
        "into Annotator A packet."
    )

if prohibited_blind_columns & set(b.columns):
    raise RuntimeError(
        "Prohibited EventGold field leaked "
        "into Annotator B packet."
    )


qc = {
    "version": "v1",
    "source_eventgold_sha256": observed_sha,
    "rows": int(len(private)),
    "unique_doc_ids": int(
        private["doc_id"].nunique()
    ),
    "unique_annotation_ids": int(
        private["annotation_id"].nunique()
    ),
    "text_rule": [
        "article_excerpt_3500",
        "article_excerpt_1800",
        "title",
    ],
    "annotation_excerpt_max_chars": 1400,
    "blind_master_columns": list(
        blind.columns
    ),
    "annotator_a_columns": list(
        a.columns
    ),
    "annotator_b_columns": list(
        b.columns
    ),
    "dapt_text_source_counts": {
        str(k): int(v)
        for k, v in (
            private["dapt_text_source"]
            .value_counts()
            .to_dict()
            .items()
        )
    },
    "doc_id_exposed_to_annotators": False,
    "original_semantic_fields_exposed": False,
    "model_predictions_included": False,
    "model_probabilities_included": False,
    "new_supervised_labels_created": False,
    "annotation_order_rule": (
        "ascending SHA256 of "
        "'EVENTGOLD35_SUPERVISED_EXTENSION_V1|' "
        "+ doc_id"
    ),
}

QC_PATH.write_text(
    json.dumps(
        qc,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


print(
    "SOURCE_EVENTGOLD_SHA256="
    + observed_sha
)

print(
    "BLIND_PACKET_ROWS="
    + str(len(blind))
)

print(
    "UNIQUE_ANNOTATION_IDS="
    + str(blind["annotation_id"].nunique())
)

print(
    "ANNOTATION_EXCERPT_MAX_CHARS="
    + str(
        int(
            blind["annotation_excerpt"]
            .str.len()
            .max()
        )
    )
)

print(
    "ANNOTATOR_A_LABEL_FIELDS_BLANK=YES"
)

print(
    "ANNOTATOR_B_LABEL_FIELDS_BLANK=YES"
)

print(
    "DOC_ID_EXPOSED_TO_ANNOTATORS=NO"
)

print(
    "ORIGINAL_SEMANTIC_FIELDS_EXPOSED=NO"
)

print(
    "MODEL_PREDICTIONS_INCLUDED=NO"
)

print(
    "NEW_SUPERVISED_LABELS_CREATED=NO"
)

print(
    "BLIND_PACKET_BUILD=PASS"
)
