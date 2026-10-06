from pathlib import Path
from collections import Counter
import json

import pandas as pd


ROOT = Path.cwd()

FINAL = (
    ROOT
    / "01_event_aware_v3"
    / "10_eventgold_final_evaluation"
)

EXT = FINAL / "eventgold35_supervised_extension_v1"

A_PATH = EXT / "EventGold35_ANNOTATOR_A_CORRECTED_35_v1.csv"
B_PATH = EXT / "EventGold35_ANNOTATOR_B_CORRECTED_35_v1.csv"

DISAGREE_PATH = (
    EXT
    / "EventGold35_DISAGREEMENTS_FOR_ADJUDICATION_AFTER_RECHECK_v1.csv"
)

ADJ_PATH = (
    EXT
    / "EventGold35_DISAGREEMENTS_ADJUDICATED_RECEIVED_v1.csv"
)

PRIVATE_PATH = EXT / "EventGold35_SUPERVISED_PRIVATE_MASTER_v1.csv"
BLIND_PATH = EXT / "EventGold35_BLIND_MASTER_v1.csv"

GOLD_PATH = EXT / "EventGold35_SUPERVISED_EXTENSION_FINAL_GOLD_v1.csv"

QC_PATH = EXT / "EventGold35_SUPERVISED_EXTENSION_FINAL_GOLD_QC_v1.json"

REPORT_PATH = EXT / "EventGold35_ADJUDICATION_AND_FINAL_GOLD_REPORT_v1.txt"


FRAME = [
    "clinical innovation and diagnostic benefit",
    "health safety risk and medical harm",
    "misinformation, deepfakes, and fraud",
    "regulation, governance, and policy",
    "public trust and accountability",
    "access, inequality, and digital divide",
    "labor automation and professional displacement",
    "mental health and therapy risk",
    "other / unclear",
]

STANCE = [
    "optimistic about AI in health",
    "cautious or mixed about AI in health",
    "critical of AI risks in health",
    "focused on regulation and governance",
    "unclear / not applicable",
]

MISINFO = [
    "direct misinformation focus",
    "indirect misinformation concern",
    "no clear misinformation relation",
    "unclear / not applicable",
]

TASKS = {
    "primary_frame": {
        "A": "human_primary_frame_A",
        "B": "human_primary_frame_B",
        "final": "final_primary_frame_validated",
        "allowed": FRAME,
    },
    "stance": {
        "A": "human_stance_A",
        "B": "human_stance_B",
        "final": "final_stance_validated",
        "allowed": STANCE,
    },
    "misinformation_relation": {
        "A": "human_misinformation_relation_A",
        "B": "human_misinformation_relation_B",
        "final": "final_misinformation_relation_validated",
        "allowed": MISINFO,
    },
}


def clean(x):
    if pd.isna(x):
        return ""

    x = str(x).replace("\r", " ").replace("\n", " ")
    return " ".join(x.split()).strip()


def clean_df(df):
    for c in df.columns:
        df[c] = df[c].map(clean)
    return df


a = clean_df(
    pd.read_csv(
        A_PATH,
        dtype=str,
        keep_default_na=False,
    )
)

b = clean_df(
    pd.read_csv(
        B_PATH,
        dtype=str,
        keep_default_na=False,
    )
)

d = clean_df(
    pd.read_csv(
        DISAGREE_PATH,
        dtype=str,
        keep_default_na=False,
    )
)

adj = clean_df(
    pd.read_csv(
        ADJ_PATH,
        dtype=str,
        keep_default_na=False,
    )
)

private = clean_df(
    pd.read_csv(
        PRIVATE_PATH,
        dtype=str,
        keep_default_na=False,
    )
)

blind = clean_df(
    pd.read_csv(
        BLIND_PATH,
        dtype=str,
        keep_default_na=False,
    )
)


# ------------------------------------------------------------
# Core structural validation.
# ------------------------------------------------------------

for name, df, n in [
    ("A corrected", a, 35),
    ("B corrected", b, 35),
    ("frozen disagreement", d, 7),
    ("received adjudication", adj, 7),
    ("private master", private, 35),
    ("blind master", blind, 35),
]:
    if len(df) != n:
        raise RuntimeError(
            f"{name}: expected {n} rows, observed {len(df)}"
        )


for name, df in [
    ("A", a),
    ("B", b),
    ("private", private),
    ("blind", blind),
]:
    if "annotation_id" not in df.columns:
        raise RuntimeError(
            f"{name}: annotation_id missing"
        )

    if df["annotation_id"].duplicated().any():
        raise RuntimeError(
            f"{name}: duplicate annotation_id"
        )


ids = blind["annotation_id"].tolist()

if a["annotation_id"].tolist() != ids:
    raise RuntimeError(
        "Corrected A IDs/order differ from frozen blind master."
    )

if b["annotation_id"].tolist() != ids:
    raise RuntimeError(
        "Corrected B IDs/order differ from frozen blind master."
    )

if private["annotation_id"].tolist() != ids:
    raise RuntimeError(
        "Private-master IDs/order differ from frozen blind master."
    )


# ------------------------------------------------------------
# Validate adjudication schema and immutable fields.
# ------------------------------------------------------------

required_adj = [
    "disagreement_id",
    "annotation_id",
    "task",
    "title",
    "annotation_excerpt",
    "annotator_A_label",
    "annotator_B_label",
    "allowed_labels",
    "adjudicated_label",
    "adjudication_notes",
]

if list(adj.columns) != required_adj:
    raise RuntimeError(
        "Received adjudication schema mismatch.\n"
        f"Observed: {list(adj.columns)}"
    )

if list(d.columns) != required_adj:
    raise RuntimeError(
        "Frozen disagreement schema mismatch."
    )


immutable = [
    "disagreement_id",
    "annotation_id",
    "task",
    "title",
    "annotation_excerpt",
    "annotator_A_label",
    "annotator_B_label",
    "allowed_labels",
]

for col in immutable:
    if adj[col].tolist() != d[col].tolist():
        raise RuntimeError(
            f"Adjudication modified frozen column: {col}"
        )


if adj["disagreement_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate disagreement_id in adjudication."
    )

if adj["annotation_id"].eq("").any():
    raise RuntimeError(
        "Blank annotation_id in adjudication."
    )

if adj["adjudicated_label"].eq("").any():
    raise RuntimeError(
        "Blank adjudicated_label detected."
    )


for i, row in adj.iterrows():

    task = row["task"]

    if task not in TASKS:
        raise RuntimeError(
            f"Unknown adjudication task: {task}"
        )

    allowed = TASKS[task]["allowed"]

    if row["adjudicated_label"] not in allowed:
        raise RuntimeError(
            "Adjudicated label outside frozen taxonomy: "
            f"{row['disagreement_id']} -> "
            f"{row['adjudicated_label']}"
        )

    row_allowed = [
        x.strip()
        for x in row["allowed_labels"].split("||")
        if x.strip()
    ]

    if row["adjudicated_label"] not in row_allowed:
        raise RuntimeError(
            "Adjudicated label not present in row allowed_labels: "
            f"{row['disagreement_id']}"
        )


# ------------------------------------------------------------
# Build lookup: exactly one final adjudication per
# task-level disagreement.
# ------------------------------------------------------------

adj_lookup = {}

for _, row in adj.iterrows():

    key = (
        row["annotation_id"],
        row["task"],
    )

    if key in adj_lookup:
        raise RuntimeError(
            f"Duplicate task-level adjudication: {key}"
        )

    adj_lookup[key] = row["adjudicated_label"]


expected_disagreement_keys = {
    (row["annotation_id"], row["task"])
    for _, row in d.iterrows()
}

if set(adj_lookup) != expected_disagreement_keys:
    raise RuntimeError(
        "Adjudication keys do not exactly match "
        "the 7 frozen task-level disagreements."
    )


# ------------------------------------------------------------
# Final labels:
# consensus if A == B;
# adjudication if A != B.
# ------------------------------------------------------------

rows = []

consensus_count = 0
adjudicated_count = 0

for i, aid in enumerate(ids):

    out = {
        "annotation_id": aid,
    }

    # Preserve private identity when present.
    if "doc_id" in private.columns:
        out["doc_id"] = private.loc[i, "doc_id"]

    if "title" in private.columns:
        out["title"] = private.loc[i, "title"]
    else:
        out["title"] = blind.loc[i, "title"]

    if "dapt_text" not in private.columns:
        raise RuntimeError(
            "Private master lacks required dapt_text."
        )

    dapt_text = private.loc[i, "dapt_text"]

    if not dapt_text:
        raise RuntimeError(
            f"Blank dapt_text for {aid}"
        )

    out["dapt_text"] = dapt_text

    for task, spec in TASKS.items():

        la = a.loc[i, spec["A"]]
        lb = b.loc[i, spec["B"]]

        if la not in spec["allowed"]:
            raise RuntimeError(
                f"A invalid {task} label for {aid}: {la}"
            )

        if lb not in spec["allowed"]:
            raise RuntimeError(
                f"B invalid {task} label for {aid}: {lb}"
            )

        if la == lb:

            final_label = la
            consensus_count += 1

        else:

            key = (aid, task)

            if key not in adj_lookup:
                raise RuntimeError(
                    f"Missing adjudication for disagreement {key}"
                )

            final_label = adj_lookup[key]
            adjudicated_count += 1

        out[spec["final"]] = final_label

    rows.append(out)


gold = pd.DataFrame(rows)


# ------------------------------------------------------------
# Final integrity checks.
# ------------------------------------------------------------

required_final = [
    "annotation_id",
    "dapt_text",
    "final_primary_frame_validated",
    "final_stance_validated",
    "final_misinformation_relation_validated",
]

for c in required_final:
    if c not in gold.columns:
        raise RuntimeError(
            f"Final gold missing required column: {c}"
        )

if len(gold) != 35:
    raise RuntimeError(
        "Final gold does not contain exactly 35 documents."
    )

if gold["annotation_id"].duplicated().any():
    raise RuntimeError(
        "Final gold contains duplicate annotation_id."
    )

if gold[required_final].eq("").any().any():
    raise RuntimeError(
        "Final gold contains required blank values."
    )


for task, spec in TASKS.items():

    invalid = sorted(
        set(gold[spec["final"]])
        - set(spec["allowed"])
    )

    if invalid:
        raise RuntimeError(
            f"Final gold invalid {task} labels: {invalid}"
        )


# dapt_text must reproduce the exact annotation excerpt prefix.
for i in range(len(gold)):

    prefix = clean(gold.loc[i, "dapt_text"][:1400])

    if prefix != blind.loc[i, "annotation_excerpt"]:
        raise RuntimeError(
            "Final dapt_text does not reproduce frozen "
            f"annotation excerpt for {gold.loc[i, 'annotation_id']}"
        )


if consensus_count != 98:
    raise RuntimeError(
        f"Expected 98 consensus task decisions, got {consensus_count}"
    )

if adjudicated_count != 7:
    raise RuntimeError(
        f"Expected 7 adjudicated task decisions, got {adjudicated_count}"
    )


gold.to_csv(
    GOLD_PATH,
    index=False,
    lineterminator="\n",
)


# ------------------------------------------------------------
# QC + report.
# ------------------------------------------------------------

distributions = {
    "primary_frame": dict(
        Counter(
            gold["final_primary_frame_validated"]
        )
    ),
    "stance": dict(
        Counter(
            gold["final_stance_validated"]
        )
    ),
    "misinformation_relation": dict(
        Counter(
            gold[
                "final_misinformation_relation_validated"
            ]
        )
    ),
}

qc = {
    "rows": 35,
    "unique_annotation_ids": int(
        gold["annotation_id"].nunique()
    ),
    "required_labels_complete": True,
    "dapt_text_complete": True,
    "dapt_text_matches_frozen_annotation_excerpt_prefix": True,
    "consensus_task_decisions": consensus_count,
    "adjudicated_task_decisions": adjudicated_count,
    "total_task_decisions": consensus_count + adjudicated_count,
    "model_predictions_used_for_adjudication": False,
    "eventgold_performance_used_for_adjudication": False,
    "final_label_distributions": distributions,
}

QC_PATH.write_text(
    json.dumps(
        qc,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


lines = [
    "EVENTGOLD35 SUPERVISED EXTENSION — FINAL HUMAN GOLD v1",
    "======================================================",
    "",
    "Documents: 35",
    "Tasks per document: 3",
    "Total task-level final labels: 105",
    f"Consensus task-level labels: {consensus_count}",
    f"Adjudicated task-level labels: {adjudicated_count}",
    "",
    "Inter-annotator agreement was computed before adjudication.",
    "Only the 7 frozen task-level disagreements were adjudicated.",
    "No model predictions, probabilities, logits, or EventGold",
    "performance results were used during annotation or adjudication.",
    "",
    "FINAL LABEL DISTRIBUTIONS",
    "-------------------------",
]

for task, dist in distributions.items():

    lines.append(task)

    for label, n in sorted(dist.items()):
        lines.append(
            f"  {label}: {n}"
        )

    lines.append("")


REPORT_PATH.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)


print("ADJUDICATION_ROWS=7")
print("ADJUDICATION_SCHEMA_VALID=YES")
print("FROZEN_COLUMNS_UNCHANGED=YES")
print("ADJUDICATION_LABELS_VALID=YES")
print("ADJUDICATION_KEYS_MATCH_FROZEN_DISAGREEMENTS=YES")
print()
print("FINAL_GOLD_ROWS=35")
print("FINAL_GOLD_UNIQUE_IDS=35")
print("FINAL_GOLD_REQUIRED_LABELS_COMPLETE=YES")
print("FINAL_GOLD_DAPT_TEXT_COMPLETE=YES")
print("FINAL_GOLD_TEXT_MATCHES_FROZEN_PACKET=YES")
print()
print(f"CONSENSUS_TASK_DECISIONS={consensus_count}")
print(f"ADJUDICATED_TASK_DECISIONS={adjudicated_count}")
print(
    "TOTAL_FINAL_TASK_DECISIONS="
    + str(consensus_count + adjudicated_count)
)
print()
print("FINAL_GOLD_VALIDATION=PASS")
print("MODEL_PREDICTIONS_USED_FOR_ADJUDICATION=NO")
print("EVENTGOLD_PERFORMANCE_USED_FOR_ADJUDICATION=NO")

print()
print("FINAL_LABEL_DISTRIBUTIONS=")
print(
    json.dumps(
        distributions,
        ensure_ascii=False,
        sort_keys=True,
    )
)
