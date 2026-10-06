from pathlib import Path
from collections import Counter
import math

import pandas as pd

ROOT = Path.cwd()

FINAL = (
    ROOT
    / "01_event_aware_v3"
    / "10_eventgold_final_evaluation"
)

EXT = FINAL / "eventgold35_supervised_extension_v1"

A_PATH = EXT / "EventGold35_ANNOTATOR_A_RECEIVED_ORIGINAL_v1.xlsx"
B_PATH = EXT / "EventGold35_ANNOTATOR_B_RECEIVED_ORIGINAL_v1.xlsx"

BLIND_PATH = EXT / "EventGold35_BLIND_MASTER_v1.csv"

A_STD = EXT / "EventGold35_ANNOTATOR_A_STANDARDIZED_v1.csv"
B_STD = EXT / "EventGold35_ANNOTATOR_B_STANDARDIZED_v1.csv"

AGREEMENT_PATH = EXT / "EventGold35_INTERANNOTATOR_AGREEMENT_v1.csv"

DISAGREE_PATH = (
    EXT
    / "EventGold35_DISAGREEMENTS_FOR_ADJUDICATION_v1.csv"
)

REPORT_PATH = (
    EXT
    / "EventGold35_INTERANNOTATOR_AGREEMENT_REPORT_v1.txt"
)


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


def clean_text(x):
    if pd.isna(x):
        return ""

    x = str(x).replace("\r", " ").replace("\n", " ")
    return " ".join(x.split()).strip()


def clean_label(x):
    return clean_text(x)


def kappa(a, b):

    if len(a) != len(b):
        raise RuntimeError("Kappa vectors differ in length.")

    n = len(a)

    if n == 0:
        raise RuntimeError("Empty kappa vectors.")

    po = sum(
        x == y
        for x, y in zip(a, b)
    ) / n

    categories = sorted(
        set(a) | set(b)
    )

    ca = Counter(a)
    cb = Counter(b)

    pe = sum(
        (ca[c] / n) * (cb[c] / n)
        for c in categories
    )

    if math.isclose(1.0 - pe, 0.0):
        return float("nan")

    return (po - pe) / (1.0 - pe)


def read_reviewer(path, suffix):

    try:
        df = pd.read_excel(
            path,
            sheet_name="Annotation",
            header=3,
            dtype=str,
            keep_default_na=False,
        )
    except ImportError as e:
        raise RuntimeError(
            "Reading XLSX requires openpyxl in this environment."
        ) from e

    required = [
        "annotation_id",
        "Title",
        "Excerpt",
        "Primary frame",
        "Stance toward AI in health",
        "Relation to misinformation",
        "Optional notes",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Reviewer {suffix}: missing columns {missing}"
        )

    df = df[required].copy()

    for c in required:
        df[c] = df[c].map(clean_text)

    if len(df) != 35:
        raise RuntimeError(
            f"Reviewer {suffix}: expected 35 rows, "
            f"observed {len(df)}."
        )

    if df["annotation_id"].duplicated().any():
        raise RuntimeError(
            f"Reviewer {suffix}: duplicate annotation_id."
        )

    if df["annotation_id"].eq("").any():
        raise RuntimeError(
            f"Reviewer {suffix}: blank annotation_id."
        )

    checks = [
        ("Primary frame", FRAME),
        ("Stance toward AI in health", STANCE),
        ("Relation to misinformation", MISINFO),
    ]

    for col, allowed in checks:

        invalid = sorted(
            set(df[col]) - set(allowed)
        )

        if invalid:
            raise RuntimeError(
                f"Reviewer {suffix}: invalid labels "
                f"in {col}: {invalid}"
            )

    return df


blind = pd.read_csv(
    BLIND_PATH,
    dtype=str,
    keep_default_na=False,
)

for c in blind.columns:
    blind[c] = blind[c].map(clean_text)

if list(blind.columns) != [
    "annotation_id",
    "title",
    "annotation_excerpt",
]:
    raise RuntimeError(
        "Frozen blind master schema mismatch."
    )

if len(blind) != 35:
    raise RuntimeError(
        "Frozen blind master does not contain 35 rows."
    )


a = read_reviewer(A_PATH, "A")
b = read_reviewer(B_PATH, "B")


# ------------------------------------------------------------
# Verify reviewer files correspond exactly to frozen blind packet.
# ------------------------------------------------------------

expected_ids = blind["annotation_id"].tolist()

if a["annotation_id"].tolist() != expected_ids:
    raise RuntimeError(
        "Reviewer A annotation order/IDs differ "
        "from frozen blind packet."
    )

if b["annotation_id"].tolist() != expected_ids:
    raise RuntimeError(
        "Reviewer B annotation order/IDs differ "
        "from frozen blind packet."
    )


for reviewer_name, df in [
    ("A", a),
    ("B", b),
]:

    title_ok = (
        df["Title"].tolist()
        == blind["title"].tolist()
    )

    excerpt_ok = (
        df["Excerpt"].tolist()
        == blind["annotation_excerpt"].tolist()
    )

    if not title_ok:
        raise RuntimeError(
            f"Reviewer {reviewer_name}: titles differ "
            "from frozen blind packet."
        )

    if not excerpt_ok:
        raise RuntimeError(
            f"Reviewer {reviewer_name}: excerpts differ "
            "from frozen blind packet."
        )


# ------------------------------------------------------------
# Standardized immutable reviewer tables.
# ------------------------------------------------------------

def standardize(df, suffix):

    out = pd.DataFrame({
        "annotation_id":
            df["annotation_id"],

        "title":
            df["Title"],

        "annotation_excerpt":
            df["Excerpt"],

        f"human_primary_frame_{suffix}":
            df["Primary frame"],

        f"human_stance_{suffix}":
            df["Stance toward AI in health"],

        f"human_misinformation_relation_{suffix}":
            df["Relation to misinformation"],

        f"human_notes_{suffix}":
            df["Optional notes"],
    })

    return out


a_std = standardize(a, "A")
b_std = standardize(b, "B")

a_std.to_csv(
    A_STD,
    index=False,
    lineterminator="\n",
)

b_std.to_csv(
    B_STD,
    index=False,
    lineterminator="\n",
)


# ------------------------------------------------------------
# Agreement before adjudication.
# ------------------------------------------------------------

task_specs = [
    (
        "primary_frame",
        "human_primary_frame_A",
        "human_primary_frame_B",
        FRAME,
    ),
    (
        "stance",
        "human_stance_A",
        "human_stance_B",
        STANCE,
    ),
    (
        "misinformation_relation",
        "human_misinformation_relation_A",
        "human_misinformation_relation_B",
        MISINFO,
    ),
]


agreement_rows = []
disagreement_rows = []

disagreement_counter = 0

for task, col_a, col_b, options in task_specs:

    va = a_std[col_a].tolist()
    vb = b_std[col_b].tolist()

    agreements = sum(
        x == y
        for x, y in zip(va, vb)
    )

    disagreements = len(va) - agreements

    raw = agreements / len(va)

    kap = kappa(
        va,
        vb,
    )

    agreement_rows.append({
        "task": task,
        "n": len(va),
        "agreements": agreements,
        "disagreements": disagreements,
        "raw_agreement": raw,
        "cohen_kappa": kap,
    })

    for i in range(len(a_std)):

        if va[i] == vb[i]:
            continue

        disagreement_counter += 1

        disagreement_rows.append({
            "disagreement_id":
                f"EG35_ADJ_{disagreement_counter:03d}",

            "annotation_id":
                a_std.loc[i, "annotation_id"],

            "task":
                task,

            "title":
                a_std.loc[i, "title"],

            "annotation_excerpt":
                a_std.loc[i, "annotation_excerpt"],

            "annotator_A_label":
                va[i],

            "annotator_B_label":
                vb[i],

            "allowed_labels":
                " || ".join(options),

            "adjudicated_label":
                "",

            "adjudication_notes":
                "",
        })


agreement = pd.DataFrame(
    agreement_rows
)

disagreement = pd.DataFrame(
    disagreement_rows
)

agreement.to_csv(
    AGREEMENT_PATH,
    index=False,
    lineterminator="\n",
)

disagreement.to_csv(
    DISAGREE_PATH,
    index=False,
    lineterminator="\n",
)


# ------------------------------------------------------------
# Human-readable report.
# ------------------------------------------------------------

lines = []

lines.append(
    "EVENTGOLD35 INTER-ANNOTATOR AGREEMENT v1"
)

lines.append(
    "========================================"
)

lines.append("")

lines.append(
    "Agreement calculated BEFORE adjudication."
)

lines.append("")

for r in agreement_rows:

    lines.append(
        f"{r['task']}: "
        f"{r['agreements']}/{r['n']} "
        f"({100*r['raw_agreement']:.2f}%), "
        f"Cohen_kappa={r['cohen_kappa']:.6f}, "
        f"disagreements={r['disagreements']}"
    )

lines.append("")

lines.append(
    f"TOTAL_TASK_LEVEL_DISAGREEMENTS="
    f"{len(disagreement_rows)}"
)

lines.append("")

lines.append(
    "No model prediction or model performance "
    "was used in this calculation."
)

REPORT_PATH.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)


print("ANNOTATOR_A_ROWS=35")
print("ANNOTATOR_B_ROWS=35")
print("ANNOTATOR_A_LABELS_VALID=YES")
print("ANNOTATOR_B_LABELS_VALID=YES")
print("ANNOTATOR_A_MATCHES_FROZEN_PACKET=YES")
print("ANNOTATOR_B_MATCHES_FROZEN_PACKET=YES")

print()

for r in agreement_rows:

    print(
        f"{r['task'].upper()}_AGREEMENT="
        f"{r['agreements']}/{r['n']} "
        f"({100*r['raw_agreement']:.2f}%)"
    )

    print(
        f"{r['task'].upper()}_KAPPA="
        f"{r['cohen_kappa']:.6f}"
    )

    print(
        f"{r['task'].upper()}_DISAGREEMENTS="
        f"{r['disagreements']}"
    )

print()

print(
    "TOTAL_TASK_LEVEL_DISAGREEMENTS="
    + str(len(disagreement_rows))
)

print(
    "AGREEMENT_COMPUTED_BEFORE_ADJUDICATION=YES"
)

print(
    "MODEL_PREDICTIONS_USED=NO"
)

print(
    "EVENTGOLD_PERFORMANCE_USED=NO"
)

print(
    "INDEPENDENT_ANNOTATION_VALIDATION=PASS"
)
