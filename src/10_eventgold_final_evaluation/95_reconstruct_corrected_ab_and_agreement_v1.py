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

BLIND_PATH = EXT / "EventGold35_BLIND_MASTER_v1.csv"

A_ORIG = EXT / "EventGold35_ANNOTATOR_A_RECEIVED_ORIGINAL_v1.xlsx"
B_ORIG = EXT / "EventGold35_ANNOTATOR_B_RECEIVED_ORIGINAL_v1.xlsx"

A_RC = EXT / "EventGold35_ANNOTATOR_A_RECHECK_8_COMPLETED_RECEIVED_v1.xlsx"
B_RC = EXT / "EventGold35_ANNOTATOR_B_RECHECK_8_COMPLETED_RECEIVED_v1.xlsx"

A_FINAL = EXT / "EventGold35_ANNOTATOR_A_CORRECTED_35_v1.csv"
B_FINAL = EXT / "EventGold35_ANNOTATOR_B_CORRECTED_35_v1.csv"

AGREEMENT_PATH = EXT / "EventGold35_INTERANNOTATOR_AGREEMENT_AFTER_RECHECK_v1.csv"

REPORT_PATH = (
    EXT
    / "EventGold35_INTERANNOTATOR_AGREEMENT_AFTER_RECHECK_REPORT_v1.txt"
)

DISAGREE_PATH = (
    EXT
    / "EventGold35_DISAGREEMENTS_FOR_ADJUDICATION_AFTER_RECHECK_v1.csv"
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

RECHECK_IDS = [
    "EG35_ANN_003",
    "EG35_ANN_007",
    "EG35_ANN_009",
    "EG35_ANN_011",
    "EG35_ANN_013",
    "EG35_ANN_016",
    "EG35_ANN_017",
    "EG35_ANN_021",
]


def clean(x):
    if pd.isna(x):
        return ""

    x = str(x).replace("\r", " ").replace("\n", " ")
    return " ".join(x.split()).strip()


def read_original(path):

    df = pd.read_excel(
        path,
        sheet_name="Annotation",
        header=3,
        dtype=str,
        keep_default_na=False,
    )

    required = [
        "annotation_id",
        "Title",
        "Excerpt",
        "Primary frame",
        "Stance toward AI in health",
        "Relation to misinformation",
        "Optional notes",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{path.name}: missing original columns {missing}"
        )

    df = df[required].copy()

    for c in required:
        df[c] = df[c].map(clean)

    if len(df) != 35:
        raise RuntimeError(
            f"{path.name}: expected 35 rows, found {len(df)}"
        )

    if df["annotation_id"].duplicated().any():
        raise RuntimeError(
            f"{path.name}: duplicate annotation IDs"
        )

    return df


def read_recheck(path):

    df = pd.read_excel(
        path,
        sheet_name="Annotation",
        header=3,
        dtype=str,
        keep_default_na=False,
    )

    required = [
        "annotation_id",
        "Title",
        "Exact frozen excerpt",
        "Primary frame",
        "Stance toward AI in health",
        "Relation to misinformation",
        "Optional notes",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{path.name}: missing recheck columns {missing}"
        )

    df = df[required].copy()

    for c in required:
        df[c] = df[c].map(clean)

    if len(df) != 8:
        raise RuntimeError(
            f"{path.name}: expected 8 rows, found {len(df)}"
        )

    if df["annotation_id"].duplicated().any():
        raise RuntimeError(
            f"{path.name}: duplicate annotation IDs"
        )

    return df


def validate_labels(df, path_name):

    specs = [
        ("Primary frame", FRAME),
        ("Stance toward AI in health", STANCE),
        ("Relation to misinformation", MISINFO),
    ]

    for col, allowed in specs:

        if df[col].eq("").any():
            ids = df.loc[
                df[col].eq(""),
                "annotation_id"
            ].tolist()

            raise RuntimeError(
                f"{path_name}: blank labels in {col}: {ids}"
            )

        invalid = sorted(
            set(df[col]) - set(allowed)
        )

        if invalid:
            raise RuntimeError(
                f"{path_name}: invalid labels in {col}: {invalid}"
            )


def cohen_kappa(a, b):

    n = len(a)

    if n == 0 or n != len(b):
        raise RuntimeError("Invalid vectors for Cohen kappa.")

    po = sum(x == y for x, y in zip(a, b)) / n

    ca = Counter(a)
    cb = Counter(b)

    categories = sorted(set(a) | set(b))

    pe = sum(
        (ca[c] / n) * (cb[c] / n)
        for c in categories
    )

    if math.isclose(1.0 - pe, 0.0):
        return float("nan")

    return (po - pe) / (1.0 - pe)


# ------------------------------------------------------------
# Frozen blind source.
# ------------------------------------------------------------

blind = pd.read_csv(
    BLIND_PATH,
    dtype=str,
    keep_default_na=False,
)

for c in blind.columns:
    blind[c] = blind[c].map(clean)

expected_schema = [
    "annotation_id",
    "title",
    "annotation_excerpt",
]

if list(blind.columns) != expected_schema:
    raise RuntimeError(
        "Frozen blind-master schema mismatch."
    )

if len(blind) != 35:
    raise RuntimeError(
        "Frozen blind master must contain 35 rows."
    )

if blind["annotation_id"].duplicated().any():
    raise RuntimeError(
        "Frozen blind master contains duplicate IDs."
    )


# ------------------------------------------------------------
# Read originals and corrective rechecks.
# ------------------------------------------------------------

a0 = read_original(A_ORIG)
b0 = read_original(B_ORIG)

ar = read_recheck(A_RC)
br = read_recheck(B_RC)

validate_labels(a0, A_ORIG.name)
validate_labels(b0, B_ORIG.name)
validate_labels(ar, A_RC.name)
validate_labels(br, B_RC.name)


# ------------------------------------------------------------
# Validate identities/order against frozen packet.
# ------------------------------------------------------------

expected_35 = blind["annotation_id"].tolist()

if a0["annotation_id"].tolist() != expected_35:
    raise RuntimeError(
        "Original A IDs/order differ from frozen packet."
    )

if b0["annotation_id"].tolist() != expected_35:
    raise RuntimeError(
        "Original B IDs/order differ from frozen packet."
    )

if ar["annotation_id"].tolist() != RECHECK_IDS:
    raise RuntimeError(
        "A recheck IDs/order differ from frozen corrective set."
    )

if br["annotation_id"].tolist() != RECHECK_IDS:
    raise RuntimeError(
        "B recheck IDs/order differ from frozen corrective set."
    )


# ------------------------------------------------------------
# Recheck rows must use exact frozen title/excerpt.
# ------------------------------------------------------------

blind_idx = blind.set_index("annotation_id")

for name, df in [("A", ar), ("B", br)]:

    for _, row in df.iterrows():

        aid = row["annotation_id"]

        if row["Title"] != blind_idx.loc[aid, "title"]:
            raise RuntimeError(
                f"{name} recheck title mismatch for {aid}"
            )

        if (
            row["Exact frozen excerpt"]
            != blind_idx.loc[aid, "annotation_excerpt"]
        ):
            raise RuntimeError(
                f"{name} recheck excerpt mismatch for {aid}"
            )


# ------------------------------------------------------------
# Reconstruct corrected 35-row independent annotations.
#
# 27 unaffected rows: retain original labels/notes.
# 8 affected rows: substitute independent corrective labels/notes.
#
# Text columns always come from the frozen blind master.
# ------------------------------------------------------------

def reconstruct(original, recheck, suffix):

    original = original.set_index("annotation_id")
    recheck = recheck.set_index("annotation_id")

    rows = []

    for _, brow in blind.iterrows():

        aid = brow["annotation_id"]

        if aid in RECHECK_IDS:

            src = recheck.loc[aid]

            frame = src["Primary frame"]
            stance = src["Stance toward AI in health"]
            misinfo = src["Relation to misinformation"]
            notes = src["Optional notes"]

            source = "corrective_recheck"

        else:

            src = original.loc[aid]

            frame = src["Primary frame"]
            stance = src["Stance toward AI in health"]
            misinfo = src["Relation to misinformation"]
            notes = src["Optional notes"]

            source = "original_annotation"

        rows.append({
            "annotation_id": aid,
            "title": brow["title"],
            "annotation_excerpt": brow["annotation_excerpt"],
            f"human_primary_frame_{suffix}": frame,
            f"human_stance_{suffix}": stance,
            f"human_misinformation_relation_{suffix}": misinfo,
            f"human_notes_{suffix}": notes,
            f"annotation_source_{suffix}": source,
        })

    out = pd.DataFrame(rows)

    if len(out) != 35:
        raise RuntimeError(
            f"Corrected reviewer {suffix} does not contain 35 rows."
        )

    return out


a = reconstruct(a0, ar, "A")
b = reconstruct(b0, br, "B")

a.to_csv(
    A_FINAL,
    index=False,
    lineterminator="\n",
)

b.to_csv(
    B_FINAL,
    index=False,
    lineterminator="\n",
)


# ------------------------------------------------------------
# Agreement BEFORE adjudication.
# ------------------------------------------------------------

tasks = [
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

counter = 0

for task, ca, cb, allowed in tasks:

    va = a[ca].tolist()
    vb = b[cb].tolist()

    agree = sum(
        x == y
        for x, y in zip(va, vb)
    )

    disagree = len(va) - agree

    raw = agree / len(va)

    kap = cohen_kappa(
        va,
        vb,
    )

    agreement_rows.append({
        "task": task,
        "n": len(va),
        "agreements": agree,
        "disagreements": disagree,
        "raw_agreement": raw,
        "cohen_kappa": kap,
    })

    for i, (x, y) in enumerate(zip(va, vb)):

        if x == y:
            continue

        counter += 1

        disagreement_rows.append({
            "disagreement_id":
                f"EG35_ADJ_{counter:03d}",

            "annotation_id":
                a.loc[i, "annotation_id"],

            "task":
                task,

            "title":
                a.loc[i, "title"],

            "annotation_excerpt":
                a.loc[i, "annotation_excerpt"],

            "annotator_A_label":
                x,

            "annotator_B_label":
                y,

            "allowed_labels":
                " || ".join(allowed),

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

lines = [
    "EVENTGOLD35 INTER-ANNOTATOR AGREEMENT AFTER EXACT-TEXT RECHECK v1",
    "================================================================",
    "",
    "Agreement was calculated before adjudication.",
    "The 8 corrective rechecks replaced only the corresponding",
    "independent labels for the mechanically identified affected rows.",
    "The remaining 27 annotations were retained unchanged.",
    "",
]

for r in agreement_rows:

    lines.append(
        f"{r['task']}: "
        f"{r['agreements']}/{r['n']} "
        f"({100*r['raw_agreement']:.2f}%), "
        f"Cohen_kappa={r['cohen_kappa']:.6f}, "
        f"disagreements={r['disagreements']}"
    )

lines.extend([
    "",
    f"TOTAL_TASK_LEVEL_DISAGREEMENTS={len(disagreement_rows)}",
    "",
    "MODEL_PREDICTIONS_USED=NO",
    "EVENTGOLD_MODEL_PERFORMANCE_USED=NO",
])

REPORT_PATH.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)


# ------------------------------------------------------------
# Console state.
# ------------------------------------------------------------

print("A_RECHECK_ROWS=8")
print("B_RECHECK_ROWS=8")
print("A_RECHECK_LABELS_VALID=YES")
print("B_RECHECK_LABELS_VALID=YES")
print("A_RECHECK_MATCHES_FROZEN_TEXT=YES")
print("B_RECHECK_MATCHES_FROZEN_TEXT=YES")
print("CORRECTED_A_ROWS=35")
print("CORRECTED_B_ROWS=35")

print()

for r in agreement_rows:

    key = r["task"].upper()

    print(
        f"{key}_AGREEMENT="
        f"{r['agreements']}/{r['n']} "
        f"({100*r['raw_agreement']:.2f}%)"
    )

    print(
        f"{key}_KAPPA="
        f"{r['cohen_kappa']:.6f}"
    )

    print(
        f"{key}_DISAGREEMENTS="
        f"{r['disagreements']}"
    )

print()

print(
    f"TOTAL_TASK_LEVEL_DISAGREEMENTS="
    f"{len(disagreement_rows)}"
)

print("AGREEMENT_COMPUTED_BEFORE_ADJUDICATION=YES")
print("MODEL_PREDICTIONS_USED=NO")
print("EVENTGOLD_PERFORMANCE_USED=NO")
print("CORRECTED_INDEPENDENT_ANNOTATION_VALIDATION=PASS")
