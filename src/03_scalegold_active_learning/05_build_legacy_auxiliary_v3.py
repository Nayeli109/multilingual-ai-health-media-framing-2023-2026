from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path.cwd()

LEGACY_PATH = (
    ROOT
    / "results/quality_sensitivity_v1/"
      "final_human_validated_quality_filtered_v1.csv"
)

STRICT_DAPT_PATH = (
    ROOT
    / "01_event_aware_v3/01_data_audit/"
      "DAPT_STRICT_EVENT_HOLDOUT_v3.csv"
)

GOLD_PATH = (
    ROOT
    / "01_event_aware_v3/00_frozen_benchmark/"
      "EventGold35_FINAL.csv"
)

EXCLUSIONS_PATH = (
    ROOT
    / "01_event_aware_v3/01_data_audit/"
      "DAPT_STRICT_EVENT_HOLDOUT_exclusions_v3.csv"
)

OUTDIR = (
    ROOT
    / "01_event_aware_v3/03_scalegold_active_learning"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

FROZEN_LEGACY_OUT = (
    OUTDIR / "LegacyGold209_quality_filtered_FROZEN_v3.csv"
)

AUX_OUT = (
    OUTDIR / "LegacyAux_leakage_free_v3.csv"
)

QC_OUT = (
    OUTDIR / "LegacyAux_leakage_free_QC_v3.json"
)

HASH_OUT = (
    OUTDIR / "LegacyAux_SHA256SUMS_v3.txt"
)


def clean_id(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


# ============================================================
# LOAD
# ============================================================

legacy = pd.read_csv(
    LEGACY_PATH,
    low_memory=False
)

strict_dapt = pd.read_csv(
    STRICT_DAPT_PATH,
    low_memory=False
)

gold = pd.read_csv(
    GOLD_PATH,
    low_memory=False
)

exclusions = pd.read_csv(
    EXCLUSIONS_PATH,
    low_memory=False
)


# ============================================================
# BASIC QC
# ============================================================

if len(legacy) != 209:
    raise RuntimeError(
        f"Expected 209 historical quality-filtered annotations; "
        f"observed {len(legacy)}."
    )

for name, df in [
    ("legacy", legacy),
    ("strict_dapt", strict_dapt),
    ("gold", gold),
    ("exclusions", exclusions),
]:
    if "doc_id" not in df.columns:
        raise RuntimeError(
            f"{name} does not contain doc_id."
        )

    df["doc_id"] = (
        df["doc_id"]
        .map(clean_id)
    )


# ============================================================
# TARGETS
# ============================================================

target_cols = [
    "final_primary_frame_validated",
    "final_stance_validated",
    "final_misinformation_relation_validated",
]

missing_targets = [
    c for c in target_cols
    if c not in legacy.columns
]

if missing_targets:
    raise RuntimeError(
        "Missing historical target columns: "
        + str(missing_targets)
    )


# ============================================================
# FREEZE ORIGINAL 209
# ============================================================

legacy.to_csv(
    FROZEN_LEGACY_OUT,
    index=False
)


# ============================================================
# LEAKAGE AUDIT
# ============================================================

gold_ids = set(gold["doc_id"])
exclusion_ids = set(exclusions["doc_id"])
strict_ids = set(strict_dapt["doc_id"])
legacy_ids = set(legacy["doc_id"])

gold_overlap = (
    legacy_ids
    & gold_ids
)

known_event_neighbor_overlap = (
    legacy_ids
    & exclusion_ids
)

strict_overlap = (
    legacy_ids
    & strict_ids
)


# ============================================================
# BUILD LEAKAGE-FREE AUX SET
#
# Inner join with strict DAPT guarantees:
# - no EventGold documents
# - no known EventGold candidate-event neighbors
# - text is available
# ============================================================

legacy_labels = legacy[
    [
        "doc_id",
        *target_cols,
    ]
].copy()

optional_legacy_cols = [
    "final_secondary_frame",
    "primary_frame_agreement",
    "stance_agreement",
    "misinformation_relation_agreement",
    "topic",
    "final_topic_label",
]

for c in optional_legacy_cols:
    if c in legacy.columns:
        legacy_labels[c] = legacy[c]


dapt_cols = [
    "doc_id",
    "dapt_text",
]

for c in [
    "title",
    "language",
    "region",
    "country",
    "year",
    "published_date",
    "topic_v1",
    "topic_label_v1",
]:
    if c in strict_dapt.columns:
        dapt_cols.append(c)


aux = legacy_labels.merge(
    strict_dapt[dapt_cols],
    on="doc_id",
    how="inner",
    validate="one_to_one",
)


# ============================================================
# TARGET COMPLETENESS
# ============================================================

for c in target_cols:

    aux[c] = (
        aux[c]
        .fillna("")
        .astype(str)
        .str.strip()
    )

complete_mask = (
    aux[target_cols]
    .ne("")
    .all(axis=1)
)

aux_complete = (
    aux.loc[complete_mask]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# FINAL LEAKAGE ASSERTIONS
# ============================================================

final_ids = set(aux_complete["doc_id"])

if final_ids & gold_ids:
    raise RuntimeError(
        "EventGold leakage remained."
    )

if final_ids & exclusion_ids:
    raise RuntimeError(
        "Known event-neighbor leakage remained."
    )

if not final_ids <= strict_ids:
    raise RuntimeError(
        "LegacyAux contains documents outside strict DAPT corpus."
    )


# ============================================================
# SAVE
# ============================================================

aux_complete.to_csv(
    AUX_OUT,
    index=False
)


# ============================================================
# DISTRIBUTIONS
# ============================================================

distributions = {}

for c in target_cols:

    distributions[c] = (
        aux_complete[c]
        .value_counts(dropna=False)
        .to_dict()
    )


qc = {
    "historical_quality_filtered_rows": int(
        len(legacy)
    ),

    "historical_eventgold_overlap": int(
        len(gold_overlap)
    ),

    "historical_known_event_neighbor_overlap": int(
        len(known_event_neighbor_overlap)
    ),

    "historical_rows_present_in_strict_dapt": int(
        len(strict_overlap)
    ),

    "legacy_aux_after_strict_join": int(
        len(aux)
    ),

    "legacy_aux_complete_targets": int(
        len(aux_complete)
    ),

    "final_eventgold_overlap": int(
        len(final_ids & gold_ids)
    ),

    "final_known_event_neighbor_overlap": int(
        len(final_ids & exclusion_ids)
    ),

    "targets": target_cols,

    "target_distributions": distributions,

    "language_distribution": (
        aux_complete["language"]
        .value_counts(dropna=False)
        .to_dict()
        if "language" in aux_complete.columns
        else None
    ),

    "region_distribution": (
        aux_complete["region"]
        .value_counts(dropna=False)
        .to_dict()
        if "region" in aux_complete.columns
        else None
    ),

    "frozen_legacy_sha256": sha256(
        FROZEN_LEGACY_OUT
    ),

    "legacy_aux_sha256": sha256(
        AUX_OUT
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


with HASH_OUT.open(
    "w",
    encoding="utf-8"
) as f:

    for path in [
        FROZEN_LEGACY_OUT,
        AUX_OUT,
        QC_OUT,
    ]:

        f.write(
            f"{sha256(path)}  "
            f"{path.relative_to(ROOT)}\n"
        )


# ============================================================
# REPORT
# ============================================================

print("==============================================")
print("LEGACY HUMAN AUXILIARY AUDIT")
print("==============================================")

print(
    "Historical quality-filtered set :",
    len(legacy)
)

print(
    "Overlap with EventGold-35       :",
    len(gold_overlap)
)

print(
    "Overlap with known neighbors    :",
    len(known_event_neighbor_overlap)
)

print(
    "Present in strict DAPT corpus   :",
    len(strict_overlap)
)

print(
    "Aux rows after strict join      :",
    len(aux)
)

print(
    "Complete auxiliary labels       :",
    len(aux_complete)
)

print(
    "Final EventGold overlap         :",
    len(final_ids & gold_ids)
)

print(
    "Final event-neighbor overlap    :",
    len(final_ids & exclusion_ids)
)


for c in target_cols:

    print("\n----------------------------------------------")
    print(c)
    print("----------------------------------------------")

    print(
        aux_complete[c]
        .value_counts(dropna=False)
        .to_string()
    )


if "language" in aux_complete.columns:

    print("\n----------------------------------------------")
    print("LANGUAGE")
    print("----------------------------------------------")

    print(
        aux_complete["language"]
        .value_counts(dropna=False)
        .to_string()
    )


if "region" in aux_complete.columns:

    print("\n----------------------------------------------")
    print("REGION")
    print("----------------------------------------------")

    print(
        aux_complete["region"]
        .value_counts(dropna=False)
        .to_string()
    )


print("\nSaved:")
print(FROZEN_LEGACY_OUT)
print(AUX_OUT)
print(QC_OUT)
print(HASH_OUT)
