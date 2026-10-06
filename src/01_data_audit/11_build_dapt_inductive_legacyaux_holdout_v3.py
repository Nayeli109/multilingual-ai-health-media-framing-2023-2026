from pathlib import Path
import hashlib
import json
import re
import unicodedata

import pandas as pd


ROOT = Path.cwd()

STRICT_PATH = (
    ROOT
    / "01_event_aware_v3/01_data_audit/"
      "DAPT_STRICT_EVENT_HOLDOUT_v3.csv"
)

LEGACY_PATH = (
    ROOT
    / "01_event_aware_v3/03_scalegold_active_learning/"
      "LegacyAux_leakage_free_v3.csv"
)

OUT_PATH = (
    ROOT
    / "01_event_aware_v3/01_data_audit/"
      "DAPT_INDUCTIVE_LEGACYAUX_HOLDOUT_v3.csv"
)

EXCL_PATH = (
    ROOT
    / "01_event_aware_v3/01_data_audit/"
      "DAPT_INDUCTIVE_LEGACYAUX_HOLDOUT_exclusions_v3.csv"
)

QC_PATH = (
    ROOT
    / "01_event_aware_v3/01_data_audit/"
      "DAPT_INDUCTIVE_LEGACYAUX_HOLDOUT_QC_v3.json"
)


def norm_text(x):
    if pd.isna(x):
        return ""

    x = unicodedata.normalize(
        "NFKC",
        str(x)
    )

    x = x.lower().strip()
    x = re.sub(r"\s+", " ", x)

    return x


def digest(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


strict = pd.read_csv(
    STRICT_PATH,
    low_memory=False
)

legacy = pd.read_csv(
    LEGACY_PATH,
    low_memory=False
)

if len(strict) != 1472:
    raise RuntimeError(
        f"Expected strict DAPT n=1472; "
        f"observed {len(strict)}."
    )

if len(legacy) != 199:
    raise RuntimeError(
        f"Expected LegacyAux n=199; "
        f"observed {len(legacy)}."
    )

for name, df in [
    ("strict", strict),
    ("legacy", legacy),
]:
    if "doc_id" not in df.columns:
        raise RuntimeError(
            f"{name}: missing doc_id."
        )

    df["doc_id"] = (
        df["doc_id"]
        .astype(str)
        .str.strip()
    )


legacy_ids = set(
    legacy["doc_id"]
)

strict_ids = set(
    strict["doc_id"]
)

if not legacy_ids <= strict_ids:
    missing = legacy_ids - strict_ids

    raise RuntimeError(
        "Some LegacyAux documents are outside "
        f"strict DAPT: {len(missing)}"
    )


# ============================================================
# DOCUMENT-ID HOLDOUT
# ============================================================

candidate = (
    strict.loc[
        ~strict["doc_id"].isin(
            legacy_ids
        )
    ]
    .copy()
)

candidate["exclusion_reason"] = ""


# ============================================================
# EXACT NORMALIZED-TEXT DUPLICATES
# ============================================================

if "dapt_text" not in candidate.columns:
    raise RuntimeError(
        "Strict DAPT lacks dapt_text."
    )

legacy_texts = set(
    legacy["dapt_text"]
    .map(norm_text)
)

legacy_texts.discard("")

candidate["_norm_text"] = (
    candidate["dapt_text"]
    .map(norm_text)
)

text_dup = (
    candidate["_norm_text"]
    .isin(legacy_texts)
    &
    candidate["_norm_text"].ne("")
)

candidate.loc[
    text_dup,
    "exclusion_reason"
] = "exact_normalized_text_duplicate"


# ============================================================
# EXACT NORMALIZED-TITLE DUPLICATES
# Conservative anti-syndication check.
# ============================================================

title_dup = pd.Series(
    False,
    index=candidate.index
)

if (
    "title" in candidate.columns
    and
    "title" in legacy.columns
):

    legacy_titles = set(
        legacy["title"]
        .map(norm_text)
    )

    legacy_titles.discard("")

    candidate["_norm_title"] = (
        candidate["title"]
        .map(norm_text)
    )

    title_dup = (
        candidate["_norm_title"]
        .isin(legacy_titles)
        &
        candidate["_norm_title"].ne("")
    )

    candidate.loc[
        title_dup
        &
        candidate[
            "exclusion_reason"
        ].eq(""),
        "exclusion_reason"
    ] = "exact_normalized_title_duplicate"

    candidate.loc[
        title_dup
        &
        candidate[
            "exclusion_reason"
        ].ne("")
        &
        ~candidate[
            "exclusion_reason"
        ].str.contains(
            "title",
            regex=False
        ),
        "exclusion_reason"
    ] += "+exact_normalized_title_duplicate"


additional_exclusions = (
    candidate.loc[
        candidate[
            "exclusion_reason"
        ].ne("")
    ]
    .copy()
)

clean = (
    candidate.loc[
        candidate[
            "exclusion_reason"
        ].eq("")
    ]
    .copy()
)


# ============================================================
# DROP AUDIT-ONLY NORMALIZED COLUMNS
# ============================================================

for d in [
    clean,
    additional_exclusions,
]:
    for c in [
        "_norm_text",
        "_norm_title",
    ]:
        if c in d.columns:
            d.drop(
                columns=c,
                inplace=True
            )


# ============================================================
# ASSERTIONS
# ============================================================

clean_ids = set(
    clean["doc_id"]
)

if clean_ids & legacy_ids:
    raise RuntimeError(
        "LegacyAux doc_id leakage remains."
    )

clean_norm_texts = set(
    clean["dapt_text"]
    .map(norm_text)
)

clean_norm_texts.discard("")

if clean_norm_texts & legacy_texts:
    raise RuntimeError(
        "Exact normalized text leakage remains."
    )


# ============================================================
# SAVE
# ============================================================

clean.to_csv(
    OUT_PATH,
    index=False
)

additional_exclusions.to_csv(
    EXCL_PATH,
    index=False
)

qc = {
    "strict_dapt_input": int(
        len(strict)
    ),

    "legacyaux_holdout_documents": int(
        len(legacy)
    ),

    "after_doc_id_holdout": int(
        len(candidate)
    ),

    "additional_exact_text_or_title_exclusions": int(
        len(additional_exclusions)
    ),

    "final_inductive_dapt_documents": int(
        len(clean)
    ),

    "final_legacyaux_doc_id_overlap": int(
        len(clean_ids & legacy_ids)
    ),

    "final_exact_normalized_text_overlap": int(
        len(
            clean_norm_texts
            & legacy_texts
        )
    ),
}

QC_PATH.write_text(
    json.dumps(
        qc,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print("=" * 70)
print("INDUCTIVE DAPT HOLDOUT QC")
print("=" * 70)

for k, v in qc.items():
    print(f"{k}: {v}")

print("")
print(
    "Output SHA256:",
    digest(OUT_PATH)
)

print("")
print("Saved:")
print(OUT_PATH)
print(EXCL_PATH)
print(QC_PATH)
