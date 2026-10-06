#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("01_event_aware_v3")

STAGEA = (
    ROOT / "07_eve_frame_stage_a_no_event" /
    "comparison_results_v3"
)

FINAL = ROOT / "10_eventgold_final_evaluation"
ERES = FINAL / "eventgold35_supervised_extension_final_results_v1"

OUT = ROOT / "12_posthoc_generalization_gap_v1"
RES = OUT / "results_v1"
RES.mkdir(parents=True, exist_ok=True)

PROTOCOL = OUT / "POSTHOC_GENERALIZATION_GAP_PROTOCOL_v1.txt"

STAGE_PRED = STAGEA / "stage_a_oof_ensemble_predictions_v3.csv"
EVENT_PRED = ERES / "eventgold35_supervised_extension_ensemble_predictions_v1.csv"

STAGE_METRICS = STAGEA / "task_metrics_comparison_v3.csv"
EVENT_METRICS = ERES / "eventgold35_supervised_extension_metrics_v1.csv"

EXPECTED_PROTOCOL_SHA = (
    "ac8457cd9d206299140c613dfe92d736"
    "d672965f87e08686daca67f6c2135fe7"
)

TASKS = [
    "primary_frame",
    "stance",
    "misinformation_relation",
]

N_CLASSES = {
    "primary_frame": 9,
    "stance": 5,
    "misinformation_relation": 4,
}

RNG_SEEDS = {
    "primary_frame": 2026100601,
    "stance": 2026100602,
    "misinformation_relation": 2026100603,
}

METRICS = [
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "weighted_f1",
]

B = 10000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metric_bundle(y, pred, k):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    cm = np.bincount(
        k * y + pred,
        minlength=k * k
    ).reshape(k, k)

    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)

    precision = np.divide(
        tp,
        predicted,
        out=np.zeros(k, dtype=float),
        where=predicted > 0,
    )

    recall = np.divide(
        tp,
        support,
        out=np.zeros(k, dtype=float),
        where=support > 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(k, dtype=float),
        where=(precision + recall) > 0,
    )

    total = support.sum()

    return {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "accuracy": float(tp.sum() / total),
        "weighted_f1": float((f1 * support).sum() / total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    }


def stratified_indices(y, rng):
    y = np.asarray(y, dtype=int)
    sampled = []

    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        sampled.append(
            rng.choice(idx, size=len(idx), replace=True)
        )

    return np.concatenate(sampled)


def js_divergence_nats(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(
            np.sum(a[mask] * np.log(a[mask] / b[mask]))
        )

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def predictive_entropy(prob):
    p = np.clip(
        np.asarray(prob, dtype=float),
        1e-15,
        1.0,
    )
    return -np.sum(p * np.log(p), axis=1)


# ------------------------------------------------------------
# FAIL-CLOSED INPUT AUDIT
# ------------------------------------------------------------

if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA:
    raise RuntimeError(
        "PROTOCOL SHA256 MISMATCH — STOP"
    )

for p in [
    STAGE_PRED,
    EVENT_PRED,
    STAGE_METRICS,
    EVENT_METRICS,
]:
    if not p.exists():
        raise FileNotFoundError(p)


stage = pd.read_csv(STAGE_PRED)
event = pd.read_csv(EVENT_PRED)

stage_ref = pd.read_csv(STAGE_METRICS)
event_ref = pd.read_csv(EVENT_METRICS)

if len(stage) != 597:
    raise RuntimeError(
        f"Expected 597 Stage-A rows, found {len(stage)}"
    )

if len(event) != 105:
    raise RuntimeError(
        f"Expected 105 EventGold rows, found {len(event)}"
    )


performance = []
bootstrap_long = []
composition = []
composition_summary = []
per_class = []
uncertainty = []
label_maps = {}


for task in TASKS:

    k = N_CLASSES[task]

    ds = (
        stage.loc[stage["task"] == task]
        .copy()
        .reset_index(drop=True)
    )

    de = (
        event.loc[event["task"] == task]
        .copy()
        .reset_index(drop=True)
    )

    if len(ds) != 199:
        raise RuntimeError(
            f"{task}: Stage-A n != 199"
        )

    if len(de) != 35:
        raise RuntimeError(
            f"{task}: EventGold n != 35"
        )

    ys = ds["gold_id"].to_numpy(dtype=int)
    ps = ds["prediction_id"].to_numpy(dtype=int)

    ye = de["gold_id"].to_numpy(dtype=int)
    pe = de["prediction_id"].to_numpy(dtype=int)

    for arr_name, arr in [
        ("Stage gold", ys),
        ("Stage prediction", ps),
        ("EventGold gold", ye),
        ("EventGold prediction", pe),
    ]:
        if arr.min() < 0 or arr.max() >= k:
            raise RuntimeError(
                f"{task}: {arr_name} class ID out of range"
            )

    # --------------------------------------------------------
    # FROZEN LABEL MAP
    # --------------------------------------------------------

    mapping = {}

    for df in [ds, de]:
        for id_col, label_col in [
            ("gold_id", "gold_label"),
            ("prediction_id", "prediction_label"),
        ]:
            pairs = (
                df[[id_col, label_col]]
                .dropna()
                .drop_duplicates()
            )

            for class_id, label in pairs.itertuples(
                index=False
            ):
                class_id = int(class_id)
                label = str(label)

                if (
                    class_id in mapping
                    and mapping[class_id] != label
                ):
                    raise RuntimeError(
                        f"{task}: inconsistent label map "
                        f"for class {class_id}"
                    )

                mapping[class_id] = label

    if set(mapping.keys()) != set(range(k)):
        raise RuntimeError(
            f"{task}: incomplete label universe: "
            f"{sorted(mapping.keys())}"
        )

    label_maps[task] = {
        str(i): mapping[i]
        for i in range(k)
    }

    # --------------------------------------------------------
    # RECOMPUTE METRICS
    # --------------------------------------------------------

    ms = metric_bundle(ys, ps, k)
    me = metric_bundle(ye, pe, k)

    # Find Stage-A authoritative row robustly.
    sr = stage_ref[
        stage_ref["task"].astype(str).eq(task)
    ].copy()

    if "system" in sr.columns:
        mask = (
            sr["system"]
            .astype(str)
            .str.contains("STAGE_A", case=False)
            &
            sr["system"]
            .astype(str)
            .str.contains("NO_EVENT", case=False)
        )
        sr = sr[mask]

    er = event_ref[
        event_ref["task"].astype(str).eq(task)
    ].copy()

    if len(sr) != 1:
        raise RuntimeError(
            f"{task}: Stage-A authoritative metric "
            f"row count = {len(sr)}"
        )

    if len(er) != 1:
        raise RuntimeError(
            f"{task}: EventGold authoritative metric "
            f"row count = {len(er)}"
        )

    for metric in METRICS:

        s_expected = float(sr.iloc[0][metric])
        e_expected = float(er.iloc[0][metric])

        if not np.isclose(
            ms[metric],
            s_expected,
            atol=1e-12,
            rtol=0,
        ):
            raise RuntimeError(
                f"{task} Stage-A {metric} mismatch: "
                f"recomputed={ms[metric]} "
                f"frozen={s_expected}"
            )

        if not np.isclose(
            me[metric],
            e_expected,
            atol=1e-12,
            rtol=0,
        ):
            raise RuntimeError(
                f"{task} EventGold {metric} mismatch: "
                f"recomputed={me[metric]} "
                f"frozen={e_expected}"
            )

    # --------------------------------------------------------
    # INDEPENDENT STRATIFIED BOOTSTRAP
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RNG_SEEDS[task]
    )

    boot = {
        m: np.empty(B, dtype=float)
        for m in METRICS
    }

    for b in range(B):

        idx_s = stratified_indices(
            ys,
            rng,
        )

        idx_e = stratified_indices(
            ye,
            rng,
        )

        bs = metric_bundle(
            ys[idx_s],
            ps[idx_s],
            k,
        )

        be = metric_bundle(
            ye[idx_e],
            pe[idx_e],
            k,
        )

        for metric in METRICS:
            boot[metric][b] = (
                be[metric] - bs[metric]
            )

    for metric in METRICS:

        gap = me[metric] - ms[metric]

        relative = (
            gap / ms[metric]
            if ms[metric] != 0
            else np.nan
        )

        lo, hi = np.quantile(
            boot[metric],
            [0.025, 0.975],
            method="linear",
        )

        performance.append({
            "task": task,
            "metric": metric,
            "legacyaux_value": ms[metric],
            "eventgold_value": me[metric],
            "gap_eventgold_minus_legacyaux": gap,
            "relative_change": relative,
            "bootstrap_ci95_low": float(lo),
            "bootstrap_ci95_high": float(hi),
            "bootstrap_replicates": B,
            "bootstrap_seed": RNG_SEEDS[task],
        })

        for b, value in enumerate(
            boot[metric],
            start=1,
        ):
            bootstrap_long.append({
                "task": task,
                "metric": metric,
                "replicate": b,
                "gap_eventgold_minus_legacyaux":
                    float(value),
            })

    # --------------------------------------------------------
    # LABEL COMPOSITION
    # --------------------------------------------------------

    cs = np.bincount(
        ys,
        minlength=k,
    )

    ce = np.bincount(
        ye,
        minlength=k,
    )

    prop_s = cs / cs.sum()
    prop_e = ce / ce.sum()

    tv = (
        0.5
        * np.abs(prop_e - prop_s).sum()
    )

    js = js_divergence_nats(
        prop_s,
        prop_e,
    )

    composition_summary.append({
        "task": task,
        "total_variation_distance": float(tv),
        "jensen_shannon_divergence_nats":
            float(js),
    })

    for c in range(k):

        composition.append({
            "task": task,
            "class_id": c,
            "class_label": mapping[c],
            "legacyaux_support": int(cs[c]),
            "legacyaux_proportion":
                float(prop_s[c]),
            "eventgold_support": int(ce[c]),
            "eventgold_proportion":
                float(prop_e[c]),
            "signed_proportion_difference_"
            "eventgold_minus_legacyaux":
                float(prop_e[c] - prop_s[c]),
            "absolute_proportion_difference":
                float(
                    abs(prop_e[c] - prop_s[c])
                ),
        })

    # --------------------------------------------------------
    # PER-CLASS DIAGNOSTICS
    # --------------------------------------------------------

    for c in range(k):

        per_class.append({
            "task": task,
            "class_id": c,
            "class_label": mapping[c],

            "legacyaux_precision":
                float(ms["precision"][c]),
            "legacyaux_recall":
                float(ms["recall"][c]),
            "legacyaux_f1":
                float(ms["f1"][c]),
            "legacyaux_support":
                int(ms["support"][c]),

            "eventgold_precision":
                float(me["precision"][c]),
            "eventgold_recall":
                float(me["recall"][c]),
            "eventgold_f1":
                float(me["f1"][c]),
            "eventgold_support":
                int(me["support"][c]),

            "f1_gap_eventgold_minus_legacyaux":
                float(
                    me["f1"][c]
                    - ms["f1"][c]
                ),
        })

    # --------------------------------------------------------
    # PREDICTIVE UNCERTAINTY
    # --------------------------------------------------------

    pcols_s = [
        f"prob_class_{i}"
        for i in range(k)
    ]

    pcols_e = [
        f"prob_{i}"
        for i in range(k)
    ]

    probs_s = ds[
        pcols_s
    ].to_numpy(dtype=float)

    probs_e = de[
        pcols_e
    ].to_numpy(dtype=float)

    for dataset, probs in [
        ("LegacyAux", probs_s),
        ("EventGold", probs_e),
    ]:

        if not np.all(
            np.isfinite(probs)
        ):
            raise RuntimeError(
                f"{task} {dataset}: "
                "non-finite probabilities"
            )

        if not np.allclose(
            probs.sum(axis=1),
            1.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{task} {dataset}: "
                "probabilities do not sum to 1"
            )

        uncertainty.append({
            "task": task,
            "dataset": dataset,
            "n_articles": len(probs),
            "mean_max_predicted_probability":
                float(
                    probs.max(axis=1).mean()
                ),
            "mean_predictive_entropy_nats":
                float(
                    predictive_entropy(
                        probs
                    ).mean()
                ),
        })


# ------------------------------------------------------------
# WRITE RESULTS
# ------------------------------------------------------------

pd.DataFrame(
    performance
).to_csv(
    RES / "performance_gap_summary_v1.csv",
    index=False,
)

pd.DataFrame(
    bootstrap_long
).to_csv(
    RES /
    "performance_gap_bootstrap_distributions_v1.csv",
    index=False,
)

pd.DataFrame(
    composition
).to_csv(
    RES / "label_composition_v1.csv",
    index=False,
)

pd.DataFrame(
    composition_summary
).to_csv(
    RES / "label_composition_summary_v1.csv",
    index=False,
)

pd.DataFrame(
    per_class
).to_csv(
    RES / "per_class_generalization_v1.csv",
    index=False,
)

pd.DataFrame(
    uncertainty
).to_csv(
    RES / "predictive_uncertainty_v1.csv",
    index=False,
)

with open(
    RES / "label_maps_v1.json",
    "w",
) as f:
    json.dump(
        label_maps,
        f,
        indent=2,
        ensure_ascii=False,
    )
    f.write("\n")


input_paths = [
    PROTOCOL,
    STAGE_PRED,
    EVENT_PRED,
    STAGE_METRICS,
    EVENT_METRICS,
]

summary = {
    "status":
        "POSTHOC_GENERALIZATION_GAP_COMPLETE",

    "analysis_type":
        "EXPLORATORY_POSTHOC_GENERALIZATION_DIAGNOSTIC",

    "model_training_performed": False,
    "model_selection_performed": False,
    "human_annotation_performed": False,

    "legacyaux_articles_per_task": 199,
    "eventgold_articles_per_task": 35,

    "tasks": TASKS,

    "primary_metric": "macro_f1",

    "secondary_metrics": [
        "balanced_accuracy",
        "accuracy",
        "weighted_f1",
    ],

    "bootstrap_replicates": B,

    "bootstrap_design":
        "independent stratified article-level "
        "resampling within each dataset and task",

    "bootstrap_seeds": RNG_SEEDS,

    "hypothesis_testing": "NONE",

    "jensen_shannon_log_base": "natural",

    "interpretation": {
        "performance_gap_is_distribution_shift_proof":
            False,
        "label_composition_is_full_shift_proof":
            False,
        "seeds_or_folds_are_inferential_observations":
            False,
        "results_may_modify_final_model":
            False,
    },

    "input_sha256": {
        str(p): sha256(p)
        for p in input_paths
    },
}

with open(
    RES / "analysis_summary_v1.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )
    f.write("\n")


# ------------------------------------------------------------
# RESULT FREEZE MANIFEST
# ------------------------------------------------------------

result_files = sorted(
    p for p in RES.iterdir()
    if (
        p.is_file()
        and p.name
        != "GENERALIZATION_GAP_RESULTS_SHA256SUMS_v1.txt"
    )
)

manifest = (
    RES /
    "GENERALIZATION_GAP_RESULTS_SHA256SUMS_v1.txt"
)

with open(manifest, "w") as f:

    for p in result_files:
        f.write(
            f"{sha256(p)}  {p.name}\n"
        )


# ------------------------------------------------------------
# TERMINAL SUMMARY
# ------------------------------------------------------------

perf = pd.DataFrame(performance)

print(
    "GENERALIZATION_GAP_ANALYSIS=COMPLETE"
)

print(
    "PROTOCOL_SHA256="
    + sha256(PROTOCOL)
)

print(
    "RESULT_FILES="
    + str(len(result_files))
)

print(
    "RESULT_MANIFEST_SHA256="
    + sha256(manifest)
)

print(
    "\n===== PRIMARY MACRO-F1 GAP ====="
)

print(
    perf[
        perf["metric"] == "macro_f1"
    ].to_string(index=False)
)

print(
    "\n===== LABEL COMPOSITION ====="
)

print(
    pd.DataFrame(
        composition_summary
    ).to_string(index=False)
)

print(
    "\n===== PREDICTIVE UNCERTAINTY ====="
)

print(
    pd.DataFrame(
        uncertainty
    ).to_string(index=False)
)
