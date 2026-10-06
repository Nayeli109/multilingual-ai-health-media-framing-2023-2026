#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(
    "01_event_aware_v3/11_posthoc_representation_analysis_v1"
)

SOURCE = ROOT / "01_posthoc_representation_topology_v1.py"
R = ROOT / "results_v1"
OUT = ROOT / "bottleneck_validation_v1" / "results_v1"

OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SOURCE_SHA = (
    "e7e3b726a237214a3e74c61cef9a503a"
    "8d0d398da7520e6120190eed6f35e7f6"
)

TARGETS = {
    0: 0.0034230537712574005,
    1: 0.0006667841225862503,
}

FILES = {
    0: (
        R / "XLMR_BASE__H0_persistence_diagram_v1.csv",
        R / "XLMR_DAPT_INDUCTIVE__H0_persistence_diagram_v1.csv",
    ),
    1: (
        R / "XLMR_BASE__H1_persistence_diagram_v1.csv",
        R / "XLMR_DAPT_INDUCTIVE__H1_persistence_diagram_v1.csv",
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_infinite(dgm):
    dgm = np.asarray(dgm, dtype=float)

    inf = np.isinf(dgm[:, 1])
    finite = dgm[~inf]
    infinite = dgm[inf]

    return finite, infinite


def bottleneck_networkx(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1, 2)

    m = len(a)
    n = len(b)

    if m == 0 and n == 0:
        return 0.0

    if m == 0:
        return float(
            np.max((b[:, 1] - b[:, 0]) / 2.0)
        )

    if n == 0:
        return float(
            np.max((a[:, 1] - a[:, 0]) / 2.0)
        )

    cross = cdist(
        a,
        b,
        metric="chebyshev",
    )

    da = (
        a[:, 1] - a[:, 0]
    ) / 2.0

    db = (
        b[:, 1] - b[:, 0]
    ) / 2.0

    costs = []

    costs.extend(
        cross.ravel().tolist()
    )

    costs.extend(
        da.tolist()
    )

    costs.extend(
        db.tolist()
    )

    costs.append(0.0)

    candidates = np.unique(
        np.asarray(costs, dtype=float)
    )

    left_A = [
        f"A{i}" for i in range(m)
    ]

    left_DB = [
        f"DB{j}" for j in range(n)
    ]

    right_B = [
        f"B{j}" for j in range(n)
    ]

    right_DA = [
        f"DA{i}" for i in range(m)
    ]

    left = left_A + left_DB
    right = right_B + right_DA

    def feasible(threshold):
        G = nx.Graph()

        G.add_nodes_from(
            left,
            bipartite=0,
        )

        G.add_nodes_from(
            right,
            bipartite=1,
        )

        for i in range(m):
            for j in range(n):
                if cross[i, j] <= threshold:
                    G.add_edge(
                        left_A[i],
                        right_B[j],
                    )

        for i in range(m):
            if da[i] <= threshold:
                G.add_edge(
                    left_A[i],
                    right_DA[i],
                )

        for j in range(n):
            if db[j] <= threshold:
                G.add_edge(
                    left_DB[j],
                    right_B[j],
                )

        for j in range(n):
            for i in range(m):
                G.add_edge(
                    left_DB[j],
                    right_DA[i],
                )

        matching = (
            nx.algorithms.bipartite
            .hopcroft_karp_matching(
                G,
                top_nodes=left,
            )
        )

        matched_left = sum(
            1
            for node in left
            if node in matching
        )

        return matched_left == len(left)

    lo = 0
    hi = len(candidates) - 1

    while lo < hi:
        mid = (lo + hi) // 2

        if feasible(candidates[mid]):
            hi = mid
        else:
            lo = mid + 1

    return float(candidates[lo])


# --------------------------------------------------------
# SOURCE INTEGRITY
# --------------------------------------------------------

observed_source_sha = sha256(SOURCE)

if observed_source_sha != EXPECTED_SOURCE_SHA:
    raise RuntimeError(
        "ORIGINAL TDA SOURCE SHA MISMATCH"
    )


# --------------------------------------------------------
# ANALYTIC UNIT TESTS
# --------------------------------------------------------

tests = [
    (
        "identical_single_point",
        np.array([[0.0, 2.0]]),
        np.array([[0.0, 2.0]]),
        0.0,
    ),
    (
        "one_point_vs_empty",
        np.array([[0.0, 2.0]]),
        np.empty((0, 2)),
        1.0,
    ),
    (
        "simple_shift",
        np.array([[0.0, 2.0]]),
        np.array([[0.0, 3.0]]),
        1.0,
    ),
    (
        "diagonal_better_than_direct",
        np.array([[0.0, 10.0]]),
        np.array([[0.0, 1.0]]),
        5.0,
    ),
]

unit_rows = []

for name, a, b, expected in tests:
    observed = bottleneck_networkx(
        a,
        b,
    )

    passed = bool(
        np.isclose(
            observed,
            expected,
            atol=1e-12,
            rtol=0,
        )
    )

    unit_rows.append({
        "test": name,
        "expected": expected,
        "observed": observed,
        "absolute_difference":
            abs(observed - expected),
        "pass": passed,
    })

    if not passed:
        raise RuntimeError(
            f"UNIT TEST FAILED: {name}"
        )


# --------------------------------------------------------
# FROZEN DIAGRAM VALIDATION
# --------------------------------------------------------

rows = []

for dim in [0, 1]:

    base_path, dapt_path = FILES[dim]

    base = pd.read_csv(
        base_path
    )[["birth", "death"]].to_numpy(
        dtype=float
    )

    dapt = pd.read_csv(
        dapt_path
    )[["birth", "death"]].to_numpy(
        dtype=float
    )

    base_finite, base_inf = (
        audit_infinite(base)
    )

    dapt_finite, dapt_inf = (
        audit_infinite(dapt)
    )

    if len(base_inf) != len(dapt_inf):
        raise RuntimeError(
            f"H{dim}: unequal number "
            "of infinite features"
        )

    if len(base_inf):
        bb = np.sort(
            base_inf[:, 0]
        )
        db = np.sort(
            dapt_inf[:, 0]
        )

        if not np.allclose(
            bb,
            db,
            atol=1e-12,
            rtol=0,
        ):
            raise RuntimeError(
                f"H{dim}: infinite-feature "
                "birth mismatch"
            )

    independent = (
        bottleneck_networkx(
            base_finite,
            dapt_finite,
        )
    )

    frozen = TARGETS[dim]

    diff = abs(
        independent - frozen
    )

    passed = bool(
        diff <= 1e-12
    )

    rows.append({
        "homology_dimension": dim,
        "base_total_points":
            len(base),
        "dapt_total_points":
            len(dapt),
        "base_finite_points":
            len(base_finite),
        "dapt_finite_points":
            len(dapt_finite),
        "base_infinite_points":
            len(base_inf),
        "dapt_infinite_points":
            len(dapt_inf),
        "frozen_bottleneck_distance":
            frozen,
        "independent_networkx_distance":
            independent,
        "absolute_difference":
            diff,
        "tolerance":
            1e-12,
        "pass":
            passed,
    })

    if not passed:
        raise RuntimeError(
            f"H{dim}: BOTTLENECK "
            "VALIDATION FAILED"
        )


unit_df = pd.DataFrame(
    unit_rows
)

validation_df = pd.DataFrame(
    rows
)

unit_df.to_csv(
    OUT /
    "analytic_unit_tests_v1.csv",
    index=False,
)

validation_df.to_csv(
    OUT /
    "independent_bottleneck_validation_v1.csv",
    index=False,
)


summary = {
    "status":
        "INDEPENDENT_BOTTLENECK_VALIDATION_PASS",

    "original_algorithm":
        "scipy.sparse.csgraph.maximum_bipartite_matching",

    "independent_algorithm":
        "networkx Hopcroft-Karp bipartite matching",

    "original_tda_source_sha256":
        observed_source_sha,

    "tolerance":
        1e-12,

    "analytic_unit_tests_passed":
        bool(unit_df["pass"].all()),

    "h0_pass":
        bool(
            validation_df.loc[
                validation_df[
                    "homology_dimension"
                ] == 0,
                "pass",
            ].iloc[0]
        ),

    "h1_pass":
        bool(
            validation_df.loc[
                validation_df[
                    "homology_dimension"
                ] == 1,
                "pass",
            ].iloc[0]
        ),

    "input_sha256": {
        str(p): sha256(p)
        for pair in FILES.values()
        for p in pair
    },
}

with open(
    OUT /
    "bottleneck_validation_summary_v1.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )
    f.write("\n")


files = sorted(
    p for p in OUT.iterdir()
    if (
        p.is_file()
        and p.name
        != "BOTTLENECK_VALIDATION_RESULTS_SHA256SUMS_v1.txt"
    )
)

manifest = (
    OUT /
    "BOTTLENECK_VALIDATION_RESULTS_SHA256SUMS_v1.txt"
)

with open(manifest, "w") as f:
    for p in files:
        f.write(
            f"{sha256(p)}  {p.name}\n"
        )


print(
    "BOTTLENECK_INDEPENDENT_VALIDATION=PASS"
)

print()
print("===== UNIT TESTS =====")
print(
    unit_df.to_string(index=False)
)

print()
print("===== FROZEN DIAGRAM VALIDATION =====")
print(
    validation_df.to_string(index=False)
)

print()
print(
    "RESULT_MANIFEST_SHA256="
    + sha256(manifest)
)
