from pathlib import Path
import os
import json
import hashlib
import math

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed

from ripser import ripser


ROOT = Path("01_event_aware_v3")
OUT = ROOT / "11_posthoc_representation_analysis_v1"
RESULTS = OUT / "results_v1"

BASE_PATH = ROOT / "05_supervised_transformers" / "threeway_representation_probe_v3" / "XLMR_BASE_embeddings.npy"
DAPT_PATH = ROOT / "05_supervised_transformers" / "threeway_representation_probe_v3" / "XLMR_DAPT_INDUCTIVE_embeddings.npy"
CONTEXT_PATH = ROOT / "08_eve_frame_stage_b_event_context" / "context_embeddings_v3" / "context_background_embeddings_xlmr_v3.npy"
PROTO_PATH = ROOT / "08_eve_frame_stage_b_event_context" / "event_prototypes_v3" / "verified_event_prototypes_xlmr_v3.npy"
PAIRED_NPZ = ROOT / "05_supervised_transformers" / "frozen_paired_inference_v3" / "paired_inference_distributions_v3.npz"
AXIS_NPZ = ROOT / "05_supervised_transformers" / "frozen_foldsafe_axis_ablation_v3" / "axis_ablation_bootstrap_distributions_v3.npz"

SEED = 20261005
N_REPLICATES = 1000
SUBSAMPLE_FRACTION = 0.80
N_JOBS = max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate(name: str, x: np.ndarray) -> None:
    if x.ndim != 2:
        raise RuntimeError(f"{name}: expected 2D matrix, got {x.shape}")
    if not np.isfinite(x).all():
        raise RuntimeError(f"{name}: non-finite values found")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise RuntimeError(f"{name}: zero-norm row found")


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def centered_gram(x: np.ndarray) -> np.ndarray:
    k = x @ x.T
    n = k.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    return h @ k @ h


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    k = centered_gram(x)
    l = centered_gram(y)
    num = np.sum(k * l)
    den = math.sqrt(np.sum(k * k) * np.sum(l * l))
    if den == 0:
        raise RuntimeError("CKA denominator is zero")
    return float(num / den)


def spectral_stats(x: np.ndarray):
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, full_matrices=False, compute_uv=False)
    eig = s**2
    eig = eig[eig > 0]
    if eig.size == 0:
        raise RuntimeError("Empty positive eigenspectrum")
    p = eig / eig.sum()
    entropy = float(-np.sum(p * np.log(p)))
    return {
        "effective_rank": float(np.exp(entropy)),
        "participation_ratio": float(eig.sum() ** 2 / np.sum(eig**2)),
        "spectral_entropy": entropy,
    }, eig


def neighbor_retention(d1: np.ndarray, d2: np.ndarray, k: int) -> float:
    if d1.shape != d2.shape:
        raise RuntimeError("Distance-matrix shape mismatch")
    vals = []
    for i in range(d1.shape[0]):
        a = np.argsort(d1[i])[1 : k + 1]
        b = np.argsort(d2[i])[1 : k + 1]
        vals.append(len(set(a).intersection(set(b))) / k)
    return float(np.mean(vals))


def finite_diagram(d: np.ndarray) -> np.ndarray:
    if len(d) == 0:
        return np.empty((0, 2), dtype=float)
    return np.asarray(d[np.isfinite(d[:, 1])], dtype=float)


def lifetimes(d: np.ndarray) -> np.ndarray:
    d = finite_diagram(d)
    if len(d) == 0:
        return np.asarray([], dtype=float)
    life = d[:, 1] - d[:, 0]
    return life[life > 0]


def persistent_entropy(life: np.ndarray) -> float:
    if len(life) == 0:
        return 0.0
    total = life.sum()
    if total <= 0:
        return 0.0
    p = life / total
    return float(-np.sum(p * np.log(p)))


def diagram_summary(name: str, dim: int, diagram: np.ndarray) -> dict:
    life = lifetimes(diagram)
    return {
        "representation": name,
        "homology_dimension": dim,
        "n_finite_features": int(len(life)),
        "total_persistence": float(life.sum()) if len(life) else 0.0,
        "max_persistence": float(life.max()) if len(life) else 0.0,
        "mean_persistence": float(life.mean()) if len(life) else 0.0,
        "persistent_entropy": persistent_entropy(life),
    }


def persistence(dist: np.ndarray):
    dist = np.asarray(dist, dtype=np.float64)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return ripser(dist, distance_matrix=True, maxdim=1, coeff=2)["dgms"]


def topo_features(dist: np.ndarray) -> dict:
    dgms = persistence(dist)
    h0 = lifetimes(dgms[0])
    h1 = lifetimes(dgms[1])
    return {
        "h0_total_persistence": float(h0.sum()) if len(h0) else 0.0,
        "h0_max_persistence": float(h0.max()) if len(h0) else 0.0,
        "h1_total_persistence": float(h1.sum()) if len(h1) else 0.0,
        "h1_max_persistence": float(h1.max()) if len(h1) else 0.0,
        "h1_feature_count": int(len(h1)),
        "h1_persistent_entropy": persistent_entropy(h1),
    }


def bottleneck_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Exact bottleneck distance for finite points via thresholded perfect matching."""
    a = finite_diagram(a)
    b = finite_diagram(b)
    m, n = len(a), len(b)

    if m == 0 and n == 0:
        return 0.0
    if m == 0:
        return float(np.max((b[:, 1] - b[:, 0]) / 2.0))
    if n == 0:
        return float(np.max((a[:, 1] - a[:, 0]) / 2.0))

    ab = cdist(a, b, metric="chebyshev")
    da = (a[:, 1] - a[:, 0]) / 2.0
    db = (b[:, 1] - b[:, 0]) / 2.0

    size = m + n
    cost = np.full((size, size), np.inf, dtype=float)
    cost[:m, :n] = ab

    for i in range(m):
        cost[i, n + i] = da[i]
    for j in range(n):
        cost[m + j, j] = db[j]

    cost[m:, n:] = 0.0
    candidates = np.unique(cost[np.isfinite(cost)])

    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        threshold = candidates[mid]
        graph = csr_matrix((cost <= threshold).astype(np.int8))
        matching = maximum_bipartite_matching(graph, perm_type="column")
        if np.all(matching != -1):
            hi = mid
        else:
            lo = mid + 1

    return float(candidates[lo])


def summarize_npz(path: Path, source: str):
    z = np.load(path)
    rows = []
    for key in z.files:
        x = np.asarray(z[key], dtype=float)
        if x.ndim != 1:
            raise RuntimeError(f"{path}:{key}: expected 1D distribution")
        rows.append(
            {
                "source": source,
                "distribution": key,
                "n": int(len(x)),
                "mean": float(np.mean(x)),
                "sd": float(np.std(x, ddof=1)),
                "q025": float(np.quantile(x, 0.025)),
                "median": float(np.quantile(x, 0.5)),
                "q975": float(np.quantile(x, 0.975)),
                "fraction_gt_0": float(np.mean(x > 0)),
                "fraction_lt_0": float(np.mean(x < 0)),
            }
        )
    return rows


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    base = np.load(BASE_PATH)
    dapt = np.load(DAPT_PATH)
    context = np.load(CONTEXT_PATH)
    proto = np.load(PROTO_PATH)

    for name, x in [
        ("XLMR_BASE", base),
        ("XLMR_DAPT_INDUCTIVE", dapt),
        ("STAGE_B_CONTEXT_BACKGROUND", context),
        ("STAGE_B_VERIFIED_EVENT_PROTOTYPES", proto),
    ]:
        validate(name, x)

    if base.shape != (199, 768):
        raise RuntimeError(f"Unexpected BASE shape: {base.shape}")
    if dapt.shape != base.shape:
        raise RuntimeError(f"BASE/DAPT mismatch: {base.shape} vs {dapt.shape}")
    if context.shape[1] != 768:
        raise RuntimeError(f"Unexpected context shape: {context.shape}")
    if proto.shape[1] != 768:
        raise RuntimeError(f"Unexpected prototype shape: {proto.shape}")

    base_n = normalize(base)
    dapt_n = normalize(dapt)
    context_n = normalize(context)
    proto_n = normalize(proto)

    d_base = pairwise_distances(base_n, metric="euclidean")
    d_dapt = pairwise_distances(dapt_n, metric="euclidean")
    d_context = pairwise_distances(context_n, metric="euclidean")
    d_proto = pairwise_distances(proto_n, metric="euclidean")

    for d in (d_base, d_dapt, d_context, d_proto):
        np.fill_diagonal(d, 0.0)

    # Geometry
    tri = np.triu_indices(len(base_n), k=1)
    rho, pval = spearmanr(d_base[tri], d_dapt[tri])

    pd.DataFrame(
        [{
            "comparison": "XLMR_DAPT_INDUCTIVE_vs_XLMR_BASE",
            "linear_cka": linear_cka(base_n, dapt_n),
            "pairwise_distance_spearman_rho": float(rho),
            "pairwise_distance_spearman_p": float(pval),
            "neighbor_retention_k5": neighbor_retention(d_base, d_dapt, 5),
            "neighbor_retention_k10": neighbor_retention(d_base, d_dapt, 10),
            "neighbor_retention_k20": neighbor_retention(d_base, d_dapt, 20),
        }]
    ).to_csv(RESULTS / "global_geometry_comparison_v1.csv", index=False)

    spectral_rows, spectra = [], {}
    for name, x in [
        ("XLMR_BASE", base_n),
        ("XLMR_DAPT_INDUCTIVE", dapt_n),
        ("STAGE_B_CONTEXT_BACKGROUND", context_n),
        ("STAGE_B_VERIFIED_EVENT_PROTOTYPES", proto_n),
    ]:
        stats, eig = spectral_stats(x)
        spectral_rows.append({"representation": name, **stats})
        spectra[name] = eig

    pd.DataFrame(spectral_rows).to_csv(
        RESULTS / "spectral_geometry_summary_v1.csv", index=False
    )

    maxlen = max(len(v) for v in spectra.values())
    eig_table = {}
    for name, eig in spectra.items():
        arr = np.full(maxlen, np.nan)
        arr[: len(eig)] = eig
        eig_table[name] = arr

    pd.DataFrame(eig_table).to_csv(
        RESULTS / "representation_eigenspectra_v1.csv", index=False
    )

    joint = np.vstack([base_n, dapt_n])
    pca = PCA(n_components=10, svd_solver="full")
    coords = pca.fit_transform(joint)

    rows = []
    for name, block in [
        ("XLMR_BASE", coords[:199]),
        ("XLMR_DAPT_INDUCTIVE", coords[199:]),
    ]:
        for idx, row in enumerate(block):
            rec = {"representation": name, "article_index": idx}
            for j, value in enumerate(row, start=1):
                rec[f"PC{j}"] = float(value)
            rows.append(rec)

    pd.DataFrame(rows).to_csv(
        RESULTS / "paired_joint_pca_coordinates_v1.csv", index=False
    )

    pd.DataFrame({
        "component": np.arange(1, 11),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(
            pca.explained_variance_ratio_
        ),
    }).to_csv(RESULTS / "joint_pca_variance_v1.csv", index=False)

    # Persistent homology
    topology_rows = []
    diagrams = {}

    for name, dist in [
        ("XLMR_BASE", d_base),
        ("XLMR_DAPT_INDUCTIVE", d_dapt),
        ("STAGE_B_CONTEXT_BACKGROUND", d_context),
        ("STAGE_B_VERIFIED_EVENT_PROTOTYPES", d_proto),
    ]:
        dgms = persistence(dist)
        diagrams[name] = dgms

        for dim in [0, 1]:
            topology_rows.append(diagram_summary(name, dim, dgms[dim]))
            pd.DataFrame(dgms[dim], columns=["birth", "death"]).to_csv(
                RESULTS / f"{name}__H{dim}_persistence_diagram_v1.csv",
                index=False,
            )

    pd.DataFrame(topology_rows).to_csv(
        RESULTS / "persistent_homology_summary_v1.csv", index=False
    )

    bottleneck_rows = []
    for dim in [0, 1]:
        bottleneck_rows.append({
            "comparison": "XLMR_DAPT_INDUCTIVE_vs_XLMR_BASE",
            "homology_dimension": dim,
            "bottleneck_distance": bottleneck_distance(
                diagrams["XLMR_BASE"][dim],
                diagrams["XLMR_DAPT_INDUCTIVE"][dim],
            ),
        })

    pd.DataFrame(bottleneck_rows).to_csv(
        RESULTS / "persistence_diagram_distances_v1.csv", index=False
    )

    # Paired topological stability
    rng = np.random.default_rng(SEED)
    n = len(base_n)
    sample_n = int(round(SUBSAMPLE_FRACTION * n))

    subsamples = [
        np.sort(rng.choice(n, size=sample_n, replace=False))
        for _ in range(N_REPLICATES)
    ]

    def one_rep(rep, idx):
        fb = topo_features(d_base[np.ix_(idx, idx)])
        fd = topo_features(d_dapt[np.ix_(idx, idx)])

        row = {"replicate": rep, "n_subsample": len(idx)}
        for metric in fb:
            row[f"BASE__{metric}"] = fb[metric]
            row[f"DAPT__{metric}"] = fd[metric]
            row[f"DELTA_DAPT_MINUS_BASE__{metric}"] = fd[metric] - fb[metric]
        return row

    bootstrap_rows = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        verbose=10,
    )(
        delayed(one_rep)(i, idx)
        for i, idx in enumerate(subsamples)
    )

    boot = pd.DataFrame(bootstrap_rows)
    boot.to_csv(
        RESULTS / "paired_topology_subsampling_distributions_v1.csv",
        index=False,
    )

    summary_rows = []
    for col in [
        c for c in boot.columns
        if c.startswith("DELTA_DAPT_MINUS_BASE__")
    ]:
        x = boot[col].to_numpy(dtype=float)
        summary_rows.append({
            "metric": col.replace("DELTA_DAPT_MINUS_BASE__", ""),
            "n_replicates": int(len(x)),
            "mean_delta": float(np.mean(x)),
            "sd_delta": float(np.std(x, ddof=1)),
            "ci95_low": float(np.quantile(x, 0.025)),
            "median_delta": float(np.quantile(x, 0.5)),
            "ci95_high": float(np.quantile(x, 0.975)),
            "fraction_delta_gt_0": float(np.mean(x > 0)),
            "fraction_delta_lt_0": float(np.mean(x < 0)),
        })

    pd.DataFrame(summary_rows).to_csv(
        RESULTS / "paired_topology_stability_summary_v1.csv", index=False
    )

    # Existing frozen 10k distributions
    existing_rows = []
    existing_rows.extend(
        summarize_npz(PAIRED_NPZ, "FROZEN_PAIRED_INFERENCE")
    )
    existing_rows.extend(
        summarize_npz(AXIS_NPZ, "FROZEN_FOLDSAFE_AXIS_ABLATION")
    )
    pd.DataFrame(existing_rows).to_csv(
        RESULTS / "existing_frozen_distribution_summary_v1.csv",
        index=False,
    )

    summary = {
        "status": "POSTHOC_REPRESENTATION_TOPOLOGY_ANALYSIS_COMPLETE",
        "analysis_type": "EXPLORATORY_MECHANISTIC_POSTHOC",
        "model_training_performed": False,
        "model_selection_performed": False,
        "human_annotation_performed": False,
        "eventgold_used_for_model_selection": False,
        "paired_articles": int(base.shape[0]),
        "representation_dimension": int(base.shape[1]),
        "homology_dimensions": [0, 1],
        "topology_subsampling_replicates": N_REPLICATES,
        "topology_subsample_fraction": SUBSAMPLE_FRACTION,
        "rng_seed": SEED,
        "parallel_workers": N_JOBS,
        "input_sha256": {
            str(BASE_PATH): sha256(BASE_PATH),
            str(DAPT_PATH): sha256(DAPT_PATH),
            str(CONTEXT_PATH): sha256(CONTEXT_PATH),
            str(PROTO_PATH): sha256(PROTO_PATH),
            str(PAIRED_NPZ): sha256(PAIRED_NPZ),
            str(AXIS_NPZ): sha256(AXIS_NPZ),
        },
    }

    with open(
        RESULTS / "analysis_summary_v1.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("POSTHOC_REPRESENTATION_TOPOLOGY_ANALYSIS_COMPLETE=YES")


if __name__ == "__main__":
    main()
