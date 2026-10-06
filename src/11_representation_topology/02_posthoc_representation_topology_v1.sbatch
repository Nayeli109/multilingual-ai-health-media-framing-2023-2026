#!/bin/bash
#SBATCH -J q1_repr_tda_v1
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH -o 01_event_aware_v3/11_posthoc_representation_analysis_v1/logs/posthoc_repr_tda_%j.out
#SBATCH -e 01_event_aware_v3/11_posthoc_representation_analysis_v1/logs/posthoc_repr_tda_%j.err

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

OUT="01_event_aware_v3/11_posthoc_representation_analysis_v1"
RESULTS="$OUT/results_v1"

mkdir -p "$RESULTS" "$OUT/logs"

export PYTHONPATH="$OUT/vendor_retry_v1:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "============================================================"
echo "POST-HOC REPRESENTATION / TOPOLOGY v1"
echo "JOBID=${SLURM_JOB_ID:-NA}"
echo "HOST=$(hostname)"
echo "START=$(date -Is)"
echo "============================================================"

echo
echo "===== ENVIRONMENT ====="
python --version

python - <<'PY'
import importlib.metadata as md
import numpy
import scipy
import sklearn
import ripser

print("numpy =", numpy.__version__)
print("scipy =", scipy.__version__)
print("sklearn =", sklearn.__version__)
print("ripser =", md.version("ripser"))
print("DEPENDENCY_IMPORT_CHECK=PASS")
PY

echo
echo "===== FROZEN PROTOCOL CHECK ====="

EXPECTED="61fe0cb10bc57e12f77a7a7e0bbf03080a6547f899591265fbf8bb93d2d08e39"
ACTUAL=$(sha256sum "$OUT/POSTHOC_REPRESENTATION_TOPOLOGY_PROTOCOL_v1.txt" | awk '{print $1}')

if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo "STOP: frozen protocol hash mismatch"
    echo "EXPECTED=$EXPECTED"
    echo "ACTUAL=$ACTUAL"
    exit 1
fi

echo "FROZEN_PROTOCOL_CHECK=PASS"

echo
echo "===== INPUT CHECK ====="

for F in \
  "01_event_aware_v3/05_supervised_transformers/threeway_representation_probe_v3/XLMR_BASE_embeddings.npy" \
  "01_event_aware_v3/05_supervised_transformers/threeway_representation_probe_v3/XLMR_DAPT_INDUCTIVE_embeddings.npy" \
  "01_event_aware_v3/08_eve_frame_stage_b_event_context/context_embeddings_v3/context_background_embeddings_xlmr_v3.npy" \
  "01_event_aware_v3/08_eve_frame_stage_b_event_context/event_prototypes_v3/verified_event_prototypes_xlmr_v3.npy" \
  "01_event_aware_v3/05_supervised_transformers/frozen_paired_inference_v3/paired_inference_distributions_v3.npz" \
  "01_event_aware_v3/05_supervised_transformers/frozen_foldsafe_axis_ablation_v3/axis_ablation_bootstrap_distributions_v3.npz"
do
    if [ ! -s "$F" ]; then
        echo "STOP: missing or empty input: $F"
        exit 1
    fi
done

echo "INPUT_CHECK=PASS"

echo
echo "===== ANALYSIS ====="

python "$OUT/01_posthoc_representation_topology_v1.py"

echo
echo "===== RESULT VALIDATION ====="

python - <<'PY'
from pathlib import Path
import json
import pandas as pd

r = Path(
    "01_event_aware_v3/"
    "11_posthoc_representation_analysis_v1/"
    "results_v1"
)

required = [
    "global_geometry_comparison_v1.csv",
    "spectral_geometry_summary_v1.csv",
    "persistent_homology_summary_v1.csv",
    "persistence_diagram_distances_v1.csv",
    "paired_topology_subsampling_distributions_v1.csv",
    "paired_topology_stability_summary_v1.csv",
    "existing_frozen_distribution_summary_v1.csv",
    "paired_joint_pca_coordinates_v1.csv",
    "joint_pca_variance_v1.csv",
    "analysis_summary_v1.json",
]

missing = [name for name in required if not (r / name).is_file()]
if missing:
    raise SystemExit(f"STOP: missing required outputs: {missing}")

with open(r / "analysis_summary_v1.json", encoding="utf-8") as f:
    summary = json.load(f)

if summary.get("status") != "POSTHOC_REPRESENTATION_TOPOLOGY_ANALYSIS_COMPLETE":
    raise SystemExit("STOP: final analysis status is not COMPLETE")

boot = pd.read_csv(r / "paired_topology_subsampling_distributions_v1.csv")
if len(boot) != 1000:
    raise SystemExit(
        f"STOP: expected 1000 topology replicates; got {len(boot)}"
    )

for name in [
    "global_geometry_comparison_v1.csv",
    "spectral_geometry_summary_v1.csv",
    "persistent_homology_summary_v1.csv",
    "persistence_diagram_distances_v1.csv",
    "paired_topology_stability_summary_v1.csv",
    "existing_frozen_distribution_summary_v1.csv",
]:
    if pd.read_csv(r / name).empty:
        raise SystemExit(f"STOP: empty required result: {name}")

print("RESULT_VALIDATION=PASS")
PY

echo
echo "===== FREEZE RESULTS ====="

find "$RESULTS" -maxdepth 1 -type f -print0 | \
  sort -z | \
  xargs -0 sha256sum \
  > "$OUT/POSTHOC_RESULTS_SHA256SUMS_v1.txt"

sha256sum \
  "$OUT/POSTHOC_REPRESENTATION_TOPOLOGY_PROTOCOL_v1.txt" \
  "$OUT/01_posthoc_representation_topology_v1.py" \
  "$OUT/02_posthoc_representation_topology_v1.sbatch" \
  "$OUT/POSTHOC_RESULTS_SHA256SUMS_v1.txt" \
  > "$OUT/POSTHOC_ANALYSIS_FREEZE_SHA256SUMS_v1.txt"

echo
echo "============================================================"
echo "FINAL_STATUS=PASS"
echo "END=$(date -Is)"
echo "============================================================"
