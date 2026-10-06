#!/bin/bash

#SBATCH --job-name=v4_cublas_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:05:00

#SBATCH --output=01_event_aware_v3/09_eve_frame_v4_architecture_sensitivity/logs_v4/cublas_smoke_%j.out
#SBATCH --error=01_event_aware_v3/09_eve_frame_v4_architecture_sensitivity/logs_v4/cublas_smoke_%j.err

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python - <<'PY'
import os

if os.environ.get(
    "CUBLAS_WORKSPACE_CONFIG"
) != ":4096:8":
    raise RuntimeError(
        "Incorrect CUBLAS_WORKSPACE_CONFIG."
    )

import torch

torch.use_deterministic_algorithms(
    True
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available."
    )

torch.manual_seed(
    20260924
)

torch.cuda.manual_seed_all(
    20260924
)

device = torch.device(
    "cuda"
)

a = torch.randn(
    512,
    512,
    device=device,
)

b = torch.randn(
    512,
    512,
    device=device,
)

c = torch.matmul(
    a,
    b,
)

torch.cuda.synchronize()

if not torch.isfinite(
    c
).all():
    raise RuntimeError(
        "Non-finite smoke-test output."
    )

print(
    "CUBLAS_DETERMINISM_SMOKE_OK"
)

print(
    "CUBLAS_WORKSPACE_CONFIG=",
    os.environ[
        "CUBLAS_WORKSPACE_CONFIG"
    ],
)

print(
    "CUDA_DEVICE=",
    torch.cuda.get_device_name(0),
)
PY
