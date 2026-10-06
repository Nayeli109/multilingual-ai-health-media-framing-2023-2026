#!/bin/bash

#SBATCH --job-name=xlmr_dapt_lr1e5_v3
#SBATCH --partition=gpu
#SBATCH --output=01_event_aware_v3/logs/xlmr_dapt_lr1e5_v3_%j.out
#SBATCH --error=01_event_aware_v3/logs/xlmr_dapt_lr1e5_v3_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --gres=gpu:a100-sxm4-40gb:1

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4

echo "=============================================="
echo "XLM-R DAPT v3"
echo "=============================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM job: ${SLURM_JOB_ID:-NA}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-NA}"
echo ""

echo "GPU:"
nvidia-smi

echo ""
echo "Memory allocation:"
echo "${SLURM_MEM_PER_NODE:-unknown} MB"

echo ""
echo "Python:"
python --version

echo ""
echo "Starting training..."

python \
01_event_aware_v3/02_domain_adaptation/13_train_xlmr_dapt_inductive_lr1e5_v3.py

echo ""
echo "=============================================="
echo "TRAINING FINISHED"
echo "=============================================="
date
