#!/bin/bash

#SBATCH --job-name=eveA_audit
#SBATCH --partition=cpu-dev
#SBATCH --output=01_event_aware_v3/logs/eveA_audit_%j.out
#SBATCH --error=01_event_aware_v3/logs/eveA_audit_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G

set -euo pipefail

cd ~/Q1_MediaDiscourse_2026
source ~/media_q1_env/bin/activate

python -u \
01_event_aware_v3/07_eve_frame_stage_a_no_event/40_audit_stage_a_v3.py
