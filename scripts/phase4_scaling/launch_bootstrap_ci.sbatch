#!/bin/bash
#SBATCH --job-name=tgv_boot2
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=logs/boot2_%j.out
#SBATCH --error=logs/boot2_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
# cell resampling (all shares) then 5-degree spatial blocks; each writes outputs/bootstrap_ci_{unit}.csv
BOOT_B=1000 BOOT_UNIT=cell   venv/bin/python scripts/phase4_scaling/bootstrap_ci.py
BOOT_B=1000 BOOT_UNIT=block5 venv/bin/python scripts/phase4_scaling/bootstrap_ci.py
