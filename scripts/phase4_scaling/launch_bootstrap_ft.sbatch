#!/bin/bash
#SBATCH --job-name=tgv_bootft
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/bootft_%j.out
#SBATCH --error=logs/bootft_%j.err
# Fine-tune vs scratch bootstrap (step 7b). Submit with BOOT_TARGET=incidence (after the americas s456 10% rerun) or
# BOOT_TARGET=civilian (after the civilian fine-tune grid), e.g. sbatch --dependency=afterok:<jobid> --export=ALL,BOOT_TARGET=incidence
# Each unit is skipped if its output already exists, so a requeue resumes instead of failing (Codex 9, L01).
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
: "${BOOT_TARGET:?pass BOOT_TARGET=incidence or civilian}"
unset BOOT_ROOT
for U in cell block5; do
  if [ -e "outputs/bootstrap_ft_${BOOT_TARGET}_${U}.csv" ]; then echo "[skip] ${U} already done"; continue; fi
  BOOT_B=1000 BOOT_UNIT=$U BOOT_TARGET=$BOOT_TARGET venv/bin/python scripts/phase4_scaling/bootstrap_ft.py
done
