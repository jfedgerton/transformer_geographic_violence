#!/bin/bash
#SBATCH --job-name=tgv_bootsh
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bootsh_%j.out
#SBATCH --error=logs/bootsh_%j.err
# 2026-10-26 limited-data (share) bootstrap for the full-source frozen-embedding grid. sbatch --export=ALL,BOOT_SIZE=M
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
: "${BOOT_SIZE:?set BOOT_SIZE}"
export BOOT_COMPS=share_0.0025,share_0.005,share_0.0075,share_0.01,share_0.025,share_0.05,share_0.1
md5sum scripts/phase4_scaling/bootstrap_ci_fullsrc.py
BOOT_B=1000 BOOT_UNIT=block5 BOOT_OUTSFX=_shares venv/bin/python scripts/phase4_scaling/bootstrap_ci_fullsrc.py
BOOT_B=1000 BOOT_UNIT=cell BOOT_OUTSFX=_shares venv/bin/python scripts/phase4_scaling/bootstrap_ci_fullsrc.py
echo BOOTSH_DONE_${BOOT_SIZE}
