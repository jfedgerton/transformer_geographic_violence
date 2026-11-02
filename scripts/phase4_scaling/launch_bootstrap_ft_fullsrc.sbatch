#!/bin/bash
#SBATCH --job-name=tgv_bootft
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bootftfs_%j.out
#SBATCH --error=logs/bootftfs_%j.err
# 2026-10-27 fine-tuning vs scratch bootstrap for the full-source share grid (bootstrap_ft_fullsrc.py), block5 then cell units.
# Usage: sbatch --export=ALL,BOOT_SIZE=M,BOOT_FT=top2 scripts/phase4_scaling/launch_bootstrap_ft_fullsrc.sbatch
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
: "${BOOT_SIZE:?set BOOT_SIZE}"; : "${BOOT_FT:?set BOOT_FT}"
md5sum scripts/phase4_scaling/bootstrap_ft_fullsrc.py
BOOT_B=1000 BOOT_UNIT=block5 venv/bin/python scripts/phase4_scaling/bootstrap_ft_fullsrc.py
BOOT_B=1000 BOOT_UNIT=cell venv/bin/python scripts/phase4_scaling/bootstrap_ft_fullsrc.py
echo BOOTFTFS_DONE_${BOOT_SIZE}_${BOOT_FT}
