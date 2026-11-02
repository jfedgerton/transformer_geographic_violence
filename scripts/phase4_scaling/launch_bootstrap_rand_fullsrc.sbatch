#!/bin/bash
#SBATCH --job-name=tgv_bootrand
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bootrand_%j.out
#SBATCH --error=logs/bootrand_%j.err
# 2026-10-26 pretrained-vs-random bootstrap for the full-source encoders (bootstrap_ci_fullsrc.py, BOOT_COMPS=random).
# Outputs carry _rand so they never collide with the persistence/nested files. sbatch --export=ALL,BOOT_SIZE=M
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
: "${BOOT_SIZE:?set BOOT_SIZE}"
export BOOT_COMPS=random
md5sum scripts/phase4_scaling/bootstrap_ci_fullsrc.py
BOOT_B=1000 BOOT_UNIT=block5 BOOT_OUTSFX=_rand venv/bin/python scripts/phase4_scaling/bootstrap_ci_fullsrc.py
BOOT_B=1000 BOOT_UNIT=cell BOOT_OUTSFX=_rand venv/bin/python scripts/phase4_scaling/bootstrap_ci_fullsrc.py
echo BOOTRAND_DONE_${BOOT_SIZE}
