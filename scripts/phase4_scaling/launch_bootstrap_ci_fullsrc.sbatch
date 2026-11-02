#!/bin/bash
#SBATCH --job-name=tgv_bootfs
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/bootfs_%j.out
#SBATCH --error=logs/bootfs_%j.err
# Full-source bootstrap (2026-10-23). sbatch --export=ALL,BOOT_SIZE=S  (BOOT_COMPS defaults to persistence,nested)
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
: "${BOOT_SIZE:?set BOOT_SIZE}"; : "${BOOT_COMPS:=persistence,nested}"; export BOOT_COMPS   # commas cannot go through --export
md5sum scripts/phase4_scaling/bootstrap_ci_fullsrc.py
# smoke: 5 replicates, separate output names; any failure stops the job before the full runs
BOOT_B=5 BOOT_UNIT=cell BOOT_OUTSFX=_smoke venv/bin/python scripts/phase4_scaling/bootstrap_ci_fullsrc.py
BOOT_B=1000 BOOT_UNIT=cell   venv/bin/python scripts/phase4_scaling/bootstrap_ci_fullsrc.py
BOOT_B=1000 BOOT_UNIT=block5 venv/bin/python scripts/phase4_scaling/bootstrap_ci_fullsrc.py
echo BOOTFS_DONE_${BOOT_SIZE}
