#!/bin/bash
# launch_country_window_fullsrc.sbatch -- written 2026-10-27 (Claude, approved plan step 3).
# Copies: scripts/phase4_scaling/launch_country_window.sbatch. Changes: runs country_window_fullsrc.py; requires SIZE;
# WIN_FT defaults to top2; passes OUTSFX; WIN_DRAWS/WIN_SHARES/WIN_B are kept only when SMOKE=1 (which requires
# OUTSFX), otherwise unset as in the original. Same resources.
# Usage: sbatch --export=ALL,SIZE=M,WIN_TARGET=incidence scripts/phase4_scaling/launch_country_window_fullsrc.sbatch
#SBATCH --job-name=tgv_windowfs
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/windowfs_%j.out
#SBATCH --error=logs/windowfs_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
: "${SIZE:?set SIZE=S|M|L}"; : "${WIN_TARGET:?pass WIN_TARGET=incidence or civilian}"
export SIZE WIN_TARGET WIN_FT=${WIN_FT:-top2} OUTSFX=${OUTSFX:-}
unset WIN_ROOT
if [ "${SMOKE:-0}" = "1" ]; then [ -n "$OUTSFX" ] || { echo "SMOKE=1 needs OUTSFX"; exit 1; }
else unset WIN_DRAWS WIN_SHARES WIN_B; fi
md5sum scripts/phase4_scaling/country_window_fullsrc.py
venv/bin/python scripts/phase4_scaling/country_window_fullsrc.py
echo WINDOWFS_DONE_${SIZE}_${WIN_TARGET}${OUTSFX}
