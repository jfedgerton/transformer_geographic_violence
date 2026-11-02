#!/bin/bash
# launch_country_slice_fullsrc.sbatch -- written 2026-10-27 (Claude, approved plan step 3).
# Copies: scripts/phase4_scaling/launch_country_slice.sbatch. Changes: runs country_slice_fullsrc.py; requires SIZE;
# SLICE_FT defaults to top2; passes OUTSFX; SLICE_SHARES/SLICE_DRAWS are kept only when SMOKE=1 (which requires
# OUTSFX), otherwise unset as in the original. Same resources.
# Usage: sbatch --export=ALL,SIZE=M,SLICE_TARGET=incidence scripts/phase4_scaling/launch_country_slice_fullsrc.sbatch
#SBATCH --job-name=tgv_slicefs
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/slicefs_%j.out
#SBATCH --error=logs/slicefs_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
: "${SIZE:?set SIZE=S|M|L}"; : "${SLICE_TARGET:?pass SLICE_TARGET=incidence or civilian}"
export SIZE SLICE_TARGET SLICE_FT=${SLICE_FT:-top2} OUTSFX=${OUTSFX:-}
unset SLICE_ROOT SLICE_ALLOW_MISSING
if [ "${SMOKE:-0}" = "1" ]; then [ -n "$OUTSFX" ] || { echo "SMOKE=1 needs OUTSFX"; exit 1; }
else unset SLICE_DRAWS SLICE_SHARES; fi
md5sum scripts/phase4_scaling/country_slice_fullsrc.py
venv/bin/python scripts/phase4_scaling/country_slice_fullsrc.py
echo SLICEFS_DONE_${SIZE}_${SLICE_TARGET}${OUTSFX}
