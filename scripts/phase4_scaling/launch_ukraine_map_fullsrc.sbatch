#!/bin/bash
# launch_ukraine_map_fullsrc.sbatch -- written 2026-10-27 (Claude, approved plan step 3).
# Copies: scripts/phase4_scaling/launch_ukraine_map.sbatch. Changes: runs export_ukraine_map_fullsrc.py (SIZE required,
# UKR_FT default top2, outputs in outputs/ukraine_maps_{SIZE}{OUTSFX}/); the R step reads the cells file from that
# directory (original: hard-coded outputs/ukraine_maps/); fig_ukraine_map.R itself is unchanged. Same resources.
# Usage: sbatch --export=ALL,SIZE=M scripts/phase4_scaling/launch_ukraine_map_fullsrc.sbatch   (UKR_SHARE=auto default)
#SBATCH --job-name=tgv_ukrmapfs
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ukraine_mapfs_%j.out
#SBATCH --error=logs/ukraine_mapfs_%j.err
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
module load r 2>/dev/null || true
command -v Rscript >/dev/null || { echo "Rscript not found (load the R module that has ggplot2 and sf)"; exit 1; }
: "${SIZE:?set SIZE=S|M|L}"
SHARE=${UKR_SHARE:-auto}; FT=${UKR_FT:-top2}; OSFX=${OUTSFX:-}
OUTD=outputs/ukraine_maps_${SIZE}${OSFX}
md5sum scripts/phase4_scaling/export_ukraine_map_fullsrc.py scripts/phase4_scaling/fig_ukraine_map.R
SIZE=$SIZE python -u scripts/phase4_scaling/export_ukraine_map_fullsrc.py --share "$SHARE" --ft "$FT" --out-dir "$OUTD" | tee logs/ukraine_mapfs_export_${SLURM_JOB_ID}.txt
CHOSEN=$(grep -o "map uses share [0-9.]*" logs/ukraine_mapfs_export_${SLURM_JOB_ID}.txt | awk '{print $4}')
test -n "$CHOSEN" || { echo "no share chosen"; exit 1; }
Rscript scripts/phase4_scaling/fig_ukraine_map.R "$OUTD/ukraine_cells_f${CHOSEN}_${SIZE}_fullsrc_${FT}.csv"
echo UKRMAPFS_DONE_${SIZE}${OSFX} share=$CHOSEN
