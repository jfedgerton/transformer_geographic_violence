#!/bin/bash
#SBATCH --job-name=tgv_ukrmap
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ukraine_map_%j.out
#SBATCH --error=logs/ukraine_map_%j.err
# Ukraine forecast maps (fig:ukraine_invasion + appendix maps). CPU only, no model is fit.
# Usage: sbatch --export=ALL,UKR_SHARE=auto,UKR_GRID_DIR=outputs/share_grid,UKR_SUFFIX= scripts/phase4_scaling/launch_ukraine_map.sbatch
#   UKR_SHARE=auto applies the fixed rule (largest Feb-Jul 2022 PR-AUC gain, fine-tuned minus scratch);
#   UKR_GRID_DIR / UKR_SUFFIX point at the full-source share-grid outputs once they exist.
set -eo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate
module load r 2>/dev/null || true
command -v Rscript >/dev/null || { echo "Rscript not found (load the R module that has ggplot2 and sf)"; exit 1; }
SHARE=${UKR_SHARE:-auto}; GRID=${UKR_GRID_DIR:-outputs/share_grid}; SUF=${UKR_SUFFIX:-}
md5sum scripts/phase4_scaling/export_ukraine_map.py scripts/phase4_scaling/fig_ukraine_map.R
python -u scripts/phase4_scaling/export_ukraine_map.py --share "$SHARE" --grid-dir "$GRID" --suffix "$SUF" | tee logs/ukraine_map_export_${SLURM_JOB_ID}.txt
CHOSEN=$(grep -o "map uses share [0-9.]*" logs/ukraine_map_export_${SLURM_JOB_ID}.txt | awk '{print $4}')
test -n "$CHOSEN" || { echo "no share chosen"; exit 1; }
Rscript scripts/phase4_scaling/fig_ukraine_map.R outputs/ukraine_maps/ukraine_cells_f${CHOSEN}${SUF}.csv
echo UKRMAP_DONE share=$CHOSEN
