#!/bin/bash
# launch_alert_chain_fullsrc.sbatch -- written 2026-10-27 (Claude, approved plan step 3). New launcher replacing, for the
# full-source encoders, launch_alert_counts.sbatch + run_cell_alerts_persistence.sh + run_cell_month_decompose.sh +
# launch_alert_profile.sbatch + launch_alert_budget_sweep.sbatch (cell_alerts_agg.py had no launcher). Runs the
# *_fullsrc.py copies in dependency order for one SIZE, then (optionally) figures_tables/fig_cell_alerts.R unchanged on
# the size-specific cell file. A step whose completion marker exists is skipped, so a requeue resumes; the scripts
# themselves refuse to overwrite. Resources = the largest of the originals (96G), time = sum of their limits, capped.
# Usage: sbatch --export=ALL,SIZE=M scripts/phase4_scaling/launch_alert_chain_fullsrc.sbatch   (OUTSFX=_smoke for smoke)
#        figure: NE_ZIP=<path to ne_50m_admin_0_countries.zip>; without it the figure step is skipped with a TODO line
#        (2026-10-27: no ne_50m zip on Roar; data_raw/shapefiles/ne_10m_admin_0/ne_10m_admin_0_countries.zip exists, untested).
#SBATCH --job-name=tgv_alertsfs
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/alertsfs_%j.out
#SBATCH --error=logs/alertsfs_%j.err
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
: "${SIZE:?set SIZE=S|M|L}"
case "$SIZE" in S|M|L) ;; *) echo "SIZE must be S, M or L"; exit 1;; esac
export SIZE OUTSFX=${OUTSFX:-}
unset FS_ROOT AP_ROOT SWEEP_ROOT
T=_${SIZE}_fullsrc${OUTSFX}
P=scripts/phase4_scaling
md5sum $P/alert_counts_fullsrc.py $P/cell_alerts_agg_fullsrc.py $P/cell_alerts_persistence_fullsrc.py \
       $P/cell_month_decompose_fullsrc.py $P/alert_profile_fullsrc.py $P/alert_budget_sweep_fullsrc.py
step() {  # step <marker> <script>
  if [ -e "$1" ]; then echo "[skip] $2: $1 exists"; return 0; fi
  echo "[run] $2 $(date)"; venv/bin/python $P/$2; test -e "$1" || { echo "[FATAL] $2 finished without $1"; exit 1; }
}
step outputs/alert_counts_persistence${T}.csv          alert_counts_fullsrc.py
step outputs/cell_alerts_incidence${T}.csv.gz          cell_alerts_agg_fullsrc.py
step outputs/cell_alerts_persistence${T}.csv.gz        cell_alerts_persistence_fullsrc.py
step outputs/cell_month_decompose${T}                  cell_month_decompose_fullsrc.py
step outputs/alert_profile_incidence${T}.DONE          alert_profile_fullsrc.py
step outputs/alert_budget_sweep${T}.csv                alert_budget_sweep_fullsrc.py
# ---- figure (unchanged R script) ----
FD=outputs/cell_alerts_fig${T}
if [ -e "$FD/fig_cell_alerts_incidence${T}.png" ]; then echo "[skip] figure: png exists"
elif [ -z "${NE_ZIP:-}" ] || [ ! -f "${NE_ZIP:-/nonexistent}" ]; then echo "[TODO] figure skipped: set NE_ZIP to ne_50m_admin_0_countries.zip (not on Roar; only data_raw/shapefiles/ne_10m_admin_0/ne_10m_admin_0_countries.zip exists, untested with fig_cell_alerts.R)"
else
  module load r 2>/dev/null || true
export R_PROFILE_USER=$PWD/$P/Rprofile_cairo  # 2026-10-27 fix: png() on headless nodes needs cairo, not X11
  if command -v Rscript >/dev/null && Rscript -e 'suppressPackageStartupMessages({library(ggplot2); library(sf); library(dplyr); library(readr); library(patchwork); library(scales)})'; then
    mkdir -p "$FD"
    Rscript figures_tables/fig_cell_alerts.R outputs/cell_alerts_incidence${T}.csv.gz "$NE_ZIP" incidence "$FD/fig_cell_alerts_incidence${T}.png"
  else echo "[TODO] figure skipped: Rscript or an R package (ggplot2 sf dplyr readr patchwork scales) is missing"; fi
fi
echo ALERTSFS_DONE_${SIZE}${OUTSFX}
