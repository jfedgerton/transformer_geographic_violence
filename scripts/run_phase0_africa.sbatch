#!/bin/bash
#SBATCH --job-name=phase0_africa_smoke
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=open
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase0_africa_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/phase0_africa_%j.err

# Stage B Phase 0 smoke test: run 01-04 on full Africa using acled_full.csv + region filter.
# This is the validation run before we add PRIO-GRID covariates, ICEWS, and scale up to GPU.

set -e
set -o pipefail

cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast

module load anaconda
source venv/bin/activate

# Make sure CONFIG points at acled_full.csv (idempotent sed)
sed -i 's|acled_export\.csv|acled_full.csv|' scripts/00_master.py
grep -n 'acled_full\|acled_export' scripts/00_master.py

echo "================================================"
echo "Phase 0 Africa smoke — Stage B"
echo "Start: $(date)"
echo "================================================"

echo "[$(date)] Step 01: ingest ACLED (full + Africa region filter)"
python -u scripts/01_ingest_acled.py

echo "[$(date)] Step 02: build PRIO-GRID gid"
python -u scripts/02_build_grid.py

echo "[$(date)] Step 03: aggregate to grid-month"
python -u scripts/03_aggregate_location_month.py

echo "[$(date)] Step 04: build sequences (streaming, 7-field schema)"
python -u scripts/04_build_sequences.py

echo "================================================"
echo "Phase 0 Africa smoke complete: $(date)"
echo ""
echo "Expected sanity checks:"
echo "  - ACLED events kept after Africa filter: a few million"
echo "  - Unique PRIO-GRID gids: ~10,000 (Africa land cells)"
echo "  - gid range: roughly 80,000 - 160,000 (Africa lat band)"
echo "  - 04 streaming should complete with ETA visible"
echo "================================================"
