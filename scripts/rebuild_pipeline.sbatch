#!/bin/bash
#SBATCH --job-name=tgv_rebuild
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=basic
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rebuild_%j.out
#SBATCH --error=logs/rebuild_%j.err
set -euo pipefail
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
cd $D
module load anaconda
source venv/bin/activate
export PHASE_MODE=global
for s in 01_ingest_acled 01b_ingest_ucdp 01c_merge_sources 02_build_grid 03_aggregate_location_month 04_build_sequences 05_build_baseline_features; do
  echo "=== $(date) RUN $s ==="
  python scripts/$s.py
  echo "=== $(date) DONE $s ==="
done
echo ALL_BUILD_DONE
