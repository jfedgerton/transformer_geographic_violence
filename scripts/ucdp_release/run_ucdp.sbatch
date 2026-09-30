#!/bin/bash
#SBATCH --job-name=tgv_ucdp
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ucdp_%j.out
#SBATCH --error=logs/ucdp_%j.err
# UCDP mini-encoder pipeline: build panel (skipped if it exists) then train S and M.
# NOTE: if data_raw has no GED csv, run 01 on a LOGIN node first (compute nodes may
# lack internet for the download).
set -euo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast
module load anaconda
source venv/bin/activate

if [ ! -f data_intermediate/ucdp_panel.parquet ]; then
  python scripts/ucdp_release/01_build_ucdp_data.py
fi
UCDP_TAG=ucdp_S UCDP_DIM=64  UCDP_LAYERS=2 UCDP_HEADS=4 python scripts/ucdp_release/02_train_ucdp_repr.py
UCDP_TAG=ucdp_M UCDP_DIM=128 UCDP_LAYERS=4 UCDP_HEADS=8 python scripts/ucdp_release/02_train_ucdp_repr.py
echo "[ucdp] both capacities trained; run 03_package_ucdp_release.py for the chosen tag"
