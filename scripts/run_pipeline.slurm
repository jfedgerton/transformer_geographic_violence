#!/bin/bash
#SBATCH --job-name=tgv_pipeline
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=open
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -e
set -o pipefail

cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast

module load anaconda
source venv/bin/activate

# Install torch (CPU wheel) if missing
python -c "import torch" 2>/dev/null || pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install scientific stack used by 02-11. pip is idempotent so safe to rerun.
pip install --no-cache-dir matplotlib seaborn scikit-learn xgboost

mkdir -p logs

echo "[$(date)] Step 1/4: 00b_subset_acled.py"
python scripts/00b_subset_acled.py

echo "[$(date)] Step 2/4: 00_master.py (runs 01 through 11)"
python scripts/00_master.py

echo "[$(date)] Step 3/4: 06b_train_spatial_repr.py"
python scripts/06b_train_spatial_repr.py

echo "[$(date)] Step 4/4: 12_export_tensor.py"
python scripts/12_export_tensor.py

echo "[$(date)] Pipeline complete."
