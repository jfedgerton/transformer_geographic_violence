#!/bin/bash
#SBATCH --job-name=priority_b_multisource
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=open
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/priority_b_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/priority_b_%j.err

# Priority B: multi-source pre-training (ACLED + UCDP-GED).
# Pipeline:
#   01  re-run ACLED ingest (recreates acled_cleaned.parquet)
#   01b ingest UCDP GED
#   01c merge ACLED + UCDP -> overwrites acled_cleaned.parquet with unified events
#   02  build grid
#   03  aggregate
#   04  build sequences (7-field, hierarchical actors — same as Priority A)
#   05  baseline features
#   06  MLM pre-training on the merged corpus
#   07  downstream classifier
#   08  evaluation

set -e
set -o pipefail

cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast

module load anaconda
source venv/bin/activate

python -c "import torch" 2>/dev/null || pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir matplotlib seaborn scikit-learn xgboost

mkdir -p logs

echo "================================================"
echo "Priority B: Multi-Source (ACLED + UCDP-GED) MLM"
echo "Start: $(date)"
echo "================================================"

echo "[$(date)] Step 01: ingest ACLED (rebuild cleaned parquet)"
python -u scripts/01_ingest_acled.py

echo "[$(date)] Step 01b: ingest UCDP GED"
python -u scripts/01b_ingest_ucdp.py

echo "[$(date)] Step 01c: merge ACLED + UCDP -> unified events"
python -u scripts/01c_merge_sources.py

echo "[$(date)] Step 02: build grid"
python -u scripts/02_build_grid.py

echo "[$(date)] Step 03: aggregate to grid-month"
python -u scripts/03_aggregate_location_month.py

echo "[$(date)] Step 04: build sequences (7-field schema)"
python -u scripts/04_build_sequences.py

echo "[$(date)] Step 05: baseline features"
python -u scripts/05_build_baseline_features.py

echo "[$(date)] Step 06: MLM pre-training on merged corpus"
python -u scripts/06_train_representation_model.py

echo "[$(date)] Step 07: downstream classifier"
python -u scripts/07_train_downstream_model.py

echo "[$(date)] Step 08: evaluation"
python -u scripts/08_evaluate_models.py

echo "================================================"
echo "Priority B complete: $(date)"
echo "Compare combined-model AUC against Priority A (0.9166)"
echo "and embedding-only AUC against Priority A (0.8406)."
echo "================================================
