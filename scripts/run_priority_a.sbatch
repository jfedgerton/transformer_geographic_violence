#!/bin/bash
#SBATCH --job-name=priority_a_hierarchical_actors
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=open
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/priority_a_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/priority_a_%j.err

# Priority A: hierarchical actor-identity representation.
# Reuses the cleaned event panel from Stage A; reruns from step 04 onward
# so the new 7-field (actor1_type / actor2_type) schema is built and the
# MLM encoder is retrained on it.

set -e
set -o pipefail

cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast

module load anaconda
source venv/bin/activate

# Idempotent: install torch CPU wheel + sklearn stack if not already present.
python -c "import torch" 2>/dev/null || pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir matplotlib seaborn scikit-learn xgboost

mkdir -p logs

echo "================================================"
echo "Priority A: Hierarchical Actor Representation"
echo "Start: $(date)"
echo "================================================"

echo "[$(date)] Step 04: build sequences (7-field schema)"
python -u scripts/04_build_sequences.py

echo "[$(date)] Step 05: baseline features"
python -u scripts/05_build_baseline_features.py

echo "[$(date)] Step 06: MLM pre-training (7-field)"
python -u scripts/06_train_representation_model.py

echo "[$(date)] Step 07: downstream classifier"
python -u scripts/07_train_downstream_model.py

echo "[$(date)] Step 08: evaluation"
python -u scripts/08_evaluate_models.py

echo "================================================"
echo "Priority A complete: $(date)"
echo "Compare AUC against MLM baseline 0.916 (5-field)."
echo "================================================"
