#!/bin/bash
#SBATCH --job-name=stage_b_phase1_africa
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/stage_b_phase1_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/logs/stage_b_phase1_%j.err

# Stage B Phase 1: baseline Africa MLM run on GPU.
# This is the first end-to-end Africa-scale Stage B run with the current
# 7-field architecture (hierarchical actors, no cell-cell attention yet,
# no PRIO-GRID covariates in main model, no calibration step).
#
# Goal: produce baseline AUC/Brier numbers to compare against ViEWS.
# Likely outcome: within 0.02-0.05 of ViEWS. If yes, the foundation-model
# framing is alive and Phase 2 iterations close the gap. If no, Phase 2
# needs more aggressive architecture work.

set -e
set -o pipefail

cd /storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast

module load anaconda
source venv/bin/activate

# Switch torch to CUDA wheel if we're not already on one.
# This is a one-time reinstall; subsequent runs are no-ops.
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "Installing CUDA torch wheel..."
    pip install --quiet --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
    python -c "import torch; print(f'torch={torch.__version__}, cuda available={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
}

# Ensure CONFIG points at acled_full.csv
sed -i 's|acled_export\.csv|acled_full.csv|' scripts/00_master.py

echo "================================================"
echo "Stage B Phase 1 — Baseline Africa MLM"
echo "Start: $(date)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)"
echo "================================================"

echo "[$(date)] Step 01: ingest ACLED (Africa region filter)"
python -u scripts/01_ingest_acled.py

echo "[$(date)] Step 01b: ingest UCDP-GED (Africa)"
python -u scripts/01b_ingest_ucdp.py

echo "[$(date)] Step 01c: merge ACLED + UCDP -> unified events"
python -u scripts/01c_merge_sources.py

echo "[$(date)] Step 02: build PRIO-GRID gid"
python -u scripts/02_build_grid.py

echo "[$(date)] Step 03: aggregate to grid-month"
python -u scripts/03_aggregate_location_month.py

echo "[$(date)] Step 04: build sequences (streaming, 7-field schema)"
python -u scripts/04_build_sequences.py

echo "[$(date)] Step 05: baseline features"
python -u scripts/05_build_baseline_features.py

echo "[$(date)] Step 06: MLM pre-training on GPU"
python -u scripts/06_train_representation_model.py

echo "[$(date)] Step 07: downstream classifier"
python -u scripts/07_train_downstream_model.py

echo "[$(date)] Step 08: evaluation (vs ViEWS-style)"
python -u scripts/08_evaluate_models.py

echo "================================================"
echo "Stage B Phase 1 complete: $(date)"
echo "Compare AUC/Brier vs ViEWS published Africa numbers (~0.91-0.93 AUC)"
echo "================================================"
